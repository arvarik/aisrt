"""Speech-to-text inference and its dedicated worker thread."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, Final

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from loguru import logger

from aisrt.hardware import ModelConfig

# Sequential decoding keeps the temperature ladder that breaks repetition loops.
_TEMPERATURE_LADDER: Final = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

_SEQUENTIAL_VAD: Final = {
    "threshold": 0.5,
    "neg_threshold": 0.35,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": 30.0,
    "min_silence_duration_ms": 1000,
    "speech_pad_ms": 200,
}

# The batched pipeline forces max_speech_duration_s itself, so it is omitted.
_BATCHED_VAD: Final = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 200,
}


def build_transcribe_options(
    batched: bool,
    translate: bool,
    language: str | None,
    batch_size: int = 0,
) -> dict[str, Any]:
    """Build the keyword arguments for one ``transcribe`` call.

    Batched inference accepts several arguments and then discards them, so this
    function passes them only on the sequential path where they take effect.

    Args:
        batched: True when the batched pipeline runs the call.
        translate: True to translate speech into English.
        language: The spoken language, or None to let the model detect it.
        batch_size: Chunks decoded together. Used only when ``batched`` is True.

    Returns:
        The keyword arguments to pass to ``transcribe``.
    """
    options: dict[str, Any] = {
        "task": "translate" if translate else "transcribe",
        "language": language,
        "beam_size": 5,
        "best_of": 5,
        # Word timestamps drive both the subtitle chunker and the hallucination
        # guard, so they are always on.
        "word_timestamps": True,
        "repetition_penalty": 1.05,
        "vad_filter": True,
        "without_timestamps": False,
    }

    if batched:
        options["batch_size"] = max(1, batch_size)
        options["vad_parameters"] = dict(_BATCHED_VAD)
        return options

    options.update(
        {
            "temperature": list(_TEMPERATURE_LADDER),
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            # Films have long silent stretches. Carrying text across them is the
            # classic trigger for a repeated-caption loop.
            "condition_on_previous_text": False,
            "prompt_reset_on_temperature": 0.5,
            # Skip a low-confidence segment that sits alone between long silences.
            "hallucination_silence_threshold": 2.0,
            "max_initial_timestamp": 1.0,
            "vad_parameters": dict(_SEQUENTIAL_VAD),
        }
    )
    return options


class STTWorker:
    """Owns one Whisper model and the single thread that runs inference.

    Inference is serialized onto one thread so that the model is never asked to
    decode two files at once, and so the event loop never blocks on it.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the worker without loading the model.

        Args:
            config: The resolved model settings.
        """
        self.config = config
        self.model: WhisperModel | BatchedInferencePipeline | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aisrt-stt")
        self._lock = threading.Lock()
        self._closed = False
        # The worker thread checks this between segments. Without it, close()
        # would either abandon a running decode or wait out the whole file,
        # which can exceed a service manager's stop timeout.
        self._stop = threading.Event()

    @property
    def model_name(self) -> str:
        """The model that produced a transcript, for the state database."""
        return self.config.model_name

    @property
    def batched(self) -> bool:
        """Whether the loaded model is the batched pipeline."""
        return isinstance(self.model, BatchedInferencePipeline)

    def __enter__(self) -> STTWorker:
        """Load the model."""
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the model and the worker thread."""
        self.close()

    def initialize(self) -> None:
        """Load the model onto the configured device.

        Calling this twice without an intervening close is a no-op, so a
        watch-mode rescan reuses the model already resident in memory.

        Raises:
            RuntimeError: If the worker was already closed. Build a new one.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "This STTWorker was closed and cannot be reused. Create a new one."
                )
            if self.model is not None:
                return
            self._login_to_huggingface()
            logger.info(
                f"Loading {self.config.model_name} on {self.config.device} "
                f"({self.config.compute_type})"
            )
            self.model = self.executor.submit(self._load_model).result()
            logger.info("Model ready.")

    def _load_model(self) -> WhisperModel | BatchedInferencePipeline:
        """Construct the model on the worker thread."""
        model = WhisperModel(
            model_size_or_path=self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads,
            # Inference is already serialized onto one thread, so extra inner
            # workers would only consume memory.
            num_workers=1,
        )
        if self.config.batched:
            logger.info(f"Using batched inference with batch_size={self.config.batch_size}.")
            return BatchedInferencePipeline(model=model)
        return model

    @staticmethod
    def _login_to_huggingface() -> None:
        """Authenticate with Hugging Face when a token is present."""
        token = os.environ.get("HF_TOKEN")
        if not token:
            return
        try:
            from huggingface_hub import login

            login(token=token)
            logger.info("Authenticated with Hugging Face.")
        except Exception as error:
            logger.warning(f"Hugging Face login failed, continuing anonymously: {error}")

    def _base_model(self) -> WhisperModel:
        """Return the underlying model, unwrapping the batched pipeline.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("The Whisper model is not loaded. Call initialize() first.")
        if isinstance(self.model, BatchedInferencePipeline):
            return self.model.model
        return self.model

    def detect_language(self, audio: np.ndarray) -> tuple[str | None, float]:
        """Identify the spoken language from the first minutes of speech.

        The check runs the encoder only, so it costs a fraction of a second. The
        voice activity filter is essential here, because the opening of a film is
        usually a logo or music rather than dialogue.

        Args:
            audio: The decoded audio.

        Returns:
            The language code and its probability. ``(None, 0.0)`` when
            detection failed.
        """
        # A missing model is a wiring error, so it propagates. Only a detection
        # failure falls back to per-window detection inside transcribe().
        model = self._base_model()
        try:
            language, probability, _ = model.detect_language(
                audio,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 1000, "speech_pad_ms": 200},
                language_detection_segments=4,
                language_detection_threshold=0.6,
            )
        except Exception as error:
            logger.debug(f"Language detection failed: {error}")
            return None, 0.0
        return language, float(probability)

    def transcribe(
        self,
        audio: np.ndarray,
        translate: bool,
        language: str | None,
        on_progress: Callable[[float], None] | None = None,
    ) -> tuple[Iterator[Any], float]:
        """Transcribe audio and return the segment stream with the duration.

        The returned iterator is lazy. Consume it on the same thread that called
        this method.

        Args:
            audio: The decoded audio.
            translate: True to translate speech into English.
            language: The spoken language, or None to let the model detect it.
            on_progress: Called with the end time of each finished segment.

        Returns:
            The segment iterator and the audio duration in seconds.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        # Read the reference once. close() may null it from another thread, and
        # a half-finished call must not fail with an attribute error.
        model = self.model
        if model is None:
            raise RuntimeError("The Whisper model is not loaded. Call initialize() first.")

        options = build_transcribe_options(
            batched=isinstance(model, BatchedInferencePipeline),
            translate=translate,
            language=language,
            batch_size=self.config.batch_size,
        )
        segments, info = model.transcribe(audio, **options)

        if on_progress is None:
            return segments, float(info.duration)

        def _tracked() -> Iterator[Any]:
            for segment in segments:
                if self._stop.is_set():
                    # A shutdown asked us to stop. Ending the generator lets the
                    # caller unwind cleanly instead of blocking the exit.
                    logger.debug("Shutdown requested mid-transcription. Stopping early.")
                    return
                on_progress(float(segment.end))
                yield segment

        return _tracked(), float(info.duration)

    def close(self) -> None:
        """Stop any running transcription, then release the model and the thread.

        The stop flag is set first so a decode already in progress unwinds at its
        next segment boundary. The model reference is dropped only after the
        executor has been told to stop, so a call that is still running does not
        fail with a confusing "model is not loaded" error.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.model = None
