"""Tests for the speech-to-text worker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aisrt.hardware import ModelConfig
from aisrt.stt import STTWorker, build_transcribe_options


def config(batch_size: int = 0) -> ModelConfig:
    """Build a model config for the tests."""
    return ModelConfig(
        model_name="tiny.en",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        batch_size=batch_size,
    )


class TestTranscribeOptions:
    """The options must match what each decoding path honours."""

    def test_word_timestamps_are_always_on(self) -> None:
        """The subtitle chunker and the hallucination guard both need them."""
        for batched in (True, False):
            assert build_transcribe_options(batched, False, None)["word_timestamps"] is True

    def test_the_task_follows_the_translate_flag(self) -> None:
        """The flag selects the Whisper task."""
        assert build_transcribe_options(False, True, None)["task"] == "translate"
        assert build_transcribe_options(False, False, None)["task"] == "transcribe"

    def test_no_initial_prompt(self) -> None:
        """A style prompt biases the model and can leak into the transcript."""
        assert "initial_prompt" not in build_transcribe_options(False, False, None)
        assert "initial_prompt" not in build_transcribe_options(True, False, None)

    def test_sequential_mode_keeps_every_guard(self) -> None:
        """These arguments only take effect on the sequential path."""
        options = build_transcribe_options(False, False, "en")
        assert options["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert options["condition_on_previous_text"] is False
        assert options["hallucination_silence_threshold"] == 2.0
        assert options["compression_ratio_threshold"] == 2.4
        assert options["log_prob_threshold"] == -1.0
        assert options["no_speech_threshold"] == 0.6

    def test_batched_mode_omits_the_ignored_arguments(self) -> None:
        """Passing an argument the batched pipeline discards is misleading."""
        options = build_transcribe_options(True, False, None, batch_size=16)
        for ignored in (
            "temperature",
            "compression_ratio_threshold",
            "log_prob_threshold",
            "no_speech_threshold",
            "condition_on_previous_text",
            "hallucination_silence_threshold",
        ):
            assert ignored not in options, f"{ignored} is discarded in batched mode"
        assert options["batch_size"] == 16

    def test_batched_vad_omits_max_speech_duration(self) -> None:
        """The batched pipeline forces this value and drops whatever is passed."""
        options = build_transcribe_options(True, False, None, batch_size=8)
        assert "max_speech_duration_s" not in options["vad_parameters"]

    def test_the_language_is_passed_through(self) -> None:
        """Pinning the language stops per-window re-detection."""
        assert build_transcribe_options(False, False, "ja")["language"] == "ja"
        assert build_transcribe_options(False, False, None)["language"] is None


class TestWorkerLifecycle:
    """The worker owns one model and one thread, explicitly."""

    @patch("aisrt.stt.WhisperModel")
    def test_loads_with_the_resolved_settings(self, whisper: MagicMock) -> None:
        """Every routed value reaches the model constructor."""
        worker = STTWorker(config())
        worker.initialize()

        whisper.assert_called_once_with(
            model_size_or_path="tiny.en",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=1,
        )
        worker.close()

    @patch("aisrt.stt.WhisperModel")
    def test_loading_twice_is_a_no_op(self, whisper: MagicMock) -> None:
        """Watch mode reuses the model already in memory."""
        worker = STTWorker(config())
        worker.initialize()
        worker.initialize()
        assert whisper.call_count == 1
        worker.close()

    @patch("aisrt.stt.BatchedInferencePipeline")
    @patch("aisrt.stt.WhisperModel")
    def test_batched_mode_wraps_the_model(self, whisper: MagicMock, batched: MagicMock) -> None:
        """A non-zero batch size selects the batched pipeline."""
        worker = STTWorker(config(batch_size=8))
        worker.initialize()
        batched.assert_called_once_with(model=whisper.return_value)
        worker.close()

    @patch("aisrt.stt.BatchedInferencePipeline")
    @patch("aisrt.stt.WhisperModel")
    def test_sequential_mode_uses_the_bare_model(
        self, whisper: MagicMock, batched: MagicMock
    ) -> None:
        """A zero batch size keeps the sequential decoder."""
        worker = STTWorker(config(batch_size=0))
        worker.initialize()
        batched.assert_not_called()
        assert worker.model is whisper.return_value
        worker.close()

    @patch("aisrt.stt.WhisperModel")
    def test_the_context_manager_loads_and_releases(self, whisper: MagicMock) -> None:
        """Using the worker as a context manager is the intended lifecycle."""
        with STTWorker(config()) as worker:
            assert worker.model is not None
        assert worker.model is None

    @patch("aisrt.stt.WhisperModel")
    def test_two_workers_are_independent(self, whisper: MagicMock) -> None:
        """There is no hidden global, so a test cannot leak into the next one."""
        first = STTWorker(config())
        second = STTWorker(config())
        assert first is not second
        first.initialize()
        assert second.model is None
        first.close()

    def test_transcribing_before_loading_raises(self) -> None:
        """The failure is explicit, not one FAILED row per file."""
        worker = STTWorker(config())
        with pytest.raises(RuntimeError, match="not loaded"):
            worker.transcribe(np.zeros(16000, dtype=np.float32), False, None)

    @patch("aisrt.stt.WhisperModel")
    def test_the_model_name_is_recorded(self, whisper: MagicMock) -> None:
        """The state database stores which model produced the subtitle."""
        worker = STTWorker(config())
        assert worker.model_name == "tiny.en"
        worker.close()


class TestHuggingFaceLogin:
    """A token is optional, and a failure must not stop the run."""

    @patch("aisrt.stt.WhisperModel")
    def test_no_token_means_no_login(
        self, whisper: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An anonymous download is the default."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        login = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(login=login)}):
            STTWorker(config()).initialize()
        login.assert_not_called()

    @patch("aisrt.stt.WhisperModel")
    def test_a_failed_login_does_not_stop_the_run(
        self, whisper: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bad token falls back to an anonymous download."""
        monkeypatch.setenv("HF_TOKEN", "bad-token")
        hub = MagicMock()
        hub.login.side_effect = RuntimeError("401 Unauthorized")
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            worker = STTWorker(config())
            worker.initialize()
        assert worker.model is not None
        worker.close()


class TestLanguageDetection:
    """Detection runs the encoder only, and never breaks the run."""

    @patch("aisrt.stt.WhisperModel")
    def test_returns_the_detected_language(self, whisper: MagicMock) -> None:
        """The code and its probability come straight from the model."""
        whisper.return_value.detect_language.return_value = ("ja", 0.93, [])
        worker = STTWorker(config())
        worker.initialize()

        language, probability = worker.detect_language(np.zeros(16000, dtype=np.float32))

        assert language == "ja"
        assert probability == pytest.approx(0.93)
        assert whisper.return_value.detect_language.call_args.kwargs["vad_filter"] is True
        worker.close()

    @patch("aisrt.stt.WhisperModel")
    def test_a_failure_is_reported_as_unknown(self, whisper: MagicMock) -> None:
        """A detection error falls back to per-window detection."""
        whisper.return_value.detect_language.side_effect = RuntimeError("boom")
        worker = STTWorker(config())
        worker.initialize()

        assert worker.detect_language(np.zeros(16000, dtype=np.float32)) == (None, 0.0)
        worker.close()


class TestTranscribe:
    """Transcription reports progress and returns the duration."""

    @patch("aisrt.stt.WhisperModel")
    def test_progress_is_reported_per_segment(self, whisper: MagicMock) -> None:
        """The caller learns how far into the audio the model has reached."""
        segments = [MagicMock(end=1.0), MagicMock(end=2.5), MagicMock(end=4.0)]
        whisper.return_value.transcribe.return_value = (iter(segments), MagicMock(duration=4.0))

        worker = STTWorker(config())
        worker.initialize()
        seen: list[float] = []
        stream, duration = worker.transcribe(
            np.zeros(16000, dtype=np.float32), False, "en", on_progress=seen.append
        )
        consumed = list(stream)

        assert len(consumed) == 3
        assert seen == [1.0, 2.5, 4.0]
        assert duration == 4.0
        worker.close()

    @patch("aisrt.stt.WhisperModel")
    def test_works_without_a_progress_callback(self, whisper: MagicMock) -> None:
        """Progress reporting is optional."""
        whisper.return_value.transcribe.return_value = (iter([]), MagicMock(duration=0.0))
        worker = STTWorker(config())
        worker.initialize()
        stream, duration = worker.transcribe(np.zeros(16000, dtype=np.float32), False, None)
        assert list(stream) == []
        assert duration == 0.0
        worker.close()
