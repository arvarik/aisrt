"""Hardware profiling and model routing."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Final

import psutil
from loguru import logger

from aisrt.config import HardwareConfig

TURBO_MODEL: Final = "large-v3-turbo"
"""Fast model. It transcribes well but cannot translate, so the router only
picks it for a transcribe task."""

LARGE_MODEL: Final = "large-v3"
MEDIUM_MODEL: Final = "medium"
SMALL_EN_MODEL: Final = "small.en"
SMALL_MODEL: Final = "small"


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """What the machine can run."""

    has_cuda: bool
    vram_gb: float
    ram_gb: float
    physical_cores: int
    is_apple_silicon: bool
    gpu_name: str | None = None
    cuda_device_count: int = 0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """The resolved settings for one Whisper model."""

    model_name: str
    device: str
    compute_type: str
    cpu_threads: int
    batch_size: int = 0
    """Batch size for batched inference. 0 means run sequentially."""

    @property
    def batched(self) -> bool:
        """Whether inference should use the batched pipeline."""
        return self.batch_size > 0


class HardwareProfiler:
    """Reads the machine's compute capability."""

    @staticmethod
    def _get_cuda_info() -> tuple[bool, float, str | None, int]:
        """Read the CUDA device count and the VRAM of the largest device.

        CTranslate2 is asked first, because a driver that NVML can see is
        useless when CTranslate2 was not built against it.

        Returns:
            A tuple of (has_cuda, vram_gb, gpu_name, device_count).
        """
        device_count = 0
        try:
            import ctranslate2

            device_count = ctranslate2.get_cuda_device_count()
        except Exception as error:
            logger.debug(f"CTranslate2 reports no usable CUDA device: {error}")
            return False, 0.0, None, 0

        if device_count <= 0:
            return False, 0.0, None, 0

        vram_gb = 0.0
        gpu_name: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                for index in range(pynvml.nvmlDeviceGetCount()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    total = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
                    if total > vram_gb:
                        vram_gb = total
                        name = pynvml.nvmlDeviceGetName(handle)
                        gpu_name = name.decode() if isinstance(name, bytes) else str(name)
            finally:
                pynvml.nvmlShutdown()
        except Exception as error:
            logger.debug(f"NVML unavailable, so VRAM is unknown: {error}")

        return True, vram_gb, gpu_name, device_count

    @classmethod
    def profile(cls) -> HardwareProfile:
        """Inspect the machine and log what was found."""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        physical_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        has_cuda, vram_gb, gpu_name, device_count = cls._get_cuda_info()

        profile = HardwareProfile(
            has_cuda=has_cuda,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            physical_cores=physical_cores,
            is_apple_silicon=is_apple_silicon,
            gpu_name=gpu_name,
            cuda_device_count=device_count,
        )

        if has_cuda:
            logger.info(
                f"Hardware: CUDA {gpu_name or 'GPU'} x{device_count}, "
                f"{vram_gb:.1f} GB VRAM, {ram_gb:.1f} GB RAM, {physical_cores} cores"
            )
        elif is_apple_silicon:
            # CTranslate2 has no Metal backend, so Apple Silicon runs on the CPU.
            logger.info(
                f"Hardware: Apple Silicon, {ram_gb:.1f} GB RAM, {physical_cores} cores. "
                "CTranslate2 has no Metal backend, so inference runs on the CPU."
            )
        else:
            logger.info(f"Hardware: CPU only, {ram_gb:.1f} GB RAM, {physical_cores} cores")
        return profile


class ModelRouter:
    """Picks the model, device, and precision that fit the machine and the task."""

    @staticmethod
    def get_config(
        profile: HardwareProfile,
        overrides: HardwareConfig,
        translate: bool = False,
    ) -> ModelConfig:
        """Resolve the model settings.

        Args:
            profile: What the machine can run.
            overrides: User settings that win over the routing table.
            translate: True when the run must translate speech into English. The
                turbo and ``*.en`` checkpoints cannot do this, so the router
                picks a multilingual checkpoint instead.

        Returns:
            The resolved settings.
        """
        if profile.has_cuda and profile.vram_gb >= 10.0:
            model_name = LARGE_MODEL
            device, compute_type, cpu_threads, batch_size = "cuda", "float16", 4, 16
        elif profile.has_cuda and profile.vram_gb >= 8.0:
            model_name = LARGE_MODEL
            device, compute_type, cpu_threads, batch_size = "cuda", "int8_float16", 4, 8
        elif profile.has_cuda and profile.vram_gb >= 6.0:
            model_name = TURBO_MODEL if not translate else MEDIUM_MODEL
            device, compute_type, cpu_threads, batch_size = "cuda", "float16", 4, 8
        elif profile.has_cuda and profile.vram_gb >= 4.0:
            model_name = TURBO_MODEL if not translate else MEDIUM_MODEL
            device, compute_type, cpu_threads, batch_size = "cuda", "int8_float16", 4, 4
        elif profile.has_cuda:
            model_name = SMALL_EN_MODEL if not translate else SMALL_MODEL
            device, compute_type, cpu_threads, batch_size = "cuda", "int8_float16", 4, 4
        elif profile.ram_gb >= 16.0:
            model_name = TURBO_MODEL if not translate else MEDIUM_MODEL
            device, compute_type = "cpu", "int8"
            cpu_threads, batch_size = profile.physical_cores, 8
        else:
            model_name = SMALL_EN_MODEL if not translate else SMALL_MODEL
            device, compute_type = "cpu", "int8"
            cpu_threads, batch_size = profile.physical_cores, 4

        routed_device = device
        if overrides.force_model:
            model_name = overrides.force_model
        if overrides.force_device and overrides.force_device != "auto":
            device = overrides.force_device
        if overrides.force_compute_type:
            compute_type = overrides.force_compute_type
        elif device != routed_device:
            # The routed precision belongs to the routed device. Pick one the
            # forced device definitely supports instead.
            compute_type = "float16" if device == "cuda" else "int8"
        if overrides.batch_size:
            batch_size = overrides.batch_size
        if overrides.prefer_accuracy:
            # Sequential decoding keeps the temperature fallback and the
            # hallucination guard that the batched pipeline discards.
            batch_size = 0
        if device == "cpu":
            cpu_threads = max(1, min(cpu_threads, profile.physical_cores))

        compute_type = _validate_compute_type(device, compute_type)

        if translate and not supports_translation(model_name):
            logger.warning(
                f"Model '{model_name}' is not trained for translation. It returns the "
                "original language. Use --force-model large-v3 or medium to translate."
            )

        config = ModelConfig(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            batch_size=batch_size,
        )
        mode = f"batched x{batch_size}" if config.batched else "sequential"
        logger.info(
            f"Model: {config.model_name} on {config.device} "
            f"({config.compute_type}, {config.cpu_threads} threads, {mode})"
        )
        return config


def supports_translation(model_name: str) -> bool:
    """Report whether a checkpoint can translate speech into English.

    Args:
        model_name: A model alias or a path to a local model directory.

    Returns:
        False for the turbo checkpoint and for every English-only checkpoint.
        A local path is assumed capable, because its training is unknown.
    """
    name = model_name.strip().lower()
    if "/" in name or "\\" in name:
        return True
    return not (name.endswith(".en") or "turbo" in name or name.startswith("distil"))


def _validate_compute_type(device: str, compute_type: str) -> str:
    """Check the precision against CTranslate2 and fall back when unsupported.

    Args:
        device: ``"cuda"`` or ``"cpu"``.
        compute_type: The requested precision.

    Returns:
        The requested precision, or a supported one when it is unavailable.
    """
    try:
        import ctranslate2

        supported = ctranslate2.get_supported_compute_types(device)
    except Exception as error:
        logger.debug(f"Cannot list compute types for {device}: {error}")
        return compute_type

    if compute_type in supported:
        return compute_type

    for candidate in ("int8_float16", "float16", "int8_float32", "int8", "float32"):
        if candidate in supported:
            logger.warning(
                f"Compute type '{compute_type}' is unsupported on {device}. "
                f"Using '{candidate}' instead."
            )
            return candidate
    return compute_type
