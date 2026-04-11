"""Hardware profiling and AI routing for the SRT Generator."""

import os
import platform
from dataclasses import dataclass

import psutil
from loguru import logger

from aisrt.config import HardwareConfig


@dataclass
class HardwareProfile:
    """System hardware capabilities."""

    has_cuda: bool
    vram_gb: float
    ram_gb: float
    physical_cores: int
    is_apple_silicon: bool


@dataclass
class ModelConfig:
    """The resolved configuration for the Whisper model."""

    model_name: str
    device: str
    compute_type: str
    cpu_threads: int


class HardwareProfiler:
    """Profiles system hardware to determine optimal STT model routing."""

    @staticmethod
    def _get_cuda_info() -> tuple[bool, float]:
        """Safely attempt to initialize pynvml to check for CUDA devices and VRAM."""
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                # We target the primary GPU (index 0)
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_gb = info.total / (1024**3)
                pynvml.nvmlShutdown()
                return True, vram_gb
            pynvml.nvmlShutdown()
        except (ImportError, Exception) as e:
            logger.debug(f"CUDA/NVML not available or failed to initialize: {e}")

        return False, 0.0

    @classmethod
    def profile(cls) -> HardwareProfile:
        """Analyze system hardware."""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        physical_cores = psutil.cpu_count(logical=False) or 1

        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        has_cuda, vram_gb = cls._get_cuda_info()

        profile = HardwareProfile(
            has_cuda=has_cuda,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            physical_cores=physical_cores,
            is_apple_silicon=is_apple_silicon,
        )
        logger.info(
            f"Hardware Profile: CUDA={has_cuda} (VRAM={vram_gb:.1f}GB), "
            f"RAM={ram_gb:.1f}GB, Cores={physical_cores}, AppleSilicon={is_apple_silicon}"
        )
        return profile


class ModelRouter:
    """Routes hardware profiles to the optimal STT model configuration."""

    @staticmethod
    def get_config(profile: HardwareProfile, overrides: HardwareConfig) -> ModelConfig:
        """Determine the best model configuration, respecting overrides."""
        # 1. Start with the logic matrix defaults
        if profile.has_cuda and profile.vram_gb >= 6.0:
            model_name = "large-v3-turbo"
            device = "cuda"
            compute_type = "float16"
            cpu_threads = 4  # Minimal threads for GPU orchestration
        elif profile.has_cuda and profile.vram_gb >= 4.0:
            model_name = "large-v3-turbo"
            device = "cuda"
            compute_type = "int8_float16"
            cpu_threads = 4
        elif profile.is_apple_silicon or profile.ram_gb > 16.0:
            model_name = "large-v3-turbo"
            device = "cpu"
            compute_type = "int8"
            cpu_threads = profile.physical_cores
        else:
            model_name = "small.en"
            device = "cpu"
            compute_type = "int8"
            cpu_threads = profile.physical_cores

        # 2. Apply user overrides
        if overrides.force_model:
            model_name = overrides.force_model
        if overrides.force_device:
            device = overrides.force_device
        if overrides.force_compute_type:
            compute_type = overrides.force_compute_type

        config = ModelConfig(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        logger.info(
            f"Routed to Model: {config.model_name}, Device: {config.device}, "
            f"Compute: {config.compute_type}, Threads: {config.cpu_threads}"
        )
        return config


def setup_thread_safety() -> None:
    """Ensure underlying C libraries do not thrash threads."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
