"""Tests for the HardwareProfiler and ModelRouter."""

from unittest.mock import MagicMock, patch

from srtgen.config import HardwareConfig
from srtgen.hardware import HardwareProfile, HardwareProfiler, ModelRouter, setup_thread_safety


def test_hardware_profile_apple_silicon() -> None:
    """Test routing on Apple Silicon with unified memory."""
    profile = HardwareProfile(
        has_cuda=False,
        vram_gb=0.0,
        ram_gb=18.0,
        physical_cores=8,
        is_apple_silicon=True,
    )
    config = HardwareConfig()
    model_cfg = ModelRouter.get_config(profile, config)

    assert model_cfg.device == "cpu"
    assert model_cfg.compute_type == "int8"
    assert model_cfg.model_name == "large-v3-turbo"
    assert model_cfg.cpu_threads == 8


def test_hardware_profile_cuda_high_vram() -> None:
    """Test routing with a beefy NVIDIA GPU."""
    profile = HardwareProfile(
        has_cuda=True,
        vram_gb=8.0,
        ram_gb=32.0,
        physical_cores=12,
        is_apple_silicon=False,
    )
    config = HardwareConfig()
    model_cfg = ModelRouter.get_config(profile, config)

    assert model_cfg.device == "cuda"
    assert model_cfg.compute_type == "float16"
    assert model_cfg.model_name == "large-v3-turbo"


def test_hardware_profile_cuda_low_vram() -> None:
    """Test routing with a smaller NVIDIA GPU."""
    profile = HardwareProfile(
        has_cuda=True,
        vram_gb=3.0,
        ram_gb=16.0,
        physical_cores=6,
        is_apple_silicon=False,
    )
    config = HardwareConfig()
    model_cfg = ModelRouter.get_config(profile, config)

    assert model_cfg.device == "cuda"
    assert model_cfg.compute_type == "int8_float16"
    assert model_cfg.model_name == "large-v3-turbo"


def test_hardware_profile_cpu_only_low_ram() -> None:
    """Test routing for a basic CPU machine with < 16GB RAM."""
    profile = HardwareProfile(
        has_cuda=False,
        vram_gb=0.0,
        ram_gb=8.0,
        physical_cores=4,
        is_apple_silicon=False,
    )
    config = HardwareConfig()
    model_cfg = ModelRouter.get_config(profile, config)

    assert model_cfg.device == "cpu"
    assert model_cfg.compute_type == "int8"
    assert model_cfg.model_name == "small.en"


def test_model_router_overrides() -> None:
    """Test that user configuration strictly overrides the matrix."""
    profile = HardwareProfile(
        has_cuda=True,
        vram_gb=8.0,
        ram_gb=32.0,
        physical_cores=12,
        is_apple_silicon=False,
    )
    overrides = HardwareConfig(
        force_device="cpu",
        force_compute_type="int8",
        force_model="tiny.en",
    )
    model_cfg = ModelRouter.get_config(profile, overrides)

    assert model_cfg.device == "cpu"
    assert model_cfg.compute_type == "int8"
    assert model_cfg.model_name == "tiny.en"


@patch("srtgen.hardware.psutil")
def test_profiler_execution(mock_psutil: MagicMock) -> None:
    """Test the HardwareProfiler actually builds a profile without crashing."""
    mock_psutil.virtual_memory.return_value.total = 16 * (1024**3)
    mock_psutil.cpu_count.return_value = 4

    with patch("srtgen.hardware.platform.system", return_value="Linux"):
        profile = HardwareProfiler.profile()
        assert profile.ram_gb == 16.0
        assert profile.physical_cores == 4
        assert not profile.is_apple_silicon


def test_setup_thread_safety() -> None:
    """Ensure environment variables are set."""
    import os

    setup_thread_safety()
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("MKL_NUM_THREADS") == "1"
