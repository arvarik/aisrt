"""Tests for the HardwareProfiler and ModelRouter."""

from unittest.mock import MagicMock, patch

from aisrt.config import HardwareConfig
from aisrt.hardware import HardwareProfile, HardwareProfiler, ModelRouter


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
        vram_gb=4.0,
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


@patch("aisrt.hardware.psutil")
def test_profiler_execution(mock_psutil: MagicMock) -> None:
    """Test the HardwareProfiler actually builds a profile without crashing."""
    mock_psutil.virtual_memory.return_value.total = 16 * (1024**3)
    mock_psutil.cpu_count.return_value = 4

    with patch("aisrt.hardware.platform.system", return_value="Linux"):
        profile = HardwareProfiler.profile()
        assert profile.ram_gb == 16.0
        assert profile.physical_cores == 4
        assert not profile.is_apple_silicon


def test_get_cuda_info_import_error() -> None:
    """Test _get_cuda_info when pynvml is not installed."""
    with patch.dict("sys.modules", {"pynvml": None}):
        has_cuda, vram = HardwareProfiler._get_cuda_info()
        assert has_cuda is False
        assert vram == 0.0


def test_get_cuda_info_init_failure() -> None:
    """Test _get_cuda_info when nvmlInit fails."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = Exception("Initialization failed")

    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        has_cuda, vram = HardwareProfiler._get_cuda_info()
        assert has_cuda is False
        assert vram == 0.0
        mock_pynvml.nvmlInit.assert_called_once()
        # Shutdown should NOT be called if Init failed
        mock_pynvml.nvmlShutdown.assert_not_called()


def test_get_cuda_info_get_count_failure() -> None:
    """Test _get_cuda_info when nvmlDeviceGetCount fails."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.side_effect = Exception("GetCount failed")

    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        has_cuda, vram = HardwareProfiler._get_cuda_info()
        assert has_cuda is False
        assert vram == 0.0
        mock_pynvml.nvmlInit.assert_called_once()
        # Shutdown SHOULD be called if Init succeeded but subsequent call failed
        mock_pynvml.nvmlShutdown.assert_called_once()
