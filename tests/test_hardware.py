"""Tests for hardware profiling and model routing."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from aisrt.config import HardwareConfig
from aisrt.hardware import (
    LARGE_MODEL,
    MEDIUM_MODEL,
    SMALL_EN_MODEL,
    SMALL_MODEL,
    TURBO_MODEL,
    HardwareProfile,
    HardwareProfiler,
    ModelRouter,
    supports_translation,
)


def profile(
    has_cuda: bool = False,
    vram_gb: float = 0.0,
    ram_gb: float = 32.0,
    cores: int = 8,
    apple: bool = False,
) -> HardwareProfile:
    """Build a hardware profile for one routing case."""
    return HardwareProfile(
        has_cuda=has_cuda,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        physical_cores=cores,
        is_apple_silicon=apple,
    )


@pytest.fixture
def permissive_compute_types() -> Iterator[None]:
    """Accept every compute type, so routing tests do not depend on the host."""
    with patch("aisrt.hardware._validate_compute_type", side_effect=lambda _d, c: c):
        yield


class TestProfiler:
    """Profiling must never raise, whatever the machine."""

    def test_no_cuda_when_ctranslate2_sees_no_device(self) -> None:
        """CTranslate2 is the authority on whether the GPU is usable."""
        with patch("ctranslate2.get_cuda_device_count", return_value=0):
            has_cuda, vram, name, count = HardwareProfiler._get_cuda_info()
        assert (has_cuda, vram, name, count) == (False, 0.0, None, 0)

    def test_no_cuda_when_ctranslate2_raises(self) -> None:
        """A broken CUDA install reports no GPU instead of crashing."""
        with patch("ctranslate2.get_cuda_device_count", side_effect=RuntimeError("no driver")):
            has_cuda, vram, _name, _count = HardwareProfiler._get_cuda_info()
        assert has_cuda is False
        assert vram == 0.0

    def test_reads_vram_from_the_largest_gpu(self) -> None:
        """With two GPUs, routing uses the one with the most memory."""
        fake_nvml = MagicMock()
        fake_nvml.nvmlDeviceGetCount.return_value = 2
        fake_nvml.nvmlDeviceGetMemoryInfo.side_effect = [
            MagicMock(total=8 * 1024**3),
            MagicMock(total=24 * 1024**3),
        ]
        fake_nvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 4090"

        with (
            patch("ctranslate2.get_cuda_device_count", return_value=2),
            patch.dict("sys.modules", {"pynvml": fake_nvml}),
        ):
            has_cuda, vram, name, count = HardwareProfiler._get_cuda_info()

        assert has_cuda is True
        assert vram == pytest.approx(24.0)
        assert name == "NVIDIA RTX 4090"
        assert count == 2

    def test_a_cuda_device_without_nvml_still_counts(self) -> None:
        """A missing NVML costs the VRAM reading, not the GPU."""
        with (
            patch("ctranslate2.get_cuda_device_count", return_value=1),
            patch.dict("sys.modules", {"pynvml": None}),
        ):
            has_cuda, vram, _name, count = HardwareProfiler._get_cuda_info()

        assert has_cuda is True
        assert vram == 0.0
        assert count == 1

    def test_profile_returns_a_usable_result(self) -> None:
        """A full profile always has at least one core and some memory."""
        with patch("ctranslate2.get_cuda_device_count", return_value=0):
            result = HardwareProfiler.profile()
        assert result.physical_cores >= 1
        assert result.ram_gb > 0


@pytest.mark.usefixtures("permissive_compute_types")
class TestRouting:
    """The routing table must match the documented matrix."""

    @pytest.mark.parametrize(
        ("machine", "model", "device", "compute"),
        [
            (profile(True, 24.0), LARGE_MODEL, "cuda", "float16"),
            (profile(True, 10.0), LARGE_MODEL, "cuda", "float16"),
            (profile(True, 8.0), LARGE_MODEL, "cuda", "int8_float16"),
            (profile(True, 6.0), TURBO_MODEL, "cuda", "float16"),
            (profile(True, 4.0), TURBO_MODEL, "cuda", "int8_float16"),
            (profile(True, 2.0), SMALL_EN_MODEL, "cuda", "int8_float16"),
            (profile(ram_gb=32.0), TURBO_MODEL, "cpu", "int8"),
            (profile(ram_gb=8.0), SMALL_EN_MODEL, "cpu", "int8"),
        ],
    )
    def test_transcribe_routing(
        self, machine: HardwareProfile, model: str, device: str, compute: str
    ) -> None:
        """Each tier resolves to its documented model and precision."""
        config = ModelRouter.get_config(machine, HardwareConfig())
        assert config.model_name == model
        assert config.device == device
        assert config.compute_type == compute

    def test_apple_silicon_runs_on_the_cpu(self) -> None:
        """CTranslate2 has no Metal backend, so Apple Silicon uses the CPU."""
        config = ModelRouter.get_config(profile(ram_gb=32.0, apple=True), HardwareConfig())
        assert config.device == "cpu"
        assert config.compute_type == "int8"

    def test_apple_silicon_with_little_memory_gets_a_small_model(self) -> None:
        """Memory is tested before the platform, so an 8 GB machine is safe."""
        config = ModelRouter.get_config(profile(ram_gb=8.0, apple=True), HardwareConfig())
        assert config.model_name == SMALL_EN_MODEL


@pytest.mark.usefixtures("permissive_compute_types")
class TestTranslationRouting:
    """Turbo returns the original language, so it must not be used to translate."""

    @pytest.mark.parametrize("machine", [profile(True, 6.0), profile(True, 4.0)])
    def test_a_turbo_tier_switches_to_a_multilingual_model(self, machine: HardwareProfile) -> None:
        """A translate run picks a checkpoint that can actually translate."""
        config = ModelRouter.get_config(machine, HardwareConfig(), translate=True)
        assert config.model_name == MEDIUM_MODEL
        assert supports_translation(config.model_name)

    def test_the_cpu_tier_switches_too(self) -> None:
        """The same rule applies without a GPU."""
        config = ModelRouter.get_config(profile(ram_gb=32.0), HardwareConfig(), translate=True)
        assert config.model_name == MEDIUM_MODEL

    def test_an_english_only_fallback_becomes_multilingual(self) -> None:
        """small.en cannot read another language at all."""
        config = ModelRouter.get_config(profile(ram_gb=8.0), HardwareConfig(), translate=True)
        assert config.model_name == SMALL_MODEL

    def test_the_large_tier_already_translates(self) -> None:
        """A large checkpoint needs no substitution."""
        config = ModelRouter.get_config(profile(True, 24.0), HardwareConfig(), translate=True)
        assert config.model_name == LARGE_MODEL

    @pytest.mark.parametrize(
        ("model", "capable"),
        [
            ("large-v3", True),
            ("medium", True),
            ("small", True),
            ("large-v3-turbo", False),
            ("turbo", False),
            ("small.en", False),
            ("distil-large-v3.5", False),
            ("/models/my-custom-model", True),
        ],
    )
    def test_capability_table(self, model: str, capable: bool) -> None:
        """Turbo, English-only, and distilled checkpoints cannot translate."""
        assert supports_translation(model) is capable


@pytest.mark.usefixtures("permissive_compute_types")
class TestOverrides:
    """A user override always wins over the routing table."""

    def test_model_device_and_compute_type(self) -> None:
        """Every forced value reaches the resolved config."""
        overrides = HardwareConfig(
            force_model="tiny.en", force_device="cpu", force_compute_type="float32"
        )
        config = ModelRouter.get_config(profile(True, 24.0), overrides)
        assert config.model_name == "tiny.en"
        assert config.device == "cpu"
        assert config.compute_type == "float32"

    def test_forcing_the_cpu_drops_a_gpu_precision(self) -> None:
        """float16 belongs to the GPU tier and must not follow the device change."""
        config = ModelRouter.get_config(profile(True, 24.0), HardwareConfig(force_device="cpu"))
        assert config.device == "cpu"
        assert config.compute_type == "int8"

    def test_auto_keeps_the_routed_device(self) -> None:
        """'auto' means "do not override"."""
        config = ModelRouter.get_config(profile(True, 24.0), HardwareConfig(force_device="auto"))
        assert config.device == "cuda"

    def test_accuracy_mode_disables_batching(self) -> None:
        """Sequential decoding keeps the guards that batching discards."""
        config = ModelRouter.get_config(profile(True, 24.0), HardwareConfig(prefer_accuracy=True))
        assert config.batch_size == 0
        assert config.batched is False

    def test_an_explicit_batch_size_enables_batching(self) -> None:
        """Asking for throughput turns the batched pipeline on."""
        config = ModelRouter.get_config(
            profile(True, 24.0), HardwareConfig(batch_size=8, prefer_accuracy=False)
        )
        assert config.batch_size == 8
        assert config.batched is True

    def test_cpu_threads_never_exceed_the_core_count(self) -> None:
        """Oversubscribing the CPU only slows inference down."""
        config = ModelRouter.get_config(
            profile(ram_gb=32.0, cores=4), HardwareConfig(force_device="cpu")
        )
        assert config.cpu_threads <= 4


class TestComputeTypeValidation:
    """An unsupported precision must fall back instead of crashing at load."""

    def test_falls_back_to_a_supported_type(self) -> None:
        """CTranslate2's supported list decides what is usable."""
        from aisrt.hardware import _validate_compute_type

        with patch("ctranslate2.get_supported_compute_types", return_value={"int8", "float32"}):
            assert _validate_compute_type("cpu", "float16") == "int8"

    def test_keeps_a_supported_type(self) -> None:
        """A valid precision passes through untouched."""
        from aisrt.hardware import _validate_compute_type

        with patch("ctranslate2.get_supported_compute_types", return_value={"int8", "float32"}):
            assert _validate_compute_type("cpu", "int8") == "int8"

    def test_an_unavailable_device_is_not_fatal(self) -> None:
        """Asking CUDA on a CPU-only host must not raise during routing."""
        from aisrt.hardware import _validate_compute_type

        with patch("ctranslate2.get_supported_compute_types", side_effect=ValueError("no cuda")):
            assert _validate_compute_type("cuda", "float16") == "float16"
