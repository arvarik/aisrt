"""Tests for the STT Worker wrapper."""

from unittest.mock import MagicMock, patch

from aisrt.hardware import ModelConfig
from aisrt.stt import STTWorker


@patch("aisrt.stt.WhisperModel")
def test_stt_worker_singleton(mock_whisper_model: MagicMock) -> None:
    """Test that STTWorker behaves as a singleton and initializes correctly."""
    # Reset singleton for clean test
    STTWorker._instance = None

    worker1 = STTWorker()
    worker2 = STTWorker()

    assert worker1 is worker2
    assert worker1.model is None

    config = ModelConfig(
        model_name="tiny.en",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
    )

    worker1.initialize(config)

    # Model should be loaded
    assert worker1.model is not None
    mock_whisper_model.assert_called_once_with(
        model_size_or_path="tiny.en",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )

    # Second initialization should do nothing
    worker2.initialize(config)
    assert mock_whisper_model.call_count == 1

    worker1.close()
