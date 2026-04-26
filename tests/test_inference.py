import pytest
from unittest.mock import MagicMock, patch
from src.config import Config
from src.inference import InferenceEngine


@pytest.fixture
def config():
    return Config(
        env="dev",
        nats_url="nats://localhost:4222",
        nats_subject="dev.v1.inference",
        nats_user="test",          # ✅ add this
        nats_password="test",      # ✅ add this
        model_path="models/tiny-llama.gguf",
        max_tokens=128,
        temperature=0.7,
        log_level="DEBUG",
        request_timeout=30,
    )


def test_run_raises_if_model_not_loaded(config):
    engine = InferenceEngine(config)
    with pytest.raises(RuntimeError, match="Model not loaded"):
        engine.run("hello")


@patch("src.inference.Llama")
def test_load_and_run(mock_llama, config):
    mock_instance = MagicMock()
    mock_instance.return_value = {
        "choices": [{"text": "  mocked output  "}]
    }
    mock_llama.return_value = mock_instance

    engine = InferenceEngine(config)
    engine.load()
    result = engine.run("What is 1+1?")

    assert result == "mocked output"