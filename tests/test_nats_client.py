import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.config import Config
from src.nats_client import NATSWorker
from src.inference import InferenceEngine


@pytest.fixture
def config():
    return Config(
        env="dev",
        nats_url="nats://localhost:4222",
        nats_subject="dev.v1.inference",
        model_path="models/tiny-llama.gguf",
        nats_user="test",          # ✅ add this
        nats_password="test",      # ✅ add this
        max_tokens=128,
        temperature=0.7,
        log_level="DEBUG",
        request_timeout=30,
    )


@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=InferenceEngine)
    engine.run.return_value = "test output"
    return engine


@pytest.mark.asyncio
async def test_message_handler_publishes_response(config, mock_engine):
    worker = NATSWorker(config, mock_engine)
    worker.nc = AsyncMock()

    msg = MagicMock()
    msg.data = json.dumps({"input": "hello", "request_id": "abc123"}).encode()
    msg.reply = "reply.subject"

    await worker._message_handler(msg)

    worker.nc.publish.assert_awaited_once()
    call_args = worker.nc.publish.call_args
    response = json.loads(call_args[0][1].decode())

    assert response["output"] == "test output"
    assert response["request_id"] == "abc123"
    assert response["error"] is None


@pytest.mark.asyncio
async def test_message_handler_handles_inference_error(config, mock_engine):
    mock_engine.run.side_effect = Exception("model crash")
    worker = NATSWorker(config, mock_engine)
    worker.nc = AsyncMock()

    msg = MagicMock()
    msg.data = json.dumps({"input": "hello", "request_id": "xyz"}).encode()
    msg.reply = "reply.subject"

    await worker._message_handler(msg)

    call_args = worker.nc.publish.call_args
    response = json.loads(call_args[0][1].decode())

    assert response["error"] == "model crash"
    assert response["output"] == ""