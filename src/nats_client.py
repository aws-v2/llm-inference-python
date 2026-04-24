import asyncio
import json
import logging
import nats
from nats.aio.client import Client as NATSClient
from src.config import Config
from src.inference import InferenceEngine

logger = logging.getLogger(__name__)


class NATSWorker:
    def __init__(self, config: Config, engine: InferenceEngine):
        self.config = config
        self.engine = engine
        self.nc: NATSClient | None = None

    async def connect(self):
        logger.info(f"Connecting to NATS at {self.config.nats_url}")

        connect_opts = dict(
            reconnect_time_wait=2,
            max_reconnect_attempts=10,
            error_cb=self._on_error,
            disconnected_cb=self._on_disconnect,
            reconnected_cb=self._on_reconnect,
        )

        # Attach credentials if provided
        if self.config.nats_user and self.config.nats_password:
            connect_opts["user"] = self.config.nats_user
            connect_opts["password"] = self.config.nats_password
            logger.info(f"Using NATS credentials for user: {self.config.nats_user}")

        self.nc = await nats.connect(self.config.nats_url, **connect_opts)
        logger.info("Connected to NATS")

    async def subscribe(self):
        subject = f"{self.config.nats_subject}.llmgateway.task.infer"
        logger.info(f"Subscribing to subject: {subject}")
        await self.nc.subscribe(subject, cb=self._message_handler)
        logger.info(f"Listening on [{subject}]")

    async def _message_handler(self, msg):
        data = {}
        try:
            data = json.loads(msg.data.decode())
            input_text = data.get("input", "")
            request_id = data.get("request_id", "")

            logger.info(f"Received request [{request_id}]")

            output = self.engine.run(input_text)

            response = {
                "output": output,
                "request_id": request_id,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            response = {
                "output": "",
                "request_id": data.get("request_id", ""),
                "error": str(e),
            }

        if msg.reply:
            await self.nc.publish(
                msg.reply,
                json.dumps(response).encode(),
            )

    async def _on_error(self, e):
        logger.error(f"NATS error: {e}")

    async def _on_disconnect(self):
        logger.warning("Disconnected from NATS")

    async def _on_reconnect(self):
        logger.info("Reconnected to NATS")

    async def drain(self):
        if self.nc:
            await self.nc.drain()