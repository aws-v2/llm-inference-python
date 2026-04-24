import asyncio
import logging
import signal
import sys
from src.config import load_config
from src.inference import InferenceEngine
from src.nats_client import NATSWorker


def setup_logging(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def run():
    config = load_config()
    setup_logging(config.log_level)

    logger = logging.getLogger("main")
    logger.info(f"Starting Python inference worker | ENV={config.env}")

    # Load model
    engine = InferenceEngine(config)
    engine.load()

    # Start NATS worker
    worker = NATSWorker(config, engine)
    await worker.connect()
    await worker.subscribe()

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    logger.info("Worker running. Waiting for messages...")
    await stop_event.wait()

    logger.info("Draining NATS connection...")
    await worker.drain()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(run())