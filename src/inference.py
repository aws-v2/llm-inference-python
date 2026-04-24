import logging
from llama_cpp import Llama
from src.config import Config

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, config: Config):
        self.config = config
        self.model: Llama | None = None

    def load(self):
        logger.info(f"Loading model from: {self.config.model_path}")
        try:
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def run(self, input_text: str) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        logger.debug(f"Running inference on input: {input_text[:80]}...")
        try:
            result = self.model(
                input_text,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=["</s>", "\n\n"],
                echo=False,
            )
            output = result["choices"][0]["text"].strip()
            logger.debug(f"Inference result: {output[:80]}...")
            return output
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise