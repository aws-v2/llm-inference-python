import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    env: str
    nats_url: str
    nats_subject: str
    nats_user: str
    nats_password: str
    model_path: str
    max_tokens: int
    temperature: float
    log_level: str
    request_timeout: int


def load_config() -> Config:
    env = os.getenv("ENV", "dev")

    profile_path = os.path.join(
        os.path.dirname(__file__), f"../profiles/{env}.env"
    )
    load_dotenv(dotenv_path=profile_path, override=False)

    nats_prefix = os.getenv("NATS_PREFIX", f"{env}.v1")

    return Config(
        env=env,
        nats_url=os.getenv("NATS_URL", "nats://localhost:4222"),
        nats_subject=f"{nats_prefix}",
        nats_user=os.getenv("NATS_USER", ""),
        nats_password=os.getenv("NATS_PASSWORD", ""),
        model_path=os.getenv("MODEL_PATH", "models/tiny-llama.gguf"),
        max_tokens=int(os.getenv("MAX_TOKENS", "512")),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        log_level=os.getenv("LOG_LEVEL", "DEBUG"),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
    )