import enum
from pathlib import Path
from tempfile import gettempdir

from pydantic_settings import BaseSettings, SettingsConfigDict

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8080
    workers_count: int = 1
    reload: bool = False
    consolelog: bool = False

    environment: str = "dev"

    log_level: LogLevel = LogLevel.DEBUG

    open_ai_api_key: str = ""
    timeout: int = 3000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="langchain_chroma_document_chatbot_",
        env_file_encoding="utf-8",
    )


settings = Settings()
