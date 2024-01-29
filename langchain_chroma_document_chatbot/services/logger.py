import logging as std_logging
from typing import Any

from google.cloud import logging  # type: ignore
from langchain_chroma_document_chatbot.settings import settings

std_logging.basicConfig(level=std_logging.INFO)


class Logger:
    logger: Any

    def __init__(self, log_name: str) -> None:
        logging_client = logging.Client()
        self.logger = logging_client.logger(log_name)

    def parse_payload(self, payload: dict) -> dict:  # type: ignore
        for key, value in payload.items():
            if not isinstance(value, str):
                payload[key] = str(value)

        return payload

    def info(self, payload: dict) -> None:  # type: ignore
        self.logger.log_struct(
            info=self.parse_payload(payload),
            severity="info",
        )
        if settings.consolelog:
            std_logging.info(payload)

    def warning(self, payload: dict) -> None:  # type: ignore
        self.logger.log_struct(
            info=self.parse_payload(payload),
            severity="WARNING",
        )
        if settings.consolelog:
            std_logging.info(payload)

    def error(self, payload: dict) -> None:  # type: ignore
        self.logger.log_struct(
            info=self.parse_payload(payload),
            severity="ERROR",
        )
        if settings.consolelog:
            std_logging.info(payload)

    def debug(self, payload: dict) -> None:  # type: ignore
        self.logger.log_struct(
            info=self.parse_payload(payload),
            severity="DEBUG",
        )


logger = Logger("langchain_chroma_document_chatbot")
