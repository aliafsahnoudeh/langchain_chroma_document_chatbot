import re
from typing import Pattern

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma_document_chatbot.services.logger import logger
from langchain_chroma_document_chatbot.settings import settings


def add_cors_middleware(app: FastAPI) -> None:
    allowed_origins_pattern: Pattern[str] | str = re.compile(r".*")

    if settings.environment == "dev":
        allowed_origins_pattern = "*"
        logger.info({"message": "CORS: *"})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[allowed_origins_pattern],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
