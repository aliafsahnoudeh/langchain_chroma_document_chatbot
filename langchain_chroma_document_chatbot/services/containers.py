"""Containers module."""

from dependency_injector import containers, providers
from langchain_chroma_document_chatbot.services.chatbot.document_chian import (
    DocumentChain,
)
from langchain_chroma_document_chatbot.services.logger import logger

from . import feedback_service
from .chatbot.main import Chatbot


class Container(containers.DeclarativeContainer):
    logger.info({"Container": "initializing"})

    wiring_config = containers.WiringConfiguration(
        modules=[
            "langchain_chroma_document_chatbot.web.api.chat.views",
            "langchain_chroma_document_chatbot.services.after_load_service",
        ],
    )

    config = providers.Configuration(yaml_files=["config.yml"])

    document_chain = providers.Singleton(
        DocumentChain,
    )

    chat_bot = providers.Singleton(
        Chatbot,
        document_chain=document_chain,
    )

    feedback_service = providers.Singleton(feedback_service.FeedbackService)

    logger.info({"Container": "initialized"})
