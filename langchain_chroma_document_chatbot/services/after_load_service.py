import asyncio

from dependency_injector.wiring import Provide, inject
from fastapi import Depends
from langchain_chroma_document_chatbot.services.chatbot.document_chian import (
    DocumentChain,
)
from langchain_chroma_document_chatbot.services.containers import Container
from langchain_chroma_document_chatbot.services.logger import logger


@inject
async def post_run_actions(  # type: ignore
    document_chain: DocumentChain = Depends(Provide[Container.document_chain]),
):
    logger.info({"message": "Running post run actions"})
    loop = asyncio.get_event_loop()
    loop.create_task(document_chain.load())
    logger.info({"message": "Post run actions finished successfully"})
