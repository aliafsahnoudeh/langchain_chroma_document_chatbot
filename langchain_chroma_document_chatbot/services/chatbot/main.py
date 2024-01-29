import json
from typing import Any

from langchain import callbacks
from langchain_chroma_document_chatbot.models.chatMessage import ChatMessage
from langchain_chroma_document_chatbot.models.tracingProject import TracingProject
from langchain_chroma_document_chatbot.services.chatbot.document_chian import (
    DocumentChain,
)
from langchain_chroma_document_chatbot.services.logger import logger


class Chatbot:
    document_chain: DocumentChain
    general_chain: Any
    topic_chain: Any

    def __init__(self, document_chain: DocumentChain) -> None:
        self.document_chain = document_chain

    def chat(
        self,
        message: ChatMessage,
    ) -> Any:  # type: ignore
        try:
            with callbacks.collect_runs() as cb:  # type: ignore
                with callbacks.tracing_v2_enabled(
                    project_name=TracingProject.langchain_chroma_document_chatbot,
                ):
                    logger.info(
                        {
                            "message": "Chatbot chat",
                            "service": "Chatbot",
                            "method": "chat",
                            "question": message.content,
                        },
                    )
                    chain = self.document_chain.get_chain()
                    for document_chunk in chain.stream(message.content):
                        yield json.dumps(
                            {
                                "content": document_chunk,
                                "runId": "",
                                "role": "ASSISTANT",
                            },
                        )

                    run_id = str(cb.traced_runs[0].id)

                    yield json.dumps(
                        {"content": "", "runId": str(run_id), "role": "ASSISTANT"},
                    )

        except Exception as exception:
            logger.error(
                {
                    "message": "Failed to chat",
                    "service": "Chatbot",
                    "method": "chat",
                    "error": str(exception),
                },
            )
            yield json.dumps(
                {
                    "content": "Sorry I couldn't figure that out.",
                    "runId": "",
                    "role": "ASSISTANT",
                },
            )
