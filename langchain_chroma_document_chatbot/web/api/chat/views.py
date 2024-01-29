from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse
from langchain_chroma_document_chatbot.models.chatMessage import ChatMessage
from langchain_chroma_document_chatbot.services.chatbot.main import Chatbot
from langchain_chroma_document_chatbot.services.containers import Container
from langchain_chroma_document_chatbot.services.feedback_service import FeedbackService
from langchain_chroma_document_chatbot.services.logger import logger
from pydantic import BaseModel

router = APIRouter()


class ChatBody(BaseModel):
    message: ChatMessage
    chat_history: list[ChatMessage]


class FeedbackBody(BaseModel):
    run_id: str
    score: bool
    feedback_key: str


@router.post(
    "/message",
    tags=["API"],
    response_model=str,
    responses={503: {"detail": "Server is busy, please try again later"}},
)
@inject
def chat(
    data: ChatBody,
    request: Request,
    chat_bot: Chatbot = Depends(Provide[Container.chat_bot]),
) -> StreamingResponse:
    logger.info(
        {
            "message": "resolver chat",
            "data": data,
            "request": request,
        },
    )
    return StreamingResponse(
        chat_bot.chat(
            message=data.message,
        ),
        media_type="text/event-stream",
    )


@router.post("/feedback", tags=["API"], response_model=str)
@inject
def feedback(
    data: FeedbackBody,
    request: Request,
    feedback_service: FeedbackService = Depends(Provide[Container.feedback_service]),
) -> Response:
    logger.info(
        {
            "message": "resolver feedback",
            "data": data,
            "request": request,
        },
    )
    return Response(
        feedback_service.persist(
            score=data.score,
            feedback_key=data.feedback_key,
            run_id=data.run_id,
        ),
    )
