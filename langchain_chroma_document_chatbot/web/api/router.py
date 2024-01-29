from fastapi.routing import APIRouter
from langchain_chroma_document_chatbot.web.api import chat, docs, monitoring

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(docs.router)
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
