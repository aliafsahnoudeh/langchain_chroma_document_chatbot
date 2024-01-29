from pydantic import BaseModel


class ChatMessage(BaseModel):
    content: str
    role: str
    runId: str | None = None
