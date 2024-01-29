import asyncio
from importlib import metadata
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import UJSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma_document_chatbot.middlewares.corsMiddleware import (
    add_cors_middleware,
)
from langchain_chroma_document_chatbot.services.after_load_service import (
    post_run_actions,
)
from langchain_chroma_document_chatbot.services.containers import Container
from langchain_chroma_document_chatbot.services.logger import logger
from langchain_chroma_document_chatbot.web.api.router import api_router
from langchain_chroma_document_chatbot.web.lifetime import (
    register_shutdown_event,
    register_startup_event,
)

APP_ROOT = Path(__file__).parent.parent


def get_app() -> FastAPI:  # noqa: WPS213
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    logger.info({"message": "get_app"})
    container = Container()

    app = FastAPI(
        title="langchain_chroma_document_chatbot",
        version=metadata.version("langchain_chroma_document_chatbot"),
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        default_response_class=UJSONResponse,
    )
    logger.info({"message": "FastAPI initialized"})
    app.container = container  # type: ignore

    add_cors_middleware(app)

    logger.info({"message": "middleware initialized"})

    register_startup_event(app)
    register_shutdown_event(app)

    app.include_router(router=api_router, prefix="/api")
    logger.info(
        {
            "message": "router initialized",
        },
    )

    app.mount(
        "/static",
        StaticFiles(directory=APP_ROOT / "static"),
        name="static",
    )

    logger.info(
        {
            "message": "app mounted",
        },
    )

    loop = asyncio.get_event_loop()
    loop.create_task(post_run_actions())

    return app
