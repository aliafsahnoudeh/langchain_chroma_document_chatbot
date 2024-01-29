import uvicorn
from langchain_chroma_document_chatbot.gunicorn_runner import GunicornApplication
from langchain_chroma_document_chatbot.services.logger import logger
from langchain_chroma_document_chatbot.settings import settings


def main() -> None:
    """Entrypoint of the application."""
    try:
        logger.info(
            {
                "message": "Starting application",
                "settings": settings.json(),
            },
        )
        if settings.reload:
            uvicorn.run(
                "langchain_chroma_document_chatbot.web.application:get_app",
                workers=settings.workers_count,
                host=settings.host,
                port=settings.port,
                reload=settings.reload,
                log_level=settings.log_level.value.lower(),
                factory=True,
            )
        else:
            # We choose gunicorn only if reload
            # option is not used, because reload
            # feature doen't work with Uvicorn workers.
            GunicornApplication(
                "langchain_chroma_document_chatbot.web.application:get_app",
                host=settings.host,
                port=settings.port,
                workers=settings.workers_count,
                factory=True,
                accesslog="-",
                loglevel=settings.log_level.value.lower(),
                timeout=settings.timeout,
                access_log_format='%r "-" %s "-" %Tf',  # noqa: WPS323
            ).run()
    except Exception as exception:
        logger.error(
            {
                "message": "Application failed",
                "settings": settings.json(),
                "exception": exception,
            },
        )


if __name__ == "__main__":
    main()
