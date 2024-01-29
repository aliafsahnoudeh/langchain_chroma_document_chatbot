# langchain_chroma_document_chatbot

This project was generated using fastapi_template.

## Poetry

This project uses poetry.

To run the project use this set of commands:

```bash
poetry install
poetry run python -m langchain_chroma_document_chatbot
```

This will start the server on the configured host.

You can find swagger documentation at `/api/docs`.

You can read more about poetry here: https://python-poetry.org/

## Docker

You can start the project with docker using this command:

```bash
docker-compose -f deploy/docker-compose.yml --project-directory . up --build
```

If you want to develop in docker with autoreload add `-f deploy/docker-compose.dev.yml` to your docker command.
Like this:

```bash
docker-compose -f deploy/docker-compose.yml -f deploy/docker-compose.dev.yml --project-directory . up --build
```

This command exposes the web application on port 8000, mounts current directory and enables autoreload.

But you have to rebuild image every time you modify `poetry.lock` or `pyproject.toml` with this command:

```bash
docker-compose -f deploy/docker-compose.yml --project-directory . build
```

## Project structure

```bash
$ tree "langchain_chroma_document_chatbot"
langchain_chroma_document_chatbot
├── conftest.py  # Fixtures for all tests.
├── db  # module contains db configurations
│   ├── dao  # Data Access Objects. Contains different classes to interact with database.
│   └── models  # Package contains different models for ORMs.
├── __main__.py  # Startup script. Starts uvicorn.
├── services  # Package for different external services such as rabbit or redis etc.
├── settings.py  # Main configuration settings for project.
├── static  # Static content.
├── tests  # Tests for project.
└── web  # Package contains web server. Handlers, startup config.
    ├── api  # Package with all handlers.
    │   └── router.py  # Main router.
    ├── application.py  # FastAPI application configuration.
    └── lifetime.py  # Contains actions to perform on startup and shutdown.
```

## Configuration

This application can be configured with environment variables.

You can create `.env` file in the root directory and place all
environment variables here.

All environment variables should start with "AI*SERVICE*" prefix.

For example if you see in your "langchain_chroma_document_chatbot/settings.py" a variable named like
`random_parameter`, you should provide the "langchain_chroma_document_chatbot_RANDOM_PARAMETER"
variable to configure the value. This behaviour can be changed by overriding `env_prefix` property
in `langchain_chroma_document_chatbot.settings.Settings.Config`.

An example of .env file:

```bash
langchain_chroma_document_chatbot_RELOAD="True"
langchain_chroma_document_chatbot_PORT="8000"
langchain_chroma_document_chatbot_ENVIRONMENT="dev"
```

You can read more about BaseSettings class here: https://pydantic-docs.helpmanual.io/usage/settings/

## Pre-commit

To install pre-commit simply run inside the shell:

```bash
pre-commit install
```

pre-commit is very useful to check your code before publishing it.
It's configured using .pre-commit-config.yaml file.

By default it runs:

-   black (formats your code);
-   mypy (validates types);
-   isort (sorts imports in all files);
-   flake8 (spots possible bugs);

You can read more about pre-commit here: https://pre-commit.com/

## Vector store data

This chatbot uses "Music Theory Book" by Mark Andrew Cook, downloaded from [here](https://2012books.lardbucket.org/pdfs/music-theory.pdf).
<br>

This book is licensed under a Creative Commons by-nc-sa 3.0 (http://creativecommons.org/licenses/by-nc-sa/
3.0/) license. See the license for more details, but that basically means you can share this book as long as you
credit the author (but see below), don't make money from it, and do make it available to everyone else under the
same terms.
This book was accessible as of December 29, 2012, and it was downloaded then by Andy Schmitz
(http://lardbucket.org) in an effort to preserve the availability of this book.
Normally, the author and publisher would be credited here. However, the publisher has asked for the customary
Creative Commons attribution to the original publisher, authors, title, and book URI to be removed. Additionally,
per the publisher's request, their name has been removed in some passages. More information is available on this
project's attribution page (http://2012books.lardbucket.org/attribution.html?utm_source=header).
For more information on the source of this book, or why it is available for free, please see the project's home page
(http://2012books.lardbucket.org/). You can browse or download additional books there.
