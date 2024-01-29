import asyncio
from typing import Any, List

from langchain.callbacks import LangChainTracer, StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document, StrOutputParser  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough  # type: ignore
from langchain_openai.chat_models import ChatOpenAI

from langchain_chroma_document_chatbot.services.logger import logger
from langchain_chroma_document_chatbot.settings import settings

resources = [
    "./qa-resources/music-theory.pdf",
]


class DocumentChain:
    retriever: VectorStoreRetriever
    prompt_template: PromptTemplate

    def __init__(
        self,
    ) -> None:
        template = """You are a music-theory chatbot assisstant.
Use the following pieces of retrieved context delimited with three backticks to answer the question delimited by three quotes.
- Try to be useful to the user and provide the best answer you can, describing stpes to take if necessary.
- Answer in three sentences or fewer, keeping it concise.
- Identify and respond in the language of the question, typically English.
- If you could not find an answer, respond with "Sorry, I couldn't figure that out".
- If you could find the answer, do not mention it was indicated in the document and instead bring the actual text to the user.
Question:
'''
{question}
'''

Context:
```
{context}
```
Answer:
"""
        self.prompt_template = PromptTemplate.from_template(template)

    async def load(self):  # type: ignore
        loop = asyncio.get_event_loop()
        loop.create_task(self.async_load_vector_database())

    async def async_load_vector_database(self) -> None:  # type: ignore
        logger.info(
            {
                "message": "async_load_documents",
                "service": "DocumentService",
                "method": "async_load_documents",
            },
        )
        self.load_vector_database()

    def get_chain(self) -> Any:
        logger.info(
            {
                "message": "Received question",
                "service": "DocumentService",
                "method": "ask",
            },
        )

        if not hasattr(self, "retriever"):  # noqa: WPS421
            self.load_vector_database()

        callbackManager = CallbackManager(
            [
                LangChainTracer(),
                StreamingStdOutCallbackHandler(),
            ],
        )

        model = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=settings.open_ai_api_key,
            callback_manager=callbackManager,
            streaming=True,
            verbose=False,
        )  # type: ignore

        return (
            {
                "context": self.retriever,
                "question": RunnablePassthrough(),
            }  # type: ignore
            | self.prompt_template
            | model
            | StrOutputParser()
        )

    def load_vector_database(self) -> None:
        logger.info(
            {
                "message": "Loading vector database",
                "service": "DocumentChain",
                "method": "load_vector_database",
            },
        )

        db = Chroma.from_documents(
            documents=self.get_local_pdf_files(),
            embedding=OpenAIEmbeddings(openai_api_key=settings.open_ai_api_key),  # type: ignore
        )
        self.retriever = db.as_retriever()
        logger.info(
            {
                "message": "Chroma was selected as vector database",
                "service": "DocumentChain",
                "method": "load_vector_database",
            },
        )

    def get_local_pdf_files(self) -> List[Document]:  # type: ignore
        logger.info(
            {
                "message": "Loading pdf Files",
                "service": "DocumentChain",
                "method": "get_local_pdf_files",
            },
        )
        documents: List[Document] = []
        for resource in resources:
            loader = PyPDFLoader(resource)
            raw_documents = loader.load()
            textSplitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0,
            )
            for raw_document in raw_documents:
                last_slash_index = raw_document.metadata["source"].rfind("/")
                title = raw_document.metadata["source"][last_slash_index + 1 : -4]
                raw_document.metadata["title"] = title
            documents = documents + textSplitter.split_documents(raw_documents)

        return documents
