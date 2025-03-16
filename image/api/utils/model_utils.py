from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def init_embedding_model():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    return embedding_model


def init_llm_model(generation_model):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model=generation_model, api_key=api_key)
    return llm