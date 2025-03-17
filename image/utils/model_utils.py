from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

def init_embedding_model(model_name):
    embedding_model = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )
    return embedding_model

def init_llm_model(model_name):
    llm = ChatOpenAI(
        model=model_name, 
        api_key=OPENAI_API_KEY
        )
    return llm