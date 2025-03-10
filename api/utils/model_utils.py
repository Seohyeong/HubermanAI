from dotenv import load_dotenv
import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings

        
class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = os.getenv("OPENROUTER_API_KEY")
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)
        
        
def init_embedding_model(model_name, device):
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    
    embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
            )
    return embedding_model


def init_llm_model(model_name, generation_model):
    load_dotenv()
    
    if model_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(model=generation_model, api_key=api_key)
    elif model_name == "cohere":
        api_key = os.getenv("COHERE_API_KEY")
        llm = ChatCohere(model=generation_model, api_key=api_key)
    elif model_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        llm = ChatOpenRouter(model=generation_model, api_key=api_key)
    return llm