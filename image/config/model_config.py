from dataclasses import dataclass

@dataclass
class Config:
    name: str = "rag_config"
    
    # model
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"
    
    # mlflow
    mlflow_host: str = "127.0.0.1"
    mlflow_port: str = "8080"
    
    # chroma server on ec2
    chroma_host: str = "35.91.77.87"
    chroma_port: str = "8000"
    query_collection_name: str = "query_collection"
    doc_collection_name: str = "doc_collection"
    
    # retriever hyperparameters
    query_top_k: int = 10
    doc_top_k: int = 3
    
    # running on docker image
    is_using_image_runtime: bool = False

def get_config(exp_name=None):
    if exp_name and hasattr(Config, exp_name):
        return getattr(Config, exp_name)()
    return Config()