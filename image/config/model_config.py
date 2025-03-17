from dataclasses import dataclass

@dataclass
class Config:
    name: str = "rag_config"
    
    # model
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"
    
    # mlflow
    host: str = "127.0.0.1"
    port: str = "8080"
    activate_mlflow: bool = False
    
    # query db
    query_db_dir: str = "chroma_db_query"
    query_db_collection_name = "huberman"
    
    # retriever db
    rag_db_dir: str = "chroma_db_docs"
    rag_db_collection_name = "huberman"
    
    # dataset
    train_data_path: str = "api/data/train.json"
    syn_test_data_path: str = "api/data/syn_test.json"
    qna_test_data_path: str = "api/data/qna_test.json"
    relevant_qs_path: str = "api/data/relevant_qs.json"
    irrelevant_qs_path: str = "api/data/irrelevant_qs.json"
    
    # query retriever
    qr_top_k: int = 10
    qr_search_type: str = "similarity"
    
    # rag retriever
    rr_top_k: int = 3
    rr_score_threshold: float = 0.5
    rr_search_type: str = "similarity_score_threshold"
    
    # running on image
    is_using_image_runtime: bool = False

def get_config(exp_name=None):
    if exp_name and hasattr(Config, exp_name):
        return getattr(Config, exp_name)()
    return Config()