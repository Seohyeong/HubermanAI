from dataclasses import dataclass, asdict

@dataclass
class Config:
    name: str = "rag_config"
    
    # model
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    generation_model: str = "gpt-3.5-turbo"
    
    # db
    db_dir: str = "chroma_db"
    db_collection_name = "huberman"
    
    # sqlite3
    history_db = "history_db.db"
    
    # dataset
    train_data_path: str = "api/data/train.json"
    syn_test_data_path: str = "api/data/syn_test.json"
    qna_test_data_path: str = "api/data/qna_test.json"
    relevant_qs_path: str = "api/data/relevant_qs.json"
    irrelevant_qs_path: str = "api/data/irrelevant_qs.json"
    
    # nemo guardrails
    guardrails_config_path: str = "api/config/guardrails_config"
    
    # retriever
    top_k: int = 3
    score_threshold: float = 0.5
    search_type: str = "similarity_score_threshold" # "similarity"
    
    
    @classmethod
    def openai(cls):
        return cls(generation_model = "gpt-3.5-turbo")
        
    @classmethod
    def cohere(cls):
        # command-r7b-12-2024
        return cls(generation_model = "command-r-plus-08-2024")
    
    @classmethod
    def llama(cls):
        return cls(generation_model = "meta-llama/Llama-3.2-3B-Instruct")
    
    def dict_(self):
        return asdict(self)
    

def get_config(exp_name=None):
    if exp_name and hasattr(Config, exp_name):
        return getattr(Config, exp_name)()
    return Config()