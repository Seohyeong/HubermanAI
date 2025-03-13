import os
from pathlib import Path
from pydantic import BaseModel, Field
import subprocess

from config import get_config


# init config
def init_config(llm_model_name):
    config = get_config(llm_model_name)
    os.environ["LANGCHAIN_TRACING"] = os.getenv("LANGSMITH_TRACING")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

    def resolve_path(config, project_dir):
        config.rag_db_dir = os.path.join(project_dir, config.rag_db_dir)
        config.query_db_dir = os.path.join(project_dir, config.query_db_dir)
        config.history_db = os.path.join(project_dir, config.history_db)
        config.train_data_path = os.path.join(project_dir, config.train_data_path)
        config.qna_test_data_path = os.path.join(project_dir, config.qna_test_data_path)
        config.syn_test_data_path = os.path.join(project_dir, config.syn_test_data_path)
        config.relevant_qs_path = os.path.join(project_dir, config.relevant_qs_path)
        config.irrelevant_qs_path = os.path.join(project_dir, config.irrelevant_qs_path)

    project_dir = str(Path(__file__).resolve().parent.parent.parent)
    resolve_path(config, project_dir)
    return config

# init mlflow
def start_mlflow_server(host: str, port: str):
    try:
        command = ["mlflow", "server", "--host", host, "--port", str(port)]
        process = subprocess.Popen(command)
        print(f"[MLflow server] running on {host}:{port}")
        return process
    except Exception as e:
        print(f"[MLflow server] error occurred : {e}")
        return None

def stop_mlflow_server(process):
    if process:
        process.terminate()
        process.wait()
        print("[MLflow server] successfully killed")
        
# doc validation (filter out duplicates)
def validate_docs(docs):
    new_docs = []
    doc_ids = set()
    for doc in docs:
        if doc.id not in doc_ids:
            new_docs.append(doc)
            doc_ids.add(doc.id)
    return new_docs

# pydantic
class RAGDoc(BaseModel):
    video_id: str
    title: str
    header: str
    time_start: str
    time_end : str
    segment_idx: str = Field(default=None)
    score: float = Field(default=None)
    
class RAGOutput(BaseModel):
    answer: str = Field(default=None)
    docs: list[RAGDoc] = Field(default=[])
    contextualized_query: str = Field(default=None)
    is_valid: bool = Field(default=None)
    
class QueryInput(BaseModel):
    session_id: str = Field(default=None)
    question: str
    chat_history: list = Field(default=[])

class QueryOutput(BaseModel):
    session_id: str
    answer: str
    docs: list[RAGDoc] = Field(default=[])
    contextualized_query: str = Field(default=None)
    is_valid: bool = Field(default=None)