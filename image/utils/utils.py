from pydantic import BaseModel


# init mlflow
def start_mlflow_server(host: str, port: str):
    import subprocess
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


# convert doc (from choroma) to RAGdoc format
def convert_to_ragdoc(docs):
    new_docs = []
    
    ids = docs["ids"][0]
    scores = docs["distances"][0]
    metadatas = docs["metadatas"][0]
    contexts = docs["documents"][0]
    
    for id, score, metadata, context in zip(ids, scores, metadatas, contexts):
        new_doc = RAGDoc(
            doc_id=id,
            score=score,
            context=context,
            # metadata
            segment_idx=metadata["segment_idx"],
            time_end=metadata["time_end"],
            time_start=metadata["time_start"],
            header=metadata["video_header"],
            video_id=metadata["video_id"],
            title=metadata["video_title"],
        )
        new_docs.append(new_doc)
    return new_docs

# pydantic
class RAGDoc(BaseModel):
    video_id: str
    title: str
    header: str
    time_start: str
    time_end : str
    segment_idx: str = None
    score: float = None
    doc_id: str = None
    context: str = None
    
class RAGOutput(BaseModel):
    answer: str = None
    docs: list[RAGDoc] = []
    contextualized_query: str = None
    is_valid: bool = None
    
class QueryInput(BaseModel):
    session_id: str = None
    question: str
    chat_history: list = []

class QueryOutput(BaseModel):
    session_id: str
    answer: str
    docs: list[RAGDoc] = []
    contextualized_query: str = None
    is_valid: bool = None
    
    
# if __name__ == "__main__":
#     project_dir = str(Path(__file__).resolve().parent.parent)
#     print(project_dir)