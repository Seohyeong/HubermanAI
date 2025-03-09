import os
import json

from langchain_core.documents import Document
from langchain_chroma import Chroma


# indexing
def create_docs(json_path):
    with open(json_path, "r", encoding = "utf-8") as f:
        data = [json.loads(line) for line in f]
        
    docs = []
    for item in data:
        doc = Document(
            page_content=item["context"],
            metadata={"video_id": item["video_id"], 
                    "video_title": item["video_title"], 
                    "video_header": item["video_header"],
                    "segment_idx": item["segment_idx"],
                    "time_start": item["time_start"],
                    "time_end": item["time_end"]},
            )
        docs.append(doc)
    return docs # len: 6165

def init_db(config, embedding_function):
    if os.path.exists(config.db_dir):
        vectorstore = Chroma(collection_name=config.db_collection_name,
                            embedding_function=embedding_function,
                            persist_directory=config.db_dir)
    else:
        docs = create_docs(config.train_data_path)
        vectorstore = Chroma.from_documents(collection_name=config.db_collection_name, 
                                            documents=docs, 
                                            embedding=embedding_function, 
                                            persist_directory=config.db_dir)
    return vectorstore