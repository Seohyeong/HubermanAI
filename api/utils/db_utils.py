import json
import os
from tqdm import tqdm
import uuid

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# query indexing
def create_questions(file_paths: list):
    questions = []
    for json_path in file_paths:
        is_relevant = False if "irrelevant_qs" in json_path else True
        with open(json_path, "r", encoding = "utf-8") as f:
            data = [json.loads(line) for line in f]
        for item in data:
            doc = Document(
                id=str(uuid.uuid4()),
                page_content=item["question"],
                metadata={"is_relevant": is_relevant}
                )
            questions.append(doc)
    return questions


def init_query_db(config, embedding_function):
    if os.path.exists(config.query_db_dir):
        vectorstore = Chroma(collection_name=config.query_db_collection_name,
                            embedding_function=embedding_function,
                            persist_directory=config.query_db_dir)
    else:
        file_paths = [config.irrelevant_qs_path, config.relevant_qs_path, 
                      config.qna_test_data_path, config.syn_test_data_path]
        questions = create_questions(file_paths)
        vectorstore = Chroma.from_documents(collection_name=config.query_db_collection_name,
                                            documents=questions, 
                                            embedding=embedding_function, 
                                            persist_directory=config.query_db_dir)
    return vectorstore


# rag indexing
def create_docs(json_path):
    with open(json_path, "r", encoding = "utf-8") as f:
        data = [json.loads(line) for line in f]
    docs = []
    for item in data:
        doc = Document(
            id=f"{item['video_id']}_{item['segment_idx']}",
            page_content=item["context"],
            metadata={"video_id": item["video_id"], 
                    "video_title": item["video_title"], 
                    "video_header": item["video_header"],
                    "segment_idx": item["segment_idx"],
                    "time_start": item["time_start"],
                    "time_end": item["time_end"]},
            )
        docs.append(doc)
    return docs
    
def init_db(config, embedding_function):
    if os.path.exists(config.rag_db_dir):
        vectorstore = Chroma(collection_name=config.rag_db_collection_name,
                            embedding_function=embedding_function,
                            persist_directory=config.rag_db_dir)
    else:
        docs = create_docs(config.train_data_path) # 6333
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, # limit: 512
                                                       chunk_overlap=10,
                                                       length_function=len)
        split_docs = text_splitter.split_documents(docs) # 68858
        for split_doc in split_docs:
            md = split_doc.metadata
            split_doc.id = f"{md['video_id']}_{md['segment_idx']}_{md['time_start']}_{md['time_end']}_{str(uuid.uuid4())}"

        vectorstore = Chroma(collection_name=config.rag_db_collection_name, 
                             embedding_function=embedding_function, 
                             persist_directory=config.rag_db_dir)
        batch_size = 100
        total_docs = len(split_docs)
        with tqdm(total=total_docs, desc="Creating vectorstore") as pbar:
            for i in range(0, total_docs, batch_size):
                batch = split_docs[i:min(i+batch_size, total_docs)]
                vectorstore.add_documents(documents=batch)
                pbar.update(len(batch))
    return vectorstore