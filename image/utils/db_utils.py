import os
import sys
import shutil

IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))
if IS_USING_IMAGE_RUNTIME:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    
from langchain_chroma import Chroma


# query indexing
def create_questions(file_paths: list):
    
    import json
    import uuid
    from langchain_core.documents import Document
    
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

# rag indexing
def create_docs(json_path):
    
    import json
    from langchain_core.documents import Document
    
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


# Reference to singleton instance of ChromaDB
QUERY_DB_INSTANCE = None
RAG_DB_INSTANCE = None

def init_query_db(config, embedding_function):
    if config.is_using_image_runtime:
        global QUERY_DB_INSTANCE
        if QUERY_DB_INSTANCE is None:
            try:
                chroma_query_db_dir = config.query_db_dir.split("/")[-1]
                copy_chroma_to_tmp(chroma_query_db_dir)
                QUERY_DB_INSTANCE = Chroma(collection_name=config.query_db_collection_name,
                                    embedding_function=embedding_function,
                                    persist_directory=f"/tmp/{chroma_query_db_dir}",
                                    collection_metadata={"hnsw:space": "cosine"})
                vectorstore = QUERY_DB_INSTANCE
            except Exception as e:
                print(f"Failed to init_query_db {str(e)}")
            print(f"Init RAG ChromaDB query_db_instance: {QUERY_DB_INSTANCE} from {chroma_query_db_dir}")
        else:
            vectorstore = QUERY_DB_INSTANCE
            print("Query vectorstore is NOT initialized!")
    else:
        if os.path.exists(config.query_db_dir):
            vectorstore = Chroma(collection_name=config.query_db_collection_name,
                                embedding_function=embedding_function,
                                persist_directory=config.query_db_dir,
                                collection_metadata={"hnsw:space": "cosine"})
        else:
            file_paths = [config.irrelevant_qs_path, config.relevant_qs_path, 
                        config.qna_test_data_path, config.syn_test_data_path]
            questions = create_questions(file_paths)
            vectorstore = Chroma.from_documents(collection_name=config.query_db_collection_name,
                                                documents=questions, 
                                                embedding=embedding_function, 
                                                persist_directory=config.query_db_dir,
                                                collection_metadata={"hnsw:space": "cosine"})
    return vectorstore
    
def init_db(config, embedding_function):
    if config.is_using_image_runtime:
        global RAG_DB_INSTANCE
        if RAG_DB_INSTANCE is None:
            try:
                chroma_rag_db_dir = config.rag_db_dir.split("/")[-1]
                copy_chroma_to_tmp(chroma_rag_db_dir)
                RAG_DB_INSTANCE = Chroma(collection_name=config.rag_db_collection_name,
                                    embedding_function=embedding_function,
                                    persist_directory=f"/tmp/{chroma_rag_db_dir}",
                                    collection_metadata={"hnsw:space": "cosine"})
                vectorstore = RAG_DB_INSTANCE
            except Exception as e:
                print(f"Failed to init_query_db {str(e)}")
            print(f"Init RAG ChromaDB {RAG_DB_INSTANCE} from {chroma_rag_db_dir}")
        else:
            vectorstore = RAG_DB_INSTANCE
            print("RAG vectorstore is NOT initialized!")
    else:
        if os.path.exists(config.rag_db_dir):
            vectorstore = Chroma(collection_name=config.rag_db_collection_name,
                                embedding_function=embedding_function,
                                persist_directory=config.rag_db_dir,
                                collection_metadata={"hnsw:space": "cosine"})
        else:  
              
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from tqdm import tqdm
            import uuid
            
            docs = create_docs(config.train_data_path) # 6333
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, # limit: 8,191
                                                        chunk_overlap=200,
                                                        length_function=len)
            split_docs = text_splitter.split_documents(docs)
            for split_doc in split_docs:
                md = split_doc.metadata
                split_doc.id = f"{md['video_id']}_{md['segment_idx']}_{md['time_start']}_{md['time_end']}_{str(uuid.uuid4())}"

            vectorstore = Chroma(collection_name=config.rag_db_collection_name, 
                                embedding_function=embedding_function, 
                                persist_directory=config.rag_db_dir,
                                collection_metadata={"hnsw:space": "cosine"})
            batch_size = 500
            total_docs = len(split_docs)
            with tqdm(total=total_docs, desc="Creating vectorstore") as pbar:
                for i in range(0, total_docs, batch_size):
                    batch = split_docs[i:min(i+batch_size, total_docs)]
                    vectorstore.add_documents(documents=batch)
                    pbar.update(len(batch))
    return vectorstore

def copy_chroma_to_tmp(db_dir):
    dst_chroma_path = f"/tmp/{db_dir}"

    if not os.path.exists(dst_chroma_path):
        os.makedirs(dst_chroma_path)

    tmp_contents = os.listdir(dst_chroma_path)
    if len(tmp_contents) == 0:
        print(f"Copying ChromaDB from {db_dir} to {dst_chroma_path}")
        os.makedirs(dst_chroma_path, exist_ok=True)
        shutil.copytree(db_dir, dst_chroma_path, dirs_exist_ok=True)
    else:
        print(f"ChromaDB already exists in {dst_chroma_path}")