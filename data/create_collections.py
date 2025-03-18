import chromadb
import chromadb.utils.embedding_functions as embedding_functions

import os
import uuid
import json
import math
from datetime import datetime
from tqdm import tqdm
import numpy as np


def create_questions(file_paths: list):
    questions = {
        "documents": [],
        "metadatas": [],
        "ids": []
    }
    for json_path in file_paths:
        is_relevant = False if "irrelevant_qs" in json_path else True
        with open(json_path, "r", encoding = "utf-8") as f:
            data = [json.loads(line) for line in f]
        for item in tqdm(data, total=len(data), desc=f"{json_path}"):
            document = item["question"]
            metadata = {"is_relevant": is_relevant}
            id = str(uuid.uuid4())
            questions["documents"].append(document)
            questions["metadatas"].append(metadata)
            questions["ids"].append(id)
    return questions


def create_query_collection(chroma_client, embedding_function, collection_name):
    # delete the existing collection if it exists
    existing_collections = chroma_client.list_collections()
    if collection_name in existing_collections:
        chroma_client.delete_collection(name=collection_name)
    
    # create a new collection
    query_collection = chroma_client.create_collection(
        name=collection_name, 
        embedding_function=embedding_function,
        metadata={
            "description": "Ask Huberman Query Analyzer DB",
            "created": str(datetime.now()),
            "hnsw:space": "cosine"
            }
        )
    
    # add to the query colletion
    query_db_file_paths = [
        "data/irrelevant_qs.json", # irrelevant
        "data/relevant_qs.json", # relevant
        "data/qna_test.json", # relevant
        "data/syn_test.json" # relevant
    ]
    print(f"creating questions from {query_db_file_paths}")
    
    questions = create_questions(query_db_file_paths)
    questions["embeddings"] = embedding_function(questions["documents"])
    
    print(f"adding to query collection")
    query_collection.add(
        documents=questions["documents"],
        embeddings=questions["embeddings"],
        metadatas=questions["metadatas"],
        ids=questions["ids"]
    )
    print(query_collection.peek())
    print(query_collection.count())


def create_docs(json_path):
    docs = {
        "documents": [],
        "metadatas": [],
        "ids": []
    }
    with open(json_path, "r", encoding = "utf-8") as f:
        data = [json.loads(line) for line in f]
    for item in tqdm(data, total=len(data), desc=f"{json_path}"):
        document = item["context"]
        metadata = {"video_id": item["video_id"], 
                    "video_title": item["video_title"], 
                    "video_header": item["video_header"],
                    "segment_idx": item["segment_idx"],
                    "time_start": item["time_start"],
                    "time_end": item["time_end"]}
        id = f"{item['video_id']}_{item['segment_idx']}"
        
        docs["documents"].append(document)
        docs["metadatas"].append(metadata)
        docs["ids"].append(id)
    return docs


# TODO: add overlap
def split_docs(docs, max_words):
    doc_lengths = []
    
    new_docs = {
        "documents": [],
        "metadatas": [],
        "ids": []
    }
    for document, metadata in tqdm(zip(docs["documents"], docs["metadatas"]), 
                                   total=len(docs["documents"]),
                                   desc="splitting documents"):
        words = document.split()
        doc_lengths.append(len(words))
        total_words = len(words)
        num_chunks = math.ceil(len(words)/max_words)
        for i in range(0, num_chunks):
            start_idx = i*max_words
            end_idx = min((i+1)*max_words,total_words)
            current_chunks = words[start_idx:end_idx]
            
            chunk_document = " ".join(current_chunks)
            chunk_metadata = metadata.copy()
            chunk_metadata["split_doc_idx"] = i
            chunk_id = f"{metadata['video_id']}_{metadata['segment_idx']}_{metadata['time_start']}_{metadata['time_end']}_{str(uuid.uuid4())}"
            
            new_docs["documents"].append(chunk_document)
            new_docs["metadatas"].append(chunk_metadata)
            new_docs["ids"].append(chunk_id)
            
    print("average doc length: ", np.mean(doc_lengths))
    return new_docs
    
    
def add_docs_in_batches(doc_collection, docs, embedding_function, batch_size):
    total_docs = len(docs["documents"])
    print(f"Total documents to process: {total_docs}")

    for i in tqdm(range(0, total_docs, batch_size), desc="adding to doc collection"):
        batch_end = min(i + batch_size, total_docs)
        batch_docs = {
            "documents": docs["documents"][i:batch_end],
            "metadatas": docs["metadatas"][i:batch_end],
            "ids": docs["ids"][i:batch_end]
        }
        batch_docs["embeddings"] = embedding_function(batch_docs["documents"])
        doc_collection.add(
            documents=batch_docs["documents"],
            embeddings=batch_docs["embeddings"],
            metadatas=batch_docs["metadatas"],
            ids=batch_docs["ids"]
        )
            
            
def create_doc_collection(chroma_client, embedding_function, collection_name, max_words):          
    # delete the existing collection if it exists
    existing_collections = chroma_client.list_collections()
    if collection_name in existing_collections:
        chroma_client.delete_collection(name=collection_name)
        
    # create a new collection
    doc_collection = chroma_client.create_collection(
        name=collection_name, 
        embedding_function=embedding_function,
        metadata={
            "description": "Ask Huberman Doc Retriever DB",
            "created": str(datetime.now()),
            "hnsw:space": "cosine"
            }
        )
    
    # add to the doc colletion
    file = "data/train.json"
    print(f"creating questions from {file}")
    
    docs = create_docs(file) # average doc length:  926.1380072635402
    # docs = split_docs(docs, max_words)
    add_docs_in_batches(doc_collection, docs, embedding_function, batch_size=1000)
    print(doc_collection.peek())
    print(doc_collection.count())
        

def main(host, port, run_create_query_collection, run_create_doc_collection):
    # create chroma client
    chroma_client = chromadb.HttpClient(host, port)
    
    # define embedding_function
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small"
                )

    if run_create_query_collection:
        print("creating query collection")
        create_query_collection(chroma_client, 
                                embedding_function,
                                collection_name="query_collection")
    if run_create_doc_collection:
        print("creating doc collection")
        create_doc_collection(chroma_client, 
                              embedding_function,
                              collection_name="doc_collection",
                              max_words=1000)
    
    
def test_retriver(host, port, collection_name, query):
    chroma_client = chromadb.HttpClient(host, port)
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small"
                )

    collection = chroma_client.get_collection(name=collection_name)
    
    # query Chroma directly. langchain vectorstore doesn't use the embedding created
    results = collection.query(
        query_embeddings=embedding_function([query]),
        query_texts=[query],
        n_results=10 
        )
    print(results)
    

if __name__ == "__main__":
    host="35.91.77.87"
    port=8000
    
    run_create_query_collection = False
    run_create_doc_collection = True
    main(host, port, run_create_query_collection, run_create_doc_collection)
    
    # query="Tell me about nicotine."
    # collection_name="query_collection"
    # test_retriver(host, port, collection_name, query)