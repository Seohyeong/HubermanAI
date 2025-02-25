from dotenv import load_dotenv
import os
import json
import re

import openai
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.data_structs import Node

import chromadb

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
json_path = 'data.json'
PERSIST_DIR = "./chroma_db"


def create_doc(context, video_id, video_title, video_header, segment_idx, time_start, time_end):
    doc = Document(
        text=context,
        metadata={'video_id': video_id, 
                  'video_title': video_title, 
                  'video_header': video_header,
                  'segment_idx': segment_idx,
                  'time_start': time_start,
                  'time_end': time_end},
        excluded_embed_metadata_keys=['video_id', 'segment_idx', 'time_start', 'time_end'],
        excluded_llm_metadata_keys=['video_id', 'segment_idx', 'time_start', 'time_end']
        )
    return doc


def create_docs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    docs = []
    for item in data:
        video_id = item['video_id']
        video_title = item['title']
        
        video_header = ''
        segment_idx = 0
        chunks = []
        for chunk in item['transcript']:
            chunk_type = list(chunk.keys())[0] # 'header' or 'segment'
            if chunk_type == 'header':
                if chunks and video_header:
                    context = ' '.join([item['text'] for item in chunks])
                    time_start = chunks[0]['timestamp']
                    time_end = chunks[-1]['timestamp']
                    doc = create_doc(context, video_id, video_title, video_header, 
                                     segment_idx, time_start, time_end)
                    docs.append(doc)
                    video_header = ''
                    segment_idx += 1
                    chunks = []
                video_header = list(chunk.values())[0]
            else:
                try:
                    timestamp, text = list(chunk.values())[0].split('\n', 1)
                except:
                    continue # {'segment': '0:07'}
                text = re.sub(r'\[.*?\] ', '', text)
                chunks.append({'timestamp': timestamp, 'text': text})  
    return docs


Settings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')



if not os.path.exists(PERSIST_DIR):
    docs = create_docs(json_path) # len: 6069
    nodes = [Node(text=document.text) for document in docs]
    
    db = chromadb.PersistentClient(path=PERSIST_DIR) # creates the directory
    chroma_collection = db.get_or_create_collection("huberman")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("huberman")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    
    
# retriever
retriever = index.as_retriever(similarity_top_k=5)
retrieved_nodes = retriever.retrieve("Are chemical sunscreens safe?")

for node in retrieved_nodes:
    print('-------------')
    print(node.score)
    print(node.text)
# response = query_engine.query("Are chemical sunscreens safe?")
# print(response)