import argparse
import chromadb
from datetime import datetime
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

from llama_index.core import Document, PromptTemplate, VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.data_structs import Node
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama

from data.prompts import TEXT_QA_PROMPT


def create_docs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    docs = []
    for item in data:
        doc = Document(
            text=item['context'],
            metadata={'video_id': item['video_id'], 
                    'video_title': item['video_title'], 
                    'video_header': item['video_header'],
                    'segment_idx': item['segment_idx'],
                    'time_start': item['time_start'],
                    'time_end': item['time_end']},
            excluded_embed_metadata_keys=['video_id', 'segment_idx', 'time_start', 'time_end'],
            excluded_llm_metadata_keys=['video_id', 'segment_idx', 'time_start', 'time_end']
            )
        docs.append(doc)
    return docs


class ChromaDB:
    def __init__(self, db_dir, db_collection_name):
        self.db_dir = db_dir
        self.db_collection_name = db_collection_name
        
        self.does_db_exist = True if os.path.exists(self.db_dir) else False

        self.db = chromadb.PersistentClient(path = db_dir) # creates a dir
        self.collection = self.db.get_or_create_collection(db_collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
    
    def index(self, nodes):
        storage_context = StorageContext.from_defaults(vector_store = self.vector_store)
        self.index = VectorStoreIndex(nodes, storage_context = storage_context)
        self.index.storage_context.persist(persist_dir = self.db_dir)
    
    def load(self):
        storage_context = StorageContext.from_defaults(vector_store = self.vector_store, 
                                                       persist_dir = self.db_dir)
        self.index = load_index_from_storage(storage_context)
    
      
class Retriever:
    def __init__(self, index, top_k):
        self.index = index
        self.top_k = top_k
        self.retriever = self.index.as_retriever(similarity_top_k = self.top_k)
    
    def retrieve(self, q: str) -> list:
        retrieved_nodes = self.retriever.retrieve(q)
        return retrieved_nodes
    
    def evaluate(self, docs: list[dict]):
        results = []
        
        for doc in tqdm(docs, total=len(docs)):
            # gt segment
            gt_segment_id = doc['doc_id']
            # retrieved result
            rt_nodes = self.retrieve(doc['question'])
            rt_scores = [node.score for node in rt_nodes]
            rt_segment_ids = [node.metadata['video_id'] + '_' + node.metadata['segment_idx'] for node in rt_nodes]
            rt_texts = [node.text for node in rt_nodes]
            # calculate reciprocal rank
            try:
                rr = 1 / (rt_segment_ids.index(gt_segment_id) + 1)
            except ValueError:
                rr = 0
            
            result = {'score': rt_scores, 
                      'rt_segment_ids': rt_segment_ids, 
                      'rt_texts': rt_texts,
                      'rr': rr}
            results.append(result)
            
        return results


class QueryEngine:
    ''' query_engine = retriever + llm'''
    def __init__(self, index, top_k):
        self.index = index
        self.top_k = top_k
        self.query_engine = self.index.as_query_engine(similarity_top_k=self.top_k)
        
    def query(self, q):
        return self.query_engine.query(q)
    
    def update_prompts(self, prompt):
        self.query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(template=prompt)})
        
    def evaluate(self, docs: list[dict]):
        correctness_evaluator = CorrectnessEvaluator(llm=Settings.llm) # 1 ~ 5
        faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm) # binary
        relavancy_evaluator = RelevancyEvaluator(llm=Settings.llm) # binary
        
        qe_results = []
        for doc in tqdm(docs, total=len(docs)):
            query = doc['question']
            response = self.query(doc['question'])
            correctness_result = correctness_evaluator.evaluate(query=query, response=response.response, reference=doc['context']) # Whether the generated answer matches that of the reference answer given the query 
            faithfulness_result = faithfulness_evaluator.evaluate_response(response=response) # Evaluates if the answer is faithful to the retrieved contexts 
            relavancy_result = relavancy_evaluator.evaluate_response(query=query, response=response) # Whether retrieved context is relevant to the query
            
            result = {
                'query': query,
                'response': response.response,
                'correct_score': float(correctness_result.feedback.split('\n')[0]),
                'faith_score': faithfulness_result.score,
                'rel_score': relavancy_result.score
            
            }
            qe_results.append(result)
        return qe_results
    
    def display_prompt(self):
        prompts_dict = self.query_engine.get_prompts()
        for k, p in prompts_dict.items():
            print('PROMPT FOR {}\n'.format(k))
            print(p.get_template())
            print('\n\n')
    
    
def main():
    parser = argparse.ArgumentParser(description='RAG')

    # flags
    parser.add_argument('--run_retriever', type=bool, default=True)
    parser.add_argument('--run_query_engine', type=bool, default=True)
    
    # embedding model and llm
    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-small-en-v1.5')
    parser.add_argument('--generation_model', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    
    # train dataset and db
    parser.add_argument('--train_data_path', type=str, default='./data/train.json')
    parser.add_argument('--db_dir', type=str, default='./chroma_db')
    parser.add_argument('--db_collection_name', type=str, default='huberman')
    
    # test dataset
    parser.add_argument('--syn_test_data_path', type=str, default='./data/syn_test.json') # for retriever
    parser.add_argument('--qna_test_data_path', type=str, default='./data/qna_test.json') # for llm
    
    parser.add_argument('--question', type=str, default='Are chemical sunscreens safe?')
    
    args = parser.parse_args()
    
    # write log file
    log_file_path = './logs/log_{}'.format(datetime.now().strftime('%H_%M_%d_%m_%Y.txt'))
    with open(log_file_path, 'a') as f:
        f.write('EMBED_MODEL: {}\n'.format(args.embedding_model))
        f.write('LLM_MODEL: {}\n\n'.format(args.generation_model))
        
    # load api keys
    load_dotenv()
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')


    # set up embedding model (retriever) and llm
    Settings.embed_model = HuggingFaceEmbedding(args.embedding_model)
    
    # # tokenizer = AutoTokenizer.from_pretrained(args.generation_model)
    # Settings.llm = HuggingFaceLLM(
    #     model_name=args.generation_model,
    #     tokenizer_name=args.generation_model,
    #     context_window=3900,
    #     max_new_tokens=500,
    #     generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    #     # stopping_ids=[tokenizer.eos_token_id],
    #     # messages_to_prompt=messages_to_prompt,
    #     # completion_to_prompt=completion_to_prompt,
    #     device_map="auto"
    # )
    
    # Settings.llm = Ollama(model="llama3.2:3b", request_timeout=100.0)


    # set up chromadb
    chroma_db = ChromaDB(args.db_dir, args.db_collection_name)
    if chroma_db.does_db_exist:
        chroma_db.load()
    else:
        docs = create_docs(args.train_data_path)
        nodes = [Node(text=document.text, metadata=document.metadata) for document in docs]
        chroma_db.index(nodes)

    if args.run_retriever:
        # init retriever
        retriever = Retriever(index = chroma_db.index, top_k = 20)
        
        # retrieved_nodes = retriever.retrieve(args.question)
        
        # evaluate on syn_test
        with open(args.syn_test_data_path, 'r', encoding='utf-8') as f:
            syn_test = [json.loads(line) for line in f]
            rt_results = retriever.evaluate(syn_test)
           
        mrr = 0 
        for result in rt_results:
            mrr += result['rr']
        mrr = mrr / len(rt_results)
        
        with open(log_file_path, 'a') as f:
            f.write('MRR on SYN_TEST: {}\n'.format(mrr))

    
    if args.run_query_engine:
        # init query_engine
        query_engine = QueryEngine(index = chroma_db.index, top_k = 10)
        # query_engine.display_prompt()
        query_engine.update_prompts(TEXT_QA_PROMPT)

        print(query_engine.query(args.question))
        
        # evaluate on qna_test
        with open(args.qna_test_data_path, 'r', encoding='utf-8') as f:
            qna_test = [json.loads(line) for line in f]
            qe_results = query_engine.evaluate(qna_test)
            
        corr, faith, rel = 0, 0, 0
        n = len(qe_results)
        for result in qe_results:
            corr += result['correct_score']
            faith += result['faith_score']
            rel += result['rel_score']
        
        corr = corr / n
        faith = faith / n
        rel = rel / n
        
        with open(log_file_path, 'a') as f:
            f.write('CORR on QNA_TEST: {}\n'.format(corr))
            f.write('FAITH on QNA_TEST: {}\n'.format(faith))
            f.write('REL on QNA_TEST: {}\n'.format(rel))
            
        with open('./logs/qe_results.json', 'w', encoding='utf-8') as f:
            for doc in qe_results:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n') 
        
if __name__ == '__main__':
    main()