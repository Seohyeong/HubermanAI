from dotenv import load_dotenv
import os
from pathlib import Path
import sqlite3

# langchain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere

# rsc
from config import get_config
from utils import create_docs, docs2str, \
    CHAT_PROMPT, QA_SYSTEM_PROMPT, CONTEXTUALIZE_Q_SYSTEM_PROMPT


def _init_config(llm_model_name):
    config = get_config(llm_model_name)
    load_dotenv()
    if llm_model_name == "openai":
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    elif llm_model_name == "cohere":
        os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    os.environ["LANGCHAIN_TRACING"] = os.getenv("LANGSMITH_TRACING")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

    def resolve_path(config, project_dir):
        config.db_dir = os.path.join(project_dir, config.db_dir)
        config.history_db = os.path.join(project_dir, config.history_db)
        config.train_data_path = os.path.join(project_dir, config.train_data_path)
        config.qna_test_data_path = os.path.join(project_dir, config.qna_test_data_path)
        config.syn_test_data_path = os.path.join(project_dir, config.syn_test_data_path)

    project_dir = str(Path(__file__).resolve().parent.parent)
    resolve_path(config, project_dir)
    return config
    

class HistoryDB():
    def __init__(self, llm_model_name):
        self.config = self._init_config(llm_model_name)
        self.db_name = self.config.history_db
        conn = self.get_db_connection()
        
        conn.execute('''CREATE TABLE IF NOT EXISTS history_db
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        query TEXT,
        response TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.close()

    def _init_config(self, llm_model_name):
        return _init_config(llm_model_name)
        
    def get_db_connection(self):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn
    
    def insert_history_db(self, session_id, query, response):
        conn = self.get_db_connection()
        conn.execute('''INSERT INTO history_db (session_id, query, response) 
                    VALUES (?, ?, ?)''',
                    (session_id, query, response))
        conn.commit()
        conn.close()

    def get_chat_history(self, session_id):
        conn = self.get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''SELECT query, response 
                    FROM history_db 
                    WHERE session_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 3''', (session_id,))
        messages = []
        for row in cursor.fetchall():
            messages.extend([
                {"role": "human", "content": row['query']},
                {"role": "ai", "content": row['response']}
            ])
        conn.close()
        return messages


class RagChatbot():
    def __init__(self, llm_model_name):
        self.llm_model_name = llm_model_name
        self.config = self._init_config(self.llm_model_name)
        
        self.embedding_function, self.llm = self._init_models()
        self.vectorstore = self._init_db()
        self.retriever = self.vectorstore.as_retriever(search_type=self.config.search_type, 
                                                       search_kwargs={"k": self.config.top_k})
        
        self.prompt = ChatPromptTemplate.from_template(CHAT_PROMPT)
        
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}")])
        self.history_aware_retriever = create_history_aware_retriever(self.llm, 
                                                                      self.retriever, 
                                                                      self.contextualize_q_prompt)
        
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [("system", QA_SYSTEM_PROMPT),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}")])
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)
        
        
    def _init_config(self, llm_model_name):
        return _init_config(llm_model_name)
    
    def _init_models(self):
        embedding_function = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
            )
        if self.llm_model_name == "openai":
            llm = ChatOpenAI(model=self.config.generation_model)
        elif self.llm_model_name == "cohere":
            llm = ChatCohere(model=self.config.generation_model)
        return embedding_function, llm
        
    def _init_db(self):
        if os.path.exists(self.config.db_dir):
            vectorstore = Chroma(collection_name=self.config.db_collection_name,
                                embedding_function=self.embedding_function,
                                persist_directory=self.config.db_dir)
        else:
            docs = create_docs(self.config.train_data_path)
            vectorstore = Chroma.from_documents(collection_name=self.config.db_collection_name, 
                                                documents=docs, 
                                                embedding=self.embedding_function, 
                                                persist_directory=self.config.db_dir)
        return vectorstore
        
    def _validate_docs(self, docs):
        new_docs = []
        doc_ids = set()
        for doc in docs:
            if doc.id not in doc_ids:
                new_docs.append(doc)
                doc_ids.add(doc.id)
        return new_docs
    
    def invoke(self, query):
        docs = self.retriever.invoke(query)
        docs = self._validate_docs(docs)
        context = docs2str(docs)
        llm_response = self.llm.invoke(self.prompt.format(context=context, question=query))
        
        return {
            "answer": llm_response.content,
            "docs": [
                {   
                    "video_id": doc.metadata.get("video_id"),
                    "title": doc.metadata.get("video_title"),
                    "header": doc.metadata.get("video_header"),
                    "time_start": doc.metadata.get("time_start"),
                    "time_end": doc.metadata.get("time_end")
                }
                for doc in docs
            ]
        }
        
    def invoke_with_history(self, query, chat_history):
        llm_response = self.rag_chain.invoke({"input": query, "chat_history": chat_history})
        docs = self._validate_docs(llm_response["context"])
        
        return {
            "answer": llm_response["answer"],
            "docs": [
                {   
                    "video_id": doc.metadata.get("video_id"),
                    "title": doc.metadata.get("video_title"),
                    "header": doc.metadata.get("video_header"),
                    "time_start": doc.metadata.get("time_start"),
                    "time_end": doc.metadata.get("time_end")
                }
                for doc in docs
            ]
        } 


def main():
    from langchain_core.messages import HumanMessage, AIMessage
    
    rag_chain = RagChatbot("cohere")
    
    q1 = "Is chemical sunscreen bad for you?"
    q2 = "Where can I find it?"
    
    chat_history = []

    print(f"QUESTION:\n{q1}\n")
    simple_output_1 = rag_chain.invoke(q1)
    print(f"SIMPLE_ANSWER:\n{simple_output_1['answer']}\n")
    history_output_1 = rag_chain.invoke_with_history(q1, chat_history)
    print(f"HISTORY_ANSWER:\n{history_output_1['answer']}\n\n")
    
    chat_history.extend([
        HumanMessage(content=q1),
        AIMessage(content=history_output_1["answer"])
    ])
    
    print(f"QUESTION:\n{q2}\n")
    simple_output_2 = rag_chain.invoke(q2)
    print(f"SIMPLE_ANSWER:\n{simple_output_2['answer']}\n")
    history_output_2 = rag_chain.invoke_with_history(q2, chat_history)
    print(f"HISTORY_ANSWER:\n{history_output_2['answer']}\n\n")


if __name__ == "__main__":
    main()