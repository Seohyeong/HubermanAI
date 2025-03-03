from dotenv import load_dotenv
import os
from pathlib import Path

# langchain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# rsc
from config import get_config
from utils import create_docs, docs2str, \
    CHAT_PROMPT, QA_SYSTEM_PROMPT, CONTEXTUALIZE_Q_SYSTEM_PROMPT


class RagChatbot():
    def __init__(self, llm_model_name):
        self.config = self._init_config(llm_model_name)
        
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
        config = get_config(llm_model_name)

        load_dotenv()
        if "openai" in llm_model_name:
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
        os.environ["LANGCHAIN_TRACING"] = os.getenv("LANGSMITH_TRACING")
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

        def resolve_path(config, project_dir):
            config.db_dir = os.path.join(project_dir, config.db_dir)
            config.train_data_path = os.path.join(project_dir, config.train_data_path)
            config.qna_test_data_path = os.path.join(project_dir, config.qna_test_data_path)
            config.syn_test_data_path = os.path.join(project_dir, config.syn_test_data_path)

        project_dir = str(Path(__file__).resolve().parent.parent)
        resolve_path(config, project_dir)
        return config
    
    def _init_models(self):
        embedding_function = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
            )
        llm = ChatOpenAI(model=self.config.generation_model)
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
        
        
    def invoke(self, query):
        docs = self.retriever.invoke(query)
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
                for doc in llm_response["context"]
            ]
        } 


def main():
    from langchain_core.messages import HumanMessage, AIMessage
    
    rag_chain = RagChatbot("openai")
    
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


if __name__ == '__main__':
    main()