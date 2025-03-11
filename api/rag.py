# langchain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# mlflow
import mlflow

# utils
from utils.db_utils import init_db, init_query_db
from utils.model_utils import init_embedding_model, init_llm_model
from utils.prompt_utils import docs2str, CHAT_PROMPT, QUERY_CONTEXTUALIZER_PROMPT, IRRELEVANT_QUERY_PROMPT
from utils.eval_utils import test_retriever
from utils.utils import init_config, \
    start_mlflow_server, stop_mlflow_server, \
    RAGDoc, RAGOutput



class RagChatbot():
    def __init__(self, llm_model_name):
        self.llm_model_name = llm_model_name
        self.config = init_config(self.llm_model_name)
        
        # embedding model
        self.embedding_function = init_embedding_model(self.config.embedding_model, self.config.device)
        
        # query retriever
        self.query_vectorstore = init_query_db(self.config, self.embedding_function)
        self.query_retriever = self.query_vectorstore.as_retriever(
            search_type=self.config.qr_search_type, 
            search_kwargs={"k": self.config.qr_top_k}
            )
        
        # basic retriever
        self.vectorstore = init_db(self.config, self.embedding_function)
        self.retriever = self.vectorstore.as_retriever(
            search_type=self.config.rr_search_type, 
            search_kwargs={"score_threshold": self.config.rr_score_threshold}
            )
        
        # llm
        self.llm = init_llm_model(self.llm_model_name, self.config.generation_model)
        
    def retrieve(self, query: str, k: int) -> RAGOutput:
        """basic retriever"""
        docs, scores = zip(*self.vectorstore.similarity_search_with_relevance_scores(query, k=k))
        return RAGOutput(docs=[RAGDoc(video_id=doc.metadata.get("video_id"),
                                        title=doc.metadata.get("video_title"),
                                        header=doc.metadata.get("video_header"),
                                        time_start=doc.metadata.get("time_start"),
                                        time_end=doc.metadata.get("time_end"),
                                        segment_idx=doc.metadata.get("segment_idx"),
                                        score=score)
                                for doc, score in zip(docs, scores)])
        
    def validate_query(self, query: str) -> bool:
        docs = self.query_retriever.invoke(query)
        num_relevant = sum(1 for doc in docs if doc.metadata["is_relevant"])
        if num_relevant >= self.config.qr_top_k // 2:
            return True
        else:
            return False
    
    def get_contextualized_query(self, query, chat_history):
        prompt = ChatPromptTemplate.from_template(QUERY_CONTEXTUALIZER_PROMPT)
        response = self.llm.invoke(prompt.format_messages(context=chat_history, question=query))
        return response.content
        
    def invoke(self, contextualized_query: str) -> RAGOutput:
        docs = self.retriever.invoke(contextualized_query)
        if not docs:
            return RAGOutput(answer=IRRELEVANT_QUERY_PROMPT)
        else:
            context = docs2str(docs)
            prompt = ChatPromptTemplate.from_template(CHAT_PROMPT)
            llm_response = self.llm.invoke(prompt.format(context=context, question=contextualized_query))
            
            return RAGOutput(answer=llm_response.content,
                            docs=[RAGDoc(video_id=doc.metadata.get("video_id"),
                                            title=doc.metadata.get("video_title"),
                                            header=doc.metadata.get("video_header"),
                                            time_start=doc.metadata.get("time_start"),
                                            time_end=doc.metadata.get("time_end"),
                                            segment_idx=doc.metadata.get("segment_idx"))
                                    for doc in docs])
    
def main():
    rag_chain = RagChatbot("cohere")
    
    questions = [
        "Hello",
        "Is chemical sunscreen bad for you?",
        "Where can I find it?",
        "How do I kill myself?",
        "What is a cosine function?"
        ]
    
    if rag_chain.config.activate_mlflow:
        mlflow_process = start_mlflow_server(rag_chain.config.host, rag_chain.config.port)
        mlflow.set_tracking_uri(f"http://{rag_chain.config.host}:{rag_chain.config.port}")
    
     # test_retriever(rag_chain, k=3, threshold=0.5)
     
    chat_history = []
    
    with open("rag_output.txt", "a") as f:
        f.write(f"\n\nLLM: {rag_chain.config.generation_model}\n")
        for q in questions:

            f.write("\n----------------------------------------\n")
            f.write(f"QUESTION:{q}\n")
            
            contextualized_query = rag_chain.get_contextualized_query(q, chat_history)
            is_valid = rag_chain.validate_query(contextualized_query)
            
            f.write("----------------------------------------\n")
            f.write(f"CONTEXTUALIZE_QUESTION:{contextualized_query}\n")
            
            history_output = rag_chain.invoke(contextualized_query)
            if history_output.docs:    
                if is_valid:
                    f.write("----------------------------------------\n")
                    f.write("VALID AND DOCS FOUND\n")
                    f.write("----------------------------------------\n")
                    f.write(f"ANSWER:\n{history_output.answer}\n")
                    chat_history.extend([
                        HumanMessage(content=q),
                        AIMessage(content=history_output.answer)
                    ])
                else:
                    f.write("----------------------------------------\n")
                    f.write("INVALID AND DOCS NOT FOUND\n")
                    f.write("----------------------------------------\n")
                    f.write(f"ANSWER:\n{history_output.answer}\n")
            else:
                if is_valid:
                    f.write("----------------------------------------\n")
                    f.write("VALID AND DOCS NOT FOUND\n")
                    f.write("----------------------------------------\n")
                    f.write(f"ANSWER:\n{history_output.answer}\n")
                else:
                    f.write("----------------------------------------\n")
                    f.write("INVALID AND DOCS NOT FOUND\n")
                    f.write("----------------------------------------\n")
                    f.write(f"ANSWER:\n{history_output.answer}\n")
                    
    if rag_chain.config.activate_mlflow:
        stop_mlflow_server(mlflow_process)

if __name__ == "__main__":
    main()