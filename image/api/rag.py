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
from logger_config import logger


class RagChatbot():
    def __init__(self, llm_model_name):
        logger.info("Initializing RAG system")
        self.llm_model_name = llm_model_name
        try:
            self.config = init_config(self.llm_model_name)
            logger.debug("Configuration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize config: {str(e)}")
            
        
        # embedding model
        logger.info("Initializing emebedding")
        try:
            self.embedding_function = init_embedding_model(self.config.embedding_model, self.config.device)
            logger.debug("Embedding model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
        
        # query retriever
        logger.info("Initializing query vectorstore and retriever")
        try:
            self.query_vectorstore = init_query_db(self.config, self.embedding_function)
            self.query_retriever = self.query_vectorstore.as_retriever(
                search_type=self.config.qr_search_type, 
                search_kwargs={"k": self.config.qr_top_k}
                )
            logger.debug("Query vectorstore and retriever initialized")
        except Exception as e:
            logger.error(f"Failed to initialize query vectorstore and retriever: {str(e)}")
        
        # basic retriever
        logger.info("Initializing RAG vectorstore and retriever")
        try:
            self.vectorstore = init_db(self.config, self.embedding_function)
            self.retriever = self.vectorstore.as_retriever(
                search_type=self.config.rr_search_type, 
                search_kwargs={"score_threshold": self.config.rr_score_threshold}
                )
            logger.debug("RAG vectorstore and retriever initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG vectorstore and retriever: {str(e)}")
            
        # llm
        logger.info("Initializing LLM")
        try:
            self.llm = init_llm_model(self.llm_model_name, self.config.generation_model)
            logger.debug("LLM initialized")
        except Exception as e:
            logger.error(f"Failed to initizlize LLM: {str(e)}")
            
    def retrieve_with_score(self, query: str, k: int) -> RAGOutput:
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
        
    def filter_docs(self, docs):
        filtered_docs = []
        video_ids = set()
        for doc in docs:
            video_id = "_".join(doc.id.split("_")[:-1])
            if video_id not in video_ids:
                filtered_docs.append(doc)
                video_ids.add(video_id)
        return filtered_docs
    
    def get_contextualized_query(self, query, chat_history):
        prompt = ChatPromptTemplate.from_template(QUERY_CONTEXTUALIZER_PROMPT)
        response = self.llm.invoke(prompt.format_messages(context=chat_history, question=query))
        return response.content
        
    def invoke(self, query: str, chat_history: list) -> RAGOutput:
        logger.info("Contextualizing the query")
        try:
            self.contextualized_query = self.get_contextualized_query(query, chat_history)
            logger.debug("Query is contextualized")
        except Exception as e:
            logger.error(f"Failed to contextualized the query: {str(e)}")
            
        logger.info("Retrieving relevant documents")
        try:
            self.docs = self.retriever.invoke(self.contextualized_query)
            logger.debug("Relevant docs retrieved")
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            
        logger.info("Validating query")
        try:
            self.is_valid = self.validate_query(self.contextualized_query)
            logger.debug("Query is validated")
        except Exception as e:
            logger.error(f"Failed to validate query: {str(e)}")
            
        if self.is_valid and self.docs:
            logger.info("Generating the response")
            try:
                context = docs2str(self.docs)
                prompt = ChatPromptTemplate.from_template(CHAT_PROMPT)
                llm_response = self.llm.invoke(prompt.format(context=context, question=self.contextualized_query))
                logger.debug("Response generated")
            except Exception as e:
                logger.error("Failed to generate the response")
                
            self.filtered_docs = self.filter_docs(self.docs)     
            
            return RAGOutput(answer=llm_response.content,
                            docs=[RAGDoc(video_id=doc.metadata.get("video_id"),
                                            title=doc.metadata.get("video_title"),
                                            header=doc.metadata.get("video_header"),
                                            time_start=doc.metadata.get("time_start"),
                                            time_end=doc.metadata.get("time_end"),
                                            segment_idx=doc.metadata.get("segment_idx"))
                                    for doc in self.filtered_docs],
                            contextualized_query=self.contextualized_query,
                            is_valid=True)
        else:
            return RAGOutput(answer=IRRELEVANT_QUERY_PROMPT, 
                             contextualized_query=self.contextualized_query,
                             is_valid=False)
    
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
            
            llm_output = rag_chain.invoke(q, chat_history)
            
            f.write("----------------------------------------\n")
            f.write(f"CONTEXTUALIZE_QUESTION:{rag_chain.contextualized_query}\n")
            
            if llm_output.docs:  
                f.write("----------------------------------------\n")
                f.write(f"DOCS FOUND\n")
                chat_history.extend([
                    HumanMessage(content=q),
                    AIMessage(content=llm_output.answer)
                ])
            else:
                f.write("----------------------------------------\n")
                f.write(f"NO DOCS FOUND\n")  
                
            f.write("----------------------------------------\n")
            f.write(f"ANSWER: {llm_output.answer}\n")          

                    
    if rag_chain.config.activate_mlflow:
        stop_mlflow_server(mlflow_process)

if __name__ == "__main__":
    main()