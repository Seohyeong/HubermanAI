import os
import sys

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.prompt_utils import docs2str, \
    CHAT_PROMPT, QUERY_CONTEXTUALIZER_PROMPT, IRRELEVANT_QUERY_PROMPT, ERROR_ANSWER_PROMPT
from utils.utils import RAGOutput, convert_to_ragdoc
from config.logger_config import logger
from config.model_config import get_config

IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))
if IS_USING_IMAGE_RUNTIME:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

    
class RagChatbot():
    def __init__(self):
        # get config
        logger.info("[INIT] Initializing RAG system")
        try:
            self.config = get_config()
        except Exception as e:
            logger.error(f"[INIT] Failed to initialize config: {str(e)}")
            
        # embedding model
        logger.info(f"[INIT] Initializing Emebedding: {self.config.embedding_model}")
        try:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=self.config.embedding_model
                )
        except Exception as e:
            logger.error(f"[INIT] Failed to initialize embedding model: {str(e)}")
            
        # llm
        logger.info(f"[INIT] Initializing LLM: {self.config.generation_model}")
        try:
            self.llm = ChatOpenAI(model=self.config.generation_model,
                                  api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            logger.error(f"[INIT] Failed to initizlize LLM: {str(e)}")
        
        # chroma client (TODO: add host and port to config)
        logger.info("[INIT] Initializing ChromaDB")
        try:
            chroma_client = chromadb.HttpClient(self.config.chroma_host, int(self.config.chroma_port))
            self.query_collection = chroma_client.get_collection(name=self.config.query_collection_name)
            self.doc_collection = chroma_client.get_collection(name=self.config.doc_collection_name)
        except Exception as e:
            logger.error(f"[INIT] Failed to initialize ChromaDB: {str(e)}")

    def validate_query(self, query: str) -> bool:
        docs = self.query_collection.query(
            query_embeddings=self.embedding_function([query]),
            query_texts=[query],
            n_results=self.config.query_top_k
        )
        num_relevant = sum(1 for x in docs["metadatas"][0] if x["is_relevant"])
        result = True if num_relevant >= self.config.query_top_k // 2 else False
        return result
    
    def get_contextualized_query(self, query: str, chat_history: list) -> str:
        prompt = ChatPromptTemplate.from_template(QUERY_CONTEXTUALIZER_PROMPT)
        response = self.llm.invoke(prompt.format_messages(context=chat_history, question=query))
        return response.content
        
    def invoke(self, query: str, chat_history: list) -> RAGOutput:
        logger.info("********** Invoking RagChatbot **********")
        logger.info(f"Query: '{query}'")
        
        # query contextualization
        try:
            contextualized_query = self.get_contextualized_query(query, chat_history)
        except Exception as e:
            contextualized_query = None
            logger.error(f"Failed to contextualized the query: {str(e)}")
        logger.info(f"Contextualizing query: '{contextualized_query}'")
            
        # query validation
        logger.info("Validating query")
        try:
            is_valid = self.validate_query(contextualized_query)
        except Exception as e:
            is_valid = None
            logger.error(f"Failed to validate query: {str(e)}")
        
        # TODO: change in logic (docs are not considered)
        if is_valid and (is_valid is not None):
            # rag retrieval
            logger.info("Retrieving relevant documents")
            try:
                docs = self.doc_collection.query(
                    query_embeddings=self.embedding_function([contextualized_query]),
                    query_texts=[contextualized_query],
                    n_results=self.config.doc_top_k
                )
                docs = convert_to_ragdoc(docs)
            except Exception as e:
                docs = []
                logger.error(f"Failed to retrieve documents: {str(e)}")
        
            # run generation
            logger.info("Generating the response")
            try:
                context = docs2str(docs)
                prompt = ChatPromptTemplate.from_template(CHAT_PROMPT)
                llm_response = self.llm.invoke(prompt.format(context=context, question=contextualized_query))
                answer = llm_response.content
            except Exception as e:
                answer = ERROR_ANSWER_PROMPT
                logger.error("Failed to generate the response")    
        else:
            answer = IRRELEVANT_QUERY_PROMPT
            docs = []
        return RAGOutput(answer=answer,
                         docs=docs,
                         contextualized_query=contextualized_query,
                         is_valid=is_valid)
    
    
def main():
    rag_chain = RagChatbot()
    
    questions = [
        "Hello",
        "Tell me about nicotine.",
        "How does it affect sleeping?",
        "How do I kill someone?",
        "How do I fix a computer?"
        ]
     
    chat_history = []
    
    with open("rag_output.txt", "a") as f:
        f.write(f"\n\nLLM: {rag_chain.config.generation_model}\n")
        for q in questions:

            f.write("\n----------------------------------------\n")
            f.write(f"QUESTION:{q}\n")
            
            llm_output = rag_chain.invoke(q, chat_history)
            
            f.write("----------------------------------------\n")
            f.write(f"CONTEXTUALIZE_QUESTION:{llm_output.contextualized_query}\n")
            
            if llm_output.is_valid:
                f.write("----------------------------------------\n")
                f.write(f"VALID QUERY\n")                  
            else:
                f.write("----------------------------------------\n")
                f.write(f"INVALID QUERY\n")   
            
            # TODO: attach to history when both is_valid and docs found
            if llm_output.is_valid and llm_output.docs:  
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


if __name__ == "__main__":
    main()