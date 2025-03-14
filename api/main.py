from fastapi import FastAPI
import uuid
import logging

from utils.utils import QueryInput, QueryOutput
from rag import RagChatbot
from logger_config import logger

# logging.basicConfig(filename="app.log", level=logging.INFO)

app = FastAPI()
rag_chain = RagChatbot("cohere")


@app.post("/chat", response_model=QueryOutput)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    
    logging.info(f"[FastAPI] Session ID: {session_id}, User Query: {query_input.question}")
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        llm_output = rag_chain.invoke(query_input.question, query_input.chat_history)
        answer = llm_output.answer
        contextualized_query = llm_output.contextualized_query
        is_valid = llm_output.is_valid
        logging.debug("[FastAPI] Processing completed successfully")
        logging.info(f"[FastAPI] Session ID: {session_id}, AI Response: {answer}")
    except Exception as e:
        logger.error(f"[FastAPI] Endpoint error: {str(e)}")
    
    return QueryOutput(session_id=session_id, answer=answer, docs=llm_output.docs,
                       contextualized_query=contextualized_query, is_valid=is_valid)
