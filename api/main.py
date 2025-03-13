from fastapi import FastAPI
import uuid
import logging

from utils.utils import QueryInput, QueryOutput
from rag import RagChatbot

logging.basicConfig(filename="app.log", level=logging.INFO)

app = FastAPI()
rag_chain = RagChatbot("cohere")


@app.post("/chat", response_model=QueryOutput)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}")
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    llm_output = rag_chain.invoke(query_input.question, query_input.chat_history)
    answer = llm_output.answer
    contextualized_query = llm_output.contextualized_query
    is_valid = llm_output.is_valid
    
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    
    return QueryOutput(session_id=session_id, answer=answer, docs=llm_output.docs,
                       contextualized_query=contextualized_query, is_valid=is_valid)
