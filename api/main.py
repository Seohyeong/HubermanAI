from fastapi import FastAPI
import uuid
import logging

from utils import QueryInput, QueryOutput
from rag import RagChatbot, HistoryDB

logging.basicConfig(filename="app.log", level=logging.INFO)

app = FastAPI()
rag_chain = RagChatbot("openai")
history_db = HistoryDB("openai")

@app.post("/chat", response_model=QueryOutput)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}")
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    chat_history = history_db.get_chat_history(session_id)
    llm_output = rag_chain.invoke_with_history(query_input.question, chat_history)
    answer = llm_output['answer']
    
    history_db.insert_history_db(session_id, query_input.question, answer)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    
    return QueryOutput(session_id=session_id, answer=answer, docs=llm_output["docs"])
