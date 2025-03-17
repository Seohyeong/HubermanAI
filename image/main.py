from fastapi import FastAPI
import uuid
from mangum import Mangum # to use aws lambda

from rag import RagChatbot
from config.logger_config import logger
from utils.utils import QueryInput, QueryOutput


app = FastAPI()

rag_chain = RagChatbot("openai")

@app.get("/")
def root():
    return {"Message": "Welcome to Ask Huberman!"}

@app.post("/chat", response_model=QueryOutput)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    
    logger.info(f"[FastAPI] Session ID: {session_id}, User Query: {query_input.question}")
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        llm_output = rag_chain.invoke(query_input.question, query_input.chat_history)
        answer = llm_output.answer
        contextualized_query = llm_output.contextualized_query
        is_valid = llm_output.is_valid
        logger.debug("[FastAPI] Processing completed successfully")
        logger.info(f"[FastAPI] Session ID: {session_id}, AI Response: {answer}")
    except Exception as e:
        logger.error(f"[FastAPI] Endpoint error: {str(e)}")
    
    return QueryOutput(session_id=session_id, answer=answer, docs=llm_output.docs,
                       contextualized_query=contextualized_query, is_valid=is_valid)


handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    # Run this as a server directly.
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("main:app", host="0.0.0.0", port=port)