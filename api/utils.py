import json
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# generating test set
SYSTEM_PROMPT = """
You are a helpful AI assistant for creating a Question and Answering dataset.
Given the CONTEXT and its HEADER,  generate a question that general people might ask, which can be answered using the CONTEXT.
Output ONLY the question.
"""

# vanilla llm invoke prompt
CHAT_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using only the information from the context, answer the query in clear and grammatically correct sentences.\n"
    "Note: The context may contain informal or ungrammatical phrases as it is extracted from a YouTube video.\n"
    "Use all relevant information from the context but do not make up any facts that are not given.\n"
    "Do not simply summarize the answer based on the given context, give a detailed and explicit answer.\n"
    "Question: {question}\n"
    "Answer: "
)

# history aware retriever prompt
CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, formulate a standalone question"
    "which can be understood without the chat history. Do NOT answer the question,"
    "just reformulate it if needed and otherwise return it as is."
)

# history aware QA prompt
QA_SYSTEM_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using only the information from the context, answer the query in clear and grammatically correct sentences.\n"
    "Note: The context may contain informal or ungrammatical phrases as it is extracted from a YouTube video.\n"
    "Use all relevant information from the context but do not make up any facts that are not given.\n"
    "Do not simply summarize the answer based on the given context, give a detailed and explicit answer.\n"
    "Format the response properly, using paragraphs where appropriate to enhance readability."
)


# prompt formatting
def docs2str(docs):
    return "\n\n".join(
        f"title: {doc.metadata['video_title']}\n"
        f"header: {doc.metadata['video_header']}\n"
        f"content: {doc.page_content}"
        for doc in docs
    )
    
    
# indexing
def create_docs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    docs = []
    for item in data:
        doc = Document(
            page_content=item['context'],
            metadata={'video_id': item['video_id'], 
                    'video_title': item['video_title'], 
                    'video_header': item['video_header'],
                    'segment_idx': item['segment_idx'],
                    'time_start': item['time_start'],
                    'time_end': item['time_end']},
            )
        docs.append(doc)
    return docs # len: 6165


# pydantic
class QueryInput(BaseModel):
    session_id: str = Field(default=None)
    question: str
    chat_history: list = Field(default=[])

class QueryOutput(BaseModel):
    session_id: str
    answer: str
    docs: list