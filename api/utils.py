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
    "Do not simply summarize the answer based on the given context, give a detailed and explicit answer but keep it concise.\n"
    "Format the response properly, using paragraphs for better readability. Use bold text and bullet points where necessary, but avoid excessive markdown formatting."
)

# answer for non relevant query
IRRELEVANT_MESSAGE = (
    "I'm here to answer questions related to science-based tools for everyday life, as discussed in Andrew Huberman's podcast. " 
    "Topics include brain health, sleep, fitness, supplementation, mental health, and more. "
    "However, I couldn't find any relevant supporting documents to answer your question.\n"
    "If you have a question in these areas, feel free to ask!"
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
    with open(json_path, "r", encoding = "utf-8") as f:
        data = [json.loads(line) for line in f]
        
    docs = []
    for item in data:
        doc = Document(
            page_content=item["context"],
            metadata={"video_id": item["video_id"], 
                    "video_title": item["video_title"], 
                    "video_header": item["video_header"],
                    "segment_idx": item["segment_idx"],
                    "time_start": item["time_start"],
                    "time_end": item["time_end"]},
            )
        docs.append(doc)
    return docs # len: 6165


# evaluating
def get_rr(gt_doc_id: str, pred_doc_ids: list[str]) -> float:
    rr= 0
    try:
        rr = 1 / (pred_doc_ids.index(gt_doc_id) + 1)
    except ValueError:
        rr = 0
    return rr

# pydantic
class RAGDoc(BaseModel):
    video_id: str
    title: str
    header: str
    time_start: str
    time_end : str
    segment_idx: str = Field(default=None)
    score: float = Field(default=None)
    
class RAGOutput(BaseModel):
    answer: str = Field(default=None)
    docs: list[RAGDoc] = Field(default=[])
    
class QueryInput(BaseModel):
    session_id: str = Field(default=None)
    question: str
    chat_history: list = Field(default=[])

class QueryOutput(BaseModel):
    session_id: str
    answer: str
    docs: list[RAGDoc] = Field(default=[])