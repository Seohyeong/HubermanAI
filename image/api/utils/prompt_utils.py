# prompt formatting
def docs2str(docs):
    return "\n\n".join(
        f"title: {doc.metadata['video_title']}\n"
        f"header: {doc.metadata['video_header']}\n"
        f"content: {doc.page_content}"
        for doc in docs
    )

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
    "Using only the information provided above, answer the query in clear and grammatically correct sentences.\n"
    "Note: The source material may contain informal or ungrammatical phrases as it is extracted from a YouTube video.\n"
    "Use all relevant information but do not make up any facts that are not given.\n"
    "Present the information directly without referring to 'the context,' 'the speaker,' or the source material.\n"
    "Provide a detailed and explicit answer of at least 300 words rather than a simple summary.\n"
    "Question: {question}\n"
    "Answer: "
)

# history aware retriever prompt
QUERY_CONTEXTUALIZER_PROMPT = (
    "Given a CHAT HISTORY and the latest user QUERY which might reference context in the CHAT HISTORY, " 
    "formulate a standalone QUERY which can be understood WITHOUT the CHAT HISTORY.\n"
    "NEVER respond or answer to the QUERY, just reformulate it if needed and otherwise return it as is. "
    "If the CHAT HISTOIRY is an empty list, output the QUERY as is.\n\n"
    "The CHAT HISTORY is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "QUERY: {question}\n"
    "OUTPUT: "
)

# answer for non relevant query
IRRELEVANT_QUERY_PROMPT = (
    "Hi! I'm here to answer questions related to science-based tools for everyday life, as discussed in Andrew Huberman's podcast. " 
    "Topics include brain health, sleep, fitness, supplementation, mental health, and more. "
    "However, I couldn't find any relevant supporting documents to answer your question.\n"
    "If you have a question in these areas, feel free to ask!"
)