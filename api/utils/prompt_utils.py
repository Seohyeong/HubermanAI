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

# query validation prompt
QUERY_VALIDATION_PROMPT = (
    "Validate the given QUERY based on the following rules:\n"
    "If the QUERY is irrelevant, output FALSE.\n"
    "If the QUERY is relevant, output TRUE.\n\n"
    "A QUERY is relevant if it pertains to Andrew Huberman's podcast, which provides science-based tools for everyday life, including topics such as:\n"
    "Brain health\n"
    "Sleep and sleep hygiene\n"
    "Fitness\n"
    "Supplementation\n"
    "Mental health\n"
    "Cold plunges\n"
    "Light and sun exposure\n"
    "Skin care\n"
    "Hormones\n\n"
    "A QUERY is irrelevant if it does not relate to these topics or falls under the following categories:\n"
    "Politics\n"
    "Illegal activities\n"
    "Self-harm\n\n" 
    "Output either TRUE or FALSE.\n"
    "QUERY: {query}\n"
    "Answer: "
)

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
HISTORY_AWARE_CHAT_PROMPT = (
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
IRRELEVANT_MESSAGE_PROMPT = (
    "I'm here to answer questions related to science-based tools for everyday life, as discussed in Andrew Huberman's podcast. " 
    "Topics include brain health, sleep, fitness, supplementation, mental health, and more. "
    "However, I couldn't find any relevant supporting documents to answer your question.\n"
    "If you have a question in these areas, feel free to ask!"
)

# relevant, irrelevant training set generation
Q_AZER_PROMPT =(
    """
    I'm training an embedding model to tell whether a user query is relevant to the RAG system I'm building. 
    My RAG system is built on Andrew Huberman's podcast, which provides science-based tools for everyday life, including topics such as Brain health, Sleep and sleep hygiene, Fitness, Supplementation, Mental health, Light and Sun exposure, Skin care, Hormones, etc. 
    I want you to create 500 relevant questions and 500 irrelevant questions. Make sure to generate questions that a general user might ask. 

    Give 2 json files, with each for relevant questions and irrelevant questions. Use the following format:

    {"question": "How do I play the guitar?"}
    {"question": "What’s the best way to make friends?"}
    {"question": "How do I become a stand-up comedian?"}
    ...
    """
)