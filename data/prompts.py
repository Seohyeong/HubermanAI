SYSTEM_PROMPT = """
You are a helpful AI assistant for creating a Question and Answering dataset.
Given the CONTEXT and its HEADER,  generate a question that general people might ask, which can be answered using the CONTEXT.
Output ONLY the question.
"""

TEXT_QA_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using only the information from the context, answer the query in clear and grammatically correct sentences.\n"
    "Note: The context may contain informal or ungrammatical phrases as it is extracted from a YouTube video.\n"
    "Use all relevant information from the context but do not make up any facts that are not given.\n"
    "Query: {query_str}\n"
    "Answer: "
)