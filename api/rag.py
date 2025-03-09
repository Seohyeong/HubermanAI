# langchain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# mlflow
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# utils
from utils.db_utils import init_db
from utils.model_utils import init_embedding_model, init_llm_model
from utils.prompt_utils import docs2str, \
    CHAT_PROMPT, \
    CONTEXTUALIZE_Q_SYSTEM_PROMPT, HISTORY_AWARE_CHAT_PROMPT, \
    IRRELEVANT_MESSAGE_PROMPT
from utils.eval_utils import test_retriever
from utils.utils import init_config, validate_docs, \
    RAGDoc, RAGOutput



class RagChatbot():
    def __init__(self, llm_model_name):
        self.llm_model_name = llm_model_name
        self.config = init_config(self.llm_model_name)
        self.embedding_function = init_embedding_model(self.config.embedding_model)
        self.llm = init_llm_model(self.llm_model_name, self.config.generation_model)
        self.vectorstore = init_db(self.config, self.embedding_function)
        
        # basic retriever
        self.retriever = self.vectorstore.as_retriever(search_type=self.config.search_type, 
                                                       search_kwargs={"score_threshold": self.config.score_threshold})
        
        # history aware retriever
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}")])
        self.history_aware_retriever = create_history_aware_retriever(self.llm, 
                                                                      self.retriever, 
                                                                      self.contextualize_q_prompt)
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [("system", HISTORY_AWARE_CHAT_PROMPT),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}")])
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)
        
    def retrieve(self, query, k):
        """basic retriever"""
        docs, scores = zip(*self.vectorstore.similarity_search_with_relevance_scores(query, k=k))
        return RAGOutput(docs=[RAGDoc(video_id=doc.metadata.get("video_id"),
                                        title=doc.metadata.get("video_title"),
                                        header=doc.metadata.get("video_header"),
                                        time_start=doc.metadata.get("time_start"),
                                        time_end=doc.metadata.get("time_end"),
                                        segment_idx=doc.metadata.get("segment_idx"),
                                        score=score)
                                for doc, score in zip(docs, scores)])
        
    def invoke(self, query):
        """basic retriever + llm"""
        docs = self.retriever.invoke(query)
        if not docs:
            return RAGOutput(answer=IRRELEVANT_MESSAGE_PROMPT)
        else:
            docs = validate_docs(docs)
            context = docs2str(docs)
            prompt = ChatPromptTemplate.from_template(CHAT_PROMPT)
            llm_response = self.llm.invoke(prompt.format(context=context, question=query))
            
            return RAGOutput(answer=llm_response.content,
                            docs=[RAGDoc(video_id=doc.metadata.get("video_id"),
                                            title=doc.metadata.get("video_title"),
                                            header=doc.metadata.get("video_header"),
                                            time_start=doc.metadata.get("time_start"),
                                            time_end=doc.metadata.get("time_end"),
                                            segment_idx=doc.metadata.get("segment_idx"))
                                    for doc in docs])
        
            
    def invoke_with_history(self, query, chat_history):
        """rag_chain = history_aware_retriever + llm"""
        llm_response = self.rag_chain.invoke({"input": query, "chat_history": chat_history})
        if "context" in llm_response.keys():
            docs = validate_docs(llm_response["context"])
        if not docs:
            return RAGOutput(answer=IRRELEVANT_MESSAGE_PROMPT)
        else:
            return RAGOutput(answer=llm_response["answer"],
                            docs=[RAGDoc(video_id=doc.metadata.get("video_id"),
                                            title=doc.metadata.get("video_title"),
                                            header=doc.metadata.get("video_header"),
                                            time_start=doc.metadata.get("time_start"),
                                            time_end=doc.metadata.get("time_end"),
                                            segment_idx=doc.metadata.get("segment_idx"))
                                    for doc in docs])


def test_with_chat_history(rag_chain):
    from langchain_core.messages import HumanMessage, AIMessage
    
    questions = [
        "Is chemical sunscreen bad for you?",
        "Where can I find it?",
        "Hello",
        "How do I kill myself?"
        ]
    
    chat_history = []

    for q in questions:
        print(f"QUESTION:\n{q}\n")
        # simple_output = rag_chain.invoke(q)
        # print(f"SIMPLE_ANSWER:\n{simple_output.answer}\n")
        history_output = rag_chain.invoke_with_history(q, chat_history)
        print(f"HISTORY_ANSWER:\n{history_output.answer}\n\n\n")
        
        # only append relevant query pair to the chat history
        # TODO: if current query is irrelevant, erase the chat history
        if history_output.docs:
            chat_history.extend([
                HumanMessage(content=q),
                AIMessage(content=history_output.answer)
            ])
    
    
def main():
    rag_chain = RagChatbot("cohere")
    
    # test_with_chat_history(rag_chain)
    test_retriever(rag_chain, k=3, threshold=0.5)

if __name__ == "__main__":
    main()