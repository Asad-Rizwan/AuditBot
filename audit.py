__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import json
from pathlib import Path
import streamlit as st
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import io
import time
from streamlit_mic_recorder import mic_recorder, speech_to_text
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
log_file = 'chatbot_logs.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2)])

logger = logging.getLogger(__name__)

API_KEY = st.secrets.get('OPENAI_API_KEY')
conversation_placeholder = st.empty()

global shush
shush = 0


class Document:
    def __init__(self, page_content, metadata):
        if page_content is None or page_content == "":
            raise ValueError("page_content must be a non-empty string")
        self.page_content = page_content
        self.metadata = metadata


def clear_chat():
    # Remove the audio file used in the whisper_stt function
    file_path = "./audio.mp3"
    if os.path.exists(file_path):
        os.remove(file_path)
    st.success("Chat history cleared. You can start a new conversation.")
    chat_history.clear()
    st.session_state.messages = []
    st.session_state.context_history = []
    conversation_placeholder.empty()
    logger.info("Chat history cleared.")
    st.rerun()


def load_documents_from_directory(directory: str) -> List[Document]:
    documents = []
    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        #         if filename.endswith(".json"):
        #             loader = JSONLoader(file_path=file_path,jq_schema='.[]',text_content=False )
        # #            documents.extend(load_json_documents(file_path))
        # #            continue  # Skip processing the rest of the loop for JSON files
        # #        else:
        # #            continue  # Skip files that are not supported
        documents.extend(loader.load())
    return documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


st.title('🤖 OpenAI Audit Chatbot')

# openai_api_key = st.secrets.get('OPENAI_API_KEY')
if not API_KEY:
    API_KEY = st.text_input('Enter OpenAI API token:', type='password')
    if not (API_KEY.startswith('sk-') and len(API_KEY) == 51):
        st.warning('Please enter your credentials!')
        st.stop()
    st.secrets['OPENAI_API_KEY'] = API_KEY
else:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

data_directory = "./data/"

if 'loaded_documents' not in st.session_state:
    loaded_documents = load_documents_from_directory(data_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(loaded_documents)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    persist_directory = 'chroma'
    if not os.path.exists(persist_directory):
        st.success("Database persisted!")
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embeddings,
                                         persist_directory=persist_directory)
        vectordb.persist()
    else:
        st.success("Database loaded!")
        vectordb = Chroma(persist_directory=persist_directory,
                          embedding_function=embeddings)

    st.session_state.loaded_documents = loaded_documents
    st.session_state.vectordb = vectordb
    st.session_state.context_history = []
else:
    loaded_documents = st.session_state.loaded_documents
    vectordb = st.session_state.vectordb

qa_system_prompt = """
Welcome! You are interacting with an audit assistant bot specializing in providing helpful responses related to audit reports, observations, implications, recommendations, and management responses. Your role is to offer accurate and informative guidance based on the provided context.

When responding to queries, adhere to the following guidelines:
- Address the user's questions directly.
- Utilize the available context data effectively to tailor responses.
- Provide concise and complete responses within 2-3 sentences.
- Strive to offer valuable information and assistance.
- For personalized advice, it's recommended to consult a professional auditor.

For instance, if a user inquires about a specific observation or recommendation, provide a detailed explanation based on the context. Similarly, if they seek information on audit best practices, you can share relevant insights.

Here's the available context data for generating responses:
<context>
{context}
</context>

Please bear in mind that while this assistant aims to be helpful, it does not replace professional audit advice. For personalized and accurate audit guidance, consulting a professional auditor is advised. If you have any queries or require further assistance, feel free to ask!
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_q_system_prompt = """Given the chat history and the latest user question,
rephrase the user's question into a standalone question that does not rely on previous context.
The reformulated question should be understandable without referencing the chat history.
Do NOT answer the question, only provide a rephrased version if necessary.
Otherwise, return the question as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo")
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
chat_history = st.session_state.context_history if 'context_history' in st.session_state else []
rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
)


def log_message(role, content):
    logger.info(f"{role.capitalize()} message: {content}")


prompt = st.chat_input("What would you like to ask?")

if prompt:
    log_message("user", prompt)  # Log user message
    new_chat_session = False
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        # docs = retriever.get_relevant_documents(prompt)
        # st.write(docs)
    with st.chat_message("assistant"):
        ai_msg = rag_chain.invoke({"question": prompt, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=prompt))
        chat_history.append(ai_msg)
        log_message("assistant", ai_msg)  # Log assistant message
        st.markdown(ai_msg)

    st.session_state.messages.append({"role": "assistant", "content": ai_msg})


if st.button("New Chat"):
    clear_chat()
