import streamlit as st
from huggingface_hub import InferenceClient
import os
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.set_page_config(
    layout="wide",
    page_title="Simple DeepSeek Web Chat",
)

api_key = st.secrets["api_key"]
model = "deepseek-r1:1.5b"
template = """
Answer the user.

User said: {question}
Context: {context}
Your answer:
"""

template_server = """
Answer the user.

Context: {context}
User said: 
"""


pdf_directory = 'pdf/'
db_directory = 'vectordb'

os.makedirs(db_directory, exist_ok=True)

embeddings = OllamaEmbeddings(model=model)
vector_store = Chroma(persist_directory=db_directory, embedding_function=embeddings)
model = OllamaLLM(model=model)

def upload_pdf(file):
    with open(pdf_directory + file.name, "wb") as f:
        f.write(file.getbuffer())
        
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents
    
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=950,
    chunk_overlap=100,
    add_start_index=True
    )
    
    return text_splitter.split_documents(documents)
    
def index_docs(documents):
    vector_store.add_documents(documents)
    vector_store.persist()
    
def retrieve_docs(query):
    return vector_store.similarity_search(query)
    
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    print(context)
    return chain.invoke({"question": question, "context": context})

def remove_think(text):
    return re.sub(r'<think>.*?</think>','',text,flags=re.DOTALL).strip()

def answer_question_simple(question):
    context = ""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    print(context)
    return chain.invoke({"question": question, "context": context})

st.sidebar.title("Use HuggingFace server")
server = st.sidebar.toggle("HuggingFace", value=True)
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)


def answer_question_server(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    client = InferenceClient(api_key=api_key)

    messages = [
	    { "role": "user", "content": template + " context: " + context + " user question about context: " + question }
    ]

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
	        messages=messages, 
	        temperature=0.5,
	        max_tokens=2048,
	        top_p=0.7,
	        stream=True
        )
        ended_thinking = False
        for chunk in stream:
            if ended_thinking:
                yield chunk.choices[0].delta.content
            if "</think" in chunk.choices[0].delta.content:
                ended_thinking = True

def answer_question_server_simple(question):
    client = InferenceClient(api_key=api_key)

    messages = [
	    { "role": "user", "content": template_server + question }
    ]

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
	        messages=messages, 
	        temperature=0.5,
	        max_tokens=2048,
	        top_p=0.7,
	        stream=True
        )
        ended_thinking = False
        for chunk in stream:
            if ended_thinking:
                yield chunk.choices[0].delta.content
            if "</think" in chunk.choices[0].delta.content:
                ended_thinking = True

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdf_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)
    
    question = st.chat_input("Question about the PDF")
    
    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        if server:
            with st.spinner("Thinking...", show_time=True):
                response = st.write_stream(answer_question_server(question, related_documents))  
        else:
            with st.spinner("Thinking...", show_time=True):
                answer = answer_question(question, related_documents)
                st.chat_message("assistant").write(answer)




question = st.chat_input("Question")

if server:
    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        with st.spinner("Thinking...", show_time=True):
            response = st.write_stream(answer_question_server_simple(question))
else:
    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        with st.spinner("Thinking...", show_time=True):
            answer = answer_question_simple(question)
            st.chat_message("assistant").write(remove_think(answer))

