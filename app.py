import os
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

model = "deepseek-r1:1.5b"
template = """
Answer the user.

User said: {question}
Context: {context}
Your answer:
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
    
