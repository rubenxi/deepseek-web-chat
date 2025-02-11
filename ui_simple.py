import streamlit as st
from huggingface_hub import InferenceClient
import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

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

model = OllamaLLM(model=model)

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

question = st.chat_input("Question")

if server:
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Thinking...", show_time=True):
            response = st.write_stream(answer_question_server_simple(question))
else:
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Thinking...", show_time=True):
            answer = answer_question_simple(question)
            st.chat_message("assistant").write(remove_think(answer))

