import streamlit as st
from huggingface_hub import InferenceClient
import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from datetime import datetime
import pickle

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

model = OllamaLLM(model=model)
date_file = "date_file.pkl"
n_file = "n.pkl"

def load_date():
    if os.path.exists(date_file):
        with open(date_file, "rb") as f:
            return pickle.load(f)
    return []


def save_date(date_today):
    with open(date_file, "wb") as f:
        pickle.dump(date_today, f)

def load_n():
    if os.path.exists(n_file):
        with open(n_file, "rb") as f:
            return pickle.load(f)
    return []


def save_n(n):
    with open(n_file, "wb") as f:
        pickle.dump(n, f)



def remove_think(text):
    return re.sub(r'<think>.*?</think>','',text,flags=re.DOTALL).strip()

def answer_question_simple(question):
    context = ""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    print(context)
    return chain.invoke({"question": question, "context": context})

st.sidebar.title("Use HuggingFace server")
api_key_user = st.sidebar.text_input("Api key", placeholder="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX", help="Set your own HuggingFace api key. You can get one here: https://huggingface.co/settings/tokens/new?tokenType=read")

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
        current_date = datetime.today().strftime('%Y-%m-%d')
        last_date = load_date()
        if current_date != last_date:
            save_date(current_date)
            save_n(0)
        if api_key_user:
            api_key = api_key_user
        elif load_n()<=5:
            api_key = st.secrets["api_key"]
        else:
            st.chat_message("assistant").write("""**⚠️ Rate Limit ⚠️**

My website uses an api key that is free, so it may hit a limit at some point

Try again tomorrow or use your own api key...
                                """)
        if api_key_user or (not api_key_user and load_n()<=5):
            st.chat_message("user").write(question)
            with st.spinner("Thinking...", show_time=True):
                try:
                    response = st.write_stream(answer_question_server_simple(question))
                    save_n(load_n()+1)
                except Exception:
                    st.chat_message("assistant").write("""**⚠️ Rate Limit ⚠️**
        
My website uses an api key that is free, so it may hit a limit at some point
    
Try again tomorrow or use your own api key...
                                                    """)
else:
    if question:
        st.chat_message("user").write(question)
        try:
            with st.spinner("Thinking...", show_time=True):
                answer = answer_question_simple(question)
                st.chat_message("assistant").write(remove_think(answer))
        except Exception as e:
            st.chat_message("assistant").write("Error: " + str(e))

