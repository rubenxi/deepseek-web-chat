# deepseek-web-chat
Simple chat built in Python to talk to DeepSeek or any model running in Ollama with PDF reading capabilities

![Screenshot from 2025-02-06 14-05-07](https://github.com/user-attachments/assets/f755bcde-7c72-4b5c-839e-fbe670552152)
![Screenshot from 2025-02-06 14-15-26](https://github.com/user-attachments/assets/6b4cb39f-d9b2-46c8-84c4-096229b90bef)


# Requirements
- Python 3
- pip
- streamlit
- langchain_core
- langchain_community
- langchain_ollama
- pdfplumber
- chromadb>=0.4.23
- Ollama running in background

# Install
Simply run in a Debian based system:
```
./install.sh
```
This will install all the packages needed and create a venv using Python3 and install all necessary libraries.

# Run
```
streamlit run ui.py
```

# How it works
You need to have Ollama running in background with a model loaded.

After that, run the app, and you can talk to the model (in this case deepseek-r1:1.5b) and it will answer.

You can also upload PDF files, the model will read, remember and use them to answer your questions. When you upload the PDF, the chat will show a new option "Question about the PDF".

The script takes the PDF file, splits it in vectors, stores them in ChromaDB and then uses them as context for the model.

# Install and run the DeepSeek model in Debian
Simply run in a Debian based system:
```
sudo apt install python3
sudo apt install python3-pip
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
ollama run deepseek-r1:1.5b
```
