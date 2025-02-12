# deepseek-web-chat
Simple chat built in Python to talk to DeepSeek or any model running in Ollama with PDF reading capabilities.  
It supports connection with local Ollama and HuggingFace.

![Screenshot from 2025-02-11 19-05-29](https://github.com/user-attachments/assets/0d390654-9a66-43af-ba85-06a43b50d5dd)
![Screenshot from 2025-02-11 19-06-57](https://github.com/user-attachments/assets/2f42a46c-3795-4b32-89fa-9d00c2fc0a56)


# Requirements
- Python 3
- pip
- streamlit
- langchain_core
- langchain_community
- langchain_ollama
- pdfplumber
- chromadb>=0.4.23
- huggingface_hub
- Ollama running in background for local connection

# Install
Simply run in a Debian based system:
```
chmod +x ./install.sh
./install.sh
```
This will install all the packages needed and create a venv using Python3 and install all necessary libraries.

# Run
```
source ./OllamaWebPDF_env/bin/activate
streamlit run ui.py
```

# How it works
You need to have Ollama running in background with a model loaded.

After that, run the app, and you can talk to the model (in this case deepseek-r1:1.5b) and it will answer.

You can also upload PDF files, the model will read, remember and use them to answer your questions. When you upload the PDF, the chat will show a new option "Question about the PDF".

The script takes the PDF file, splits it in vectors, stores them in ChromaDB and then uses them as context for the model.

# Works online!
There's an option to enable HuggingFace servers so it uses an online model (DeepSeek-R1-Distill-Qwen-32B).

# Install and run the DeepSeek model in Debian
Simply run in a Debian based system:
```
sudo apt install python3
sudo apt install python3-pip
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
ollama run deepseek-r1:1.5b
```
