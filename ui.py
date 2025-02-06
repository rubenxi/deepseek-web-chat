import app
import streamlit as st


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    app.upload_pdf(uploaded_file)
    documents = app.load_pdf(app.pdf_directory + uploaded_file.name)
    chunked_documents = app.split_text(documents)
    app.index_docs(chunked_documents)
    
    question = st.chat_input("Question about the PDF")
    
    if question:
        st.chat_message("user").write(question)
        related_documents = app.retrieve_docs(question)
        answer = app.answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)


question = st.chat_input("Question")
    
if question:
    st.chat_message("user").write(question)
    related_documents = app.retrieve_docs(question)
    answer = app.answer_question_simple(question)
    st.chat_message("assistant").write(app.remove_think(answer))


