import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index with safe deserialization
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Load language model
qa_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Create the retrieval-based QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), chain_type="stuff")

# Streamlit UI
st.title("PDF Chatbot")
query = st.text_input("Ask a question:")
if query:
    answer = qa.invoke(query)
    st.write(answer)
