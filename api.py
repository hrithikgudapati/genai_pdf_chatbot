from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline

# --------------- Setup ---------------
app = FastAPI()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Load smaller, fast model
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), chain_type="stuff")


# --------------- Request Schema ---------------
class Question(BaseModel):
    query: str


# --------------- API Route ---------------
@app.post("/ask")
async def ask_question(payload: Question):
    answer = qa.invoke(payload.query)
    return {"question": payload.query, "answer": answer}
