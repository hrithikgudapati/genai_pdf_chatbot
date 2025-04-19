from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline

# --------------- Setup ---------------
app = FastAPI()

# Load lightweight model
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=256)

# Dummy summary from your PDFs
pdf_summary_text = """
These documents discuss redemption through the blood of Jesus,
the biblical idea of deliverance from slavery, and spiritual restoration.
Key references include Ephesians 1:7 and Deuteronomy 28.
"""

# --------------- Request Schema ---------------
class Question(BaseModel):
    query: str

# --------------- API Route ---------------
@app.post("/ask")
async def ask_question(payload: Question):
    prompt = f"Based on the following text:\n{pdf_summary_text}\nAnswer this: {payload.query}"
    result = qa_pipeline(prompt)[0]['generated_text']
    return {"question": payload.query, "answer": result}
