from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline

# --------------- Setup ---------------
app = FastAPI()

# Load lightweight model (works on Render free)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=256)

# Static preloaded summary from PDFs (you can enhance later)
pdf_summary_text = """
This collection of PDFs discusses redemption through the blood of Jesus Christ,
the biblical concept of grace, and deliverance from spiritual slavery.
It explores Ephesians 1:7, Deuteronomy 28, and other core scriptures.
"""

# --------------- Request Schema ---------------
class Question(BaseModel):
    query: str

# --------------- API Route ---------------
@app.post("/ask")
async def ask_question(payload: Question):
    query = payload.query
    prompt = f"Based on the following notes:\n{pdf_summary_text}\nAnswer this: {query}"
    result = qa_pipeline(prompt)[0]['generated_text']
    return {"question": query, "answer": result}
