from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Step 1: Load PDFs
def extract_text_from_pdfs(pdf_paths):
    full_text = ""
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
    return full_text

pdf_files = ["data/pdf1.pdf", "data/pdf2.pdf", "data/pdf3.pdf"]
docs_text = extract_text_from_pdfs(pdf_files)

# Step 2: Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(docs_text)

# Step 3: Create embeddings and store in FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(chunks, embedding_model)
db.save_local("faiss_index")

# Step 4: Setup QA
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
qa_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=qa_pipeline)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), chain_type="stuff")

# Step 5: Ask question
query = input("Ask a question: ")
answer = qa.run(query)
print("Answer:", answer)
