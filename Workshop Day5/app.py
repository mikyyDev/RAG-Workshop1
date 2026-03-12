import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# -----------------------------
# Embedding Model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Vector Database
# -----------------------------
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# -----------------------------
# Language Model (LLM)
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------
# Request Schema
# -----------------------------
class QuestionRequest(BaseModel):
    question: str


# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend is running"}

# -----------------------------
# Upload Endpoint
# -----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded PDF temporarily
        temp_path = "temp.pdf"
        with open(temp_path, "wb") as pdf_file:
            pdf_file.write(await file.read())

        # Load and split PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        document_chunks = splitter.split_documents(documents)

        # Add metadata for traceability
        for i, doc in enumerate(document_chunks):
            doc.metadata["chunk_id"] = i

        # Store chunks in vector DB
        vector_db.add_documents(document_chunks)
        vector_db.persist()

        return {"message": "Document uploaded successfully"}
    except Exception as e:
        return {"message": f"Upload failed: {str(e)}"}

# -----------------------------
# Ask Endpoint
# -----------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        # Retrieve relevant chunks
        retrieved_docs = vector_db.similarity_search(request.question, k=3)

        if not retrieved_docs:
            return {
                "question": request.question,
                "answer": "Not in document"
            }

        # Build context
        context = "\n\n---\n\n".join(
            [f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)]
        )

        system_prompt = (
            "You are a document assistant. Answer the user's question using ONLY the provided sources. "
            "If the answer is not found in the sources, respond with 'Not in document'. "
            "Always include the source numbers used in your response."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {request.question}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Generate response
        response = llm.invoke(messages)

        return {
            "question": request.question,
            "answer": response.content
        }

    except Exception as e:
        return {"question": request.question, "answer": f"Error: {str(e)}"}