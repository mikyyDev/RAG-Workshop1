# ================================
# RAG WORKSHOP PIPELINE (Corrected)
# ================================

# Install first:
# pip install langchain langchain-community langchain-huggingface langchain-groq chromadb pypdf python-dotenv

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

# ================================
# 1. Load Environment Variables
# ================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================================
# 2. Load PDF Document
# ================================
print("\nLoading PDF...\n")

loader = PyPDFLoader("data/miky_cv.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# ================================
# 3. Split Text into Chunks
# ================================
print("\nSplitting document...\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # slightly smaller chunks for better granularity
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# Debug: print first few chunks to confirm name/email are captured
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}:\n{chunk.page_content}\n")

# ================================
# 4. Create Embeddings
# ================================
print("\nLoading embedding model...\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================================
# 5. Store in Vector Database
# ================================
print("\nCreating vector database...\n")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Use hybrid retriever (MMR balances keyword + semantic similarity)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k":10})
print("Vector DB Ready")

# ================================
# 6. Initialize LLM
# ================================
print("\nConnecting to Groq LLM...\n")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

print("LLM Ready")

# ================================
# 7. Ask Questions
# ================================
print("\nRAG Assistant Ready")
print("Type 'exit' to stop\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(query)

    print("\nRetrieved Context\n")
    for i, doc in enumerate(retrieved_docs):
        print(f"Chunk {i+1}:")
        print(doc.page_content[:200])  # preview first 200 chars
        print("------")

    # Combine context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Prompt
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.
If the answer is not in the context, say "I could not find it in the document".

Context:
{context}

Question:
{query}
"""

    # LLM Response
    response = llm.invoke(prompt)

    print("\nAI Answer:\n")
    print(response.content)
    print("\n==============================\n")