# ================================
# RAG WORKSHOP PIPELINE - Improved Output Style
# ================================

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
    chunk_size=550,
    chunk_overlap=120
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ================================
# 4. Create Embeddings & Vector Store
# ================================
print("\nCreating embeddings and vector database...\n")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_miky"
)

# Using MMR for better diversity
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)
print("Vector DB Ready")

# ================================
# 5. Initialize LLM
# ================================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)
print("LLM Ready\n")

# ================================
# 6. Main Interaction Loop
# ================================
print("RAG CV Assistant ready! (type 'exit' to quit)\n")

while True:
    query = input("Ask: ").strip()

    if query.lower() in ["exit", "quit", "q"]:
        print("Goodbye!\n")
        break

    if not query:
        continue

    # Retrieve
    retrieved_docs = retriever.invoke(query)

    # Debug: show what was retrieved
    print("\nRetrieved chunks:")
    for i, doc in enumerate(retrieved_docs, 1):
        preview = doc.page_content.replace("\n", " ").strip()[:180]
        print(f"  {i}. {preview}…")
    print("-" * 70)

    # Combine context
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Very strict check: is there actually useful content?
    has_content = any(len(doc.page_content.strip()) > 30 for doc in retrieved_docs)

    if not has_content or not context.strip():
        print("\nAnswer:")
        print("You asked something outside of the context / not mentioned in the CV.")
        print("-" * 70)
        continue

    # Better prompt with bullet-point preference
    prompt = f"""You are a helpful assistant that answers questions about Mikyy's CV.

Rules you MUST follow:
- ONLY use information that appears in the provided context.
- If the question is not answerable from the context → do NOT try to guess or use outside knowledge.
- When you have relevant information → format your answer using bullet points (- or •) when it makes sense (skills, experience, education, achievements, etc.).
- When it's a short fact → one clear sentence is okay.
- Be concise, professional and natural.
- Never apologize or say "based on the context" — just give the answer.

Context:
{context}

Question: {query}

Answer:"""

    # Get response
    response = llm.invoke(prompt)
    answer_text = response.content.strip()

    print("\nAnswer:")
    print(answer_text)
    print("-" * 70)