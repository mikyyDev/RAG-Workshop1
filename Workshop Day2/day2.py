import sys

# Ensure stdout can handle emojis on Windows console
sys.stdout.reconfigure(encoding='utf-8')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# -----------------------------------------
# 1️⃣ My Personal Documents (About Mikyy)
# -----------------------------------------

documents = [
    "I am Mikyy (Michael Alula). I am a 3rd year Information Systems student at Addis Ababa University.",
    "I am currently working as a React and Next.js frontend developer. I build responsive and modern web applications.",
    "My goal is to become a strong full-stack developer and contribute to real AI-powered products in Ethiopia.",
    "I joined IShub community because I am very curious about how chatbots and large language models work.",
    "I am actively learning Retrieval-Augmented Generation (RAG) to make LLMs give more accurate and grounded answers.",
    "So far I have built a simple RAG system using LangChain, Chroma vector database, HuggingFace embeddings, and Groq LLM.",
    "My biggest achievement in 2025 is understanding how vector similarity search and embeddings turn text into meaning.",
    "I love clean code, good UI/UX design, and solving real problems with technology.",
    "In the future I want to combine my frontend skills with AI to create smart and beautiful web experiences."
]

# Convert strings to LangChain Document objects
docs = [Document(page_content=text) for text in documents]

# -----------------------------------------
# 2️⃣ Load Embedding Model
# -----------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------------
# 3️⃣ Create Vector Store (about me)
# -----------------------------------------

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./my_chroma_db"    
)

print("✅ My personal vector database created successfully.")

# -----------------------------------------
# 4️⃣ Similarity Search Function
# -----------------------------------------

def ask_about_mikyy(query, top_k=2):
    print(f"\n🔎 Question: {query}")
    print(f"Top similar facts: {top_k}")

    results = vectorstore.similarity_search(query, k=top_k)

    for i, result in enumerate(results):
        print(f"\nFact {i+1}:")
        print(result.page_content)
        print("-" * 60)

# -----------------------------------------
# 5️⃣ Run Example Questions about me
# -----------------------------------------

print("\n" + "="*50)
print("   Mikyy Personal RAG - Ask anything about me!")
print("="*50)

ask_about_mikyy("Who is Mikyy?", top_k=2)
ask_about_mikyy("What is Mikyy studying?", top_k=1)
ask_about_mikyy("What does Mikyy do as a job?", top_k=2)
ask_about_mikyy("Why is Mikyy learning RAG?", top_k=3)
ask_about_mikyy("What are Mikyy's goals for the future?", top_k=2)
ask_about_mikyy("What has Mikyy achieved recently?", top_k=2)
ask_about_mikyy("What technologies does Mikyy use?", top_k=3)