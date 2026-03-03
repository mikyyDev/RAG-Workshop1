import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -----------------------------
# Step 1: Load environment variables
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# -----------------------------
# Step 2: Sample documents
# -----------------------------
documents = [
  {
    "title": "I am Michael Alula",
    "content": (
        "I am Michael Alula. I am a 3rd year Information Systems student. "
        "I am currently working as a React and Next.js frontend developer. "
        "I am curious about how chatbots work, which is why I joined IShub "
        "to learn about RAG (Retrieval-Augmented Generation). "
        "The reason I am learning RAG is to understand how chatbots retrieve information "
        "and generate useful answers."
    )
},
    {
        "title": "Universities in Ethiopia",
        "content": (
            "Some universities in Ethiopia include Addis Ababa University, "
            "Haramaya University, Jimma University, Mekelle University, "
            "Bahir Dar University, Hawassa University, and Arba Minch University."
        )
    }
]

# -----------------------------
# Step 3: Improved Retriever
# -----------------------------
def retrieve_doc(query: str) -> str:
    # Normalize query: lowercase, strip punctuation
    query_words = re.findall(r'\w+', query.lower())
    matched_docs = []
    for doc in documents:
        if any(word in doc["content"].lower() for word in query_words):
            matched_docs.append(doc["content"])
    return "\n".join(matched_docs) if matched_docs else "No document found."

# -----------------------------
# Step 4: Initialize Groq LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=groq_api_key
)

# -----------------------------
# Step 5: Test Questions
# -----------------------------
test_questions = [
    "Who is Michael Alula?",
    "What universities are in Ethiopia?",
    "Why are you learning RAG?"
]

for i, user_question in enumerate(test_questions, 1):
    print(f"Question {i}: {user_question}")
    print("=" * 70)

    # WITHOUT retrieval
    print("Answer WITHOUT retrieval:")
    print("-" * 60)
    print(llm.invoke(user_question).content.strip())
    print("-" * 60)

    # WITH retrieval (RAG)
    print("\nAnswer WITH retrieval (RAG):")
    print("-" * 60)

    context = retrieve_doc(user_question)

    rag_prompt = f"""
You are a helpful assistant. Answer the question using ONLY the provided context.
If the information is not in the context, respond only with: "I don't know."

Context:
{context}

Question: {user_question}

Answer:
"""

    print(llm.invoke(rag_prompt).content.strip())
    print("-" * 60)
    print()  # empty line between questions