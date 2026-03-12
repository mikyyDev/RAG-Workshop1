import streamlit as st
import requests
from requests import RequestException

API_URL = "http://127.0.0.1:8000"

# Initialize chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("📚 RAG Assistant")

# ---------------------------
# PDF Upload Section
# ---------------------------
pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if pdf_file is not None:
    file_data = {
        "file": (
            pdf_file.name,
            pdf_file.getvalue(),
            "application/pdf"
        )
    }
    try:
        upload_response = requests.post(f"{API_URL}/upload", files=file_data, timeout=120)
        if upload_response.status_code == 200:
            st.success(upload_response.json().get("message", "Document uploaded successfully"))
        else:
            st.error(upload_response.text)
    except RequestException as e:
        st.error(f"Could not reach backend at {API_URL}. Error: {e}")

# ---------------------------
# Question Input Section
# ---------------------------
user_question = st.text_input("Ask a question about the document")

if st.button("Ask Question") and user_question:
    try:
        ask_response = requests.post(
            f"{API_URL}/ask",
            json={"question": user_question},
            timeout=120
        )
        ask_response.raise_for_status()
        result = ask_response.json()
        answer_text = result.get("answer", "No answer received.")

        st.write("### Answer")
        st.write(answer_text)

        # Save Q&A in history
        st.session_state["history"].append({
            "question": user_question,
            "answer": answer_text
        })

    except RequestException as e:
        st.error(f"Could not reach backend at {API_URL}. Error: {e}")
    except ValueError:
        st.error("Backend returned an invalid response format.")

# ---------------------------
# Chat-style History Section
# ---------------------------
st.divider()
st.subheader("💬 Conversation History")

for record in st.session_state["history"]:
    st.markdown(f"**You:** {record['question']}")
    st.markdown(f"**Assistant:** {record['answer']}")
    st.write("---")