
import streamlit as st
import requests
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/chat"
)

# Page configuration
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("Medical Document AI Assistant")
st.caption("Ask questions based on your medical literature.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Chat Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("context"):
            with st.expander("View Retrieved Context"):
                st.write(message["context"])

# Chat input
question = st.chat_input(
    "Ask a question about your documents..."
)

if question:

    # Previous conversation (before current question)
    history = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages
    ]

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    # Save user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            try:
                # Send question to FastAPI
                response = requests.post(
                    API_URL,
                    json={
                        "question": question,
                        "top_k": top_k,
                        "history": history
                    },
                    timeout=180
                )

                response.raise_for_status()
                result = response.json()

                answer = result["answer"]
                context = result["context"]

                st.markdown(answer)

                with st.expander("View Retrieved Context"):
                    st.write(context)

                # Save successful assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "context": context
                })

            except requests.exceptions.RequestException as e:
                st.error(
                    "Unable to connect to the chatbot backend."
                )
                st.caption(str(e))

            except (KeyError, ValueError) as e:
                st.error(
                    "The backend returned an unexpected response."
                )
                st.caption(str(e))
