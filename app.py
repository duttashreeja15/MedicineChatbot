
import streamlit as st
from src.retrieval_and_generation import ask_pdf

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("Medical Document AI Assistant")
st.caption("Ask questions based on your uploaded medical literature.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
question = st.chat_input("Ask a question about your documents...")

if question:
    # Display and store user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                history = st.session_state.messages[:-1]

                answer, context = ask_pdf(
                    question,
                    top_k=5,
                    history=history
                )

                st.markdown(answer)

                # Show retrieved context
                with st.expander("View retrieved context"):
                    st.write(context)

            except Exception as e:
                answer = "Unable to generate an answer."
                st.error(str(e))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# Clear history button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
