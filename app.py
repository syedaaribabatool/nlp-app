
import streamlit as st
import requests

st.set_page_config(
    page_title="Ollama AI Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Ollama AI Chat")
st.caption("Chat with a locally running Ollama LLM")

OLLAMA_URL = "http://localhost:11434/api/generate"

models = [
    "llama3.2",
    "gemma3",
    "qwen2.5"
]

model = st.selectbox(
    "Select Model",
    models
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask something...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:

                answer = response.json()["response"]

                st.write(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            else:
                st.error("Ollama server is not responding.")
