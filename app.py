"""
Streamlit front-end for a local Ollama server.

Tabs:
  1. Chat              - streaming chat with any pulled Ollama model
  2. Review Sentiment  - classify one McDonald's review as positive / neutral / negative
  3. Batch (CSV)       - classify a whole CSV column and download the results

Run:  streamlit run app.py
Needs Ollama running at OLLAMA_URL (default http://localhost:11434).
"""

import io
import json
import os

import pandas as pd
import requests
import streamlit as st

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LABELS = ["positive", "neutral", "negative"]

st.set_page_config(page_title="Ollama Review Assistant", page_icon="🍟", layout="wide")


# ------------------------------------------------------------------ Ollama helpers
@st.cache_data(ttl=30)
def list_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def chat_stream(model, messages, temperature):
    """Yield text chunks from Ollama's streaming /api/chat endpoint."""
    with requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": True,
              "options": {"temperature": temperature}},
        stream=True, timeout=600,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(data["error"])
            yield data.get("message", {}).get("content", "")
            if data.get("done"):
                break


SENTIMENT_SYSTEM = (
    "You classify McDonald's customer reviews. "
    "Reply ONLY with JSON: "
    '{"sentiment": "positive" | "neutral" | "negative", "reason": "<one short sentence>"}'
)


def classify(model, review):
    """Return (label, reason) for one review using Ollama's JSON mode."""
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "stream": False, "format": "json",
              "options": {"temperature": 0},
              "messages": [{"role": "system", "content": SENTIMENT_SYSTEM},
                           {"role": "user", "content": str(review)[:2000]}]},
        timeout=300,
    )
    r.raise_for_status()
    content = r.json()["message"]["content"]
    try:
        data = json.loads(content)
        label = str(data.get("sentiment", "")).lower().strip()
        reason = data.get("reason", "")
    except json.JSONDecodeError:
        label, reason = content.lower(), ""
    label = next((l for l in LABELS if l in label), "unknown")
    return label, reason


# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚙️ Settings")
models = list_models()
if not models:
    st.sidebar.error(f"Ollama is not reachable at {OLLAMA_URL}, or no model is pulled yet.")
    st.title("🍟 Ollama Review Assistant")
    st.warning("Start Ollama and pull a model (see the Colab notebook), then refresh this page.")
    st.stop()

model = st.sidebar.selectbox("Model", models)
temperature = st.sidebar.slider("Temperature (chat)", 0.0, 1.5, 0.7, 0.1)
system_prompt = st.sidebar.text_area(
    "System prompt (chat)",
    "You are a helpful assistant. Answer clearly and concisely.",
    height=100,
)
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
st.sidebar.caption(f"Backend: {OLLAMA_URL}")

st.title("🍟 Ollama Review Assistant")
tab_chat, tab_one, tab_batch = st.tabs(["💬 Chat", "⭐ Review Sentiment", "📄 Batch (CSV)"])

# ------------------------------------------------------------------ tab 1: chat
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            history = [{"role": "system", "content": system_prompt}] + st.session_state.messages
            try:
                reply = st.write_stream(chat_stream(model, history, temperature))
            except Exception as e:
                reply = f"⚠️ Error: {e}"
                st.error(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# ------------------------------------------------------------------ tab 2: single review
with tab_one:
    review = st.text_area("Paste a customer review", height=150,
                          placeholder="The fries were cold and the staff was rude...")
    if st.button("Analyze", type="primary", disabled=not review.strip()):
        with st.spinner("Thinking..."):
            try:
                label, reason = classify(model, review)
                icon = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(label, "⚪")
                st.subheader(f"{icon} {label.capitalize()}")
                if reason:
                    st.write(reason)
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------------------------------------------------------ tab 3: batch CSV
with tab_batch:
    up = st.file_uploader("Upload a CSV (e.g. McDonald_s_Reviews.csv)", type="csv")
    if up:
        raw = up.getvalue()
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")

        cols = df.columns.tolist()
        text_col = st.selectbox("Review column", cols,
                                index=cols.index("review") if "review" in cols else 0)
        n = st.number_input("How many rows to classify", 1, len(df), min(50, len(df)),
                            help="LLMs take ~0.5-2 s per review on a T4, so start small.")

        if st.button("Run batch", type="primary"):
            sub = df.head(int(n)).copy()
            labels, reasons = [], []
            bar = st.progress(0.0, text="Classifying...")
            for i, txt in enumerate(sub[text_col].astype(str)):
                try:
                    l, r = classify(model, txt)
                except Exception as e:
                    l, r = "error", str(e)
                labels.append(l)
                reasons.append(r)
                bar.progress((i + 1) / len(sub), text=f"Classifying... {i + 1}/{len(sub)}")
            sub["predicted_sentiment"] = labels
            sub["reason"] = reasons
            st.session_state.batch_result = sub

        if "batch_result" in st.session_state:
            res = st.session_state.batch_result
            c1, c2 = st.columns([1, 2])
            with c1:
                st.bar_chart(res["predicted_sentiment"].value_counts())
            with c2:
                show = [text_col, "predicted_sentiment", "reason"]
                if "rating" in res.columns:
                    show.insert(1, "rating")
                st.dataframe(res[show], width="stretch", height=300)
            st.download_button("⬇️ Download results CSV",
                               res.to_csv(index=False).encode("utf-8"),
                               "sentiment_results.csv", "text/csv")
