import os
import sys
import pathlib

import streamlit as st

sys.path.append(str(pathlib.Path(__file__).parent))

from src.agent import run_agent

st.set_page_config(page_title="AI Stock Agent", page_icon="🤖", layout="wide")

st.title("AI Stock Analysis Agent 🤖")
st.markdown(
    "Chat with an AI agent that autonomously fetches data, trains models, "
    "and synthesizes technical + sentiment signals into a recommendation."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
data_dir        = st.sidebar.text_input("Data Directory",        "data")
checkpoints_dir = st.sidebar.text_input("Checkpoints Directory", "checkpoints")

# Anthropic key: secrets → env var → manual input
api_key = ""
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input(
        "Anthropic API Key", type="password", placeholder="sk-ant-..."
    )

# OpenAI key (optional backup): secrets → env var → manual input
openai_api_key = ""
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key (backup)", type="password", placeholder="sk-..."
    )

if not api_key and not openai_api_key:
    st.warning(
        "Enter at least one API key in the sidebar "
        "(Anthropic is primary, OpenAI is the fallback)."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Example prompts**")
st.sidebar.markdown(
    "- Should I buy NVDA today?\n"
    "- What is your outlook on AAPL?\n"
    "- Analyze TSLA and give me a recommendation\n"
    "- Retrain the model for MSFT with 50 epochs"
)

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about any stock (e.g. 'Should I buy PLTR today?')"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                events = list(run_agent(
                    prompt, api_key, data_dir, checkpoints_dir,
                    openai_api_key=openai_api_key
                ))
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.stop()

        # Show any provider-switch notice
        for e in events:
            if e["type"] == "notice":
                st.info(e["text"])

        # Display tool calls as expandable sections
        tool_starts = [e for e in events if e["type"] == "tool_start"]
        tool_ends   = {e["tool"]: e for e in events if e["type"] == "tool_end"}

        if tool_starts:
            st.markdown("**Agent actions:**")
            for ts in tool_starts:
                tool_name = ts["tool"]
                label = {
                    "fetch_data":    "📥 Fetched market data",
                    "train_model":   "🧠 Trained LSTM model",
                    "predict":       "📊 Ran price forecast",
                    "get_sentiment": "📰 Analyzed news sentiment",
                }.get(tool_name, f"🔧 {tool_name}")

                with st.expander(label, expanded=False):
                    st.markdown("**Input**")
                    st.json(ts["input"])
                    if tool_name in tool_ends:
                        st.markdown("**Result**")
                        st.json(tool_ends[tool_name]["result"])

        # Display final response
        final_response = next(
            (e["text"] for e in events if e["type"] == "response"), "No response."
        )
        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
