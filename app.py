import os
import sys
import pathlib

import streamlit as st

sys.path.append(str(pathlib.Path(__file__).parent))

from src.agent import run_agent

st.set_page_config(page_title="AI Stock Agent", page_icon="📈", layout="wide")

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Recommendation banner ── */
.rec-banner {
    padding: 18px 24px;
    border-radius: 8px;
    margin: 16px 0 8px 0;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-align: center;
}
.rec-strong-buy  { background:#d1f0da; color:#155724; border:1px solid #b7dfca; }
.rec-buy         { background:#dce8fb; color:#1a3f6f; border:1px solid #c0d7f5; }
.rec-hold        { background:#ebebeb; color:#444;    border:1px solid #d4d4d4; }
.rec-sell        { background:#fef3cd; color:#7a5700; border:1px solid #f5e3a0; }
.rec-strong-sell { background:#fadadd; color:#7b1a24; border:1px solid #f1b8be; }

/* ── Signal pills ── */
.pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
}
.pill-up       { background:#d1f0da; color:#155724; }
.pill-down     { background:#fadadd; color:#7b1a24; }
.pill-positive { background:#d1f0da; color:#155724; }
.pill-negative { background:#fadadd; color:#7b1a24; }
.pill-neutral  { background:#ebebeb; color:#444;    }

/* ── Small section labels ── */
.label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    color: #999;
    margin: 0 0 4px 0;
}

.vis-divider { border:none; border-top:1px solid #e4e4e4; margin:20px 0 12px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("AI Stock Analysis Agent")
st.caption(
    "Fetches live data · trains an LSTM · analyzes news sentiment · "
    "synthesizes a recommendation — all from a single question."
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

# OpenAI key (optional backup)
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
    st.warning("Enter at least one API key in the sidebar (Anthropic primary, OpenAI fallback).")
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
        if msg.get("visual"):
            st.markdown(msg["visual"], unsafe_allow_html=True)
        st.markdown(msg["content"])

# ── Helper: render visual dashboard ──────────────────────────────────────────
def render_dashboard(pred: dict, sent: dict) -> str:
    """Render structured tool results as visual cards. Returns the HTML written."""
    has_pred = pred and "error" not in pred
    has_sent = sent and "error" not in sent and "verdict" in sent

    if not has_pred and not has_sent:
        return ""

    # ── Price metrics ────────────────────────────────────────────────────────
    if has_pred:
        ticker      = pred.get("ticker", "")
        last_close  = pred.get("last_close", 0)
        pred_close  = pred.get("predicted_next_close", 0)
        ret_pct     = pred.get("predicted_return_pct", 0)
        direction   = pred.get("direction", "UP")
        prob        = pred.get("probability_up", 50)
        last_date   = pred.get("last_date", "")

        delta_sign  = "+" if ret_pct >= 0 else ""
        dir_class   = "pill-up" if direction == "UP" else "pill-down"
        dir_arrow   = "↑" if direction == "UP" else "↓"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{ticker} Last Close ({last_date})", f"${last_close:,.2f}")
        with col2:
            st.metric("Predicted Next Close", f"${pred_close:,.2f}", f"{delta_sign}{ret_pct:.2f}%")
        with col3:
            st.markdown(
                f'<p class="label">Technical Signal</p>'
                f'<span class="pill {dir_class}">'
                f'{dir_arrow} {direction} &nbsp;·&nbsp; {prob:.0f}% confidence'
                f'</span>',
                unsafe_allow_html=True
            )

    # ── Sentiment ────────────────────────────────────────────────────────────
    if has_sent:
        verdict     = sent.get("verdict", "NEUTRAL")
        score       = sent.get("score", 0)
        headlines   = sent.get("headlines", [])
        sent_class  = {
            "POSITIVE": "pill-positive",
            "NEGATIVE": "pill-negative",
        }.get(verdict, "pill-neutral")
        score_sign  = "+" if score >= 0 else ""

        st.markdown(
            f'<p class="label" style="margin-top:12px">News Sentiment</p>'
            f'<span class="pill {sent_class}">'
            f'{verdict} &nbsp;·&nbsp; score {score_sign}{score:.2f}'
            f'</span>',
            unsafe_allow_html=True
        )

        if headlines:
            with st.expander("Top headlines", expanded=False):
                for h in headlines:
                    st.markdown(f"- {h}")

    # ── Final recommendation ─────────────────────────────────────────────────
    if has_pred and has_sent:
        verdict   = sent.get("verdict", "NEUTRAL")
        direction = pred.get("direction", "UP")

        if direction == "UP" and verdict == "POSITIVE":
            rec, cls = "STRONG BUY", "rec-strong-buy"
        elif direction == "DOWN" and verdict == "NEGATIVE":
            rec, cls = "STRONG SELL", "rec-strong-sell"
        elif direction == "UP":
            rec, cls = "BUY", "rec-buy"
        elif direction == "DOWN":
            rec, cls = "SELL", "rec-sell"
        else:
            rec, cls = "HOLD", "rec-hold"

        st.markdown(
            f'<div class="rec-banner {cls}">{rec}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="vis-divider">', unsafe_allow_html=True)
    return ""   # visual is rendered inline; nothing to store


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about any stock (e.g. 'Should I buy PLTR today?')"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is working..."):
            try:
                events = list(run_agent(
                    prompt, api_key, data_dir, checkpoints_dir,
                    openai_api_key=openai_api_key
                ))
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.stop()

        # Provider-switch notice
        for e in events:
            if e["type"] == "notice":
                st.info(e["text"])

        # Tool call log (collapsible)
        tool_starts = [e for e in events if e["type"] == "tool_start"]
        tool_ends   = {e["tool"]: e["result"] for e in events if e["type"] == "tool_end"}

        if tool_starts:
            with st.expander("Agent steps", expanded=False):
                for ts in tool_starts:
                    name = ts["tool"]
                    label = {
                        "fetch_data":    "📥 fetch_data",
                        "train_model":   "🧠 train_model",
                        "predict":       "📊 predict",
                        "get_sentiment": "📰 get_sentiment",
                    }.get(name, f"🔧 {name}")
                    st.markdown(f"**{label}** — input: `{ts['input']}`")
                    if name in tool_ends:
                        st.json(tool_ends[name])

        # ── Visual dashboard ─────────────────────────────────────────────────
        pred_result = tool_ends.get("predict", {})
        sent_result = tool_ends.get("get_sentiment", {})
        render_dashboard(pred_result, sent_result)

        # ── Agent text response ──────────────────────────────────────────────
        final_response = next(
            (e["text"] for e in events if e["type"] == "response"), "No response."
        )
        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
