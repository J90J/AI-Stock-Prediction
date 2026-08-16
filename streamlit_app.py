import os
import sys
import pathlib

import streamlit as st

sys.path.append(str(pathlib.Path(__file__).parent))

from src.agent import run_agent

st.set_page_config(page_title="AI Stock Agent", page_icon="📈", layout="wide")

# ── Global styles: amber-on-black Bloomberg-terminal homage ────────────────
st.markdown("""
<style>
.stApp {
    font-family: 'Courier New', 'IBM Plex Mono', monospace;
    background-image: repeating-linear-gradient(
        180deg,
        rgba(255,176,0,0.025) 0px,
        rgba(255,176,0,0.025) 1px,
        transparent 1px,
        transparent 3px
    );
}

/* Title in classic terminal amber, boxed like a Bloomberg panel header */
h1 {
    font-family: 'Courier New', monospace !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    background: linear-gradient(120deg, #FFD98A, #FF9E00 55%, #C97800 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    filter: drop-shadow(0 0 14px rgba(255, 158, 0, 0.35));
}
h1::after {
    content: "\2588";
    color: #FF9E00;
    -webkit-text-fill-color: #FF9E00;
    animation: cursorBlink 1s step-end infinite;
    margin-left: 6px;
    font-size: 0.7em;
    vertical-align: middle;
}
@keyframes cursorBlink { 50% { opacity: 0; } }

/* ── Ticker tape ── */
.ticker-wrap {
    overflow: hidden;
    white-space: nowrap;
    border-top: 1px solid #332815;
    border-bottom: 1px solid #332815;
    background: #0F0C06;
    padding: 7px 0;
    margin: 4px 0 22px 0;
}
.ticker-track {
    display: inline-block;
    white-space: nowrap;
    animation: tickerScroll 32s linear infinite;
}
.ticker-item {
    display: inline-block;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
    padding: 0 26px;
    border-right: 1px solid #332815;
}
.ticker-item .sym { color: #D69A2D; font-weight: 700; }
.ticker-up   { color: #4ADE80; }
.ticker-down { color: #F87171; }
@keyframes tickerScroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ── Recommendation banner ── */
.rec-banner {
    padding: 20px 24px;
    border-radius: 4px;
    margin: 18px 0 10px 0;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-align: center;
    font-family: 'Courier New', monospace;
    text-transform: uppercase;
    border: 1px solid;
}
.rec-strong-buy  { background:rgba(20,60,35,0.55); color:#4ADE80; border-color:#2f9e5f; box-shadow:0 0 20px rgba(74,222,128,0.25); }
.rec-buy         { background:rgba(20,50,35,0.4);  color:#86EFAC; border-color:#2f9e5f; }
.rec-hold        { background:rgba(50,38,10,0.55); color:#E0A030; border-color:#8a6218; }
.rec-sell        { background:rgba(60,45,15,0.45); color:#FCD34D; border-color:#a3811f; }
.rec-strong-sell { background:rgba(65,20,25,0.55); color:#F87171; border-color:#a33a3a; box-shadow:0 0 20px rgba(248,113,113,0.25); }

/* ── Signal pills ── */
.pill {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    white-space: nowrap;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.4px;
    border: 1px solid;
}
.pill-up       { background:rgba(20,60,35,0.5); color:#4ADE80; border-color:#2f9e5f; }
.pill-down     { background:rgba(65,20,25,0.5); color:#F87171; border-color:#a33a3a; }
.pill-positive { background:rgba(20,60,35,0.5); color:#4ADE80; border-color:#2f9e5f; }
.pill-negative { background:rgba(65,20,25,0.5); color:#F87171; border-color:#a33a3a; }
.pill-neutral  { background:rgba(50,38,10,0.5); color:#E0A030; border-color:#8a6218; }

/* ── Small section labels ── */
.label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #D69A2D;
    margin: 0 0 4px 0;
    font-family: 'Courier New', monospace;
}

.vis-divider { border:none; border-top:1px solid #332815; margin:22px 0 14px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("AI Stock Analysis Agent")
st.caption(
    "Fetches live data · trains an LSTM · analyzes news sentiment · "
    "synthesizes a recommendation — all from a single question."
)

# ── Ticker tape (decorative, Bloomberg-style) ───────────────────────────────
_TICKERS = [
    ("NVDA", "+2.41%", True), ("AAPL", "-0.82%", False), ("TSLA", "+5.06%", True),
    ("MSFT", "+1.17%", True), ("AMZN", "-1.53%", False), ("GOOGL", "+0.64%", True),
    ("META", "+3.28%", True), ("PLTR", "-2.09%", False), ("AMD", "+1.94%", True),
    ("NFLX", "-0.41%", False),
]
_ticker_items = "".join(
    f'<span class="ticker-item"><span class="sym">{sym}</span> '
    f'<span class="{"ticker-up" if up else "ticker-down"}">{chg}</span></span>'
    for sym, chg, up in _TICKERS
)
st.markdown(
    f'<div class="ticker-wrap"><div class="ticker-track">{_ticker_items}{_ticker_items}</div></div>',
    unsafe_allow_html=True,
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
