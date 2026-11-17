# ==============================================================================
# STEP 1: IMPORTS & MODULE EXPLANATIONS
# ==============================================================================

import json                     # Reads downloaded_data.json to display a local preview in the UI
from pathlib import Path        # Locates downloaded_data.json inside the current project folder
import requests                 # Sends HTTP POST requests from Streamlit to our FastAPI backend
import streamlit as st          # Builds the interactive web interface (inputs, buttons, columns, cards)

# ==============================================================================
# STEP 2: APP CONFIGURATION & STYLING
# ==============================================================================

BACKEND_URL = "http://127.0.0.1:8000"                       # Local FastAPI server address
LOCAL_JSON_PATH = Path(__file__).parent / "downloaded_data.json"  # Path to local saved JSON file

# Configure browser tab title, icon, and wide two-column layout
st.set_page_config(
    page_title="World Country Trivia RAG Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject custom CSS for the gradient hero banner and styled trivia answer card
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.25rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        opacity: 0.92;
        font-size: 0.98rem;
    }
    .answer-box {
        background-color: #f8fafc;
        border-left: 5px solid #0f766e;
        padding: 1.15rem 1.4rem;
        border-radius: 8px;
        margin-top: 0.75rem;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render top hero banner
st.markdown(
    """
    <div class="main-header">
        <h1>🌍 World Country Trivia — AI RAG Explorer</h1>
        <p>LangChain • LangSmith Tracing • Local FAISS Vector DB • OpenAI • FastAPI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# STEP 3: INITIALIZE STREAMLIT SESSION STATE (PERSISTS DATA ACROSS CLICKS)
# ==============================================================================

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []    # Holds the 3 AI-suggested trivia questions
if "ingest_info" not in st.session_state:
    st.session_state.ingest_info = None          # Holds ingestion metadata (file name, record count)
if "active_question" not in st.session_state:
    st.session_state.active_question = ""        # Tracks the currently selected or typed question
if "rag_result" not in st.session_state:
    st.session_state.rag_result = None           # Stores the latest answer + retrieved sources
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}           # Caches past Q&A pairs in memory (0 tokens on repeat clicks)

# ==============================================================================
# STEP 4: TOP STATUS METRICS BAR
# ==============================================================================

m1, m2, m3, m4 = st.columns(4)
m1.metric("Vector Store", "FAISS (512-dim)")
m2.metric("Tracing", "LangSmith Active")
m3.metric(
    "Countries Indexed",
    st.session_state.ingest_info["records_indexed"] if st.session_state.ingest_info else "0",
)
m4.metric(
    "Local JSON File",
    st.session_state.ingest_info["saved_file"] if st.session_state.ingest_info else "Not Saved",
)

st.divider()

# ==============================================================================
# STEP 5: TWO-COLUMN MAIN LAYOUT (LEFT = INGESTION, RIGHT = TRIVIA Q&A)
# ==============================================================================

left_col, right_col = st.columns([1, 1.35], gap="large")

# -------------------- LEFT COLUMN: JSON DOWNLOAD & FAISS --------------------
with left_col:
    with st.container(border=True):
        st.subheader("📥 Step 1: Load & Index Country JSON")
        st.caption("Downloads the JSON locally and builds a 512-dim FAISS vector index.")

        # Pre-filled fast REST Countries v5 endpoint (fetches 100 countries with only needed fields)
        default_url = (
            "https://api.restcountries.com/countries/v5"
            "?limit=100&response_fields=names,capitals,currencies,languages,borders,region,subregion,population,landlocked"
        )
        json_url = st.text_input("JSON Endpoint URL", value=default_url)
        api_key = st.text_input(
            "Authorization API Key",
            type="password",
            placeholder="Paste your rc_live_... API key",
        )

        # Trigger POST /ingest on the FastAPI backend
        if st.button("⬇️ Download JSON & Build FAISS Index", type="primary", use_container_width=True):
            if not json_url.strip():
                st.error("Please enter a valid JSON URL.")
            else:
                with st.spinner("Downloading JSON, indexing into FAISS, and generating trivia..."):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/ingest",
                            json={"json_url": json_url, "api_key": api_key},
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.ingest_info = data
                            st.session_state.suggested_questions = data.get("suggested_questions", [])
                            st.session_state.rag_result = None
                            st.session_state.answer_cache = {}
                            st.rerun()  # Refresh UI to immediately show the 3 suggested trivia buttons
                        else:
                            st.error(f"Error: {resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"Backend connection error: {e}")

        # Display confirmation banner once JSON is downloaded & indexed
        if st.session_state.ingest_info:
            info = st.session_state.ingest_info
            st.success(
                f"✅ Saved **`{info['saved_file']}`** locally ({info['records_indexed']} records). "
                f"Status: *{info['index_note']}*."
            )

    # Expandable preview of the locally saved JSON file
    if LOCAL_JSON_PATH.exists():
        with st.expander("📄 Preview Local JSON (`downloaded_data.json`)", expanded=False):
            try:
                with open(LOCAL_JSON_PATH, "r", encoding="utf-8") as f:
                    preview = json.load(f)
                st.json(preview if isinstance(preview, dict) else preview[:2])
            except Exception as e:
                st.warning(f"Preview unavailable: {e}")

# -------------------- RIGHT COLUMN: AI TRIVIA & QUERY -----------------------
with right_col:
    with st.container(border=True):
        st.subheader("🎯 Step 2: World Country Trivia Q&A")

        # Display the 3 AI-suggested trivia buttons once JSON is indexed
        if st.session_state.suggested_questions:
            st.markdown("#### 💡 AI-Suggested Trivia Questions")
            st.caption("Click any question below to populate the query box:")
            for idx, q in enumerate(st.session_state.suggested_questions, start=1):
                if st.button(f"Q{idx}: {q}", key=f"sugg_q_{idx}", use_container_width=True):
                    st.session_state.active_question = q
        else:
            st.info("👈 Download and index the JSON on the left to unlock 3 AI-suggested trivia questions.")

        st.markdown("---")

        # Question submission form
        with st.form(key="query_form", clear_on_submit=False):
            user_question = st.text_input(
                "Ask a country trivia question:",
                value=st.session_state.active_question,
                placeholder="e.g., What is the capital and currency of Canada?",
            )
            submitted = st.form_submit_button("🔍 Ask Trivia AI", type="primary", use_container_width=True)

        # Handle question submission (uses session cache first to save OpenAI tokens)
        if submitted:
            clean_q = user_question.strip()
            st.session_state.active_question = clean_q
            if not clean_q:
                st.warning("Please enter or select a trivia question first.")
            elif clean_q in st.session_state.answer_cache:
                st.session_state.rag_result = st.session_state.answer_cache[clean_q]
            else:
                with st.spinner("Searching FAISS & generating trivia answer..."):
                    try:
                        q_resp = requests.post(
                            f"{BACKEND_URL}/query",
                            json={"question": clean_q},
                            timeout=30,
                        )
                        if q_resp.status_code == 200:
                            result = q_resp.json()
                            st.session_state.rag_result = result
                            st.session_state.answer_cache[clean_q] = result
                        else:
                            st.error(f"Query Error: {q_resp.json().get('detail', q_resp.text)}")
                    except Exception as e:
                        st.error(f"Backend connection error: {e}")

        # Render the generated answer and the top-3 FAISS source records
        if st.session_state.rag_result:
            st.markdown("### 🏆 Trivia Answer")
            st.markdown(
                f'<div class="answer-box">{st.session_state.rag_result["answer"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("")
            with st.expander("🔎 View Retrieved FAISS Country Records"):
                for idx, src in enumerate(st.session_state.rag_result["sources"], start=1):
                    st.markdown(f"**Match {idx}: {src['country']}** ({src['region']})")
                    st.code(src["content"], language="text")