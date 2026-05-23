import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="PaperLens — AI Document Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --ink:       #0A0B0E;
    --surface:   #111318;
    --surface2:  #181C24;
    --surface3:  #1E2430;
    --border:    rgba(255,255,255,0.07);
    --border2:   rgba(255,255,255,0.12);
    --gold:      #E8B86D;
    --gold-dim:  #A07840;
    --gold-glow: rgba(232,184,109,0.12);
    --text:      #EDF0F7;
    --muted:     #6B7591;
    --muted2:    #3D4560;
    --accent:    #7C9FE4;
    --green:     #5DBF8A;
    --green-bg:  rgba(93,191,138,0.08);
    --green-border: rgba(93,191,138,0.2);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--ink) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 60% 40% at 15% 0%, rgba(124,159,228,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 35% at 85% 5%, rgba(232,184,109,0.05) 0%, transparent 55%),
        var(--ink) !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container {
    max-width: 900px !important;
    padding: 0 2rem 6rem !important;
    margin: 0 auto;
}

/* ── TOP NAV BAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 3.5rem;
}
.topbar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-logo .logo-icon {
    width: 42px; height: 42px;
    background: var(--gold);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.topbar-logo .logo-icon svg { width: 22px; height: 22px; }
.topbar-logo .logo-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
}
.topbar-logo .logo-text span { color: var(--gold); }
.topbar-tag {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
}

/* ── HERO ── */
.hero {
    padding: 1rem 0 3rem;
    position: relative;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1.5rem;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
}
.hero-eyebrow .bar {
    width: 28px; height: 1px;
    background: var(--gold-dim);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 4.6rem);
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: var(--text);
    margin-bottom: 1.2rem;
    max-width: 700px;
}
.hero h1 em {
    font-style: italic;
    color: var(--gold);
}
.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: var(--muted);
    max-width: 520px;
    line-height: 1.75;
    margin-bottom: 2.5rem;
}
.hero-features {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 0.78rem;
    font-weight: 400;
    color: var(--muted);
}
.hero-pill svg { width: 13px; height: 13px; color: var(--gold); flex-shrink: 0; }

/* ── STEP LABEL ── */
.step-label {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}
.step-num {
    width: 26px; height: 26px;
    border-radius: 50%;
    border: 1px solid var(--gold-dim);
    background: var(--gold-glow);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--gold);
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}
.step-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
}
.step-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed rgba(232,184,109,0.2) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s, background 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(232,184,109,0.45) !important;
    background: rgba(232,184,109,0.02) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--muted2) !important;
    font-size: 0.88rem !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: var(--gold-glow) !important;
    border: 1px solid rgba(232,184,109,0.3) !important;
    color: var(--gold) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(232,184,109,0.18) !important;
}

/* ── STATS STRIP ── */
.stats-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 1.4rem 0 1.8rem;
}
.stat-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.stat-tile::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--gold-dim), transparent);
}
.stat-tile .val {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
    margin-bottom: 4px;
}
.stat-tile .val small {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--muted2);
    font-weight: 400;
}
.stat-tile .lbl {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted2);
}

/* ── SUCCESS ALERT ── */
[data-testid="stAlert"] {
    background: var(--green-bg) !important;
    border: 1px solid var(--green-border) !important;
    border-radius: 10px !important;
    color: #8DE8B5 !important;
    font-size: 0.88rem !important;
}

/* ── TEXT INPUT ── */
[data-testid="stTextInput"] > label { display: none !important; }
[data-testid="stTextInput"] {
    position: relative;
}
[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 300 !important;
    padding: 1rem 1.3rem !important;
    height: 56px !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    caret-color: var(--gold);
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(232,184,109,0.4) !important;
    box-shadow: 0 0 0 3px rgba(232,184,109,0.07) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: var(--muted2) !important;
    font-style: italic;
    font-weight: 300;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] p {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
    font-style: italic;
}

/* ── ANSWER CARD ── */
.answer-wrap {
    margin-top: 1.8rem;
}
.answer-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}
.answer-meta .ai-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--gold);
    animation: glow-pulse 2.5s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes glow-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(232,184,109,0.4); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 5px rgba(232,184,109,0); }
}
.answer-meta .ai-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
}
.answer-meta .ai-model {
    font-size: 0.68rem;
    color: var(--muted2);
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
}
.answer-box {
    background: var(--surface);
    border: 1px solid rgba(232,184,109,0.12);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    position: relative;
}
.answer-box::before {
    content: '';
    position: absolute;
    top: 0; left: 24px; right: 24px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
}
.answer-text {
    font-size: 1.03rem;
    font-weight: 300;
    line-height: 1.85;
    color: #C8D0E0;
}

/* ── QUOTE MARK ── */
.answer-box .qmark {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    color: rgba(232,184,109,0.12);
    line-height: 1;
    float: left;
    margin-right: 4px;
    margin-top: -8px;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-top: 1rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted2) !important;
    padding: 0.85rem 1.1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--muted) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 1rem 1.2rem 1.4rem !important;
    font-size: 0.8rem !important;
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.75 !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    border-top: 1px solid var(--border) !important;
}

/* ── EMPTY STATE ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 1rem 4rem;
    text-align: center;
    gap: 1rem;
}
.empty-ring {
    width: 80px; height: 80px;
    border-radius: 50%;
    border: 1.5px dashed rgba(232,184,109,0.2);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.5rem;
    animation: spin-slow 20s linear infinite;
}
@keyframes spin-slow {
    to { transform: rotate(360deg); }
}
.empty-ring svg { width: 30px; height: 30px; animation: spin-slow 20s linear infinite reverse; }
.empty-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--muted2);
}
.empty-sub {
    font-size: 0.88rem;
    color: var(--muted2);
    max-width: 320px;
    line-height: 1.6;
    opacity: 0.7;
}

/* ── BOTTOM TAG ── */
.bottom-tag {
    text-align: center;
    padding: 2rem 0 0;
    font-size: 0.7rem;
    color: var(--muted2);
    letter-spacing: 0.08em;
    opacity: 0.5;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(232,184,109,0.15); border-radius: 10px; }

[data-testid="stMarkdownContainer"] p { color: var(--muted); line-height: 1.7; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── TOP NAV ──
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">
    <div class="logo-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="#0A0B0E" stroke-width="2.2" stroke-linecap="round">
        <circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="22" y2="22"/>
        <line x1="8" y1="11" x2="14" y2="11"/><line x1="11" y1="8" x2="11" y2="14"/>
      </svg>
    </div>
    <span class="logo-text">Paper<span>Lens</span></span>
  </div>
  <span class="topbar-tag">Powered by Mistral AI</span>
</div>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">
    <span class="bar"></span>
    AI Document Intelligence
    <span class="bar"></span>
  </div>
  <h1>Ask anything.<br><em>From any document.</em></h1>
  <p class="hero-sub">
    Upload a PDF and unlock instant, context-aware answers —
    no scrolling, no skimming, no guesswork.
  </p>
  <div class="hero-features">
    <span class="hero-pill">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
      Semantic search
    </span>
    <span class="hero-pill">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
      MMR retrieval
    </span>
    <span class="hero-pill">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
      ChromaDB vector store
    </span>
    <span class="hero-pill">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
      Mistral embeddings
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── STEP 1: UPLOAD ──
st.markdown("""
<div class="step-label">
  <div class="step-num">01</div>
  <span class="step-title">Upload your document</span>
  <div class="step-line"></div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

embedding_model = MistralAIEmbeddings()

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    file_size_kb = round(os.path.getsize(temp_pdf_path) / 1024, 1)

    # Stats strip
    st.markdown(f"""
    <div class="stats-strip">
      <div class="stat-tile">
        <div class="val">{len(docs)}</div>
        <div class="lbl">Pages</div>
      </div>
      <div class="stat-tile">
        <div class="val">{len(chunks)}</div>
        <div class="lbl">Text Chunks</div>
      </div>
      <div class="stat-tile">
        <div class="val">{file_size_kb}<small> KB</small></div>
        <div class="lbl">File Size</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Indexing document…"):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    st.success(f"✦  **{uploaded_file.name}** — indexed and ready to query.")

    llm = ChatMistralAI(model="mistral-small-latest")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise AI assistant. "
         "Answer ONLY using the provided context. "
         "If the answer isn't there, say: \"I could not find the answer in the document.\""),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    # ── STEP 2: ASK ──
    st.markdown("""
    <div class="step-label" style="margin-top:2.2rem">
      <div class="step-num">02</div>
      <span class="step-title">Ask your question</span>
      <div class="step-line"></div>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Question",
        placeholder="What is the key finding in this document?",
        label_visibility="collapsed"
    )

    if query:
        with st.spinner("Retrieving relevant passages and composing answer…"):
            retrieved_docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            final_prompt = prompt.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)

        # ── STEP 3: ANSWER ──
        st.markdown("""
        <div class="step-label" style="margin-top:2.2rem">
          <div class="step-num">03</div>
          <span class="step-title">Answer</span>
          <div class="step-line"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="answer-wrap">
          <div class="answer-meta">
            <span class="ai-dot"></span>
            <span class="ai-label">PaperLens Response</span>
            <span class="ai-model">mistral-small-latest</span>
          </div>
          <div class="answer-box">
            <span class="qmark">"</span>
            <div class="answer-text">{response.content}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Source passages used for this answer"):
            st.write(context)

    st.markdown('<div class="bottom-tag">PaperLens · RAG · Mistral AI · ChromaDB</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-ring">
        <svg viewBox="0 0 24 24" fill="none" stroke="rgba(232,184,109,0.3)" stroke-width="1.5">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
        </svg>
      </div>
      <div class="empty-heading">No document loaded</div>
      <div class="empty-sub">Upload a PDF above to start asking questions about its contents</div>
    </div>
    """, unsafe_allow_html=True)