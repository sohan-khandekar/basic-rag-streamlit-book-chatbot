import io
import os
import re
import html
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader


st.set_page_config(
    page_title="BookMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()

GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 250
TOP_K = 4


def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.25), transparent 32%),
                radial-gradient(circle at top right, rgba(168, 85, 247, 0.22), transparent 30%),
                linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
            color: #e5e7eb;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1150px;
        }

        .hero {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 30px;
            padding: 36px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.40);
            backdrop-filter: blur(18px);
            margin-bottom: 26px;
        }

        .hero-title {
            font-size: 3.4rem;
            font-weight: 800;
            line-height: 1.05;
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }

        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1.1rem;
            line-height: 1.65;
            max-width: 900px;
        }

        .upload-card {
            background: rgba(30, 41, 59, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 26px;
            padding: 28px;
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.34);
            margin-bottom: 24px;
        }

        .upload-title {
            color: #f8fafc;
            font-size: 1.45rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .upload-text {
            color: #cbd5e1;
            font-size: 0.98rem;
            line-height: 1.55;
            margin-bottom: 16px;
        }

        .info-card {
            background: rgba(15, 23, 42, 0.74);
            border: 1px solid rgba(96, 165, 250, 0.24);
            border-radius: 22px;
            padding: 22px;
            min-height: 165px;
            box-shadow: 0 14px 45px rgba(0, 0, 0, 0.30);
        }

        .info-card h3 {
            color: #f8fafc;
            font-size: 1.08rem;
            margin-bottom: 8px;
        }

        .info-card p {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .status-success {
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.35);
            color: #bbf7d0;
            border-radius: 18px;
            padding: 15px 17px;
            font-weight: 700;
            margin: 12px 0;
        }

        .status-warning {
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.35);
            color: #fde68a;
            border-radius: 18px;
            padding: 15px 17px;
            font-weight: 700;
            margin: 12px 0;
        }

        .status-error {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.35);
            color: #fecaca;
            border-radius: 18px;
            padding: 15px 17px;
            font-weight: 700;
            margin: 12px 0;
        }

        .chat-section {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 26px;
            padding: 28px;
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.32);
            margin-top: 24px;
        }

        .source-card {
            background: rgba(30, 41, 59, 0.78);
            border-left: 4px solid #60a5fa;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 14px;
            color: #dbeafe;
            line-height: 1.55;
        }

        .source-card small {
            color: #93c5fd;
            font-weight: 800;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 16px;
            border: none;
            padding: 0.85rem 1rem;
            font-weight: 800;
            color: white;
            background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
            box-shadow: 0 14px 36px rgba(124, 58, 237, 0.34);
            transition: 0.25s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 44px rgba(124, 58, 237, 0.48);
        }

        div[data-testid="stFileUploader"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px dashed rgba(147, 197, 253, 0.55);
            border-radius: 18px;
            padding: 14px;
        }

        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        [data-testid="stMetricLabel"] {
            color: #cbd5e1;
        }

        .stChatMessage {
            border-radius: 20px;
        }

        hr {
            border-color: rgba(148, 163, 184, 0.22);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "chunks": [],
        "embeddings": None,
        "indexed_file_name": None,
        "total_pages": 0,
        "chat_history": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def get_client():
    api_key = get_api_key()

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages_from_pdf(uploaded_file) -> List[Dict]:
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []

    for page_index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = clean_text(text)

        if text:
            pages.append(
                {
                    "page": page_index + 1,
                    "text": text,
                }
            )

    return pages


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    chunks = []
    chunk_id = 1

    for page in pages:
        page_no = page["page"]
        text = page["text"]

        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if len(chunk_text) > 80:
                chunks.append(
                    {
                        "id": chunk_id,
                        "page": page_no,
                        "text": chunk_text,
                    }
                )
                chunk_id += 1

            if end >= len(text):
                break

            start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def create_document_embeddings(client, chunks: List[Dict]) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []

    progress_bar = st.progress(0)
    status = st.empty()

    batch_size = 16

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )

        batch_embeddings = [embedding.values for embedding in result.embeddings]
        all_embeddings.extend(batch_embeddings)

        completed = min(start + batch_size, len(texts))
        progress = completed / len(texts)

        progress_bar.progress(progress)
        status.write(f"Creating book memory: {completed}/{len(texts)} chunks indexed")

    embeddings = np.array(all_embeddings, dtype=np.float32)
    embeddings = normalize_embeddings(embeddings)

    progress_bar.progress(1.0)
    status.write("Book memory created successfully.")

    return embeddings


def create_query_embedding(client, question: str) -> np.ndarray:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )

    vector = np.array(result.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def retrieve_relevant_chunks(client, question: str) -> List[Tuple[Dict, float]]:
    query_embedding = create_query_embedding(client, question)

    similarities = np.dot(st.session_state.embeddings, query_embedding)
    top_indices = similarities.argsort()[::-1][:TOP_K]

    retrieved = []

    for index in top_indices:
        chunk = st.session_state.chunks[int(index)]
        score = float(similarities[int(index)])
        retrieved.append((chunk, score))

    return retrieved


def build_prompt(question: str, retrieved_chunks: List[Tuple[Dict, float]]) -> str:
    context_blocks = []

    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"""
SOURCE {i}
Page: {chunk["page"]}
Similarity Score: {score:.4f}

Text:
{chunk["text"]}
"""
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are BookMind AI, a helpful textbook assistant.

Answer the user's question using ONLY the provided book context.

Rules:
1. Explain in simple beginner-friendly language.
2. Use the uploaded book context only.
3. If the context is insufficient, clearly say that the uploaded book context does not contain enough information.
4. Mention relevant page numbers from the provided sources.
5. Use examples and bullet points when useful.
6. Do not hallucinate or invent facts outside the context.

BOOK CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt


def generate_answer(
    client, question: str, retrieved_chunks: List[Tuple[Dict, float]]
) -> str:
    prompt = build_prompt(question, retrieved_chunks)

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return response.text or "I could not generate an answer. Please try again."


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📚 BookMind AI</div>
            <div class="hero-subtitle">
                Upload a textbook PDF and ask questions directly from its contents.
                The chatbot searches your book first, then generates an answer using the most relevant pages.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_area(client):
    st.markdown(
        """
        <div class="upload-card">
            <div class="upload-title">Upload your book</div>
            <div class="upload-text">
                Add a PDF textbook or notes file. After uploading, click the indexing button once.
                Then you can start asking questions from the book.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1])

    with left:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            label_visibility="collapsed",
        )

        process_button = st.button("🚀 Process and Index Book")

        if process_button:
            process_book(client, uploaded_file)

    with right:
        if st.session_state.indexed_file_name:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>✅ Book Ready</h3>
                    <p><b>File:</b> {html.escape(st.session_state.indexed_file_name)}</p>
                    <p><b>Pages read:</b> {st.session_state.total_pages}</p>
                    <p><b>Chunks created:</b> {len(st.session_state.chunks)}</p>
                    <p>You can now ask questions below.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="info-card">
                    <h3>📌 How it works</h3>
                    <p>1. Upload your PDF book.</p>
                    <p>2. The app extracts text from pages.</p>
                    <p>3. It creates searchable chunks.</p>
                    <p>4. Gemini answers using relevant book context.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def process_book(client, uploaded_file):
    if uploaded_file is None:
        st.markdown(
            """
            <div class="status-warning">
                ⚠️ Please upload a PDF file first.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        with st.spinner("Reading PDF pages..."):
            pages = extract_pages_from_pdf(uploaded_file)

        if not pages:
            st.markdown(
                """
                <div class="status-error">
                    ❌ Could not extract readable text from this PDF. It may be a scanned PDF.
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        with st.spinner("Splitting book into chunks..."):
            chunks = chunk_pages(pages)

        if not chunks:
            st.markdown(
                """
                <div class="status-error">
                    ❌ No useful text chunks were created.
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        with st.spinner("Creating searchable book memory..."):
            embeddings = create_document_embeddings(client, chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.indexed_file_name = uploaded_file.name
        st.session_state.total_pages = len(pages)
        st.session_state.chat_history = []

        st.markdown(
            f"""
            <div class="status-success">
                ✅ Book indexed successfully. Pages read: {len(pages)}. Chunks created: {len(chunks)}.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as error:
        st.markdown(
            f"""
            <div class="status-error">
                ❌ Failed to process book: {html.escape(str(error))}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sources(retrieved_chunks: List[Tuple[Dict, float]]):
    with st.expander("View sources used"):
        for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
            safe_text = html.escape(chunk["text"][:850])

            st.markdown(
                f"""
                <div class="source-card">
                    <small>Source {i} | Page {chunk["page"]} | Similarity: {score:.4f}</small>
                    <br><br>
                    {safe_text}...
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat(client):
    st.markdown(
        """
        <div class="chat-section">
            <h2>💬 Ask questions from your book</h2>
            <p style="color:#cbd5e1;">
                Example: Explain the balance sheet equation. What is depreciation?
                What is the difference between financial accounting and management accounting?
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.indexed_file_name:
        st.markdown(
            """
            <div class="status-warning">
                ⚠️ Upload and index a book first. The chat will start after the book is ready.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["question"])

        with st.chat_message("assistant"):
            st.markdown(message["answer"])
            render_sources(message["sources"])

    question = st.chat_input("Ask a question from your uploaded book...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.spinner("Searching the book..."):
                retrieved_chunks = retrieve_relevant_chunks(client, question)

            with st.spinner("Generating answer..."):
                answer = generate_answer(client, question, retrieved_chunks)

            with st.chat_message("assistant"):
                st.markdown(answer)
                render_sources(retrieved_chunks)

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": retrieved_chunks,
                }
            )

        except Exception as error:
            st.markdown(
                f"""
                <div class="status-error">
                    ❌ Failed to answer: {html.escape(str(error))}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_footer():
    st.markdown("---")
    st.markdown(
        """
        <p style="text-align:center; color:#94a3b8; font-size:0.9rem;">
            Built with Streamlit, Gemini API, PDF parsing, embeddings, and Retrieval-Augmented Generation.
        </p>
        """,
        unsafe_allow_html=True,
    )


def main():
    load_css()
    init_state()

    client = get_client()

    render_hero()

    if client is None:
        st.markdown(
            """
            <div class="status-error">
                ❌ GEMINI_API_KEY not found.
                <br><br>
                For local use, add it inside your <b>.env</b> file:
                <br>
                GEMINI_API_KEY=your_actual_api_key_here
                <br><br>
                For Streamlit Cloud, add it in app secrets:
                <br>
                GEMINI_API_KEY = "your_actual_api_key_here"
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    render_upload_area(client)
    render_chat(client)
    render_footer()


if __name__ == "__main__":
    main()
