# Book RAG Chatbot using Streamlit + Gemini

This project lets you upload a PDF book and ask questions from that book using Retrieval-Augmented Generation.

## Files

- `app.py` - main Streamlit app
- `requirements.txt` - Python packages to install
- `.env.example` - example API key file

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Add your Gemini API key inside `.env`:

```env
GEMINI_API_KEY=your_real_key_here
```

## How it works

1. Upload a PDF.
2. The app extracts text from each page.
3. The text is split into overlapping chunks.
4. Gemini Embedding creates a vector for each chunk.
5. Your question is embedded too.
6. The most similar chunks are retrieved.
7. Gemini answers using only those retrieved chunks.
