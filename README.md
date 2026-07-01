#  NYAYA AI — RAG & Agentic AI for Civic Legal Awareness

NYAYA AI is a multilingual civic-awareness assistant that helps people in India understand their legal rights in plain language. It combines real NLP-based issue classification, retrieval-augmented generation (RAG) over a curated legal knowledge base, and an agentic chat flow — with a built-in evaluation suite to measure RAG quality against a baseline (no-retrieval) model.

> **Disclaimer:** NYAYA AI is an educational civic-awareness tool, not a lawyer, court, or government authority. Its output is informational only and must not be treated as legal advice.

---

##  Features

- **Multilingual input detection** — Unicode script analysis across 8 Indian scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam) plus Hinglish detection, with no external language-detection library required.
- **Rule + NLP-based issue classification** — TF-IDF semantic matching against a domain corpus (fundamental rights, consumer protection, labour law, cybercrime, etc.) to identify the most relevant legal domain(s) for a user's situation.
- **Retrieval-Augmented Generation (RAG)** — A hybrid retriever (TF-IDF + ChromaDB semantic search via Sentence-Transformers) pulls relevant passages from a 198-entry legal knowledge base, grounding Gemini's explanations in real legal text instead of letting it hallucinate.
- **Agentic chat** — Multi-step reasoning trace (plan → retrieve → self-check) for conversational follow-ups, with citations back to source passages.
- **Self-verification** — RAG explanations are automatically self-checked for factual consistency against retrieved sources before being returned.
- **Built-in evaluation framework** — Compares RAG vs. baseline (no-retrieval) responses across grounding, actionability, hallucination, and relevance, using dual LLM judges with Cohen's kappa inter-rater reliability, plus a regex-based hallucination F1 score against a gold dataset.
- **Risk & entity extraction** — Flags risk level (low/medium/high) and pulls out money amounts, dates, locations, organizations, and evidence indicators from free-text complaints.
- Handled Gemini API rate-limit/downtime failures by adding a local knowledge-base fallback, so the app answers reliably instead of showing a connection error.
---

##  Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌───────────────────┐
│  index.html │ ───▶ │  FastAPI (app.py) │ ───▶ │  rag_engine.py     │
│  (frontend) │      │  - classification │      │  - hybrid retrieval│
└─────────────┘      │  - API routes     │      │  - Gemini calls    │
                      └──────────────────┘      │  - self-check      │
                                                  │  - eval suite       │
                                                  └─────────┬──────────┘
                                                            │
                                          ┌─────────────────┴─────────────────┐
                                          │   ChromaDB (vector store, auto-    │
                                          │   rebuilt from in-code knowledge   │
                                          │   base if not present)             │
                                          └─────────────────────────────────────┘
```

The Chroma vector store is **not** checked into the repo — it's rebuilt automatically on first run from the knowledge base defined in `rag_engine.py`, so there's nothing extra to set up locally.

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| LLM | Google Gemini (`google-genai`) |
| RAG / Retrieval | ChromaDB, Sentence-Transformers, TF-IDF (scikit-learn), LangChain core |
| NLP | Custom Unicode script detection, TF-IDF semantic classification |
| Frontend | Static HTML/CSS/JS (`index.html`) served directly by FastAPI |
| Deployment | Docker, Hugging Face Spaces |

---

##  Getting Started

### Prerequisites
- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Local setup

```
git clone https://github.com/Divyanshu409/NYAYA-AI-BY-RAG-AND-AGENTIC-AI.git
cd NYAYA-AI-BY-RAG-AND-AGENTIC-AI

python -m venv venv
venv\Scripts\activate        # on Windows
# source venv/bin/activate   # on macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Run the app:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```


##  Project Structure

```
.
├── app.py              # FastAPI app, routes, request/response schemas, classification logic
├── rag_engine.py        # Knowledge base, hybrid retriever, Gemini calls, self-check, eval suite
├── index.html            # Frontend
├── Dockerfile             # HF Spaces / Docker deployment config
├── requirements.txt        # Python dependencies
└── run.bat                  # Local Windows launch script
```

---

##  Limitations

- Educational tool only — not a substitute for professional legal advice.
- Risk/classification logic is rule + TF-IDF based, not a fine-tuned legal model.
- Gemini-generated explanations, even when grounded via RAG, can still be incomplete or imprecise for complex legal situations.

---

