import os, io, re, json, time, pathlib
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
import requests

# ----------------------- Config -----------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
CHAT_MODEL   = os.getenv("MODEL", "gpt-4o-mini")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TOP_K        = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE", "1200"))  # a bit larger for better context

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env")

DATA_DIR = pathlib.Path("./data")
DATA_DIR.mkdir(exist_ok=True)
EMB_NPY = DATA_DIR / "embeddings.npy"
CHUNKS_JSON = DATA_DIR / "chunks.json"

# ----------------------- App & CORS -----------------------
app = FastAPI(title="Policy Chatbot API", version="1.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve /static (for chat.html, dashboard.html, etc.)
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------- Tiny Vector Store -----------------------
KB: Dict[str, Optional[object]] = {"chunks": [], "embeds": None}

def _clean_text(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"\n{3,}", "\n\n", s)  # collapse extra blank lines
    return s.strip()

def chunk_text(txt: str, max_chars: int) -> List[str]:
    """Paragraph-first chunking; preserves short headings; keeps chunks < max_chars."""
    lines = [l.strip() for l in txt.splitlines()]
    parts, buf = [], []
    for ln in lines:
        candidate = ("\n".join(buf + [ln])).strip()
        if len(candidate) > max_chars and buf:
            parts.append("\n".join(buf).strip())
            buf = [ln] if ln else []
        else:
            buf.append(ln)
    if buf:
        parts.append("\n".join(buf).strip())
    # keep headings; only drop empty strings
    parts = [p for p in parts if p]
    return parts

def _headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def embed_batch(batch: List[str]) -> List[List[float]]:
    """One embeddings call with retry on 429/5xx."""
    payload = {"model": EMBED_MODEL, "input": batch}
    backoff = 2
    for attempt in range(5):
        r = requests.post(f"{OPENAI_BASE}/embeddings", json=payload, headers=_headers(), timeout=60)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
    r.raise_for_status()  # raise last error

def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        vecs.extend(embed_batch(texts[i:i + batch_size]))
    return np.array(vecs, dtype=np.float32)

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return A @ b

def llm_chat(messages: List[Dict]) -> str:
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.1}
    r = requests.post(f"{OPENAI_BASE}/chat/completions", json=payload, headers=_headers(), timeout=120)
    if r.status_code == 401:
        return "OpenAI authentication failed (401). Check your API key."
    if r.status_code == 429:
        return "Rate limit (429). Please retry in a moment."
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ----------------------- Helpers: read files -----------------------
def read_docx_bytes(data: bytes) -> str:
    """
    High-fidelity DOCX extractor:
    - paragraphs (headings, lists)
    - tables (cells joined with ' | ')
    """
    try:
        from docx import Document  # pip install python-docx
    except Exception:
        return ""
    f = io.BytesIO(data)
    doc = Document(f)

    parts: List[str] = []

    # paragraphs
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            parts.append(txt)

    # tables
    for tbl in doc.tables:
        for row in tbl.rows:
            cells = [c.text.strip() for c in row.cells]
            row_txt = " | ".join([c for c in cells if c])
            if row_txt:
                parts.append(row_txt)

    return "\n".join(parts)

def read_txt_bytes(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")

def persist_kb(chunks: List[str], embeds: np.ndarray):
    CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(EMB_NPY, embeds)

def load_persisted_kb() -> bool:
    if CHUNKS_JSON.exists() and EMB_NPY.exists():
        try:
            chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
            embeds = np.load(EMB_NPY)
            if isinstance(chunks, list) and isinstance(embeds, np.ndarray):
                KB["chunks"] = chunks
                KB["embeds"] = embeds
                return True
        except Exception:
            return False
    return False

# Attempt to load persisted KB on startup
load_persisted_kb()

# ----------------------- API: Basic -----------------------
@app.get("/")
def home():
    return {
        "ok": True,
        "msg": "Policy Chatbot API",
        "endpoints": ["/api/health", "/api/kb/status", "/api/kb/preview", "/api/kb/load (POST)", "/api/chat (POST)", "/static/chat.html", "/docs"],
    }

@app.get("/api/health")
def health():
    return {"ok": True, "model": CHAT_MODEL, "embed_model": EMBED_MODEL, "kb_loaded": KB["embeds"] is not None}

@app.get("/api/kb/status")
def kb_status():
    return {
        "ok": KB["embeds"] is not None and len(KB["chunks"]) > 0,
        "chunks": len(KB["chunks"]) if KB["chunks"] else 0,
        "persisted": CHUNKS_JSON.exists() and EMB_NPY.exists(),
    }

@app.get("/api/kb/preview")
def kb_preview(n: int = 5):
    """Peek at the first N chunks to verify parsing."""
    if KB["embeds"] is None or not KB["chunks"]:
        return {"ok": False, "msg": "KB not loaded"}
    n = max(1, min(n, len(KB["chunks"])))
    return {
        "ok": True,
        "total_chunks": len(KB["chunks"]),
        "first_sizes": [len(KB["chunks"][i]) for i in range(n)],
        "first_chunks": KB["chunks"][:n],
    }

# ----------------------- API: KB Load -----------------------
@app.post("/api/kb/load")
async def kb_load(file: UploadFile = File(None), text: str = Form(None)):
    """
    Load/refresh the knowledge base.
    - Option A: send 'text' (Form field)
    - Option B: upload a file (docx or txt)
    """
    raw = ""
    if text and text.strip():
        raw = text
    elif file:
        data = await file.read()
        name = (file.filename or "").lower()
        if name.endswith(".docx"):
            raw = read_docx_bytes(data)
        elif name.endswith(".txt"):
            raw = read_txt_bytes(data)
        else:
            return {"ok": False, "msg": "Unsupported file type. Use .docx or .txt"}
    else:
        return {"ok": False, "msg": "Provide text or upload a .docx/.txt file."}

    raw = _clean_text(raw)
    if not raw:
        return {"ok": False, "msg": "Document was empty after parsing."}

    chunks = chunk_text(raw, max_chars=CHUNK_SIZE)
    embeds = embed_texts(chunks)

    KB["chunks"] = chunks
    KB["embeds"] = embeds
    persist_kb(chunks, embeds)

    return {"ok": True, "chunks": len(chunks)}

# ----------------------- API: Chat -----------------------
class ChatIn(BaseModel):
    message: str

SYSTEM_PROMPT = (
    "You are 'Conflict Resolution Assistant'. "
    "Answer ONLY using the policy context provided. "
    "If the answer is not clearly supported by the policy, reply: "
    "'I don't have that information in the policy.' Be concise."
)

@app.post("/api/chat")
def api_chat(inp: ChatIn):
    q = (inp.message or "").strip()
    if not q:
        return {"reply": "Please enter a question."}

    if KB["embeds"] is None or len(KB["chunks"]) == 0:
        return {"reply": "The policy hasn’t been loaded yet. Please upload it to /api/kb/load."}

    try:
        qvec = embed_texts([q])[0]
    except requests.HTTPError as e:
        if e.response.status_code == 401:
            return {"reply": "OpenAI authentication failed (401). Check your API key."}
        if e.response.status_code == 429:
            return {"reply": "Rate limit (429). Please retry in a moment."}
        return {"reply": f"Embedding error: {e.response.status_code}"}
    except Exception as e:
        return {"reply": f"Embedding error: {str(e)}"}

    sims = cosine_sim_matrix(KB["embeds"], qvec)
    idx = np.argsort(-sims)[:TOP_K]
    context = "\n\n---\n\n".join(KB["chunks"][int(i)] for i in idx)

    user_prompt = (
        "Answer the user's question strictly from the policy context below.\n\n"
        "### Policy Context\n"
        f"{context}\n\n"
        "### Question\n"
        f"{q}\n\n"
        "### Instructions\n"
        "- Quote or paraphrase the relevant rule.\n"
        "- If not in the context, say you don't have that information.\n"
        "- Keep under 120 words."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    answer = llm_chat(messages)
    return {"reply": answer}
