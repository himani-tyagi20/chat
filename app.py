"""Real-time PDF ingestion -> Qdrant vectors -> grounded, cited answers."""

import logging
import os
import re
import uuid
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from llm import NO_ANSWER, StubLLM, get_llm

log = logging.getLogger("uvicorn.error")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # ONNX, runs locally
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 5))
# Calibration knob: cosine floor below which we treat retrieval as "not in the document".
# Measured on bge-small (see DESIGN.md): on-topic 0.78-0.90, off-topic 0.36-0.50.
# 0.62 sits in the middle of that gap; tune per corpus without touching code.
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.62))

app = FastAPI(title="OneRx PDF RAG")

# Local Qdrant by default (no server needed); set QDRANT_URL to point at a real one.
qdrant = QdrantClient(url=os.getenv("QDRANT_URL")) if os.getenv("QDRANT_URL") else QdrantClient(":memory:")

embedder = TextEmbedding(EMBED_MODEL)  # ONNX runtime, no torch, cached on first run


def chunk_page(text: str) -> list[str]:
    """Fixed-size character windows with overlap, cut back to the last whitespace."""
    # PDF extraction breaks sentences across lines; collapse all whitespace so a
    # sentence survives as one string for both retrieval and citation.
    text = re.sub(r"\s+", " ", text).strip()
    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + CHUNK_CHARS // 2, end)  # don't split mid-word
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP  # overlap < CHUNK_CHARS//2, so start always advances
    return out


class AnswerRequest(BaseModel):
    doc_id: str
    question: str
    mode: Literal["extract", "llm"] = "extract"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf uploads are accepted")
    path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(path, "wb") as fh:
        fh.write(await file.read())
    try:
        reader = PdfReader(path)
        texts, metas = [], []
        for page_no, page in enumerate(reader.pages, start=1):
            for piece in chunk_page(page.extract_text() or ""):
                metas.append({"page": page_no, "chunk_id": f"c-{len(texts):04d}"})
                texts.append(piece)
        if not texts:
            raise HTTPException(422, "no extractable text in PDF (scanned image? needs OCR)")
        doc_id = uuid.uuid4().hex[:12]
        vectors = list(embedder.embed(texts))
        qdrant.create_collection(
            collection_name=f"doc_{doc_id}",
            vectors_config=models.VectorParams(
                size=len(vectors[0]), distance=models.Distance.COSINE
            ),
        )
        qdrant.upsert(
            collection_name=f"doc_{doc_id}",
            points=[
                models.PointStruct(id=i, vector=v.tolist(), payload={**m, "text": t})
                for i, (v, m, t) in enumerate(zip(vectors, metas, texts))
            ],
        )
        return {"doc_id": doc_id, "pages": len(reader.pages), "chunks": len(texts)}
    finally:
        os.remove(path)


@app.post("/answer")
def answer(req: AnswerRequest):
    collection = f"doc_{req.doc_id}"
    if not qdrant.collection_exists(collection):
        raise HTTPException(404, f"unknown doc_id {req.doc_id}")

    qvec = next(iter(embedder.query_embed(req.question)))  # bge adds its query prefix here
    hits = qdrant.query_points(
        collection_name=collection, query=qvec.tolist(), limit=TOP_K, with_payload=True
    ).points
    hits = [h for h in hits if h.score >= MIN_SCORE]
    llm, mode = get_llm(req.mode)
    if not hits:
        return _abstain(mode)

    chunks = [
        {"text": h.payload["text"], "page": h.payload["page"], "chunk_id": h.payload["chunk_id"]}
        for h in hits
    ]
    try:
        text = llm.answer(req.question, chunks)
    except Exception:
        # The model can still vanish between the availability check and the call: pulled
        # halfway, deleted, Ollama restarting, generation timing out. Degrading to the
        # extractive answer beats 500ing on what is an optional enhancement.
        log.warning("llm mode failed, falling back to extract", exc_info=True)
        llm, mode = StubLLM(), "extract"
        text = llm.answer(req.question, chunks)

    if not text or NO_ANSWER in text:
        return _abstain(mode)

    # Grounding gate: keep only citations that point at chunks we actually retrieved.
    cited = set(re.findall(r"c-\d{4}", text)) & {c["chunk_id"] for c in chunks}
    if not cited:
        return _abstain(mode)
    return {
        "answer": text,
        "citations": [
            {"page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks if c["chunk_id"] in cited
        ],
        "abstained": False,
        "mode": mode,
    }


def _abstain(mode: str):
    return {
        "answer": "The document does not contain enough information to answer that.",
        "citations": [],
        "abstained": True,
        "mode": mode,
    }
