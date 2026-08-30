# OneRx — PDF → Vectors → Grounded Answers

FastAPI service that ingests a PDF **at runtime** (parse → chunk → embed → Qdrant), then answers
questions using only that document's retrieved chunks, with page/chunk citations, and abstains when
the document doesn't support an answer.

No paid API key needed: embeddings are local ONNX, and the LLM sits behind an interface with an
extractive stub used by default.

## Run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --port 8000
```

First start downloads the ONNX embedding model (~130 MB) into the HF cache; after that it runs offline.

## Ingest a PDF

```bash
curl -s -F "file=@/path/to/document.pdf" http://localhost:8000/ingest
```

```json
{ "doc_id": "abc123def456", "pages": 12, "chunks": 84 }
```

## Ask a question

```bash
curl -s -X POST http://localhost:8000/answer -H 'content-type: application/json' -d '{"doc_id":"abc123def456","question":"What is the refund window?"}'
```

```json
{
  "answer": "Customers may request a refund within 30 days of purchase. [c-0004]",
  "citations": [{ "page": 2, "chunk_id": "c-0004" }],
  "abstained": false
}
```

When the document doesn't contain the answer:

```json
{
  "answer": "The document does not contain enough information to answer that.",
  "citations": [],
  "abstained": true
}
```

Interactive docs at http://localhost:8000/docs.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Builds a 3-page PDF on the fly, ingests it, checks three cited answers land on the right pages, and
checks an out-of-document question abstains.

## Configuration

All optional, via environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `MIN_SCORE` | `0.62` | Cosine floor for "this is in the document". Lower = fewer abstentions. |
| `TOP_K` | `5` | Chunks retrieved per question. |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` | `1000` / `150` | Chunk window and overlap, in characters. |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed ONNX model. |
| `QDRANT_URL` | _(unset)_ | Point at a Qdrant server; unset uses the embedded local instance. |
| `USE_REAL_LLM` + `ANTHROPIC_API_KEY` | _(unset)_ | Set `USE_REAL_LLM=1` with a key to swap the stub for Claude (`pip install anthropic`). |

## Layout

- `app.py` — endpoints, chunking, Qdrant, retrieval, abstention
- `llm.py` — `LLM` interface, offline `StubLLM`, optional `ClaudeLLM`
- `test_app.py` — end-to-end tests
- `DESIGN.md` — chunking, retrieval, grounding, abstention
