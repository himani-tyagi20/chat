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

Then open **http://localhost:8000** — drop a PDF in, ask questions, get cited answers. The document
lives for that browser session; "New PDF" clears it.

Every answer is tagged with the mode that produced it (`extracted` or `LLM`) and the pages it came
from, so you can always see where a claim originated.

## Docker (app + Qdrant + Ollama)

```bash
docker compose up -d
```

Three services plus a one-shot `ollama-pull` that fetches the model into a named volume and exits.
The app is on http://localhost:8000; Qdrant and Ollama stay on the internal network.

The first `up` downloads qwen2.5 (~4.7 GB) — watch it with `docker compose logs -f ollama-pull`.
Until it lands, LLM mode answers extractively and says so rather than failing. `ollama list` shows
nothing until the pull is fully complete, which is expected mid-download.

```bash
docker compose logs -f app     # service logs
docker compose down            # stop; add -v to also drop the model and Qdrant data
```

Qdrant data and the Ollama model persist in named volumes, so a restart doesn't re-download or lose
indexed documents — unlike the local run, where vectors are in-memory.

## Two answer modes

Toggle in the UI, or send `"mode"` in the API call.

| Mode | What it does | Needs |
|---|---|---|
| `extract` (default) | Returns the best-matching sentences from the retrieved chunks, verbatim. Cannot hallucinate — it only copies. | nothing |
| `llm` | Sends only the retrieved chunks to a local open-source model via Ollama. | Ollama running |

For `llm` mode, install [Ollama](https://ollama.com) and pull any open-source model:

```bash
ollama pull qwen2.5
```

Nothing to configure — the service finds it on `localhost:11434`. Point elsewhere with `OLLAMA_URL`,
or choose another model with `OLLAMA_MODEL` — any Ollama tag works:

```bash
OLLAMA_MODEL=qwen2.5:14b .venv/bin/uvicorn app:app --port 8000
```

Reasoning variants (`qwen3`, `deepseek-r1`) are supported too: their `<think>` blocks are stripped
before the answer is returned, so discarded reasoning can't leak into the response or be mistaken for
a citation.

If `llm` is requested and no model is reachable, the request still answers in `extract` mode and the
response says so (`"mode": "extract"`) rather than failing.

## Ingest a PDF

```bash
curl -s -F "file=@/path/to/document.pdf" http://localhost:8000/ingest
```

```json
{ "doc_id": "abc123def456", "pages": 12, "chunks": 84 }
```

## Ask a question

```bash
curl -s -X POST http://localhost:8000/answer -H 'content-type: application/json' -d '{"doc_id":"abc123def456","question":"What is the refund window?","mode":"extract"}'
```

```json
{
  "answer": "Customers may request a refund within 30 days of purchase. [c-0004]",
  "citations": [{ "page": 2, "chunk_id": "c-0004" }],
  "abstained": false,
  "mode": "extract"
}
```

When the document doesn't contain the answer:

```json
{
  "answer": "The document does not contain enough information to answer that.",
  "citations": [],
  "abstained": true,
  "mode": "extract"
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
| `OLLAMA_URL` | `http://localhost:11434` | Where to reach Ollama for `llm` mode. |
| `OLLAMA_MODEL` | `qwen2.5` | Which open-source model to use. |

## Layout

- `app.py` — endpoints, chunking, Qdrant, retrieval, abstention
- `llm.py` — `LLM` interface, extractive `StubLLM`, `OllamaLLM`
- `static/index.html` — the frontend, one file, no build step
- `test_app.py` — end-to-end tests
- `DESIGN.md` — chunking, retrieval, grounding, abstention

## Running it on GitHub

`.github/workflows/test.yml` runs the test suite on every push — that part works out of the box.

GitHub **Pages cannot host this**: Pages serves static files only, and this needs a live Python
process for parsing, embedding and retrieval. To run the whole service from the repo, use a
Codespace (`uvicorn app:app` then open the forwarded port) or deploy the container anywhere that runs
Python — Fly, Render, Railway, Cloud Run.
