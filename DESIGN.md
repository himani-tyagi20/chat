# DESIGN

## Chunking

Per page, not per document. `pypdf` extracts each page's text; whitespace (including the line breaks
PDF extraction sprinkles mid-sentence) is collapsed to single spaces, then the page is cut into
~1000-character windows with 150 characters of overlap, backing each cut up to the nearest space so
words and usually sentences stay intact.

Chunking per page means the page number is known by construction — no offset bookkeeping to map a
chunk back to where it came from. 1000 characters is roughly a paragraph or two: large enough that a
fact and its subject stay together, small enough that a retrieved chunk is mostly signal. The overlap
covers facts that straddle a boundary. Chunk ids are document-global (`c-0000`, `c-0001`, …) so a
citation is unambiguous on its own.

Trade-off accepted: a fixed window ignores document structure (headings, tables, lists). Semantic or
layout-aware splitting would retrieve better on structured documents; it is not worth the complexity
until the corpus proves it.

## Embeddings and vector store

**Embeddings:** `BAAI/bge-small-en-v1.5` via `fastembed` — ONNX Runtime, no PyTorch, ~130 MB, 384
dimensions, runs on CPU and offline after the first download. Queries go through `query_embed`, which
applies the retrieval prefix BGE was trained with; passages are embedded plain. Getting that
asymmetry right matters more for score separation than model size does.

**Store:** Qdrant, one collection per document (`doc_<id>`), cosine distance. Embedded local mode by
default, so `pip install` and run — no Docker, no server. Setting `QDRANT_URL` switches to a real
Qdrant server with no other code change. A collection per document makes retrieval leak-proof between
documents and makes deletion a single call.

Ingestion is fully at request time: parse, chunk, embed, upsert, all inside `POST /ingest`. Nothing
is pre-baked. Vectors live in the local instance's memory, so a restart clears them; pointing at a
Qdrant server makes them durable.

## Grounding

Three gates, and the answer must pass all of them.

1. **Retrieval.** The model only ever sees chunks retrieved from the requested `doc_id`'s collection.
2. **Generation.** The `LLM` interface takes `(question, chunks)` — there is no path by which
   untethered text reaches the model. The default `StubLLM` is extractive: it scores each sentence of
   each retrieved chunk by token overlap with the question and returns the best two **verbatim**,
   each tagged with its chunk id. It cannot invent a fact because it only copies. The optional
   `ClaudeLLM` gets a system prompt restricting it to the excerpts and requiring `[c-NNNN]` markers.
3. **Citation.** After generation, citation markers are parsed out of the answer and intersected with
   the ids actually retrieved. Anything else is discarded. If nothing survives, the response abstains
   instead of returning an uncited claim.

Gate 3 is what keeps grounding honest when the stub is swapped for a real model: a hallucinated
citation points at a chunk that was never retrieved, so it is dropped, and an answer with no valid
citation left becomes an abstention.

## Abstention

"Not in the document" is decided in two places, because the two failure modes are different.

**Retrieval-side (topic absent).** Cosine scores below `MIN_SCORE` (default 0.62) are dropped; if
nothing survives, abstain before the model is called. Measured on the test corpus with bge-small:

| Question | Best score |
|---|---|
| "How long do customers have to request a refund?" | 0.895 |
| "When is the support desk open?" | 0.857 |
| "How long are access logs retained?" | 0.781 |
| "What is the company's revenue in Q3?" | 0.504 |
| "Maximum dosage of ibuprofen for a child?" | 0.490 |
| "Who won the 2018 world cup?" | 0.357 |

On-topic 0.78–0.90, off-topic 0.36–0.50, and the threshold sits in the middle of the gap. That gap
narrows on longer and more homogeneous corpora, so `MIN_SCORE` is an environment variable, not a
constant — it is the one number to re-measure against a new document set.

**Answer-side (topic present, answer absent).** A chunk can be about the right subject and still not
contain the answer. The stub requires a minimum question/sentence token overlap and returns
`NO_ANSWER` otherwise; a real LLM is instructed to return the same sentinel; and the citation gate
above turns an uncitable answer into an abstention. Any of these produces the same response shape:
a fixed safe message, empty citations, `abstained: true`. Nothing is invented on the abstain path
because that path never renders model text.

The bias is deliberately toward abstaining: a wrong cited answer is worse than an admitted gap.

## What was deliberately left out

- **No reranker.** Vector top-k with a score floor is enough at this document scale; a cross-encoder
  is the first upgrade if precision falls short on a longer PDF.
- **No OCR.** A scanned PDF yields no extractable text, and `/ingest` returns 422 saying so rather
  than silently indexing nothing.
- **No persistence or auth.** In-scope for a service, out of scope for a take-home; the `QDRANT_URL`
  switch is the seam where persistence lands.
- **No chunk-level dedup or table extraction.** Both matter for real clinical PDFs, neither is
  needed to demonstrate the retrieval → grounding → abstention path.
