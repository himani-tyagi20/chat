"""Two answer modes behind one interface.

`extract` copies sentences out of the retrieved chunks — no model, always available.
`llm` calls a local open-source model through Ollama's HTTP API (stdlib only, no SDK).
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

NO_ANSWER = "NO_ANSWER"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")

SYSTEM = """You answer strictly from the numbered excerpts given to you.
Rules:
- Use ONLY facts stated in the excerpts. No outside knowledge, no inference beyond the text.
- Cite the chunk id in square brackets after every claim, e.g. [c-0031].
- Keep the answer to two sentences at most.
- If the excerpts do not contain the answer, reply with exactly: NO_ANSWER
"""

STOP = frozenset(
    "the and for are was were what when where who whom which how why can may must with "
    "from that this these those there has have had does did you your our their its any "
    "all not but into than then they them his her out per via".split()
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _tokens(text: str) -> set[str]:
    """Content words only — stopwords would make overlap counts meaningless."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


def strip_thinking(text: str) -> str:
    """Reasoning models (qwen3, deepseek-r1) prepend <think>…</think>; it is not the answer.

    Left unstripped it would leak into the response and, worse, its chunk ids would be parsed
    as citations for reasoning the model discarded.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)  # unclosed opener
    return text.strip()


def format_excerpts(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['chunk_id']}] (page {c['page']})\n{c['text']}" for c in chunks)


class LLM(Protocol):
    def answer(self, question: str, chunks: list[dict]) -> str:
        """Return an answer citing [chunk_id]s, or NO_ANSWER."""


class StubLLM:
    """Extractive: returns the best-matching sentences verbatim, so it cannot invent facts."""

    # Relevance is already decided by the MIN_SCORE retrieval gate in app.py; this only needs
    # to require that the extracted sentence shares a content word with the question.
    def __init__(self, min_overlap: int = 1):
        self.min_overlap = min_overlap

    def answer(self, question: str, chunks: list[dict]) -> str:
        q = _tokens(question)
        scored = [
            (len(q & _tokens(s)), s, c["chunk_id"])
            for c in chunks
            for s in _sentences(c["text"])
        ]
        best = sorted(scored, key=lambda x: -x[0])[:2]
        if not best or best[0][0] < self.min_overlap:
            return NO_ANSWER
        return " ".join(f"{s} [{cid}]" for score, s, cid in best if score > 0)


class OllamaLLM:
    """Any open-source model served by Ollama (llama3.2, qwen2.5, mistral, phi3...)."""

    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL, timeout: int = 120):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout

    def available(self) -> bool:
        """A reachable server is not enough — the model itself must be pulled.

        Ollama answers /api/tags happily while holding no models at all, then 404s on
        /api/chat. Checking only reachability turns a still-downloading model into a 500.
        """
        try:
            body = urllib.request.urlopen(f"{self.url}/api/tags", timeout=2).read()
        except (urllib.error.URLError, OSError):
            return False
        names = {m.get("name", "") for m in json.loads(body).get("models", [])}
        # "qwen2.5" should match the "qwen2.5:latest" that `ollama pull qwen2.5` installs.
        return self.model in names or (
            ":" not in self.model and any(n.split(":")[0] == self.model for n in names)
        )

    def answer(self, question: str, chunks: list[dict]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},  # deterministic: this is extraction, not writing
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"Excerpts:\n{format_excerpts(chunks)}\n\nQuestion: {question}",
                    },
                ],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return strip_thinking(json.loads(resp.read())["message"]["content"])


def get_llm(mode: str) -> tuple[LLM, str]:
    """Return (llm, mode_actually_used) — falls back to extract if no model is reachable."""
    if mode == "llm":
        ollama = OllamaLLM()
        if ollama.available():
            return ollama, "llm"
        return StubLLM(), "extract"  # nothing running; answer anyway rather than 503
    return StubLLM(), "extract"
