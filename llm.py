"""LLM seam: a stub that runs offline, and a real Claude client when a key is present."""

import os
import re
from typing import Protocol

NO_ANSWER = "NO_ANSWER"

SYSTEM = """You answer strictly from the numbered excerpts given to you.
Rules:
- Use ONLY facts present in the excerpts. No outside knowledge, no inference beyond the text.
- Cite the chunk id in square brackets after every claim, e.g. [c-0031].
- If the excerpts do not contain the answer, reply with exactly: NO_ANSWER
"""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


STOP = frozenset(
    "the and for are was were what when where who whom which how why can may must with "
    "from that this these those there has have had does did you your our their its any "
    "all not but into than then they them his her out per via".split()
)


def _tokens(text: str) -> set[str]:
    """Content words only — stopwords would make overlap counts meaningless."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


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


class ClaudeLLM:
    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic  # lazy: only needed when a key is configured

        self.client = anthropic.Anthropic()
        self.model = model

    def answer(self, question: str, chunks: list[dict]) -> str:
        excerpts = "\n\n".join(
            f"[{c['chunk_id']}] (page {c['page']})\n{c['text']}" for c in chunks
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Excerpts:\n{excerpts}\n\nQuestion: {question}"}],
        )
        return msg.content[0].text.strip()


def get_llm() -> LLM:
    if os.getenv("ANTHROPIC_API_KEY") and os.getenv("USE_REAL_LLM") == "1":
        return ClaudeLLM(os.getenv("LLM_MODEL", "claude-sonnet-5"))
    return StubLLM()
