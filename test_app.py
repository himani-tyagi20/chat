"""End-to-end: build a PDF, ingest it, ask an answerable and an unanswerable question."""

import io
import textwrap
import urllib.error

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from app import app, chunk_page

PAGES = [
    "Onboarding Guide. The support desk is open Monday to Friday, 9am to 6pm IST. "
    "Escalations after hours go to the on-call rotation via the paging system.",
    "Refund Policy. Customers may request a refund within 30 days of purchase. "
    "Refunds are processed to the original payment method within 7 business days.",
    "Security. All uploaded files are encrypted at rest using AES-256. "
    "Access logs are retained for 90 days and reviewed every quarter.",
]


def make_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    for text in PAGES:
        y = 720
        for line in textwrap.wrap(text, 80):
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="module")
def client_and_doc():
    client = TestClient(app)
    r = client.post("/ingest", files={"file": ("test.pdf", make_pdf(), "application/pdf")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pages"] == 3 and body["chunks"] >= 3
    return client, body["doc_id"]


def test_chunking_covers_text_and_terminates():
    text = "word " * 800
    chunks = chunk_page(text)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]  # overlap carried across the boundary


@pytest.mark.parametrize(
    "question,expect_page,expect_word",
    [
        ("How long do customers have to request a refund?", 2, "30 days"),
        ("How long are access logs retained?", 3, "90 days"),
        ("When is the support desk open?", 1, "9am"),
        # short question, few content words — must not trip the answer-side overlap gate
        ("What is the refund window?", 2, "30 days"),
    ],
)
def test_grounded_answers_are_cited(client_and_doc, question, expect_page, expect_word):
    client, doc_id = client_and_doc
    body = client.post("/answer", json={"doc_id": doc_id, "question": question}).json()
    assert body["abstained"] is False, body
    assert expect_word in body["answer"]
    assert expect_page in [c["page"] for c in body["citations"]]


def test_abstains_when_not_in_document(client_and_doc):
    client, doc_id = client_and_doc
    body = client.post(
        "/answer",
        json={"doc_id": doc_id, "question": "What is the maximum dosage of ibuprofen for a child?"},
    ).json()
    assert body["abstained"] is True
    assert body["citations"] == []


def test_llm_mode_falls_back_to_extract_when_no_model_running(client_and_doc, monkeypatch):
    """Asking for the LLM with nothing served must still answer, flagged as extracted."""
    monkeypatch.setattr("llm.OllamaLLM.available", lambda self: False)
    client, doc_id = client_and_doc
    body = client.post(
        "/answer",
        json={"doc_id": doc_id, "question": "What is the refund window?", "mode": "llm"},
    ).json()
    assert body["mode"] == "extract"
    assert body["abstained"] is False


def test_llm_mode_uses_model_and_drops_invented_citations(client_and_doc, monkeypatch):
    """A citation pointing at a chunk we never retrieved must not survive."""
    monkeypatch.setattr("llm.OllamaLLM.available", lambda self: True)
    monkeypatch.setattr(
        "llm.OllamaLLM.answer",
        lambda self, q, chunks: f"Refunds take 7 business days [{chunks[0]['chunk_id']}]. "
        "Also the CEO earns $2M [c-9999].",
    )
    client, doc_id = client_and_doc
    body = client.post(
        "/answer",
        json={"doc_id": doc_id, "question": "What is the refund window?", "mode": "llm"},
    ).json()
    assert body["mode"] == "llm"
    assert [c["chunk_id"] for c in body["citations"]] == ["c-0001"]  # c-9999 dropped
    assert all(c["chunk_id"] != "c-9999" for c in body["citations"])


def test_llm_no_answer_sentinel_abstains(client_and_doc, monkeypatch):
    monkeypatch.setattr("llm.OllamaLLM.available", lambda self: True)
    monkeypatch.setattr("llm.OllamaLLM.answer", lambda self, q, chunks: "NO_ANSWER")
    client, doc_id = client_and_doc
    body = client.post(
        "/answer",
        json={"doc_id": doc_id, "question": "What is the refund window?", "mode": "llm"},
    ).json()
    assert body["abstained"] is True and body["citations"] == []


def test_thinking_blocks_are_stripped_before_citation_parsing():
    """Qwen3-style reasoning must not reach the answer, nor contribute citations."""
    from llm import strip_thinking

    assert strip_thinking("<think>maybe c-0007?</think>Refunds take 30 days [c-0001]") == (
        "Refunds take 30 days [c-0001]"
    )
    assert strip_thinking("hmm c-0007</think>Answer [c-0001]") == "Answer [c-0001]"
    assert strip_thinking("  plain answer [c-0002]  ") == "plain answer [c-0002]"


def test_reasoning_citations_do_not_leak(client_and_doc, monkeypatch):
    monkeypatch.setattr("llm.OllamaLLM.available", lambda self: True)
    monkeypatch.setattr(
        "llm.OllamaLLM.answer",
        lambda self, q, chunks: __import__("llm").strip_thinking(
            "<think>could be [c-0000] or [c-0002]</think>"
            f"Refunds take 30 days [{chunks[0]['chunk_id']}]."
        ),
    )
    client, doc_id = client_and_doc
    body = client.post(
        "/answer", json={"doc_id": doc_id, "question": "What is the refund window?", "mode": "llm"}
    ).json()
    assert "<think>" not in body["answer"]
    assert [c["chunk_id"] for c in body["citations"]] == ["c-0001"]


@pytest.mark.parametrize(
    "tags,model,expected",
    [
        ({"models": []}, "qwen2.5", False),  # server up, nothing pulled -> the 404 case
        ({"models": [{"name": "qwen2.5:latest"}]}, "qwen2.5", True),  # bare name matches :latest
        ({"models": [{"name": "qwen2.5:14b"}]}, "qwen2.5", True),
        ({"models": [{"name": "qwen2.5:7b"}]}, "qwen2.5:14b", False),  # wrong tag is not a match
        ({"models": [{"name": "llama3.2:latest"}]}, "qwen2.5", False),
    ],
)
def test_available_requires_the_model_not_just_the_server(monkeypatch, tags, model, expected):
    import io as _io
    import json as _json

    from llm import OllamaLLM

    monkeypatch.setattr(
        "llm.urllib.request.urlopen",
        lambda *a, **k: _io.BytesIO(_json.dumps(tags).encode()),
    )
    assert OllamaLLM(model=model).available() is expected


def test_llm_transport_failure_degrades_instead_of_500(client_and_doc, monkeypatch):
    """Model vanishing mid-flight must not surface as an Internal Server Error."""
    monkeypatch.setattr("llm.OllamaLLM.available", lambda self: True)

    def boom(self, q, chunks):
        raise urllib.error.HTTPError("http://ollama/api/chat", 404, "Not Found", {}, None)

    monkeypatch.setattr("llm.OllamaLLM.answer", boom)
    client, doc_id = client_and_doc
    r = client.post(
        "/answer", json={"doc_id": doc_id, "question": "What is the refund window?", "mode": "llm"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "extract" and body["abstained"] is False
    assert "30 days" in body["answer"]


def test_frontend_is_served(client_and_doc):
    client, _ = client_and_doc
    r = client.get("/")
    assert r.status_code == 200 and "Chat with a PDF" in r.text


def test_unknown_doc_id_404s(client_and_doc):
    client, _ = client_and_doc
    r = client.post("/answer", json={"doc_id": "nope", "question": "anything"})
    assert r.status_code == 404
