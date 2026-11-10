"""Quality gates for New Light OCR JSON outputs (no Ollama requests).

Run: python -m pytest tests/test_ocr_quality.py -v -rs
Select pages: NEWLIGHT_QA_START=35 NEWLIGHT_QA_END=39 python -m pytest ...
Missing pages fail by default; override with NEWLIGHT_QA_ALLOW_MISSING=1.
"""
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
START = int(os.getenv("NEWLIGHT_QA_START", "35"))
END = int(os.getenv("NEWLIGHT_QA_END", "39"))
ALLOW_MISSING = os.getenv("NEWLIGHT_QA_ALLOW_MISSING") == "1"
LANGUAGES = {"en", "fr", "und"}
UNCERTAINTY = re.compile(r"\[(?:unclear|illegible)\]", re.I)


def normalized(text):
    return " ".join(text.split()).casefold()


def load_page(page):
    path = PROCESSED / f"page_{page:04d}.json"
    if not path.is_file():
        if ALLOW_MISSING:
            pytest.skip(f"Missing OCR output: {path}")
        pytest.fail(f"Missing OCR output: {path}; run OCR for page {page}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        pytest.fail(f"Invalid UTF-8 JSON in {path}: {exc}")


@pytest.mark.parametrize("page", range(START, END + 1))
def test_page_schema_and_content(page):
    data = load_page(page)
    assert data.get("pdf_page") == page
    assert isinstance(data.get("body"), list) and data["body"], "Empty body"
    assert isinstance(data.get("footnotes"), list)
    assert isinstance(data.get("validation_warnings", []), list)
    assert isinstance(data.get("review_required", False), bool)
    for kind in ("body", "footnotes"):
        for entry in data[kind]:
            assert isinstance(entry, dict)
            assert entry.get("language") in LANGUAGES, (kind, entry)
            assert isinstance(entry.get("text"), str) and entry["text"].strip()
            assert not UNCERTAINTY.search(entry["text"]), (page, kind)
    for note in data["footnotes"]:
        assert note.get("marker") is None or isinstance(note["marker"], str)
        for key in ("continues_from_previous_page", "continues_on_next_page"):
            assert note.get(key) in (True, False, None), (page, key)


@pytest.mark.parametrize("page", range(START, END + 1))
def test_no_obvious_footnote_duplication(page):
    data = load_page(page)
    body = normalized(" ".join(x["text"] for x in data["body"]))
    for note in data["footnotes"]:
        text = normalized(note["text"])
        if len(text) >= 50:
            assert text[:50] not in body, f"Page {page}: footnote {note.get('marker')} duplicated in body"


@pytest.mark.parametrize("page", range(START, END + 1))
def test_no_repeated_body_passages(page):
    data = load_page(page)
    seen = set()
    for passage in data["body"]:
        text = normalized(passage["text"])
        if len(text) >= 60:
            assert text not in seen, f"Page {page}: repeated body passage"
            seen.add(text)


def test_page_38_financial_table():
    if not START <= 38 <= END:
        pytest.skip("Page 38 outside selected range")
    data = load_page(38)
    body = " ".join(x["text"] for x in data["body"])
    assert data["footnotes"] == [], "Page 38 should have no editorial footnotes"
    for amount in ("1500", "982", "618", "364", "1864"):
        assert amount in body.replace(",", ""), f"Missing financial amount {amount}"
    assert "Notwithstanding" in body, "Narrative after table is missing"


def test_quality_summary():
    """Print a review queue; warnings are not automatically test failures."""
    missing, review = [], []
    for page in range(START, END + 1):
        path = PROCESSED / f"page_{page:04d}.json"
        if not path.exists():
            missing.append(page)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            review.append((page, ["invalid JSON"]))
            continue
        warnings = data.get("validation_warnings", [])
        if warnings or data.get("review_required"):
            review.append((page, warnings or ["review_required=true"]))
    print(f"\nOCR QA pages {START}-{END}: missing={missing}; review={review}")
    assert not missing or ALLOW_MISSING, f"Missing OCR pages: {missing}"
