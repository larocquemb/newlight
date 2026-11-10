
"""Tests for the New Light OCR pipeline."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newlight.ocr import normalize_text, validate_result


def test_preserves_french_accents():
    text = "François avoit connoissance de la rivière."
    assert "François" in text
    assert "avoit" in text
    assert "rivière" in text


def test_no_footnote_duplication():
    result = {
        "body": [
            {
                "language": "en",
                "text": "The voyageurs crossed the river."
            }
        ],
        "footnotes": [
            {
                "marker": "1",
                "language": "en",
                "text": (
                    "This is a historical footnote containing "
                    "more than sixty characters of explanation."
                ),
                "continues_from_previous_page": False,
                "continues_on_next_page": False,
            }
        ],
    }

    assert validate_result(result) == []


def test_detects_footnote_duplication():
    note = (
        "This historical footnote contains enough text "
        "to detect accidental duplication in the body."
    )

    result = {
        "body": [{"language": "en", "text": note}],
        "footnotes": [{"marker": "1", "text": note}],
    }

    warnings = validate_result(result)
    assert len(warnings) == 1
    assert "duplicated" in warnings[0]


@pytest.mark.parametrize("page", range(35, 40))
def test_page_output(page):
    path = ROOT / "data/processed" / f"page_{page:04d}.json"

    if not path.exists():
        pytest.skip(f"Page {page} not processed")

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["pdf_page"] == page
    assert isinstance(data["body"], list)
    assert isinstance(data["footnotes"], list)

    for passage in data["body"]:
        assert passage["language"] in ("en", "fr", "und")
        assert isinstance(passage["text"], str)

    for note in data["footnotes"]:
        assert isinstance(note["text"], str)


def test_page_38_no_duplication():
    path = ROOT / "data/processed/page_0038.json"

    if not path.exists():
        pytest.skip("Page 38 not processed")

    data = json.loads(path.read_text(encoding="utf-8"))

    assert validate_result(data) == []
