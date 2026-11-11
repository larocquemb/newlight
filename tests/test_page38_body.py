"""Regression for the small-type BODY expense table on PDF page 38.

Excerpts/amounts visually read from data/processed/pdf_pages/page_0038.png, printed page 4.
These checks test table retention, not full transcription accuracy or vision.
Set NEWLIGHT_QA_OUTPUT to a generated-output directory to check actual OCR.
"""
import json
import os
from pathlib import Path

import pytest

REQUIRED_NARRATIVE = (
    "The snow was then melted in many",
    "During my journey, which occupied",
    "7th. I saw",
    "16th. Ducks and other spring birds. 20th.",
    "24th. The meadows on fire. 27th.",
    "In a few days",
)


def check_page38_completeness(page):
    body = " ".join(page["body"].split())
    for phrase in REQUIRED_NARRATIVE:
        assert phrase in body, f"Missing source passage: {phrase}"
    assert "[migratorius]" not in body, "Spurious bracket inserted inside Ectopistes migratorius"


def check_page38_table(page):
    body = " ".join(page["body"].split())
    assert page["pdf_page"] == 38
    assert page["printed_page_number"] == "4"
    assert page["running_headers"] == ["ADVANCE OF SPRING IN MANITOBA."]
    assert "ADVANCE OF SPRING IN MANITOBA." not in body
    assert "4" not in [line.strip() for line in page["body"].splitlines()]
    assert page["footnotes"] == [], "Page 38 has an embedded table, not editorial footnotes"
    assert "This campaign my house expenses for 17 persons were" in body
    assert "Total cost" in body
    assert "Notwithstanding this heavy expense and loss" in body
    table = body.split("This campaign", 1)[1].split("Notwithstanding", 1)[0]
    for amount in ("1500", "982", "618", "364", "1864"):
        assert amount in table.replace(",", ""), f"Missing expense-table amount: {amount}"


@pytest.fixture
def table_excerpts():
    # Excerpts only, not a complete reference transcription or model output.
    return {"pdf_page": 38, "printed_page_number": "4",
            "running_headers": ["ADVANCE OF SPRING IN MANITOBA."],
            "footnotes": [], "body": (
        "This campaign my house expenses for 17 persons were, . 1500 skins\n"
        "I gave out in debts in the fall, to the Saulteur Indians, 982\n"
        "Received in the course of the season, . . 618\n"
        "Loss, . . . 364\nTotal cost, . . . 1864 skins\n"
        "Notwithstanding this heavy expense and loss, amounting")}


def test_table_belongs_in_body(table_excerpts):
    check_page38_table(table_excerpts)


def test_rejects_table_misclassified_as_footnote(table_excerpts):
    table_excerpts["footnotes"] = [{"text": table_excerpts["body"]}]
    with pytest.raises(AssertionError, match="embedded table"):
        check_page38_table(table_excerpts)


def test_rejects_missing_table_amount(table_excerpts):
    table_excerpts["body"] = table_excerpts["body"].replace("982", "")
    with pytest.raises(AssertionError, match="982"):
        check_page38_table(table_excerpts)


def test_generated_page38_table():
    directory = os.environ.get("NEWLIGHT_QA_OUTPUT")
    if not directory:
        pytest.skip("Set NEWLIGHT_QA_OUTPUT to check generated page 38")
    path = Path(directory) / "page_0038.json"
    assert path.is_file(), f"Missing OCR page: {path}"
    page = json.loads(path.read_text(encoding="utf-8"))
    check_page38_table(page)
    check_page38_completeness(page)


@pytest.mark.parametrize("missing", REQUIRED_NARRATIVE)
def test_rejects_observed_missing_passages(missing):
    body = "\n".join(phrase for phrase in REQUIRED_NARRATIVE if phrase != missing)
    with pytest.raises(AssertionError, match="Missing source passage"):
        check_page38_completeness({"body": body})


def test_rejects_observed_added_bracket():
    with pytest.raises(AssertionError, match="Spurious bracket"):
        check_page38_completeness({"body": "\n".join(REQUIRED_NARRATIVE) + "\n[migratorius]"})
