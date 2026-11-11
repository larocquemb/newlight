"""Short-page regression: PDF39 / printed page5, visually inspected source.

Whitespace and apostrophe style are ignored only for excerpt matching;
the separate North-/west. assertion checks the visible printed line break.
"""
import json
import os
from pathlib import Path

import pytest

from ocr_parse import Page, validate_page


def check_page39(page):
    # Saved artifacts also contain provenance metadata outside the model schema.
    validate_page(json.dumps({key: page[key] for key in Page.model_fields}), 39)
    body = " ".join(page["body"].split()).replace("’", "'")
    assert page["footnotes"] == []
    assert page["printed_page_number"] == "5"
    assert page["running_headers"] == ["PROFIT OF THE FIRST CAMPAIGN."]
    assert "5" not in [line.strip() for line in page["body"].splitlines()], (
        "Printed page number leaked into body"
    )
    assert body.startswith("strongly opposed by a neighbor, of the T Association from Montreal")
    assert "£700, Halifax" in body
    assert "1½ canoes, containing 40 pieces of goods." in body
    assert "This campaign was my first coup d'essai" in body
    assert "North-\nwest." in page["body"], "Printed line-end hyphen/break lost"


@pytest.fixture
def short_page():
    return {"pdf_page": 39, "body": (
        "strongly opposed by a neighbor, of the T Association from\n"
        "Montreal, I had a clear profit of upward of £700, Halifax\n"
        "currency, on the outfit of loading 1½ canoes, containing 40\n"
        "pieces of goods.\n"
        "This campaign was my first coup d’essai in the North-\nwest."),
        "footnotes": [], "running_headers": ["PROFIT OF THE FIRST CAMPAIGN."],
        "printed_page_number": "5", "uncertainties": []}


def test_short_page_is_complete_without_notes(short_page):
    check_page39(short_page)


def test_rejects_header_in_body(short_page):
    short_page["body"] = "PROFIT OF THE FIRST CAMPAIGN.\n\n" + short_page["body"]
    with pytest.raises(ValueError, match="Running header duplicated"):
        check_page39(short_page)


def test_rejects_reflowed_split_word(short_page):
    short_page["body"] = short_page["body"].replace("North-\nwest.", "North-west.")
    with pytest.raises(AssertionError, match="line-end"):
        check_page39(short_page)


@pytest.mark.parametrize("printed", ["39", None, ""])
def test_rejects_wrong_or_missing_printed_number(short_page, printed):
    short_page["printed_page_number"] = printed
    with pytest.raises(AssertionError):
        check_page39(short_page)


def test_rejects_wrong_header(short_page):
    short_page["running_headers"] = ["ADVANCE OF SPRING IN MANITOBA."]
    with pytest.raises(AssertionError):
        check_page39(short_page)


@pytest.mark.parametrize("location", ["start", "end"])
def test_rejects_printed_number_in_body(short_page, location):
    short_page["body"] = ("5\n" + short_page["body"] if location == "start"
                          else short_page["body"] + "\n5")
    with pytest.raises(AssertionError, match="page number leaked"):
        check_page39(short_page)


def test_generated_page39():
    directory = os.environ.get("NEWLIGHT_QA_OUTPUT")
    if not directory:
        pytest.skip("Set NEWLIGHT_QA_OUTPUT to check generated page 39")
    path = Path(directory) / "page_0039.json"
    assert path.is_file(), f"Missing OCR page: {path}"
    check_page39(json.loads(path.read_text(encoding="utf-8")))
