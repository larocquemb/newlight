"""PDF40–42 layout regressions, derived from visually inspected source images.

Synthetic fixtures contain reference excerpts only, not full transcriptions.
NEWLIGHT_QA_OUTPUT enables actual-output checks for all three pages.
"""
import json
import os
from pathlib import Path

import pytest


def normalized(text):
    return " ".join(text.split())


def check_notes(page, number):
    assert page["pdf_page"] == number
    assert page["printed_page_number"] == str(number - 34)
    notes = page["footnotes"]
    expected = {40: ["1", "2"], 41: [None], 42: [None, "3", "4"]}[number]
    # Superscript glyphs and ordinary digits identify the same printed marker.
    markers = [n["marker"].translate(str.maketrans("¹²³⁴", "1234"))
               if n["marker"] is not None else None for n in notes]
    assert markers == expected, f"PDF{number}: expected distinct notes {expected}, got {markers}"
    assert all(n["text"].strip() for n in notes), "Empty note text"
    body = normalized(page["body"])
    if number == 40:
        assert "CHAPTER II." in body
        assert page["running_headers"] == [], "Chapter title belongs in body, not running headers"
        assert "On the regular Rainy River route" in normalized(notes[0]["text"])
        assert "1801, pp." in normalized(notes[0]["text"]), "Footnote 1 fragmented or incomplete"
        assert "Grand Portage, of the French and English" in normalized(notes[1]["text"])
        assert normalized(notes[1]["text"]).endswith("too shallow for")
        assert notes[1]["continues_on_next_page"] is True
        assert "On the regular Rainy River route" not in body
    elif number == 41:
        assert "Merchandise" in body and "28 pieces" in body, "Cargo table missing from body"
        text = normalized(notes[0]["text"])
        assert text.startswith("vessels to land")
        assert "Pigeon r., also called Dove r." in text, "Second paragraph of continued note missing"
        assert text.endswith("Rivière")
        assert notes[0]["continues_from_previous_page"] is True
        assert notes[0]["continues_on_next_page"] is True
        assert "vessels to land" not in body, "Continued footnote leaked into body"
        assert "Pigeon r., also called Dove r." not in body
    else:
        assert normalized(notes[0]["text"]).startswith("aux Tourtres")
        assert "Portage du Perdrix" in normalized(notes[1]["text"])
        assert "Otherwise Big Rock or Great Stone portage" in normalized(notes[2]["text"])
        assert notes[0]["continues_from_previous_page"] is True
        assert notes[0]["continues_on_next_page"] is False
        assert "aux Tourtres" not in body, "Continued footnote leaked into body"
        assert "Portage du Perdrix" not in body, "Note 3 leaked into body"


def note(marker, text, previous=False, following=False):
    return {"marker": marker, "text": text, "continues_from_previous_page": previous,
            "continues_on_next_page": following}


@pytest.fixture
def excerpts():
    return {
        40: {"pdf_page": 40, "printed_page_number": "6", "running_headers": [],
             "body": "CHAPTER II.\nTHE RED RIVER BRIGADE OF 1800.", "footnotes": [
                 note("1", "On the regular Rainy River route\n1801, pp. xlviii–lxii."),
                 note("2", "Grand Portage, of the French and English\ntoo shallow for", following=True)]},
        41: {"pdf_page": 41, "printed_page_number": "7", "body": "Merchandise\nTotal, 28 pieces",
             "footnotes": [note(None, "vessels to land\n\nPigeon r., also called Dove r., derives these names from the F. phrase Rivière", True, True)]},
        42: {"pdf_page": 42, "printed_page_number": "8", "body": "Equipage for the voyage:",
             "footnotes": [note(None, "aux Tourtres", True),
                           note("3", "Portage du Perdrix"),
                           note("4", "Otherwise Big Rock or Great Stone portage")]},
    }


@pytest.mark.parametrize("number", [40, 41, 42])
def test_known_note_boundaries(excerpts, number):
    check_notes(excerpts[number], number)


def test_rejects_note_split_into_line_objects(excerpts):
    page = excerpts[40]
    page["footnotes"].insert(1, note(None, "Another line of the same note"))
    with pytest.raises(AssertionError, match="distinct notes"):
        check_notes(page, 40)


def test_rejects_continued_note_in_body(excerpts):
    page = excerpts[41]
    page["body"] += "\nvessels to land"
    with pytest.raises(AssertionError, match="leaked"):
        check_notes(page, 41)


def test_rejects_empty_final_note(excerpts):
    page = excerpts[42]
    page["footnotes"][2]["text"] = ""
    with pytest.raises(AssertionError, match="Empty note"):
        check_notes(page, 42)


@pytest.mark.parametrize("number", [40, 41, 42])
def test_generated_note_boundaries(number):
    directory = os.environ.get("NEWLIGHT_QA_OUTPUT")
    if not directory:
        pytest.skip("Set NEWLIGHT_QA_OUTPUT to check generated notes on PDF40–42")
    path = Path(directory) / f"page_{number:04d}.json"
    assert path.is_file(), f"Missing OCR page: {path}"
    check_notes(json.loads(path.read_text(encoding="utf-8")), number)
