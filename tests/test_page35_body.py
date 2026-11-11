"""Page identity and body/note separation regression checks.

Reference excerpts manually read from data/processed/pdf_pages/page_0035.png. These checks
are not a complete transcription-accuracy score. To check generated output:
NEWLIGHT_QA_OUTPUT=data/processed/parsed/single-page-test python -m pytest tests/test_page35_body.py -q
"""
import os
from pathlib import Path

import pytest


def check_page35_body(text):
    # Ignore whitespace only for excerpt matching; do not rewrite the output.
    body = " ".join(text.split())
    assert "AUTUMN, 1799. While building at Rivière Terre" in body, (
        "Page 35 opening is missing: expected AUTUMN, 1799 and Rivière Terre"
    )
    assert "Dauphin [Riding] mountain, my Russia sheeting tent was" in body
    assert "E. of us, they could distinctly perceive this light, which they" in body
    assert "A LITTLE TEMPERANCE TRACT" not in body, "Text from another page leaked into body"
    assert "Henry’s Journal opens thus abruptly" not in body, "Footnote 1 leaked into body"
    assert "Henry's Journal opens thus abruptly" not in body, "Footnote 1 leaked into body"


@pytest.fixture
def reference_excerpts():
    # Only the inspected excerpts needed to identify this page, not invented
    # text for gaps between them or a claim of a full reference transcription.
    return (
        "AUTUMN, 1799. While building at Rivière Terre\n"
        "Dauphin [Riding] mountain, my Russia sheeting tent was\n"
        "E. of us, they could distinctly perceive this light, which they"
    )


def test_page35_reference_excerpts(reference_excerpts):
    check_page35_body(reference_excerpts)


def test_rejects_observed_wrong_page():
    with pytest.raises(AssertionError, match="opening is missing"):
        check_page35_body("A LITTLE TEMPERANCE TRACT.\nwhatever article of dry goods")


def test_rejects_footnote_in_body(reference_excerpts):
    with pytest.raises(AssertionError, match="Footnote 1 leaked"):
        check_page35_body(reference_excerpts + "\nHenry’s Journal opens thus abruptly")


def test_generated_page35_body():
    directory = os.environ.get("NEWLIGHT_QA_OUTPUT")
    if not directory:
        pytest.skip("Set NEWLIGHT_QA_OUTPUT to check generated page 35 text")
    path = Path(directory) / "page_0035.body.txt"
    assert path.is_file(), f"Missing OCR body: {path}"
    check_page35_body(path.read_text(encoding="utf-8"))
