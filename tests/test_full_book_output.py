"""Optional generated-book QA, distinct from offline parser regression tests.

NEWLIGHT_QA_OUTPUT selects saved OCR to validate. Require all PDF pages 35–480
only when NEWLIGHT_QA_REQUIRE_COMPLETE=1; partial recovery remains testable.
Neither check establishes transcription accuracy against the scanned images.
"""
import json
import os
from pathlib import Path

import pytest

from ocr_parse import Page, validate_page


@pytest.fixture
def generated_output():
    directory = os.environ.get('NEWLIGHT_QA_OUTPUT')
    if not directory:
        pytest.skip('Set NEWLIGHT_QA_OUTPUT to audit generated OCR')
    root = Path(directory)
    assert root.is_dir(), f'Missing OCR output directory: {root}'
    return root


def test_generated_saved_book_pages_are_valid_and_sidecars_match(generated_output):
    # Inspect a snapshot of complete page JSONs, excluding footnote sidecars.
    pages = [(n, generated_output/f'page_{n:04d}.json') for n in range(35, 481)]
    pages = [(n, path) for n, path in pages if path.is_file()]
    assert pages, f'No book pages found in {generated_output}'
    for number, path in pages:
        saved = json.loads(path.read_text(encoding='utf-8'))
        page, _ = validate_page(json.dumps({key: saved[key] for key in Page.model_fields}), number)
        body = path.with_suffix('.body.txt')
        notes = path.with_suffix('.footnotes.json')
        assert body.is_file(), f'Missing OCR body: {body}'
        assert notes.is_file(), f'Missing OCR footnotes: {notes}'
        assert body.read_text(encoding='utf-8') == page.body, f'Body sidecar differs: {body}'
        assert json.loads(notes.read_text(encoding='utf-8')) == saved['footnotes'], (
            f'Footnote sidecar differs: {notes}')


def test_generated_full_book_is_complete():
    if os.environ.get('NEWLIGHT_QA_REQUIRE_COMPLETE') != '1':
        pytest.skip('Set NEWLIGHT_QA_REQUIRE_COMPLETE=1 after recovery to require all 446 pages')
    directory = os.environ.get('NEWLIGHT_QA_OUTPUT')
    assert directory, 'NEWLIGHT_QA_OUTPUT is required for the full-book completeness check'
    root = Path(directory)
    missing = [n for n in range(35, 481) if not (root/f'page_{n:04d}.json').is_file()]
    assert not missing, f'{len(missing)}/446 PDF pages unfinished: {missing}'
