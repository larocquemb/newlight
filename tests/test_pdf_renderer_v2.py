"""Geometry regressions from actual scanned pages; no OCR/model process required."""
import copy
from pathlib import Path

import pytest

from pdf_renderer_v2 import classify_lines, group_regions, parse_tsv

FIXTURES = Path(__file__).parent / 'fixtures/renderer_v2'


def layout(number, scale=1):
    lines = parse_tsv((FIXTURES / f'page_{number:04d}.tsv').read_text())
    if scale != 1:
        lines = copy.deepcopy(lines)
        for line in lines:
            line['bbox'] = [x*scale for x in line['bbox']]
            line['letter_height'] *= scale
            for word in line['words']:
                word['bbox'] = [x*scale for x in word['bbox']]
    lines, metrics, warnings = classify_lines(lines, 1403*scale, 2325*scale)
    return group_regions(lines)


@pytest.mark.parametrize('scale', [1, 1.5, 2])
def test_same_size_table_and_continued_note_are_separate(scale):
    regions = layout(41, scale)
    assert [r['role'] for r in regions] == [
        'running_header', 'page_number', 'body', 'table', 'footnote']
    table, note = regions[-2:]
    assert 'Merchandise' in table['draft_text'] and '28 pieces' in table['draft_text']
    assert note['draft_text'].startswith('vessels to land')
    assert 'Pigeon r., also called Dove r.' in note['draft_text']
    assert note['marker_candidate'] is None
    assert table['bbox_px'][0] > note['bbox_px'][0]


def test_chapter_titles_are_body_and_notes_are_not_split_per_line():
    regions = layout(40)
    assert [r['role'] for r in regions] == ['body', 'footnote', 'footnote', 'page_number']
    assert 'CHAPTER II.' in regions[0]['draft_text']
    assert [r['marker_candidate'] for r in regions[1:3]] == ['1', '2']
    assert len(regions[1]['lines']) > 10
    assert regions[2]['draft_text'].endswith('too shallow for')
    assert regions[-1]['draft_text'] == '6'


def test_three_notes_include_their_short_final_lines():
    regions = layout(42)
    notes = [r for r in regions if r['role'] == 'footnote']
    assert [r['marker_candidate'] for r in notes] == [None, '3', '4']
    assert notes[0]['draft_text'].startswith('aux Tourtres')
    assert notes[-1]['draft_text'].endswith('right, 485 yards, N. 4o° W.')
    assert len(notes[-1]['lines']) == 2


def test_financial_table_does_not_swallow_following_narrative():
    regions = layout(38)
    assert [r['role'] for r in regions] == ['running_header', 'page_number', 'body', 'table', 'body']
    assert len(regions[-2]['lines']) == 5
    assert regions[-1]['draft_text'].startswith('Notwithstanding')
    assert '1864' in regions[-2]['draft_text']


def test_short_page_final_word_is_not_misclassified_as_note():
    regions = layout(39)
    assert [r['role'] for r in regions] == ['running_header', 'page_number', 'body']
    assert regions[-1]['draft_text'].endswith('North-\nwest.')


def test_no_text_is_explicitly_flagged():
    lines, metrics, warnings = classify_lines([], 1403, 2325)
    assert lines == [] and metrics == {}
    assert warnings == ['No narrative lines detected']


def test_numeric_table_values_inside_page_are_not_page_numbers():
    regions = layout(38)
    numbers = [r['draft_text'] for r in regions if r['role'] == 'page_number']
    assert numbers == ['4']
    assert all('1500' not in r['draft_text'] for r in regions if r['role'] == 'page_number')
