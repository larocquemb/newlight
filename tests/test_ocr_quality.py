"""Offline tests for the current parser; no model calls or generated data needed."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ocr_parse as ocr


def page_data(number=35):
    return {"pdf_page": number, "body": "  François avoit connois-\nsance.¹\n\nFin.  ",
            "footnotes": [{"marker": None, "text": "  Note continuée.\nFin.  ",
                           "continues_from_previous_page": True,
                           "continues_on_next_page": None}],
            "running_headers": ["RIVIÈRE."], "printed_page_number": "1", "uncertainties": []}


def response(data, reason="stop"):
    if isinstance(data, dict):
        data = data.copy()
        data["body_lines"] = data.pop("body").split("\n")
    content = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return SimpleNamespace(message=SimpleNamespace(content=content), done_reason=reason,
                           model_dump=lambda **_: {"message": {"content": content}, "done_reason": reason})


@pytest.fixture
def run(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for n in range(35, 41):
        (images / f"page_{n:04d}.png").write_bytes(f"synthetic image {n}".encode())
    client = Mock()
    client.show.return_value.model_dump.return_value = {"capabilities": ["vision"]}
    client.chat.side_effect = [response(page_data(n)) for n in range(35, 41)]
    constructor = Mock(return_value=client)
    monkeypatch.setattr(ocr, "Client", constructor)
    output = tmp_path / "output"
    args = ["--start", "35", "--end", "40", "--images", str(images),
            "--output", str(output), "--host", "http://test.invalid:11434"]
    return SimpleNamespace(images=images, output=output, args=args, client=client, constructor=constructor)


def test_preserves_exact_text_and_separates_notes(tmp_path):
    original = page_data()
    page, warnings = ocr.validate_page(json.dumps(original), 35)
    assert warnings == []
    ocr.save_page(tmp_path, page, {"review_required": True})
    assert (tmp_path / "page_0035.body.txt").read_bytes() == original["body"].encode()
    assert json.loads((tmp_path / "page_0035.footnotes.json").read_text()) == original["footnotes"]
    stored = json.loads((tmp_path / "page_0035.json").read_text())
    assert stored["body"] == original["body"]
    assert stored["running_headers"] == original["running_headers"]
    assert stored["printed_page_number"] == original["printed_page_number"]


@pytest.mark.parametrize("change", [{"pdf_page": 36}, {"pdf_page": "35"},
    {"body": ["legacy passage"]}, {"body": " \n "}, {"extra": "unexpected"}, {"footnotes": None}])
def test_rejects_wrong_page_or_invalid_schema(change):
    with pytest.raises(ValueError):
        ocr.validate_page(json.dumps(page_data() | change), 35)


@pytest.mark.parametrize("change", [{"text": " \n"}, {"marker": 1}, {"marker": ""},
                                     {"marker": "  "}, {"continues_on_next_page": "true"}])
def test_rejects_invalid_notes(change):
    data = page_data()
    data["footnotes"][0].update(change)
    with pytest.raises(ValueError):
        ocr.validate_page(json.dumps(data), 35)


def test_duplicate_warning_does_not_modify_text():
    data = page_data()
    note = "A continued editorial note with more than sixty characters for duplicate detection."
    data["footnotes"][0]["text"] = note
    data["body"] = "BODY\n" + note.upper()
    data["uncertainties"] = ["Font boundary ambiguous"]
    page, warnings = ocr.validate_page(json.dumps(data), 35)
    assert warnings[0] == "Font boundary ambiguous"
    assert any("duplication" in w for w in warnings)
    assert page.body == data["body"]
    assert page.footnotes[0].text == note


@pytest.mark.parametrize("line", ["RIVIÈRE.", "  rivière.  ", "\tRIVIÈRE.\t"])
@pytest.mark.parametrize("location", ["start", "end"])
def test_rejects_running_header_duplicated_as_body_line(line, location):
    data = page_data()
    data["body"] = (line + "\n\n" + data["body"] if location == "start"
                    else data["body"] + "\n\n" + line)
    with pytest.raises(ValueError, match="Running header duplicated"):
        ocr.validate_page(json.dumps(data), 35)


def test_header_words_in_narrative_are_not_removed_or_rejected():
    data = page_data() | {"body": "The heading RIVIÈRE. describes the next section."}
    page, warnings = ocr.validate_page(json.dumps(data), 35)
    assert page.body == data["body"]
    assert warnings == []


def test_chapter_title_is_body_when_it_is_not_a_running_header():
    data = page_data() | {"body": "CHAPTER I.\n\nAUTUMN, 1799.", "running_headers": []}
    page, _ = ocr.validate_page(json.dumps(data), 35)
    assert page.body == data["body"]


@pytest.mark.parametrize("printed", [None, "5", "iv", "XII", "105"])
def test_printed_number_is_preserved_separately_from_pdf_index(tmp_path, printed):
    data = page_data(39) | {"printed_page_number": printed}
    page, _ = ocr.validate_page(json.dumps(data), 39)
    ocr.save_page(tmp_path, page, {})
    saved = json.loads((tmp_path / "page_0039.json").read_text())
    assert saved["pdf_page"] == 39
    assert saved["printed_page_number"] == printed
    assert saved["body"] == data["body"]
    assert (tmp_path / "page_0039.body.txt").read_text() == data["body"]


@pytest.mark.parametrize("change", [
    {"running_headers": "RIVIÈRE."}, {"running_headers": [5]},
    {"running_headers": None}, {"printed_page_number": 5},
    {"printed_page_number": True}, {"printed_page_number": ["5"]},
])
def test_header_and_page_number_require_strict_schema(change):
    with pytest.raises(ValueError):
        ocr.validate_page(json.dumps(page_data() | change), 35)


def test_short_narrative_without_notes_is_valid():
    data = page_data(39) | {"body": "A short final paragraph.\nNorth-\nwest.", "footnotes": []}
    page, warnings = ocr.validate_page(json.dumps(data), 39)
    assert warnings == []
    assert page.body == data["body"]


def test_dangling_hyphen_flags_review_without_inventing_continuation():
    data = page_data(39) | {"body": "This campaign was in the North-\n", "footnotes": []}
    page, warnings = ocr.validate_page(json.dumps(data), 39)
    assert any("omitted final line" in warning for warning in warnings)
    assert page.body == data["body"]


def test_header_separation_preserves_short_final_word():
    body = "strongly opposed by a neighbor.\nThis campaign was in the North-\nwest."
    data = page_data(39) | {"body": "RIVIÈRE.\n" + body}
    content, _ = ocr.separate_running_headers(json.dumps(data))
    page, warnings = ocr.validate_page(content, 39)
    assert page.body == body
    assert warnings == []


@pytest.mark.parametrize("margin", ["4\n\nRIVIÈRE.\n", "RIVIÈRE.\n\n4\n"])
@pytest.mark.parametrize("edge", ["start", "end"])
def test_separates_number_and_header_in_either_order(margin, edge):
    narrative = "Actual narrative.\nNorth-\nwest."
    data = page_data(38) | {"printed_page_number": "4"}
    data["body"] = margin + narrative if edge == "start" else narrative + "\n" + margin
    content, changes = ocr.separate_running_headers(json.dumps(data))
    page, _ = ocr.validate_page(content, 38)
    assert page.body == narrative + ("\n" if edge == "end" else "")
    assert changes[0]["removed_text"] == margin
    assert page.printed_page_number == "4"


def test_does_not_remove_standalone_narrative_number_without_header():
    data = page_data(38) | {"printed_page_number": "4", "body": "4\nAn inventory row."}
    content, changes = ocr.separate_running_headers(json.dumps(data))
    assert changes == []
    assert json.loads(content)["body"] == data["body"]


@pytest.mark.parametrize("edge", ["start", "end"])
def test_separates_declared_margin_header_without_rewriting_narrative(edge):
    original = page_data()
    data = original.copy()
    data["body"] = ("RIVIÈRE.\n" + original["body"] if edge == "start"
                    else original["body"] + "\nRIVIÈRE.")
    content, changes = ocr.separate_running_headers(json.dumps(data))
    page, _ = ocr.validate_page(content, 35)
    # The newline before an end header remains part of the preserved body.
    assert page.body == original["body"] + ("\n" if edge == "end" else "")
    assert changes == [{"edge": edge, "removed_text": "RIVIÈRE.\n" if edge == "start" else "RIVIÈRE."}]
    assert page.running_headers == original["running_headers"]


def test_separation_does_not_guess_chapter_titles_or_remove_interior_lines():
    data = page_data() | {"body": "CHAPTER I.\nRIVIÈRE.\nNarrative follows."}
    content, changes = ocr.separate_running_headers(json.dumps(data))
    assert changes == []
    with pytest.raises(ValueError, match="Running header duplicated"):
        ocr.validate_page(content, 35)


def test_repaired_header_keeps_raw_response_and_audit(run):
    data = page_data()
    data["body"] = "RIVIÈRE.\n" + data["body"]
    run.client.chat.side_effect = [response(data)] + [response(page_data(n)) for n in range(36, 41)]
    ocr.main(run.args)
    evidence = next((run.output / "runs").iterdir())
    raw = json.loads((evidence / "page_0035.response.json").read_text())
    assert "\n".join(json.loads(raw["message"]["content"])["body_lines"]) == data["body"]
    saved = json.loads((run.output / "page_0035.json").read_text())
    assert saved["body"] == page_data()["body"]
    assert saved["header_separations"] == [{"edge": "start", "removed_text": "RIVIÈRE.\n"}]


def test_line_array_keeps_hyphenation_and_paragraph_gaps_exactly():
    data = page_data(39)
    del data["body"]
    data["body_lines"] = ["A first line.", "", "This campaign was in the North-", "west."]
    page = ocr.Page.model_validate_json(ocr.transcription_to_page(json.dumps(data)))
    assert page.body == "A first line.\n\nThis campaign was in the North-\nwest."


def test_indented_table_spacing_survives_decode_and_save(tmp_path):
    data = page_data(41)
    del data["body"]
    data["body_lines"] = ["Narrative introducing the cargo:", "",
                          "    Merchandise, 90 pounds each,    5 bales",
                          "    High wine, 9 gallons each,     10 kegs",
                          "        Total,                    28 pieces"]
    data["footnotes"][0]["text"] = "vessels to land, and separated by Hat point"
    content, changes = ocr.separate_running_headers(ocr.transcription_to_page(json.dumps(data)))
    page, _ = ocr.validate_page(content, 41)
    ocr.save_page(tmp_path, page, {})
    assert changes == []
    assert (tmp_path / "page_0041.body.txt").read_text() == "\n".join(data["body_lines"])
    notes = json.loads((tmp_path / "page_0041.footnotes.json").read_text())
    assert notes[0]["text"] == data["footnotes"][0]["text"]


def test_layout_measurement_ignores_dotted_leaders_for_letter_height(tmp_path, monkeypatch):
    tsv = ("level\tblock_num\tpar_num\tline_num\tleft\ttop\theight\ttext\n"
           "5\t1\t1\t1\t200\t600\t21\tMerchandise\n"
           "5\t1\t1\t1\t400\t610\t2\t.\n"
           "5\t1\t1\t1\t900\t600\t21\tbales\n"
           "5\t2\t1\t1\t120\t1100\t21\tvessels\n")
    process = Mock(return_value=SimpleNamespace(stdout=tsv, stderr="diagnostic"))
    monkeypatch.setattr(ocr.subprocess, "run", process)
    measured = ocr.measure_layout(tmp_path / "image.png", tmp_path, "page_0041")
    assert measured[0] == {"x": 200, "y": 600, "height": 21, "draft": "Merchandise . bales"}
    assert measured[1]["x"] == 120
    assert (tmp_path / "page_0041.layout.tsv").read_text() == tsv
    assert (tmp_path / "page_0041.layout.stderr.txt").read_text() == "diagnostic"


def test_layout_hints_are_optional_and_passed_to_model(run, monkeypatch):
    measured = [{"x": 120, "y": 1100, "height": 21, "draft": "vessels to land"}]
    measure = Mock(return_value=measured)
    monkeypatch.setattr(ocr, "measure_layout", measure)
    ocr.main(run.args + ["--layout-hints"])
    assert measure.call_count == 6
    message = run.client.chat.call_args.kwargs["messages"][0]["content"]
    assert json.dumps(measured, ensure_ascii=False) in message
    assert "untrusted source data" in message
    assert message.endswith("TARGET PDF PAGE: 40.")


def test_context_windows_and_individual_targets(run):
    ocr.main(run.args)
    assert run.client.chat.call_count == 6
    for n, call in zip(range(35, 41), run.client.chat.call_args_list):
        message = call.kwargs["messages"][0]
        window = range(35, 40) if n < 40 else [40]
        assert message["images"] == [f"synthetic image {p}".encode() for p in window]
        assert message["content"].endswith(f"TARGET PDF PAGE: {n}.")
        saved = json.loads((run.output / f"page_{n:04d}.json").read_text())
        assert saved["context_pages"] == list(window)
        assert saved["verification_status"] == "unverified_ocr"
        assert saved["review_required"] is True
    evidence = next((run.output / "runs").iterdir())
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert manifest["status"] == "complete_unverified"
    assert manifest["inputs"]["35"]["sha256"] == ocr.sha256(run.images / "page_0035.png")


@pytest.mark.parametrize("bad", [response(page_data(), "length"), response("{broken JSON")])
def test_failure_retained_and_later_pages_continue(run, bad):
    run.client.chat.side_effect = [bad] + [response(page_data(n)) for n in range(36, 41)]
    with pytest.raises(SystemExit, match=r"1 page\(s\) failed"):
        ocr.main(run.args + ["--num-predict", "16384"])
    assert not (run.output / "page_0035.json").exists()
    assert (run.output / "page_0040.json").exists()
    evidence = next((run.output / "runs").iterdir())
    assert json.loads((evidence / "page_0035.response.json").read_text()) == bad.model_dump()
    assert (evidence / "page_0035.error.json").exists()
    assert json.loads((evidence / "manifest.json").read_text())["status"] == "failed"
    assert run.client.chat.call_args.kwargs["options"]["num_predict"] == 16384


def test_resume_skips_without_rewriting(run):
    ocr.main(run.args)
    before = (run.output / "page_0035.json").read_bytes()
    run.client.chat.reset_mock()
    ocr.main(run.args)
    run.client.chat.assert_not_called()
    assert (run.output / "page_0035.json").read_bytes() == before


def test_changed_input_requires_explicit_reprocessing(run):
    ocr.main(run.args)
    (run.images / "page_0035.png").write_bytes(b"different source")
    run.client.chat.reset_mock()
    with pytest.raises(ValueError, match="incompatible/stale"):
        ocr.main(run.args)
    run.client.chat.assert_not_called()


def test_overwrite_preserves_previous_run_evidence(run):
    ocr.main(run.args)
    old = next((run.output / "runs").iterdir())
    before = (old / "page_0035.json").read_bytes()
    run.client.chat.side_effect = [response(page_data(n) | {"body": "New transcription"}) for n in range(35, 41)]
    ocr.main(run.args + ["--overwrite"])
    assert (old / "page_0035.json").read_bytes() == before
    assert (run.output / "page_0035.body.txt").read_text() == "New transcription"
    assert len(list((run.output / "runs").iterdir())) == 2


def test_missing_images_fail_before_connecting(run):
    (run.images / "page_0035.png").unlink()
    with pytest.raises(SystemExit) as exc:
        ocr.main(run.args)
    assert exc.value.code == 2
    run.constructor.assert_not_called()


def test_model_requires_vision_support(run):
    run.client.show.return_value.model_dump.return_value = {"capabilities": ["completion"]}
    with pytest.raises(SystemExit) as exc:
        ocr.main(run.args)
    assert exc.value.code == 2
    run.client.chat.assert_not_called()


@pytest.mark.parametrize("environment,explicit,expected", [
    (None, None, "http://arsene:11434"),
    ("http://environment:11434", None, "http://environment:11434"),
    ("http://environment:11434", "http://explicit:11434", "http://explicit:11434"),
])
def test_host_default_and_overrides(run, monkeypatch, environment, explicit, expected):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    if environment is not None:
        monkeypatch.setenv("OLLAMA_HOST", environment)
    args = run.args[:-2]  # Remove the fixture's explicit --host.
    if explicit is not None:
        args += ["--host", explicit]
    ocr.main(args)
    assert run.constructor.call_args.kwargs["host"] == expected


@pytest.mark.parametrize("extra,expected", [([], 1.1), (["--repeat-penalty", "1.15"], 1.15)])
def test_repetition_penalty_sent_and_recorded(run, extra, expected):
    ocr.main(run.args + extra)
    for call in run.client.chat.call_args_list:
        assert call.kwargs["options"]["repeat_penalty"] == expected
    evidence = next((run.output / "runs").iterdir())
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert manifest["options"]["repeat_penalty"] == expected


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_repetition_penalty_fails_before_connecting(run, value):
    with pytest.raises(SystemExit) as exc:
        ocr.main(run.args + ["--repeat-penalty", value])
    assert exc.value.code == 2
    run.constructor.assert_not_called()


@pytest.mark.parametrize('text', [
    '(ordinary parentheses)', '[editorial insertion]', '{literal braces}',
    '[an insertion (with parentheses)]', '(text\ncontinued on next line)',
])
def test_matching_bracket_pairs(text):
    assert ocr.bracket_pair_warnings(text, 'Body') == []


@pytest.mark.parametrize('text', ['[text)', '(text]', '{text]', '([crossed)]'])
def test_mixed_or_crossed_bracket_pairs_are_flagged(text):
    assert any('mismatched bracket pair' in w for w in ocr.bracket_pair_warnings(text, 'Body'))


@pytest.mark.parametrize('text', ['[continued', 'continued)', '{continued', 'continued]'])
def test_unmatched_boundary_is_a_warning_not_text_repair(text):
    data = page_data() | {'body': text}
    page, warnings = ocr.validate_page(json.dumps(data), 35)
    assert any('continuation' in w for w in warnings)
    assert page.body == text


def test_note_and_body_brackets_do_not_balance_each_other():
    data = page_data() | {'body': 'The body has an opening ['}
    data['footnotes'][0]['text'] = 'A separate note has a closing ]'
    _, warnings = ocr.validate_page(json.dumps(data), 35)
    assert any('Body: unclosed' in w for w in warnings)
    assert any('Footnote 0: unmatched closing' in w for w in warnings)


@pytest.mark.parametrize('scope', ['body', 'footnote'])
def test_curly_brace_confusion_is_flagged_without_replacement(scope):
    data = page_data()
    if scope == 'body':
        data['body'] = 'An apparent {insertion}.'
    else:
        data['footnotes'][0]['text'] = 'An apparent {insertion}.'
    page, warnings = ocr.validate_page(json.dumps(data), 35)
    assert any('curly brace' in w.casefold() for w in warnings)
    actual = page.body if scope == 'body' else page.footnotes[0].text
    assert '{insertion}' in actual
