"""Offline tests for crop routing, assembly and input/evidence integrity."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import ocr_parse
import ocr_regions as regions


def dump(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


def response(lines, marker=None, reason='stop'):
    content = json.dumps({'lines': lines, 'marker': marker, 'uncertainties': []})
    return SimpleNamespace(message=SimpleNamespace(content=content), done_reason=reason,
                           model_dump=lambda **_: {'message': {'content': content}, 'done_reason': reason})


@pytest.fixture
def crop_run(tmp_path, monkeypatch):
    root = tmp_path / 'renderer'
    root.mkdir()
    dump(root / 'manifest.json', {'renderer_version': 2, 'status': 'complete_unverified',
                                 'source_sha256': 'source-pdf-hash'})
    roles = ['running_header', 'page_number', 'body', 'table', 'footnote']
    layouts = []
    for n in [40, 41]:
        full = root / f'page_{n:04d}.png'
        full.write_bytes(b'full image, never sent to model')
        items = []
        for i, role in enumerate(roles, 1):
            crop = root / f'page_{n:04d}.region_{i:02d}.{role}.png'
            crop.write_bytes(f'{n}:{role}'.encode())
            items.append({'id': i, 'role': role, 'crop': crop.name,
                          'crop_sha256': regions.sha256(crop), 'bbox_px': [1,2,20,30],
                          'marker_candidate': '9' if role == 'footnote' else None})
        layout = {'pdf_page': n, 'full_image': full.name, 'image_sha256': regions.sha256(full),
                  'regions': items, 'warnings': []}
        dump(root / f'page_{n:04d}.layout.json', layout)
        layouts.append(layout)
    client = Mock()
    client.show.return_value.model_dump.return_value = {'capabilities': ['vision']}
    def replies():
        return [response(['A HEADER.']), response(['6']), response(['François in the North-', 'west.']),
                response(['    Merchandise     5 bales', '    Total          28 pieces']),
                response(['A continued note.', '', 'Another paragraph.'])]
    client.chat.side_effect = replies() + replies()
    constructor = Mock(return_value=client)
    monkeypatch.setattr(regions, 'Client', constructor)
    output = tmp_path / 'parsed'
    args = ['--start','40','--end','41','--renderer-run',str(root),'--output',str(output),
            '--images',str(tmp_path / 'nonexistent-pdf-pages')]
    return SimpleNamespace(root=root, output=output, args=args, client=client,
                           constructor=constructor, layouts=layouts, replies=replies)


def test_routes_single_crops_and_preserves_assembly_and_raw_evidence(crop_run):
    r = crop_run
    ocr_parse.main(r.args)
    assert r.client.chat.call_count == 10
    for call, n in zip(r.client.chat.call_args_list, [40]*5+[41]*5):
        msg = call.kwargs['messages'][0]
        assert len(msg['images']) == 1
        assert msg['images'][0].startswith(f'{n}:'.encode())
        assert b'full image' not in msg['images'][0]
    page = json.loads((r.output / 'page_0040.json').read_text())
    assert page['body'] == ('François in the North-\nwest.\n\n'
                            '    Merchandise     5 bales\n    Total          28 pieces')
    assert page['footnotes'][0]['marker'] is None  # Never copy OCR candidate 9.
    assert page['footnotes'][0]['text'] == 'A continued note.\n\nAnother paragraph.'
    assert page['running_headers'] == ['A HEADER.']
    assert page['printed_page_number'] == '6'
    assert page['footnotes'][0]['continues_on_next_page'] is True
    next_page = json.loads((r.output / 'page_0041.json').read_text())
    assert next_page['footnotes'][0]['continues_from_previous_page'] is True
    assert next_page['footnotes'][0]['continues_on_next_page'] is None
    assert page['verification_status'] == 'unverified_ocr'
    run = next((r.output / 'runs').iterdir())
    assert len(list(run.glob('*.response.json'))) == 10
    assert (run / 'page_0040.layout.json').is_file()
    assert len(page['regions']) == 5
    assert page['regions'][0]['crop_sha256'] == r.layouts[0]['regions'][0]['crop_sha256']


def test_rejects_modified_crop_before_connecting(crop_run):
    r = crop_run
    (r.root / r.layouts[0]['regions'][0]['crop']).write_bytes(b'modified')
    with pytest.raises(ValueError, match='Crop hash mismatch'):
        ocr_parse.main(r.args)
    r.constructor.assert_not_called()


def test_rejects_outside_run_paths(crop_run):
    outside = crop_run.root.parent / 'outside.png'
    outside.write_bytes(b'data')
    with pytest.raises(ValueError, match='out-of-run'):
        regions.source_file(crop_run.root, '../outside.png')


def test_resume_preserves_existing_files_and_avoids_model_calls(crop_run):
    r = crop_run
    ocr_parse.main(r.args)
    before = (r.output / 'page_0040.json').read_bytes()
    r.client.chat.reset_mock()
    ocr_parse.main(r.args)
    r.client.chat.assert_not_called()
    assert (r.output / 'page_0040.json').read_bytes() == before


def test_failure_does_not_save_partial_page_and_later_pages_continue(crop_run):
    r = crop_run
    r.client.chat.side_effect = [response(['truncated'], reason='length')] + r.replies()
    with pytest.raises(SystemExit, match=r'1 page\(s\) failed'):
        ocr_parse.main(r.args)
    assert not (r.output / 'page_0040.json').exists()
    assert (r.output / 'page_0041.json').exists()
    run = next((r.output / 'runs').iterdir())
    assert (run / 'page_0040.region_01.response.json').exists()
    assert (run / 'page_0040.error.json').exists()


def test_different_mode_is_not_silently_skipped(crop_run):
    r = crop_run
    r.output.mkdir()
    dump(r.output / 'page_0040.json', {'parser_version': 1})
    r.client.chat.side_effect = r.replies()
    with pytest.raises(SystemExit, match=r'1 page\(s\) failed'):
        ocr_parse.main(r.args)
    assert json.loads((r.output / 'page_0040.json').read_text()) == {'parser_version': 1}


@pytest.mark.parametrize('role,counts', [('body', [4,4,2]), ('table', [1]*10)])
def test_crop_chunking_uses_geometry_and_table_rows(tmp_path, role, counts):
    import pymupdf
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0,0,100,100), False)
    pix.clear_with(255)
    pix.set_dpi(72,72)
    pix.save(tmp_path / 'full.png')
    output = tmp_path / 'run'
    output.mkdir()
    layout = {'full_image': 'full.png', 'width_px': 100, 'height_px': 100, 'dpi': 72}
    region = {'id': 1, 'role': role, 'crop': 'full.png', 'bbox_px': [20,5,90,88],
              'lines': [{'bbox': [40,5+i*8,80,9+i*8]} for i in range(10)]}
    chunks = regions.region_chunks(tmp_path, 40, layout, region, output, 4)
    assert [count for _, _, count in chunks] == counts
    assert all(path.is_file() for path, _, _ in chunks)
    assert chunks[0][1] == 0
    assert all(pymupdf.Pixmap(str(path)).width == 78 for path, _, _ in chunks)


@pytest.mark.parametrize('text,declared,expected', [
    ('³ Portage du Perdrix', None, (' Portage du Perdrix', '3')),
    ('[¹] On the regular route', '1', (' On the regular route', '1')),
    ('vessels to land', None, ('vessels to land', None)),
])
def test_marker_separation_uses_only_visible_transcribed_text(text, declared, expected):
    corrected, marker, _ = regions.separate_visible_marker(text, declared)
    assert (corrected, marker) == expected


def test_conflicting_marker_requires_review():
    with pytest.raises(ValueError, match='Conflicting'):
        regions.separate_visible_marker('³ Portage du Perdrix', '4')


@pytest.mark.parametrize('raw,expected', [('^1','1'), ('²','2'), ('[3]','3'), (None,None)])
def test_marker_notation_is_normalized_without_ocr_candidate(raw, expected):
    assert regions.canonical_marker(raw) == expected


def test_missing_measured_lines_fail_instead_of_saving_incomplete_text(crop_run, monkeypatch):
    r = crop_run
    crop = r.root / r.layouts[0]['regions'][0]['crop']
    monkeypatch.setattr(regions, 'region_chunks', Mock(return_value=[(crop,0,2)]))
    r.client.chat.side_effect = [response(['Only one line'])]*2
    with pytest.raises(SystemExit, match=r'2 page\(s\) failed'):
        ocr_parse.main(r.args)
    assert not (r.output / 'page_0040.json').exists()
    run = next((r.output / 'runs').iterdir())
    error = json.loads((run / 'page_0040.error.json').read_text())
    assert 'returned 1 of 2 measured lines' in error['error']


def test_interrupt_preserves_completed_page_checkpoint(crop_run):
    r = crop_run
    r.client.chat.side_effect = r.replies() + [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        ocr_parse.main(r.args)
    saved = json.loads((r.output / 'page_0040.json').read_text())
    assert saved['continuation_proposals_pending'] is True
    assert not (r.output / 'page_0041.json').exists()
    run = next((r.output / 'runs').iterdir())
    assert json.loads((run / 'manifest.json').read_text())['status'] == 'interrupted'


def test_marker_inspection_reads_enlarged_glyph_without_copying_candidate(tmp_path):
    import pymupdf
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0,0,100,100), False)
    pix.clear_with(255)
    pix.set_dpi(72,72)
    pix.save(tmp_path / 'full.png')
    run = tmp_path / 'run'
    run.mkdir()
    layout = {'full_image':'full.png','width_px':100,'height_px':100,'dpi':72}
    region = {'id':1,'marker_candidate':'9','lines':[{'words':[{'bbox':[10,10,20,30]}]}]}
    client = Mock()
    payload = {'marker':'4','uncertainties':[]}
    client.chat.return_value = SimpleNamespace(done_reason='stop',
        message=SimpleNamespace(content=json.dumps(payload)), model_dump=lambda **_: payload)
    marker, evidence = regions.inspect_marker(client, 'test-model', {}, tmp_path, 42, layout, region, run)
    assert marker == '4'  # Independent read, not candidate 9.
    crop = pymupdf.Pixmap(evidence['crop'])
    assert (crop.width,crop.height) == (72,112)
    assert '9' not in client.chat.call_args.kwargs['messages'][0]['content']
    assert (run / evidence['response']).is_file()


def test_one_page_retry_uses_neighbor_without_rewriting_it(crop_run):
    r = crop_run
    ocr_parse.main(r.args)
    before = (r.output / 'page_0040.json').read_bytes()
    r.client.chat.side_effect = r.replies()
    args = r.args.copy()
    args[args.index('--start')+1] = '41'
    ocr_parse.main(args + ['--overwrite'])
    saved = json.loads((r.output / 'page_0041.json').read_text())
    assert saved['footnotes'][0]['continues_from_previous_page'] is True
    assert (r.output / 'page_0040.json').read_bytes() == before
    manifest = json.loads((r.output / 'runs' / saved['run_id'] / 'manifest.json').read_text())
    assert manifest['continuation_context'][0]['sha256'] == hashlib.sha256(before).hexdigest()


def test_coverage_detects_observed_within_line_omission():
    draft = '[or Calling river]. The snow was then melted in many'
    lines = [{'words':[{'text':w,'confidence':95} for w in draft.split()]}]
    result = regions.coverage_diagnostics('[or Calling river].', lines)
    assert result['suspect']
    assert 'snow' in result['missing_draft_anchors']
    complete = regions.coverage_diagnostics(draft, lines)
    assert not complete['suspect']


def test_coverage_normalization_is_diagnostic_only():
    text = '  Rivière Qu’Appelle.  '
    lines = [{'words':[{'text':'Riviere','confidence':95}, {'text':'Qu’Appelle.','confidence':95}]}]
    assert not regions.coverage_diagnostics(text,lines)['suspect']
    assert text == '  Rivière Qu’Appelle.  '


def test_visual_retry_never_copies_draft_words(tmp_path):
    import pymupdf
    pix = pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,100),False)
    pix.clear_with(255)
    pix.set_dpi(72,72)
    pix.save(tmp_path/'full.png')
    run = tmp_path/'run'
    run.mkdir()
    layout = {'full_image':'full.png','width_px':100,'height_px':100,'dpi':72}
    selected = [{'words':[{'text':f'WRONG{i}','bbox':[5+i*15,10,15+i*15,25]}
                          for i in range(5)]}]
    def fragment(text):
        payload = {'text':text,'uncertainties':[]}
        return SimpleNamespace(done_reason='stop',message=SimpleNamespace(content=json.dumps(payload)),
                               model_dump=lambda **_:payload)
    client = Mock()
    client.chat.side_effect = [fragment('Read from image'),fragment('only.')]
    text, evidence, _ = regions.retry_word_fragments(client,'model',{},tmp_path,38,layout,
                                                    selected,run,'test')
    assert text == 'Read from image only.'
    assert len(evidence) == 2
    assert all('WRONG' not in c.kwargs['messages'][0]['content'] for c in client.chat.call_args_list)
    assert all((run/e['response']).is_file() for e in evidence)


def test_unresolved_coverage_disagreement_fails_and_preserves_responses(crop_run,monkeypatch):
    r = crop_run
    r.layouts[0]['regions'][0]['lines'] = [{'words':[
        {'text':w,'confidence':95} for w in 'A HEADER has additional visible words'.split()]}]
    dump(r.root/'page_0040.layout.json',r.layouts[0])
    monkeypatch.setattr(regions,'retry_word_fragments',Mock(return_value=('Still incomplete',[],[])))
    r.client.chat.side_effect = [response(['A HEADER.'])] + r.replies()
    with pytest.raises(SystemExit,match=r'1 page\(s\) failed'):
        ocr_parse.main(r.args)
    assert not (r.output/'page_0040.json').exists()
    assert (r.output/'page_0041.json').exists()
    run = next((r.output/'runs').iterdir())
    assert (run/'page_0040.region_01.response.json').is_file()
    assert (run/'page_0040.region_01.coverage.json').is_file()


def test_punctuation_ocr_error_is_not_a_missing_word_alarm():
    lines = [{'words':[{'text':'{Mouse','confidence':95}, {'text':'river],','confidence':95}]}]
    assert not regions.coverage_diagnostics('[Mouse river],',lines)['suspect']


def test_added_opening_bracket_triggers_visual_reread():
    lines = [{'words':[{'text':'migratorius]','confidence':95}]}]
    assert regions.coverage_diagnostics('[migratorius]',lines)['suspect']


def test_low_confidence_ditto_hallucination_does_not_force_inserted_word():
    words = [{'text':w,'confidence':95} for w in 'Received in the course of the season 618'.split()]
    words.append({'text':'aad','confidence':22})  # Actual OCR misread of ditto marks.
    assert not regions.coverage_diagnostics('Received in the course of the season, . 618',
                                            [{'words':words}])['suspect']


def test_merged_ocr_word_go_to_is_not_false_failure():
    source = [{'words':[{'text':w,'confidence':95} for w in 'goto bed Several Indians assured me of the same circum'.split()]}]
    assert not regions.coverage_diagnostics('go to bed. Several Indians assured me of the same circum-',source)['suspect']


def test_missing_trailing_the_is_still_detected():
    source = [{'words':[{'text':w,'confidence':95} for w in 'W of Portage la Prairie The'.split()]}]
    assert regions.coverage_diagnostics('W. of Portage la Prairie.',source)['suspect']


def test_retry_splits_incomplete_two_word_span_to_individual_visual_reads(tmp_path):
    import pymupdf
    pix = pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,100),False)
    pix.clear_with(255)
    pix.set_dpi(72,72)
    pix.save(tmp_path/'full.png')
    run = tmp_path/'run'
    run.mkdir()
    layout = {'full_image':'full.png','width_px':100,'height_px':100,'dpi':72}
    selected = [{'words':[{'text':'Prairie.','confidence':95,'bbox':[5,10,45,25]},
                           {'text':'The','confidence':95,'bbox':[55,10,80,25]}]}]
    def fragment(text):
        data = {'text':text,'uncertainties':[]}
        return SimpleNamespace(done_reason='stop',message=SimpleNamespace(content=json.dumps(data)),
                               model_dump=lambda **_:data)
    client = Mock()
    client.chat.side_effect = [fragment('Prairie.'),fragment('Prairie.'),fragment('The')]
    text,evidence,_ = regions.retry_word_fragments(client,'model',{},tmp_path,35,layout,selected,run,'retry')
    assert text == 'Prairie. The'
    assert client.chat.call_count == 3
    assert len(evidence) == 3
    assert any('.left.' in e['response'] for e in evidence)
    assert any('.right.' in e['response'] for e in evidence)


def test_unresolved_single_word_is_not_filled_from_draft(tmp_path):
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,100),False)
    pix.clear_with(255);pix.set_dpi(72,72);pix.save(tmp_path/'full.png')
    run=tmp_path/'run';run.mkdir()
    payload={'text':'Other','uncertainties':[]}
    client=Mock();client.chat.return_value=SimpleNamespace(done_reason='stop',
        message=SimpleNamespace(content=json.dumps(payload)),model_dump=lambda **_:payload)
    selected=[{'words':[{'text':'The','confidence':95,'bbox':[10,10,50,30]}]}]
    layout={'full_image':'full.png','width_px':100,'height_px':100,'dpi':72}
    with pytest.raises(ValueError,match='single-word'):
        regions.retry_word_fragments(client,'model',{},tmp_path,35,layout,selected,run,'word')
    assert len(list(run.glob('*.response.json')))==1
