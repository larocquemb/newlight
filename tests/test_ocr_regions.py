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


def test_missing_only_preserves_earlier_version_source_matched_output(crop_run):
    r=crop_run;ocr_parse.main(r.args)
    path=r.output/'page_0040.json';saved=json.loads(path.read_text())
    saved['region_pipeline_version']=2;dump(path,saved)
    before=path.read_bytes();r.client.chat.reset_mock()
    ocr_parse.main(r.args+['--missing-only'])
    r.client.chat.assert_not_called()
    assert path.read_bytes()==before


def test_missing_only_still_rejects_source_mismatch(crop_run):
    r=crop_run;ocr_parse.main(r.args)
    path=r.output/'page_0040.json';saved=json.loads(path.read_text())
    saved['source_sha256']='other image';dump(path,saved);r.client.chat.reset_mock()
    with pytest.raises(SystemExit,match='1 page'):
        ocr_parse.main(r.args+['--missing-only'])
    r.client.chat.assert_not_called()


def test_recovery_index_tracks_success_and_keeps_failed_page_missing(crop_run):
    r=crop_run;r.output.mkdir()
    dump(r.output/'recovery_manifest.json',{'scope':[40,41],'pages':[]})
    r.client.chat.side_effect=[response(['truncated'],reason='length')]+r.replies()
    with pytest.raises(SystemExit,match='1 page'):
        ocr_parse.main(r.args+['--missing-only'])
    index=json.loads((r.output/'recovery_manifest.json').read_text())
    assert index['present_pages']==1 and index['missing_pages']==[40]
    assert index['status']=='incomplete_unverified' and index['recovery_run_status']=='failed'
    assert index['recovered_pages']==[41]
    for file in index['pages'][0]['files']:
        assert regions.sha256(Path(file['destination_path']))==file['sha256']
        assert regions.sha256(Path(file['source_path']))==file['sha256']


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


def test_italic_date_and_numeric_ocr_confusion_are_diagnostic_only():
    for draft,visual,confidence in [('7¢h.','7th.',63),('4o','40',95)]:
        assert not regions.coverage_diagnostics(visual,[{'words':[
            {'text':draft,'confidence':confidence}]}])['suspect']


@pytest.mark.parametrize('text',['[6]18','6 18','619'])
def test_amount_must_be_contiguous_and_exact(text):
    line={'words':[{'text':'618','confidence':97}]}
    assert regions.coverage_diagnostics(text,[line])['suspect']
    assert not regions.coverage_diagnostics('618',[line])['suspect']


def test_tiny_leader_dots_are_not_counted_as_numeric_words():
    line={'letter_height':21,'words':[
        {'text':'Canal','confidence':95,'bbox':[0,0,40,21]},
        {'text':'tobacco','confidence':95,'bbox':[50,0,120,21]},
        *[{'text':'0','confidence':60,'bbox':[x,15,x+4,20]} for x in (130,150,170)],
        {'text':'1','confidence':74,'bbox':[180,0,190,21]},
        {'text':'bale','confidence':95,'bbox':[200,0,240,21]}]}
    assert not regions.coverage_diagnostics('Canal tobacco 1 bale',[line])['suspect']
    assert regions.coverage_diagnostics('Canal tobacco',[line])['suspect']


@pytest.mark.parametrize('second,accepted',[('19th.',True),('10th.',False)])
def test_ocr_anchor_disagreement_requires_two_matching_visual_reads(tmp_path,second,accepted):
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,100),False)
    pix.clear_with(0);pix.set_dpi(72,72);pix.save(tmp_path/'full.png')
    run=tmp_path/'run';run.mkdir()
    def fragment(value):
        payload={'text':value,'uncertainties':[]}
        return SimpleNamespace(done_reason='stop',message=SimpleNamespace(content=json.dumps(payload)),
                               model_dump=lambda **_:payload)
    client=Mock();client.chat.side_effect=[fragment('19th.'),fragment(second),fragment('18th.'),fragment('17th.'),fragment('16th.')]
    selected=[{'words':[{'text':'roth','confidence':95,'bbox':[10,10,50,30]}]}]
    layout={'full_image':'full.png','width_px':100,'height_px':100,'dpi':72}
    if accepted:
        text,evidence,uncertainties=regions.retry_word_fragments(client,'model',{},tmp_path,40,layout,selected,run,'date')
        assert text=='19th.' and uncertainties
        adjudicated=[e['adjudicated_draft_word'] for e in evidence if 'adjudicated_draft_word' in e]
        assert regions.coverage_diagnostics(text,selected)['suspect']
        assert not regions.coverage_diagnostics(text,selected,adjudicated)['suspect']
        assert (run/'date.retry_line_01.span_01.confirm.response.json').is_file()
    else:
        with pytest.raises(ValueError,match='single-word'):
            regions.retry_word_fragments(client,'model',{},tmp_path,40,layout,selected,run,'date')


@pytest.mark.parametrize('text',['UTUMN, 1799.','AUTUMN, 1799.'])
def test_initial_read_uses_word_context_and_never_duplicates_letter(tmp_path,text):
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,100),False)
    pix.clear_with(0);pix.set_dpi(72,72);pix.save(tmp_path/'full.png')
    run=tmp_path/'run';run.mkdir()
    payload={'text':'AUTUMN,','uncertainties':[]}
    client=Mock();client.chat.return_value=SimpleNamespace(done_reason='stop',
        message=SimpleNamespace(content=json.dumps(payload)),model_dump=lambda **_:payload)
    line={'bbox':[25,20,99,30],'letter_height':10,
          'words':[{'text':'UTUMN,','bbox':[25,20,60,30]}]}
    layout={'full_image':'full.png','width_px':100,'height_px':100,'dpi':72}
    region={'role':'body','bbox_px':[0,0,100,100]}
    fixed,evidence=regions.inspect_initial(client,'model',{},tmp_path,layout,region,[line],text,run,'initial')
    assert fixed=='AUTUMN, 1799.'
    assert evidence['letter']=='A'
    with pymupdf.open(evidence['crop']) as crop:
        assert crop[0].rect.width>60  # Includes ordinary letters beside initial.


def test_separated_footnote_marker_is_not_a_missing_word(crop_run, monkeypatch):
    r = crop_run
    note = r.layouts[0]['regions'][-1]
    note['lines'] = [{'words': [{'text': w, 'confidence': 96}
                    for w in '10 Or Bottle portage'.split()]}]
    dump(r.root/'page_0040.layout.json', r.layouts[0])
    r.client.chat.side_effect = (r.replies()[:-1] + [response(['Or Bottle portage'], marker='10')]
                                + r.replies())
    retry = Mock(side_effect=AssertionError('Separated marker must not trigger reread'))
    monkeypatch.setattr(regions, 'retry_word_fragments', retry)
    ocr_parse.main(r.args)
    page = json.loads((r.output/'page_0040.json').read_text())
    assert page['footnotes'][0]['marker'] == '10'
    assert page['footnotes'][0]['text'] == 'Or Bottle portage'
    retry.assert_not_called()


def test_repetition_abort_retries_once_and_retains_both_attempts(tmp_path):
    client = Mock()
    client.chat.side_effect = [RuntimeError('token repeat limit reached'), response(['Recovered.'])]
    result = regions.visual_chat(client, tmp_path, 'test', model='m',
        messages=[{'role':'user', 'content':'Read image', 'images':[b'image']}],
        format=regions.RegionLines.model_json_schema(), options={'num_predict':4096})
    assert result.done_reason == 'stop'
    assert client.chat.call_count == 2
    retry = client.chat.call_args.kwargs
    assert retry['options']['num_predict'] == 1024
    assert retry['messages'][0]['images'] == [b'image']
    assert (tmp_path/'test.attempt_01.error.json').is_file()
    assert (tmp_path/'test.attempt_02.response.json').is_file()


def test_repetition_retry_is_bounded_and_other_errors_not_retried(tmp_path):
    for error, calls in [('token repeat limit reached',2), ('CUDA out of memory',1)]:
        client=Mock(); client.chat.side_effect=RuntimeError(error)
        with pytest.raises(RuntimeError, match=error):
            regions.visual_chat(client,tmp_path,'test',model='m',messages=[{'role':'user','content':'Read'}],
                format={},options={})
        assert client.chat.call_count == calls


@pytest.mark.parametrize('third,expected', [('1½','1½'), ('I½','I½'), ('I 1/2','I½'), ('2½',None)])
def test_fraction_disagreement_requires_two_exact_visual_readings(tmp_path, third, expected):
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,100),False)
    pix.clear_with(0);pix.set_dpi(72,72);pix.save(tmp_path/'full.png')
    run=tmp_path/'run';run.mkdir()
    def fragment(text):
        payload={'text':text,'uncertainties':[]}
        return SimpleNamespace(done_reason='stop',message=SimpleNamespace(content=json.dumps(payload)),
                               model_dump=lambda **_:payload)
    client=Mock();client.chat.side_effect=[fragment('I½'),fragment('1½'),fragment(third),fragment('3½'),fragment('4½')]
    selected=[{'words':[{'text':'134','confidence':95,'bbox':[10,10,50,30]}]}]
    layout={'full_image':'full.png','width_px':100,'height_px':100,'dpi':72}
    if expected:
        text,evidence,_=regions.retry_word_fragments(client,'model',{},tmp_path,43,layout,selected,run,'fraction')
        assert text==expected
        assert any(e.get('visual_consensus')==expected for e in evidence)
    else:
        with pytest.raises(ValueError,match='single-word'):
            regions.retry_word_fragments(client,'model',{},tmp_path,43,layout,selected,run,'fraction')
    assert client.chat.call_count==(3 if expected else 5)
    assert (run/'fraction.retry_line_01.span_01.confirm_02.response.json').is_file()


def test_blank_indent_with_ordinary_uppercase_word_is_not_a_drop_cap(tmp_path):
    import pymupdf
    doc=pymupdf.open(); page=doc.new_page(width=100,height=100)
    page.insert_text((25,29),'THE route',fontsize=10)
    page.get_pixmap(dpi=72).save(tmp_path/'full.png'); doc.close()
    run=tmp_path/'run';run.mkdir()
    line={'bbox':[25,20,99,30],'letter_height':10,
          'words':[{'text':'THE','bbox':[25,20,45,30]}]}
    layout={'full_image':'full.png','width_px':100,'height_px':100,'dpi':72}
    region={'role':'body','bbox_px':[0,0,100,100],'lines':[line]}
    client=Mock()
    text,evidence=regions.inspect_initial(client,'m',{},tmp_path,layout,region,[line],'THE route',run,'plain')
    assert text=='THE route' and evidence is None
    client.chat.assert_not_called()


@pytest.mark.parametrize('left,right,equal', [
    ('I½','I 1/2',True), ('1½','1 1/2',True), ('¾','3⁄4',True),
    ('I½','1½',False), ('618','6 18',False), ('[Nov','Nov',False),
    ('word.','word',False), ('go to','goto',False)])
def test_visual_consensus_only_normalizes_fraction_encoding(left,right,equal):
    assert (regions.visual_read_key(left)==regions.visual_read_key(right)) is equal


def test_empty_measured_line_gets_visual_retry_without_using_ocr_draft(crop_run, monkeypatch):
    r=crop_run
    r.layouts[0]['regions'][0]['lines']=[{'words':[{'text':'WRONG','confidence':20}]}]
    dump(r.root/'page_0040.layout.json',r.layouts[0])
    r.client.chat.side_effect=[response([''])]+r.replies()[1:]+r.replies()
    reread=Mock(return_value=('A HEADER.',[{'response':'retained-visual-read'}],[]))
    monkeypatch.setattr(regions,'retry_word_fragments',reread)
    ocr_parse.main(r.args)
    page=json.loads((r.output/'page_0040.json').read_text())
    assert page['running_headers']==['A HEADER.']
    assert page['regions'][0]['chunks'][0]['empty_response_retries']
    assert reread.call_count==1


def test_declared_header_leaking_from_edge_crop_is_separated_and_audited(crop_run):
    r=crop_run
    replies=r.replies();replies[2]=response(['A HEADER.','','François in the North-','west.'])
    r.client.chat.side_effect=replies+r.replies()
    ocr_parse.main(r.args)
    page=json.loads((r.output/'page_0040.json').read_text())
    assert page['body'].lstrip().startswith('François')
    assert next(e for e in page['regions'] if e['role']=='body')['body_edge_header_separations'][0]['removed_text']=='A HEADER.\n'


@pytest.mark.parametrize('confirmed,accepted',[('22',True),('2',True),('23',False),(None,False)])
def test_conflicting_marker_needs_independent_enlarged_glyph(monkeypatch,tmp_path,confirmed,accepted):
    inspect=Mock(return_value=(confirmed,{'response':'glyph.response.json'}))
    monkeypatch.setattr(regions,'inspect_marker',inspect)
    args=(Mock(),'m',{},tmp_path,175,{}, {},tmp_path,'[22] Goose r.','2')
    if accepted:
        text,marker,removed,evidence=regions.separate_confirmed_marker(*args)
        assert text==' Goose r.' and marker==confirmed
        assert evidence['conflicting_transcribed_markers']==['2','22']
        assert removed=='[22]'
    else:
        with pytest.raises(ValueError,match='does not resolve'):
            regions.separate_confirmed_marker(*args)
    inspect.assert_called_once()


@pytest.mark.parametrize('far_apart,overlaps_header',[(False,False),(True,False),(False,True)])
def test_split_folio_regions_merge_only_when_adjacent_and_clear(tmp_path,far_apart,overlaps_header):
    import pymupdf,copy
    doc=pymupdf.open();p=doc.new_page(width=200,height=300)
    p.insert_text((30,30),'346',fontsize=12);p.get_pixmap(dpi=72).save(tmp_path/'full.png');doc.close()
    parts=[{'id':1,'role':'page_number','bbox_px':[40,20,48,32],'crop':'full.png','lines':[]},
           {'id':2,'role':'page_number','bbox_px':[30 if not far_apart else 150,20,39 if not far_apart else 159,32], 'crop':'full.png','lines':[]}]
    header={'id':3,'role':'running_header','bbox_px':[50 if overlaps_header else 80,20,190,32]}
    layout={'regions':parts+[header],'full_image':'full.png','width_px':200,'height_px':300,'dpi':72}
    before=copy.deepcopy(layout);run=tmp_path/'run';run.mkdir()
    result=regions.merge_number_fragments(tmp_path,layout,run,380)
    assert layout==before
    if far_apart or overlaps_header:
        assert result==layout['regions']
    else:
        assert len(result)==2
        assert result[0]['merged_source_region_ids']==[1,2]
        assert Path(result[0]['derived_crop']).is_file()
        assert result[0]['bbox_px'][0]<30
        assert (run/'page_0380.number_merge.json').is_file()


def test_empty_visual_retry_still_fails_page_and_later_pages_continue(crop_run,monkeypatch):
    r=crop_run
    r.layouts[0]['regions'][0]['lines']=[{'words':[{'text':'Visible','confidence':95}]}]
    dump(r.root/'page_0040.layout.json',r.layouts[0])
    r.client.chat.side_effect=[response([''])]+r.replies()
    monkeypatch.setattr(regions,'retry_word_fragments',Mock(return_value=('',[],[])))
    with pytest.raises(SystemExit,match='1 page'):
        ocr_parse.main(r.args)
    assert not (r.output/'page_0040.json').exists()
    assert (r.output/'page_0041.json').is_file()
    run=next((r.output/'runs').iterdir())
    assert (run/'page_0040.region_01.response.json').is_file()
    assert 'empty region transcription after visual retry' in (run/'page_0040.error.json').read_text()


def test_interior_header_duplicate_is_not_silently_removed(crop_run):
    r=crop_run
    replies=r.replies();replies[2]=response(['Opening narrative.','A HEADER.','More narrative.'])
    r.client.chat.side_effect=replies+r.replies()
    with pytest.raises(SystemExit,match='1 page'):
        ocr_parse.main(r.args)
    assert not (r.output/'page_0040.json').exists()
    assert (r.output/'page_0041.json').is_file()
    run=next((r.output/'runs').iterdir())
    assert 'Running header duplicated in body' in (run/'page_0040.error.json').read_text()


def test_faded_sepia_ink_is_visible_even_when_red_channel_exceeds100():
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,20,20),False)
    pix.clear_with(255)
    pix.set_rect(pymupdf.IRect(5,5,15,15),(106,94,58))
    assert not any(v<100 for v in pix.samples[::pix.n])
    assert regions.ink_pixels(pix)>=100
    pix.clear_with(255)
    assert regions.ink_pixels(pix)==0


def test_currency_symbol_counts_as_visible_token_without_allowing_its_omission():
    lines=[{'words':[{'text':'b','confidence':86},{'text':'s.','confidence':49},
                     {'text':'ad,','confidence':46}]}]
    assert regions.coverage_diagnostics('s. d.',lines)['suspect']
    result=regions.coverage_diagnostics('£ s. d.',lines,{'b':'£'})
    assert not result['suspect']
    assert result['draft_token_count']==result['transcribed_token_count']==3
    assert regions.coverage_diagnostics('$ s. d.',[{'words':[
        {'text':'£','confidence':96},{'text':'s.','confidence':96},{'text':'d.','confidence':96}]}])['suspect']


def test_visually_adjudicated_punctuation_is_not_still_counted_as_missing_word():
    lines=[{'words':[{'text':'7]','confidence':96},{'text':'river','confidence':96}]}]
    assert regions.coverage_diagnostics('?], river',lines)['suspect']
    assert not regions.coverage_diagnostics('?], river',lines,{'7]':'?],'})['suspect']
    assert regions.coverage_diagnostics('?],',lines,{'7]':'?],'})['suspect']


def test_fragment_crop_expands_past_narrow_ocr_box_but_stays_between_neighbors():
    line={'bbox':[251,732,1233,778],'letter_height':27,'words':[
        {'text':'rst.','bbox':[520,741,575,764]},
        {'text':'1','bbox':[607,732,623,778]},
        {'text':'set','bbox':[648,742,694,764]}]}
    x0,y0,x1,y1=regions.fragment_box([line['words'][1]],line,1373,2159)
    assert 575<x0<605 and 625<x1<648
    assert y0<732 and y1>778


@pytest.mark.parametrize('footer_y,excluded',[(1982,True),(1900,False)])
def test_header_folio_excludes_only_isolated_mark_below_all_text(footer_y,excluded):
    import copy
    layout={'height_px':2159,'regions':[
        {'id':1,'role':'running_header','bbox_px':[276,194,1007,223]},
        {'id':2,'role':'page_number','bbox_px':[1108,196,1142,224]},
        {'id':3,'role':'body','bbox_px':[127,269,1156,1534]},
        {'id':4,'role':'footnote','bbox_px':[126,1579,1140,1937]},
        {'id':5,'role':'page_number','bbox_px':[425,footer_y,435,footer_y+18]}]}
    before=copy.deepcopy(layout)
    result=regions.folio_regions(layout)
    assert bool(result[-1].get('folio_exclusion_reason'))==excluded
    assert result[1]==layout['regions'][1]
    assert layout==before
    assert len(result)==len(layout['regions'])  # Candidate retained, never erased.


def test_repetition_after_bounded_retry_uses_visual_fragments_and_retains_failure(crop_run,monkeypatch):
    r=crop_run
    r.layouts[0]['regions'][0]['lines']=[{'words':[{'text':'A','confidence':96},
                                                        {'text':'HEADER.','confidence':96}]}]
    dump(r.root/'page_0040.layout.json',r.layouts[0])
    error=RuntimeError('token repeat limit reached')
    r.client.chat.side_effect=[error,error]+r.replies()[1:]+r.replies()
    reread=Mock(return_value=('A HEADER.',[{'response':'actual-visual-response.json'}],[]))
    monkeypatch.setattr(regions,'retry_word_fragments',reread)
    ocr_parse.main(r.args)
    page=json.loads((r.output/'page_0040.json').read_text())
    run=next((r.output/'runs').iterdir())
    chunk=page['regions'][0]['chunks'][0]
    assert chunk['response'].endswith('.fragment_assembly.json')
    assert chunk['repetition_response_retries']
    assert (run/chunk['response']).is_file()
    assert (run/'page_0040.region_01.attempt_01.error.json').is_file()
    assert (run/'page_0040.region_01.attempt_02.error.json').is_file()
    assert not (run/'page_0040.region_01.response.json').exists()
    reread.assert_called_once()


def test_visual_read_key_allows_apostrophe_typography_but_preserves_comma():
    assert regions.visual_read_key('M’Kie,')==regions.visual_read_key("M'Kie,")
    assert regions.visual_read_key('M’Kie,')!=regions.visual_read_key('M’Kie')


@pytest.mark.parametrize('draft,original,reads,expected',[
    ('May roth. Two men.','May 10th. Two men.',['May 10th. Two men.']*2,'May 10th. Two men.'),
    ('May roth.','May ioth,',['May 10th.','May 10th.'],'May 10th.'),
    ('W of Portage la Prairie The','W of Portage la Prairie',['W of Portage la Prairie']*2,None),
    ('618','619',['619']*2,None),
    ('618','6 18',['6 18']*2,None),
    ('migratorius','[migratorius]',['[migratorius]']*2,None),
])
def test_complete_line_consensus_preserves_words_and_numeric_guards(tmp_path,draft,original,reads,expected):
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,120,30),False)
    pix.clear_with(255);pix.set_dpi(72,72);crop=tmp_path/'line.png';pix.save(crop)
    run=tmp_path/'run';run.mkdir()
    def reply(text):
        payload={'lines':[text],'uncertainties':[]}
        return SimpleNamespace(done_reason='stop',message=SimpleNamespace(content=json.dumps(payload)),
                               model_dump=lambda **_:payload)
    client=Mock();client.chat.side_effect=[reply(text) for text in reads]
    selected=[{'bbox':[0,0,120,30],'words':[{'text':word,'confidence':96} for word in draft.split()]}]
    text,evidence,uncertainties=regions.reread_complete_line(client,'m',{},crop,selected,original,run,'line',72)
    assert text==expected
    assert evidence and (run/evidence[0]['response']).is_file()
    if expected:
        assert uncertainties and evidence[-1]['complete_line_consensus']==expected
    else:
        assert client.chat.call_count==2


def test_complete_line_punctuation_is_preserved_after_false_ocr_word_adjudication():
    text='aiguilles [rails ?], 5'
    selected=[{'words':[{'text':w,'confidence':96} for w in 'aiguilles [rails 7], 5'.split()]}]
    confirmations=[{'transcription':text,'done_reason':'stop'}]
    assert regions.prefer_complete_line(text,confirmations,selected,{'7],':'?'})==text
    assert regions.prefer_complete_line(text,[],selected,{'7],':'?'}) is None


def test_preferred_full_line_must_not_contradict_confirmed_numeric_word():
    selected=[{'words':[{'text':'618','confidence':96}]}]
    confirmations=[{'transcription':'618','done_reason':'stop'}]
    assert regions.prefer_complete_line('618',confirmations,selected,{'618':'619'}) is None


def test_repeated_fragment_abort_splits_only_until_smaller_reads_succeed(tmp_path):
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,40),False)
    pix.clear_with(255);pix.set_dpi(72,72);pix.save(tmp_path/'full.png')
    run=tmp_path/'run';run.mkdir()
    def fragment(value):
        payload={'text':value,'uncertainties':[]}
        return SimpleNamespace(done_reason='stop',message=SimpleNamespace(content=json.dumps(payload)),
                               model_dump=lambda **_:payload)
    client=Mock();client.chat.side_effect=[RuntimeError('token repeat limit'),RuntimeError('token repeat limit'),
                                         fragment('LEFT'),fragment('RIGHT')]
    line={'bbox':[5,10,90,30],'letter_height':20,'words':[
        {'text':'LEFT','confidence':95,'bbox':[5,10,40,30]},
        {'text':'RIGHT','confidence':95,'bbox':[50,10,90,30]}]}
    layout={'full_image':'full.png','width_px':100,'height_px':40,'dpi':72}
    text,evidence,_=regions.retry_word_fragments(client,'m',{},tmp_path,1,layout,[line],run,'repeat')
    assert text=='LEFT RIGHT' and client.chat.call_count==4
    assert len(evidence)==2
    assert (run/'repeat.retry_line_01.span_01.split.json').is_file()
    assert (run/'repeat.retry_line_01.span_01.attempt_02.error.json').is_file()


def test_genuine_opening_bracket_can_be_adjudicated_without_ignoring_unreviewed_additions():
    line={'words':[{'text':w,'confidence':96} for w in 'fone of the Red Lake Ojibways].'.split()]}
    assert regions.coverage_diagnostics('[one of the Red Lake Ojibways].',[line])['suspect']
    assert not regions.coverage_diagnostics('[one of the Red Lake Ojibways].',[line],{'fone':'[one'})['suspect']
    assert regions.coverage_diagnostics('[migratorius]',[{'words':[{'text':'migratorius]','confidence':96}]}])['suspect']


@pytest.mark.parametrize('bad_hash',[False,True])
def test_source_reviews_are_bound_to_original_image_and_layout(crop_run,bad_hash):
    import copy
    r=crop_run;layout=r.layouts[0];before=copy.deepcopy(layout)
    path=r.root/'page_0040.layout.json'
    review={'image_sha256': 'wrong' if bad_hash else layout['image_sha256'],
            'layout_sha256':regions.sha256(path),'basis':'Agent visual review of table row',
            'region_roles':{'5':'table'}}
    data={'pages':{'40':review},'sha256':'reviews-file-hash'}
    if bad_hash:
        with pytest.raises(ValueError,match='hash mismatch'):
            regions.apply_source_reviews(layout,path,data)
    else:
        result=regions.apply_source_reviews(layout,path,data)
        assert result['regions'][-1]['role']=='table'
        assert result['regions'][-1]['marker_candidate'] is None
        assert result['source_review']==review
    assert layout==before


def test_reviewed_dagger_is_retained_with_explicit_annotation_evidence(tmp_path):
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,40),False)
    pix.clear_with(255);pix.set_dpi(72,72);pix.save(tmp_path/'full.png')
    run=tmp_path/'run';run.mkdir();box=[10,10,20,30]
    layout={'full_image':'full.png','width_px':100,'height_px':40,'dpi':72,
        'image_sha256':'image-hash','reviews_sha256':'reviews-hash','reviewed_words':[
            {'draft_text':'t+','bbox_px':box,'text':'†','basis':'Agent visual inspection of retained source crop'}]}
    selected=[{'bbox':box,'words':[{'text':'t+','confidence':60,'bbox':box}]}]
    client=Mock()
    text,evidence,uncertainties=regions.retry_word_fragments(client,'m',{},tmp_path,233,layout,selected,run,'dagger')
    assert text=='†' and uncertainties
    assert evidence[0]['reviewed_transcription']=='†'
    assert evidence[0]['reviews_sha256']=='reviews-hash'
    assert 'visual_consensus' not in evidence[0]  # Not misrepresented as a model read.
    client.chat.assert_not_called()


def test_source_review_split_must_cover_every_original_line(crop_run):
    import copy
    r=crop_run;layout=copy.deepcopy(r.layouts[0]);layout['regions'][2]['lines']=[
        {'bbox':[1,10,20,20],'text':'continuation'}, {'bbox':[1,25,20,35],'text':'2 note'}]
    path=r.root/'page_0040.layout.json';dump(path,layout)
    review={'image_sha256':layout['image_sha256'],'layout_sha256':regions.sha256(path),'basis':'Visual note boundaries',
        'split_regions':[{'region_id':3,'parts':[{'id':3,'start':0,'stop':1}]}]}
    with pytest.raises(ValueError,match='retain every line'):
        regions.apply_source_reviews(layout,path,{'pages':{'40':review},'sha256':'review-hash'})


def test_reviewed_table_crop_excludes_ink_from_previous_row(tmp_path):
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,100),False)
    pix.clear_with(255);pix.set_dpi(72,72)
    pix.set_rect(pymupdf.IRect(10,30,90,38),(255,0,0))
    pix.set_rect(pymupdf.IRect(10,42,90,50),(0,0,255))
    pix.save(tmp_path/'full.png');run=tmp_path/'run';run.mkdir()
    above={'bbox':[10,28,90,45],'words':[{'text':'Previous','confidence':96,'bbox':[10,30,90,38]}]}
    current={'bbox':[10,35,90,55],'words':[{'text':'Sugar','confidence':96,'bbox':[10,42,90,50]}]}
    region={'id':2,'role':'table','source_original_role':'footnote','source_review_basis':'Visual review',
            'bbox_px':[10,35,90,55],'crop':'full.png','lines':[current]}
    layout={'full_image':'full.png','width_px':100,'height_px':100,'dpi':72,
            'regions':[{'id':1,'role':'table','lines':[above]},region]}
    crop,_,_=regions.region_chunks(tmp_path,255,layout,region,run,1)[0]
    result=pymupdf.Pixmap(str(crop));rgb=list(zip(result.samples[0::3],result.samples[1::3],result.samples[2::3]))
    assert not any(r>200 and g<30 and b<30 for r,g,b in rgb)
    assert any(b>200 and r<30 and g<30 for r,g,b in rgb)


def test_reviewed_table_role_routes_to_body_and_records_review_snapshot(crop_run):
    r=crop_run;path=r.root/'page_0040.layout.json'
    review={'version':1,'pages':{'40':{'image_sha256':r.layouts[0]['image_sha256'],
        'layout_sha256':regions.sha256(path),'basis':'Visual table-row inspection','region_roles':{'5':'table'}}}}
    reviews=r.root/'reviews.json';dump(reviews,review)
    ocr_parse.main(r.args+['--reviews',str(reviews)])
    page=json.loads((r.output/'page_0040.json').read_text())
    assert page['footnotes']==[]
    assert 'A continued note.' in page['body']  # Mock visual text routed by corrected role.
    assert page['source_reviews_sha256']==regions.sha256(reviews)
    run=next((r.output/'runs').iterdir())
    assert json.loads((run/'source_reviews.json').read_text())==review
    assert page['regions'][-1]['role']=='table'


def test_dot_dominated_table_row_uses_full_glyph_height_for_filtering():
    line={'letter_height':7,'words':[
        {'text':'7','bbox':[0,0,10,15],'confidence':96},
        {'text':'“grease','bbox':[20,0,80,7],'confidence':53},
        {'text':'3','bbox':[100,10,104,14],'confidence':57},
        {'text':'00','bbox':[120,10,136,15],'confidence':49},
        {'text':'7','bbox':[150,0,160,15],'confidence':96}]}
    assert [w['text'] for w in regions.diagnostic_words([line])]==['7','“grease','7']
    assert not regions.coverage_diagnostics('7 “ grease . . . 7',[line])['suspect']


def test_extraneous_neighboring_row_text_triggers_retry():
    line={'words':[{'text':w,'confidence':96} for w in '95 Bags of Pemmican'.split()]}
    check=regions.coverage_diagnostics('95 Bags of Pemmican 10 Kegs of Sugar',[line])
    assert check['suspect'] and check['excess_transcribed_tokens']
    assert not regions.coverage_diagnostics('95 Bags of Pemmican',[line])['suspect']


def test_reviewed_marker_preserves_source_number_instead_of_model_misreading(tmp_path):
    region={'reviewed_marker':'24'}
    text,marker,removed,evidence=regions.separate_confirmed_marker(
        Mock(),'m',{},tmp_path,255,{},region,tmp_path,'^4 Of the persons named','4')
    assert marker=='24' and text==' Of the persons named' and removed=='^4'
    assert evidence['transcribed_prefix']=='4' and evidence['resolved_marker']=='24'
    assert evidence['basis']=='Hash-checked source review'


def test_numeric_table_columns_and_fractions_are_not_extraneous_prose():
    draft='Upper English river 4,581|14|1014 148 C.&D.'
    line={'words':[{'text':w,'confidence':60} for w in draft.split()]}
    result=regions.coverage_diagnostics('Upper English river 4,581 14 10¼ 148 C. & D.',[line])
    assert not result['excess_transcribed_tokens']


def test_reviewed_rotated_header_is_sent_whole_without_fake_line_anchors(tmp_path):
    import pymupdf
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,200,100),False)
    pix.clear_with(255);pix.set_dpi(72,72);pix.save(tmp_path/'full.png')
    run=tmp_path/'run';run.mkdir()
    layout={'full_image':'full.png','width_px':200,'height_px':100,'dpi':72}
    region={'id':4,'role':'table','bbox_px':[0,0,180,60],'rotation':90,'whole_region':True,'lines':[]}
    crop,first,count=regions.region_chunks(tmp_path,293,layout,region,run,1)[0]
    result=pymupdf.Pixmap(str(crop))
    assert result.width==60 and result.height==180 and first==count==0


def test_fused_ocr_words_and_roman_style_numbers_are_not_inserted_prose():
    prose=[{'words':[{'text':w,'confidence':60} for w in "May31st. Atten o'clockIembarked in my own canoe".split()]}]
    assert not regions.coverage_diagnostics("May 31st. At ten o'clock I embarked in my own canoe",prose)['excess_transcribed_tokens']
    row=[{'words':[{'text':w,'confidence':60} for w in 'Batchewoinan bay 2581 2 11 V'.split()]}]
    assert not regions.coverage_diagnostics('Batchewoinan bay 258 I 2 II V.',row)['excess_transcribed_tokens']


def test_overwrite_updates_recovery_index_to_new_retained_source(crop_run):
    r=crop_run;r.output.mkdir()
    dump(r.output/'recovery_manifest.json',{'scope':[40,41],'pages':[]})
    ocr_parse.main(r.args)
    r.client.chat.side_effect=r.replies()+r.replies()
    ocr_parse.main(r.args+['--overwrite'])
    index=json.loads((r.output/'recovery_manifest.json').read_text())
    page=json.loads((r.output/'page_0040.json').read_text())
    assert index['active_recovery_run']==page['run_id']
    assert index['recovery_run_status']=='complete_unverified'
    assert index['present_pages']==2
    for entry in index['pages']:
        assert entry['run_id']==page['run_id']
        for item in entry['files']:
            assert regions.sha256(Path(item['source_path']))==item['sha256']
            assert regions.sha256(Path(item['destination_path']))==item['sha256']


def test_low_confidence_draft_words_are_not_hallucinated_extra_prose():
    text='Cut Bank cr. traverses this loop southeastwardly, to fall into Mouse r. near Wood'
    words=[{'text':word,'confidence':20 if index<2 else 96} for index,word in enumerate(text.split())]
    assert not regions.coverage_diagnostics(text,[{'words':words}])['suspect']
    assert regions.coverage_diagnostics(text+' inserted foreign words',[{'words':words}])['suspect']


def test_visually_adjudicated_words_are_not_rejected_again_as_insertions():
    lines=[{'words':[{'text':'GAVGo9','confidence':96}]}]
    assert regions.coverage_diagnostics('Keg salt',lines)['suspect']
    assert not regions.coverage_diagnostics('Keg salt',lines,{'GAVGo9':'Keg salt'})['suspect']


def test_table_consensus_ignores_leaders_but_preserves_number_separation_and_brackets():
    assert regions.full_line_key('Pemmican.....12 | 35..188',True)==regions.full_line_key('Pemmican..12...35....188.',True)
    assert regions.full_line_key('3 5 0',True)!=regions.full_line_key('35 0',True)
    assert regions.full_line_key('Pemmican [12] 35',True)!=regions.full_line_key('Pemmican 12 35',True)
    assert regions.full_line_key('Text...',False)!=regions.full_line_key('Text.',False)


def test_small_ocr_boxes_still_explain_words_for_insertion_diagnostics():
    words=[{'text':'Macdonel','confidence':96,'bbox':[0,0,80,20]},
           {'text':'[szc]','confidence':96,'bbox':[85,0,120,20]},
           {'text':'men','confidence':96,'bbox':[125,15,155,20]},
           {'text':'was','confidence':96,'bbox':[160,15,180,20]}]
    result=regions.coverage_diagnostics('Macdonel [sic] men was',[{'letter_height':20,'words':words}])
    assert not result['excess_transcribed_tokens']
    assert result['suspect']  # The sic/szc disagreement still requires visual adjudication.


def test_assembled_validated_spans_do_not_reapply_untrusted_draft_vocabulary():
    lines=[{'words':[{'text':w,'confidence':60} for w in '3 USES SE hoc'.split()]}]
    assert regions.coverage_diagnostics('5 Kegs sugar 2 3',lines)['suspect']
    assert not regions.coverage_diagnostics('5 Kegs sugar 2 3',lines,check_insertions=False)['suspect']
    amount=[{'words':[{'text':'618','confidence':96}]}]
    assert regions.coverage_diagnostics('619',amount,check_insertions=False)['suspect']
