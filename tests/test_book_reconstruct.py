"""Deterministic export checks; synthetic inputs, no inference required."""
import json
import zipfile

import pytest

from newlight.book_reconstruct import audit_docx, enrich, styled_runs, table_row, split_blocks, text_frame, mark_page_continuations
from newlight.book_reconstruct import audit_footnote_pagination, source_page_footnotes, custom_properties_xml, justify_footnote_last_line
from newlight.book_reconstruct import export_plain_text, main
from newlight.book_reconstruct import publish_export


@pytest.mark.parametrize('tamper',[False,True])
def test_table_audit_preserves_tabs_and_excludes_only_generated_note_mark(tmp_path,tamper):
    ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    path=tmp_path/'table.docx'
    cells=['147\t63','600','']
    model={'pages':[{'printed_page_number':None,'blocks':[{'kind':'table','rows':[{'cells':cells}]}]}],
           'footnotes':[{'id':1,'marker':'10','text':'Note.','references':[{}]}]}
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('word/document.xml',f'''<w:document xmlns:w="{ns}"><w:body><w:tbl><w:tr>
          <w:tc><w:p><w:r><w:footnoteReference w:id="1" w:customMarkFollows="1"/></w:r>
          <w:r><w:t>10</w:t></w:r><w:r><w:t>147</w:t><w:tab/><w:t>63</w:t></w:r></w:p></w:tc>
          <w:tc><w:p><w:r><w:t>{'601' if tamper else '600'}</w:t></w:r></w:p></w:tc>
          <w:tc><w:p/></w:tc></w:tr></w:tbl></w:body></w:document>''')
        archive.writestr('word/footnotes.xml',f'<w:footnotes xmlns:w="{ns}"><w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r><w:r><w:t>Note.</w:t></w:r></w:p></w:footnote></w:footnotes>')
    if tamper:
        with pytest.raises(ValueError,match='Table cell content differs'):audit_docx(path,model)
    else:assert audit_docx(path,model)['native_tables']==1


def test_stable_export_updates_without_changing_retained_run(tmp_path):
    source=tmp_path/'dated.txt';source.write_text('First version')
    stable=tmp_path/'book'/'newlight.txt'
    publish_export(source,stable)
    second=tmp_path/'second.txt';second.write_text('Second version')
    publish_export(second,stable)
    assert stable.read_text()=='Second version'
    assert source.read_text()=='First version'


def test_review_mode_still_requires_every_page(tmp_path):
    with pytest.raises(SystemExit):
        main(['--both','--allow-review-issues','--ocr-dir',str(tmp_path),
              '--renderer-run',str(tmp_path),'--start','35','--end','36'])


def test_reviewed_layout_snapshot_is_used_and_provenance_checked(tmp_path):
    ocr,layout=page_pair(tmp_path)
    source=ocr/'page_0040.json';page=json.loads(source.read_text())
    page.update(source_review={'basis':'test review'},run_id='review-run')
    source.write_text(json.dumps(page))
    snapshot=ocr/'runs'/'review-run'/'page_0040.layout.json'
    snapshot.parent.mkdir(parents=True)
    effective=json.loads((layout/snapshot.name).read_text())
    effective['regions'][0]['bbox_px'][0]=120
    snapshot.write_text(json.dumps(effective))
    model=enrich(ocr,layout,40,41)
    assert model['pages'][0]['provenance']['layout_path']==str(snapshot.resolve())
    assert model['pages'][0]['geometry']['left_pt']==pytest.approx(120*72/200)
    effective['image_sha256']='wrong image';snapshot.write_text(json.dumps(effective))
    with pytest.raises(ValueError,match='reviewed layout provenance mismatch'):
        enrich(ocr,layout,40,41)


def test_plain_text_preserves_every_body_and_note_segment(tmp_path):
    import hashlib
    source = tmp_path / 'ocr'
    source.mkdir()
    pages = []
    for number, marker in [(35, '1'), (36, None)]:
        page = {'pdf_page': number, 'body': '  François  [river].¹\nNorth-\nwest.\n',
                'printed_page_number': str(number - 34), 'running_headers': ['JOURNAL.'],
                'footnotes': [{'marker': marker, 'text': '  Note\ncontinued.  ',
                               'continues_from_previous_page': marker is None}]}
        (source / f'page_{number:04d}.json').write_text(json.dumps(page))
        pages.append(page)
    output = export_plain_text(source, tmp_path / 'out', 35, 36)
    text = output.read_text(encoding='utf-8')
    audit = json.loads((output.parent / 'text-audit.json').read_text())
    assert audit['complete'] is True
    assert audit['source_pages'] == 2
    assert audit['footnote_segments'] == 2
    assert audit['text_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    for page, entry in zip(pages, audit['pages']):
        assert text[slice(*entry['body'])] == page['body']
        assert text[slice(*entry['footnotes'][0]['text'])] == page['footnotes'][0]['text']
        assert text[slice(*entry['running_headers'][0])] == 'JOURNAL.'
    assert '[Footnote unnumbered; source segment 1]' in text
    assert text.index('PDF page 35') < text.index('PDF page 36')


def test_plain_text_requires_complete_range_unless_explicit_draft(tmp_path):
    page = {'pdf_page': 35, 'body': 'Text.', 'footnotes': []}
    (tmp_path / 'page_0035.json').write_text(json.dumps(page))
    with pytest.raises(ValueError, match='Missing OCR pages: 36'):
        export_plain_text(tmp_path, tmp_path / 'out', 35, 36)
    assert not (tmp_path / 'out').exists()
    output = export_plain_text(tmp_path, tmp_path / 'out', 35, 36, available_only=True)
    assert 'INCOMPLETE: missing PDF pages 36' in output.read_text()


def test_plain_text_cli_needs_no_renderer_and_rejects_wrong_page_identity(tmp_path):
    path = tmp_path / 'page_0035.json'
    path.write_text(json.dumps({'pdf_page': 36, 'body': 'Wrong page.', 'footnotes': []}))
    args = ['--text-only', '--ocr-dir', str(tmp_path), '--start', '35', '--end', '35',
            '--output', str(tmp_path / 'out')]
    with pytest.raises(ValueError, match='identity/body'):
        main(args)
    path.write_text(json.dumps({'pdf_page': 35, 'body': 'Correct page.', 'footnotes': []}))
    main(args)
    assert len(list((tmp_path / 'out').glob('runs/*/*.txt'))) == 1


@pytest.mark.parametrize('text,continued,expected',[
    ('A complete sentence.',True,False),
    ('A complete sentence.\n',True,False),
    ('too shallow for',True,True),
    ('A complete sentence.',False,False),
])
def test_footnote_final_line_with_period_is_never_forced_to_stretch(text,continued,expected):
    assert justify_footnote_last_line({'text':text,'continues_on_next_page':continued}) is expected


def test_custom_word_properties_have_native_names_types_and_unique_ids():
    from xml.etree import ElementTree as ET
    values={'SourceEditor':'Elliott Coues','SourcePublicationYear':1897,
            'ReviewRequired':True,'TextWidthPt':325.0}
    root=ET.fromstring(custom_properties_xml(values))
    assert [p.get('name') for p in root]==list(values)
    assert [p.get('pid') for p in root]==['2','3','4','5']
    assert [p[0].tag.rsplit('}',1)[-1] for p in root]==['lpwstr','i4','bool','r8']
    assert [p[0].text for p in root]==['Elliott Coues','1897','true','325.0']


def test_continued_note_must_start_on_its_source_page():
    continuation='vessels to land, and separated by Hat point from Wauswaugoning bay), in which'
    model={'pages':[{'pdf_page':40},{'pdf_page':41}],
           'footnotes':[{'marker':'2','parts':[
               {'pdf_page':40,'text':'too shallow for'},
               {'pdf_page':41,'text':continuation}]}]}
    assert audit_footnote_pagination(model,['too shallow for',continuation])['status']=='passed'
    result=audit_footnote_pagination(model,['too shallow for\n'+continuation,'Table'])
    assert result['status']=='failed'
    assert result['issues'][0]['actual_rendered_pages']==[1]
    assert audit_footnote_pagination(model,['too shallow for','Table'])['status']=='failed'
    assert audit_footnote_pagination(model,['too shallow for',continuation,'Extra'])['status']=='failed'


@pytest.mark.parametrize('justify_last_line', [True, False])
def test_continued_footnote_requires_layout_break_without_changing_text(tmp_path, justify_last_line):
    ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    path=tmp_path/'continued.docx'
    text='on Grand Portage bay (too shallow for'
    model={'pages':[], 'footnote_numbering':'automatic', 'footnote_line_breaks':'preserved',
           'footnotes':[{'id':1,'text':text,'references':[{}], 'continues_on_next_page':True}]}
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('word/document.xml',f'<w:document xmlns:w="{ns}"><w:body><w:p><w:r><w:footnoteReference w:id="1"/></w:r></w:p></w:body></w:document>')
        ending='<w:r><w:br/></w:r>' if justify_last_line else ''
        archive.writestr('word/footnotes.xml',f'<w:footnotes xmlns:w="{ns}"><w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r><w:r><w:t> {text}</w:t></w:r>{ending}</w:p></w:footnote></w:footnotes>')
    if justify_last_line:
        assert audit_docx(path,model)['status']=='passed'
    else:
        with pytest.raises(ValueError,match='final-line justification break'):audit_docx(path,model)


def page_pair(tmp_path, second_body='Next page.³', chapter=False):
    ocr = tmp_path / 'ocr'
    layout = tmp_path / 'layout'
    ocr.mkdir()
    layout.mkdir()
    for number, body, notes in [
        (40, 'CHAPTER II.\nText.¹ More.²', [
            {'marker': '1', 'text': 'First note.'},
            {'marker': '2', 'text': 'Beginning'}]),
        (41, ('CHAPTER III.\n' if chapter else '') + second_body, [
            {'marker': None, 'text': 'continued.'},
            {'marker': '3', 'text': 'Third note.'}]),
    ]:
        data = dict(pdf_page=number, body=body, footnotes=notes,
                    running_headers=['RUNNING HEADER'], printed_page_number=str(number-34),
                    source_sha256='image')
        (ocr / f'page_{number:04d}.json').write_text(json.dumps(data))
        geometry = dict(pdf_page=number, image_sha256='image', dpi=200,
                        source_page_rect=[0, 0, 505, 837],
                        metrics={'prose_left_px': 100, 'body_letter_height_px': 20},
                        regions=[{'role': 'body', 'bbox_px': [100, 200, 1000, 1800],
                                  'lines': []}])
        (layout / f'page_{number:04d}.layout.json').write_text(json.dumps(geometry))
    return ocr, layout


def test_note_continuation_links_to_one_logical_native_note(tmp_path):
    ocr, layout = page_pair(tmp_path)
    model = enrich(ocr, layout, 40, 41)
    assert len(model['footnotes']) == 3
    note = model['footnotes'][1]
    assert note['text'] == 'Beginning\ncontinued.'
    assert [p['pdf_page'] for p in note['parts']] == [40, 41]
    assert note['references'] == [{'pdf_page': 40, 'marker': '2'}]
    assert model['pages'][1]['running_headers'] == ['RUNNING HEADER']
    assert all('RUNNING HEADER' not in b.get('text', '') for b in model['pages'][1]['blocks'])


def test_source_page_segments_preserve_text_and_reference_numbering(tmp_path):
    ocr, layout = page_pair(tmp_path)
    model = enrich(ocr, layout, 40, 41)
    segments = source_page_footnotes(model)
    assert [n['id'] for n in segments] == [1, 2, 3, 4]
    assert [n['logical_id'] for n in segments] == [1, 2, 2, 3]
    assert [n['marker'] for n in segments if not n['continuation']] == ['1', '2', '3']
    assert segments[2]['references'] == [{'pdf_page': 41, 'continuation': True}]
    assert segments[2]['text'] == 'continued.'
    assert [n['continues_on_next_page'] for n in segments] == [False, True, False, False]
    for note in model['footnotes']:
        assert '\n'.join(n['text'] for n in segments if n['logical_id']==note['id']) == note['text']
    # Segmenting must never mutate the logical source graph or its original IDs.
    assert model['footnotes'][1]['text'] == 'Beginning\ncontinued.'
    assert [n['id'] for n in model['footnotes']] == [1, 2, 3]


def test_missing_selected_page_rejected(tmp_path):
    ocr, layout = page_pair(tmp_path)
    with pytest.raises(ValueError, match='Missing OCR pages: .42'):
        enrich(ocr, layout, 40, 42)


def test_available_only_records_gaps_and_retains_orphan_notes_and_unresolved_markers(tmp_path):
    ocr, layout=page_pair(tmp_path,second_body='Unknown reference.⁴')
    (ocr/'page_0040.json').unlink()
    model=enrich(ocr,layout,40,42,available_only=True)
    assert model['missing_pages']==[40,42]
    assert [p['pdf_page'] for p in model['pages']]==[41]
    assert model['footnote_numbering']=='source_custom'
    assert [n['text'] for n in model['footnotes']]==['continued.','Third note.']
    assert all(n['anchor_at_page_start'] for n in model['footnotes'])
    assert 'Unknown reference.⁴' in model['pages'][0]['blocks'][0]['text']
    assert {'orphan_note_continuation','note_reference_unlocated','unresolved_body_reference'} <= {i['kind'] for i in model['review_issues']}


def test_available_native_ids_follow_actual_anchor_order(tmp_path):
    ocr,layout=page_pair(tmp_path)
    path=ocr/'page_0040.json';page=json.loads(path.read_text())
    page['body']='CHAPTER II.\nText.¹ No second reference.'
    path.write_text(json.dumps(page))
    model=enrich(ocr,layout,40,41,available_only=True)
    segments=source_page_footnotes(model)
    assert [(n['id'],n['marker']) for n in segments[:2]]==[(1,'2'),(2,'1')]
    assert segments[0]['anchor_at_page_start']


def test_repeated_ocr_reference_does_not_duplicate_a_native_note(tmp_path):
    ocr,layout=page_pair(tmp_path,second_body='Repeated.³ Again.³')
    model=enrich(ocr,layout,40,41,available_only=True)
    note=next(n for n in model['footnotes'] if n['marker']=='3')
    assert len(note['references'])==1
    assert 'Repeated.³ Again.³' in model['pages'][1]['blocks'][0]['text']
    assert any(i['kind']=='repeated_body_reference_retained_as_text' for i in model['review_issues'])


def test_reference_without_note_rejected(tmp_path):
    ocr, layout = page_pair(tmp_path, second_body='Unknown.⁴')
    with pytest.raises(ValueError, match='reference 4 has no note'):
        enrich(ocr, layout, 40, 41)


def test_continuation_cannot_cross_chapter(tmp_path):
    ocr, layout = page_pair(tmp_path, chapter=True)
    with pytest.raises(ValueError, match='no adjacent preceding note'):
        enrich(ocr, layout, 40, 41)


def test_image_provenance_mismatch_rejected(tmp_path):
    ocr, layout = page_pair(tmp_path)
    path = ocr / 'page_0040.json'
    data = json.loads(path.read_text())
    data['source_sha256'] = 'different scan'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='image provenance mismatch'):
        enrich(ocr, layout, 40, 41)


def test_reviewed_correction_audited_and_original_unchanged(tmp_path):
    ocr, layout = page_pair(tmp_path)
    before = (ocr / 'page_0041.json').read_bytes()
    model = enrich(ocr, layout, 40, 41, {'pages': {'41': {'corrections': [
        {'scope': 'footnote', 'marker': None, 'before': 'continued.', 'after': 'continued here.'}
    ]}}})
    assert model['footnotes'][1]['text'].endswith('continued here.')
    assert model['corrections'][0]['scope'] == 'footnote'
    assert (ocr / 'page_0041.json').read_bytes() == before


def test_styles_preserve_line_breaks_and_every_character():
    text = 'Birds [Ectopistes\nmigratorius] and North-\nwest.'
    runs = styled_runs(text, ['Ectopistes migratorius'])
    assert ''.join(r['text'] for r in runs) == text
    assert [r['text'] for r in runs if r['italic']] == ['Ectopistes\nmigratorius']


def test_mixed_case_part_label_is_a_centred_heading():
    layout={'dpi':200,'metrics':{'prose_left_px':100,'body_letter_height_px':28},
            'regions':[{'role':'body','lines':[]}]}
    blocks=split_blocks('Part I.\nNarrative text.',layout,[])
    assert blocks[0]['kind']=='heading'
    assert blocks[0]['text']=='Part I.'
    assert blocks[1]['kind']=='paragraph'


def test_measured_indent_starts_paragraph_but_page_continuation_is_flush():
    text='Continued from preceding page.\nA new paragraph begins here.\nIts second line continues.'
    lines=[{'text':line,'bbox':[left,100+i*40,1000,130+i*40]}
           for i,(line,left) in enumerate(zip(text.splitlines(),[100,140,100]))]
    layout={'dpi':200,'metrics':{'prose_left_px':100,'body_letter_height_px':30},
            'regions':[{'role':'body','lines':lines}]}
    blocks=split_blocks(text,layout,[])
    assert len(blocks)==2
    assert blocks[0]['first_line_indent_pt']==0
    assert blocks[1]['first_line_indent_pt']==pytest.approx(14.4)
    assert blocks[1]['text'].endswith('Its second line continues.')


def test_drop_cap_offsets_do_not_split_opening_into_false_paragraphs():
    text='THE route from Grand Portage.\nwell known to require description.\nonly keep memoranda.'
    lines=[{'text':line,'bbox':[left,100+i*40,1000,130+i*40]}
           for i,(line,left) in enumerate(zip(text.splitlines(),[170,185,100]))]
    layout={'dpi':200,'metrics':{'prose_left_px':100,'body_letter_height_px':30},
            'regions':[{'role':'body','lines':lines}]}
    blocks=split_blocks(text,layout,[])
    assert len(blocks)==1 and blocks[0]['first_line_indent_pt']==0
    assert blocks[0]['text']==text


def test_narrower_text_frame_preserves_column_center_and_source_geometry():
    original={'width_pt':505,'left_pt':100,'right_pt':35}
    frame=text_frame(original,325)
    assert frame=={'width_pt':505,'left_pt':122.5,'right_pt':57.5}
    assert original=={'width_pt':505,'left_pt':100,'right_pt':35}
    assert (frame['left_pt']+505-frame['right_pt'])/2==285
    assert text_frame(original,400)==original


@pytest.mark.parametrize('width',[0,-1,float('inf'),float('nan')])
def test_invalid_text_width_rejected(width):
    with pytest.raises(ValueError,match='finite and positive'):
        text_frame({'width_pt':505,'left_pt':100,'right_pt':35},width)


@pytest.mark.parametrize('ending,next_kind,next_indent,continued',[
    ('to 1864 skins, mostly prime goods, and although I was','paragraph',0,True),
    ('The campaign ended.','paragraph',0,False),
    ('We then pro-','paragraph',0,True),
    ('An incomplete phrase','paragraph',14,False),
    ('An incomplete phrase','heading',0,False),
])
def test_only_continued_page_endings_receive_full_final_line(ending,next_kind,next_indent,continued):
    block={'kind':'paragraph','text':ending}
    pages=[{'blocks':[block]},{'blocks':[{'kind':next_kind,'text':'strongly opposed',
                                        'first_line_indent_pt':next_indent}]}]
    mark_page_continuations(pages)
    assert block['continues_on_next_page'] is continued
    assert block['text']==ending


def test_table_uses_final_quantity_not_weight_in_label():
    row = table_row('High wine, 9 gallons each, . . . 10 kegs')
    assert row['cells'] == ['High wine, 9 gallons each', '10', 'kegs']
    with pytest.raises(ValueError, match='no readable quantity'):
        table_row('Unreadable row')


@pytest.mark.parametrize('broken', [False, True])
def test_native_note_graph_audit_detects_dangling_word_reference(tmp_path, broken):
    path = tmp_path / 'structure.docx'
    ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('word/document.xml', f'<w:document xmlns:w="{ns}"><w:body>'
                         '<w:p><w:r><w:footnoteReference w:id="1"/></w:r></w:p>'
                         '</w:body></w:document>')
        archive.writestr('word/footnotes.xml', f'<w:footnotes xmlns:w="{ns}">'
                         f'<w:footnote w:id="{2 if broken else 1}"/></w:footnotes>')
    model = {'pages': [], 'footnotes': [{'id': 1, 'references': [{}]}]}
    if broken:
        with pytest.raises(ValueError, match='footnote graph'):
            audit_docx(path, model)
    else:
        assert audit_docx(path, model)['native_notes'] == 1


@pytest.mark.parametrize('defect',[None,'custom_mark','typed_number','missing_note_ref'])
def test_automatic_word_footnotes_require_real_marks_on_both_ends(tmp_path,defect):
    ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    path=tmp_path/'automatic.docx'
    custom=' w:customMarkFollows="1"' if defect=='custom_mark' else ''
    typed='<w:t>1</w:t>' if defect=='typed_number' else ''
    note_ref='' if defect=='missing_note_ref' else '<w:r><w:footnoteRef/></w:r>'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('word/document.xml',f'<w:document xmlns:w="{ns}"><w:body>'
                         f'<w:p><w:r><w:footnoteReference w:id="1"{custom}/>{typed}</w:r></w:p>'
                         '</w:body></w:document>')
        archive.writestr('word/footnotes.xml',f'<w:footnotes xmlns:w="{ns}">'
                         f'<w:footnote w:id="1"><w:p>{note_ref}</w:p></w:footnote></w:footnotes>')
    model={'pages':[],'footnotes':[{'id':1,'references':[{}]}],'footnote_numbering':'automatic'}
    if defect:
        with pytest.raises(ValueError,match='custom mark|automatic reference mark'):
            audit_docx(path,model)
    else:
        assert audit_docx(path,model)['native_notes']==1


@pytest.mark.parametrize('formatting',['preserved','flattened','dehyphenated'])
def test_native_footnote_audit_preserves_printed_compound_and_line_break(tmp_path,formatting):
    ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    path=tmp_path/'wrapping.docx'
    source='the usual first camping-\nground, at la Prairie or the Meadow.'
    content={
        'preserved':'<w:t> the usual first camping-</w:t><w:br/><w:t>ground, at la Prairie or the Meadow.</w:t>',
        'flattened':'<w:t> the usual first camping- ground, at la Prairie or the Meadow.</w:t>',
        'dehyphenated':'<w:t> the usual first campingground, at la Prairie or the Meadow.</w:t>',
    }[formatting]
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('word/document.xml',f'<w:document xmlns:w="{ns}"><w:body>'
                         '<w:p><w:r><w:footnoteReference w:id="1"/></w:r></w:p></w:body></w:document>')
        archive.writestr('word/footnotes.xml',f'<w:footnotes xmlns:w="{ns}">'
                         f'<w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r><w:r>{content}</w:r></w:p></w:footnote></w:footnotes>')
    model={'pages':[],'footnotes':[{'id':1,'references':[{}],'text':source}],
           'footnote_numbering':'automatic','footnote_line_breaks':'preserved'}
    if formatting=='preserved':
        assert audit_docx(path,model)['native_notes']==1
    else:
        with pytest.raises(ValueError,match='line breaks or hyphens'):
            audit_docx(path,model)


@pytest.mark.parametrize('missing_rule',[False,True])
def test_native_total_rule_and_ditto_cells_are_audited(tmp_path,missing_rule):
    ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    path=tmp_path/'table.docx'
    rows=[{'cells':['Loss','364','"']},
          {'cells':['Total cost','1864','skins'],'top_rule_columns':[1,2]}]
    native=[]
    for row in rows:
        cells=[]
        for index,text in enumerate(row['cells']):
            border=('<w:tcBorders><w:top w:val="single"/></w:tcBorders>'
                    if index in row.get('top_rule_columns',[]) and not missing_rule else '')
            cells.append(f'<w:tc><w:tcPr>{border}</w:tcPr><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>')
        native.append('<w:tr>'+''.join(cells)+'</w:tr>')
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('word/document.xml',f'<w:document xmlns:w="{ns}"><w:body><w:tbl>'+''.join(native)+'</w:tbl></w:body></w:document>')
        archive.writestr('word/footnotes.xml',f'<w:footnotes xmlns:w="{ns}"/>')
    model={'pages':[{'printed_page_number':None,'blocks':[{'kind':'table','rows':rows}]}],'footnotes':[]}
    if missing_rule:
        with pytest.raises(ValueError,match='table ruling'):audit_docx(path,model)
    else:
        assert audit_docx(path,model)['native_tables']==1
