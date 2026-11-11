#!/usr/bin/env python3
"""Enrich OCR with layout/style provenance and build an editable DOCX.

Run document authoring with the Codex document runtime (python-docx/lxml).
Missing pages and unresolved footnote links fail closed; OCR remains unverified.
"""
import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from .run_paths import create_run_directory
import zipfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SUPERS = '⁰¹²³⁴⁵⁶⁷⁸⁹'
REF_RE = re.compile(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+|\[\^(\d+)\]|\^(\d+)')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_export(source, destination):
    """Atomically refresh a stable filename; the original run remains retained."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        shutil.copyfile(source, staging)
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)


def normalize_docx(source, soffice):
    """Round-trip through LibreOffice for Word interoperability; retain raw input."""
    destination = source.parent / 'normalized'
    destination.mkdir(exist_ok=True)
    profile = source.parent / 'lo-profile'
    result = subprocess.run([str(soffice), '-env:UserInstallation=' + profile.resolve().as_uri(),
                             '--headless', '--norestore', '--convert-to', 'docx',
                             '--outdir', str(destination.resolve()), str(source.resolve())],
                            capture_output=True, text=True, timeout=180)
    (source.parent / 'normalization.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    normalized = destination / source.name
    if result.returncode or not normalized.is_file():
        raise ValueError('Word compatibility normalization failed; see normalization.log')
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    def contents(path):
        with zipfile.ZipFile(path) as archive:
            roots = [ET.fromstring(archive.read(name)) for name in ('word/document.xml', 'word/footnotes.xml')]
        def text(root):
            return ''.join(item.text or '' if item.tag=='{'+ns['w']+'}t' else '\xad'
                           for item in root.iter() if item.tag in
                           ('{'+ns['w']+'}t','{'+ns['w']+'}softHyphen'))
        notes=[]
        for note in roots[1]:
            if note.get('{'+ns['w']+'}type'):continue
            first=note.find('w:p/w:r',ns)
            if first is not None:
                style=first.find('w:rPr/w:rStyle',ns)
                # LibreOffice writes the generated custom mark as a styled
                # text run. Exclude only that leading mark, never note prose.
                if style is not None and style.get('{'+ns['w']+'}val')=='FootnoteCharacters':
                    note.find('w:p',ns).remove(first)
            notes.append(' '.join(text(note).split()))
        return (' '.join(text(roots[0]).split()),notes,
                len(roots[0].findall('.//w:footnoteReference', ns)))
    if contents(source) != contents(normalized):
        raise ValueError('Normalization changed body/note text or native footnote reference count')
    raw = source.parent / 'raw'
    raw.mkdir()
    shutil.copyfile(source, raw / source.name)
    shutil.copyfile(normalized, source)
    return {'status': 'passed', 'method': 'LibreOffice DOCX round-trip',
            'raw_sha256': digest(raw / source.name), 'normalized_sha256': digest(source)}


def normal(text):
    return ' '.join(text.casefold().split())


def export_plain_text(ocr_dir, output_dir, start, end, available_only=False):
    """Export saved OCR verbatim, without layout dependencies or inferred links."""
    pages, missing = [], []
    for number in range(start, end + 1):
        path = ocr_dir / f'page_{number:04d}.json'
        if not path.is_file():
            missing.append(number)
            continue
        source = path.read_bytes()
        page = json.loads(source)
        if page.get('pdf_page') != number or not isinstance(page.get('body'), str):
            raise ValueError(f'Invalid OCR identity/body: {path}')
        notes = page.get('footnotes')
        headers = page.get('running_headers', [])
        if (not isinstance(notes, list)
                or any(not isinstance(n, dict) or not isinstance(n.get('text'), str) for n in notes)
                or not isinstance(headers, list) or any(not isinstance(h, str) for h in headers)):
            raise ValueError(f'Invalid OCR footnotes/headers: {path}')
        pages.append((page, path, hashlib.sha256(source).hexdigest()))
    if missing and not available_only:
        raise ValueError('Missing OCR pages: ' + ', '.join(map(str, missing)))
    if not pages:
        raise ValueError('No OCR pages available in selected range')

    chunks, offset, evidence = [], 0, []

    def append(text):
        nonlocal offset
        begin = offset
        chunks.append(text)
        offset += len(text)
        return [begin, offset]

    append('New Light on the Early History of the Greater Northwest — Volume I\n'
           'Combined OCR transcription — unverified.\n'
           'Original line breaks, spelling and page-local footnote segments are retained.\n'
           'Tables retain OCR text; plain text does not reproduce fonts or native footnotes.\n')
    if missing:
        append('INCOMPLETE: missing PDF pages ' + ', '.join(map(str, missing)) + '\n')
    for page, path, checksum in pages:
        number = page['pdf_page']
        printed = page.get('printed_page_number')
        append(f'\n\n=== PDF page {number} | printed page {printed if printed is not None else "unknown"} ===\n')
        item = {'pdf_page': number, 'source': str(path.resolve()), 'source_sha256': checksum,
                'run_id': page.get('run_id'), 'body': None, 'footnotes': [], 'running_headers': []}
        if page.get('running_headers'):
            append('\n[Running headers]\n')
            for header in page['running_headers']:
                item['running_headers'].append(append(header))
                append('\n')
        append('\n[Body]\n')
        item['body'] = append(page['body'])
        append('\n')
        for index, note in enumerate(page['footnotes']):
            label = 'unnumbered' if note.get('marker') is None else str(note['marker'])
            append(f'\n[Footnote {label}; source segment {index + 1}]\n')
            span = append(note['text'])
            append('\n')
            item['footnotes'].append({'text': span, 'marker': note.get('marker'),
                                     'continues_from_previous_page': note.get('continues_from_previous_page'),
                                     'continues_on_next_page': note.get('continues_on_next_page')})
        evidence.append(item)
    text = ''.join(chunks)
    # Compare all content spans before writing; offsets are Unicode characters.
    for (page, _, _), item in zip(pages, evidence):
        assert text[slice(*item['body'])] == page['body']
        for note, saved in zip(page['footnotes'], item['footnotes']):
            assert text[slice(*saved['text'])] == note['text']
    run = create_run_directory(output_dir)
    output = run / f'newlight-pages-{start}-{end}-{run.name}.txt'
    output.write_bytes(text.encode('utf-8'))
    audit = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_range': [start, end],
             'source_pages': len(pages), 'missing_pages': missing, 'complete': not missing,
             'source_ocr_status': 'unverified; text copied without corrections or inferred note links',
             'footnote_segments': sum(len(p['footnotes']) for p, _, _ in pages),
             'body_characters': sum(len(p['body']) for p, _, _ in pages),
             'output': str(output.resolve()), 'text_sha256': digest(output),
             'code_sha256': digest(Path(__file__)), 'offset_unit': 'Unicode characters',
             'pages': evidence}
    (run / 'text-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return output


def marker_number(value):
    if value is None:
        return None
    text = value.translate(str.maketrans(SUPERS,'0123456789'))
    digits = re.sub(r'[^0-9]','',text)
    if not digits:
        raise ValueError(f'Non-numeric footnote marker {value!r}')
    return digits


def styled_runs(text, phrases):
    boundaries = {0,len(text)}
    intervals = []
    for phrase in phrases:
        pattern = re.escape(phrase).replace(r'\ ',r'\s+')
        for match in re.finditer(pattern,text,re.IGNORECASE):
            intervals.append((match.start(),match.end()))
            boundaries.update((match.start(),match.end()))
    edges = sorted(boundaries)
    return [{'text':text[a:b],'italic':any(x<=a and b<=y for x,y in intervals)}
            for a,b in zip(edges,edges[1:]) if a<b]


def table_row(text):
    matches = list(re.finditer(r'\d+(?:[½¼¾])?',text))
    if not matches:
        raise ValueError(f'Table row has no readable quantity: {text!r}')
    match = matches[-1]
    label = re.sub(r'[\s.,:·]+$','',text[:match.start()])
    return {'cells':[label,match.group(),text[match.end():].strip()], 'source_text':text}


def paragraph_indent(text, layout, lines):
    """Measure first-line indentation, leaving page continuations flush left."""
    best=max(lines,key=lambda line:SequenceMatcher(None,normal(line['text']),normal(text)).ratio(),default=None)
    if not best or SequenceMatcher(None,normal(best['text']),normal(text)).ratio()<=.6:
        return 0.0
    inset=best['bbox'][0]-layout['metrics']['prose_left_px']
    height=layout['metrics']['body_letter_height_px']
    return inset*72/layout['dpi'] if height*.6<inset<height*1.7 else 0.0


def text_frame(geometry, target_width):
    """Inset both margins equally; retain original scan geometry in the model."""
    if not math.isfinite(target_width) or target_width<=0:
        raise ValueError('Text width must be finite and positive')
    frame=dict(geometry)
    available=frame['width_pt']-frame['left_pt']-frame['right_pt']
    inset=max(0,(available-target_width)/2)
    frame['left_pt']+=inset;frame['right_pt']+=inset
    return frame


def split_blocks(text, layout, italics):
    """Keep source line strings; use table-region labels to select native rows."""
    tables = [r for r in layout['regions'] if r['role']=='table']
    source_lines = [line for r in layout['regions'] if r['role']=='body' for line in r['lines']]
    pending = text.splitlines()
    blocks, prose = [], []
    def flush():
        if prose:
            value='\n'.join(prose)
            blocks.append({'kind':'paragraph','text':value,'runs':styled_runs(value,italics),
                           'first_line_indent_pt':paragraph_indent(prose[0],layout,source_lines)})
            prose.clear()
    index=0
    while index<len(pending):
        line=pending[index]
        matched=None
        for table in tables:
            if not table['lines']:continue
            prefix=' '.join(table['lines'][0]['text'].split()[:3])
            if normal(line).startswith(normal(prefix)):
                matched=table;break
        if matched:
            flush();rows=[]
            while index<len(pending) and len(rows)<len(matched['lines']):
                if pending[index].strip():
                    try:row=table_row(pending[index])
                    except ValueError:
                        # Preserve headers and ambiguous OCR quantities verbatim.
                        row={'cells':[pending[index],'',''],'source_text':pending[index],
                             'quantity_unresolved':True}
                    rows.append(row)
                index+=1
            if len(rows)!=len(matched['lines']):raise ValueError('Incomplete table rows')
            blocks.append({'kind':'table','rows':rows,'bbox_px':matched['bbox_px']})
            continue
        if not line.strip():flush();index+=1;continue
        if re.fullmatch(r'CHAPTER\s+[IVXLCDM]+\.',line):
            flush();blocks.append({'kind':'heading','level':1,'text':line,'runs':styled_runs(line,italics)})
        elif re.fullmatch(r'PART\s+[IVXLCDM]+\.',line,re.IGNORECASE):
            flush();blocks.append({'kind':'heading','level':2,'text':line,'runs':styled_runs(line,italics)})
        elif line.isupper() and len(line)>8:
            flush();blocks.append({'kind':'heading','level':2,'text':line,'runs':styled_runs(line,italics)})
        else:
            # A visibly indented new narrative line can start a paragraph.
            best=max(source_lines,key=lambda x:SequenceMatcher(None,normal(x['text']),normal(line)).ratio(),default=None)
            if best and prose:
                score=SequenceMatcher(None,normal(best['text']),normal(line)).ratio()
                inset=best['bbox'][0]-layout['metrics']['prose_left_px']
                if paragraph_indent(line,layout,source_lines)>0:flush()
            prose.append(line)
        index+=1
    flush()
    return blocks


def mark_page_continuations(pages):
    """Identify unfinished prose at a source-page boundary; never alter text."""
    for index,page in enumerate(pages):
        if not page['blocks'] or page['blocks'][-1]['kind']!='paragraph':continue
        block=page['blocks'][-1]
        tail=REF_RE.sub('',block['text']).rstrip().rstrip('”"’]')
        unfinished=bool(tail) and (tail[-1].isalpha() or tail.endswith('-'))
        if index+1<len(pages):
            following=pages[index+1]['blocks']
            unfinished=unfinished and bool(following) and following[0]['kind']=='paragraph' and not following[0].get('first_line_indent_pt',0)
        block['continues_on_next_page']=bool(unfinished)


def enrich(ocr_dir, renderer_run, start, end, annotations=None, available_only=False):
    annotations=annotations or {}
    missing=[n for n in range(start,end+1) if not (ocr_dir/f'page_{n:04d}.json').is_file()]
    if missing and not available_only:raise ValueError(f'Missing OCR pages: {missing}; full-book export requires all selected pages')
    pages=[];notes=[];active=None;chapter=0;by_key={};by_page={};audit=[];issues=[]
    for number in range(start,end+1):
        if number in missing:continue
        source=ocr_dir/f'page_{number:04d}.json'
        layout_path=renderer_run/f'page_{number:04d}.layout.json'
        raw=json.loads(source.read_text());layout=json.loads(layout_path.read_text())
        if raw['pdf_page']!=number or layout['pdf_page']!=number:raise ValueError('Page identity mismatch')
        if raw.get('source_sha256')!=layout['image_sha256']:raise ValueError(f'Page {number}: image provenance mismatch')
        if raw.get('source_review'):
            reviewed=ocr_dir/'runs'/raw['run_id']/f'page_{number:04d}.layout.json'
            if not reviewed.is_file():raise ValueError(f'Page {number}: reviewed layout snapshot missing')
            effective=json.loads(reviewed.read_text())
            if effective['pdf_page']!=number or effective['image_sha256']!=layout['image_sha256']:
                raise ValueError(f'Page {number}: reviewed layout provenance mismatch')
            layout_path=reviewed;layout=effective
        data=json.loads(json.dumps(raw));config=annotations.get('pages',{}).get(str(number),{})
        if available_only and str(data['printed_page_number'])!=str(number-34):
            issues.append({'pdf_page':number,'kind':'printed_page_number_normalized',
                           'original':data['printed_page_number'],'value':str(number-34),
                           'basis':'Book pagination: PDF35 is printed1; fixed 34-page front matter offset'})
            data['printed_page_number']=str(number-34)
        for fix in config.get('corrections',[]):
            scope=fix.get('scope','body')
            if scope not in ('body','running_headers','footnote'):raise ValueError('Unsupported correction scope')
            target=None
            if scope=='footnote':
                candidates=[n for n in data['footnotes'] if marker_number(n['marker'])==fix.get('marker')]
                if len(candidates)!=1:raise ValueError('Correction must identify one footnote')
                target=candidates[0];original=target['text']
            else:original=data[scope] if scope=='body' else '\n'.join(data[scope])
            if fix['before'] in original:
                changed=original.replace(fix['before'],fix['after'],-1 if fix.get('all') else 1)
                if scope=='footnote':target['text']=changed
                else:data[scope]=changed if scope=='body' else changed.split('\n')
                audit.append({'pdf_page':number,**fix,'basis':'source-image inspection supplied in annotations'})
        if re.search(r'(?m)^CHAPTER\s+[IVXLCDM]+\.',data['body']):chapter+=1
        for note in data['footnotes']:
            marker=marker_number(note['marker'])
            if marker is None:
                if active is None or active['chapter']!=chapter or active['parts'][-1]['pdf_page'] not in (number-1,number):
                    if not available_only:raise ValueError(f'Page {number}: continuation has no adjacent preceding note in this chapter')
                    active={'id':len(notes)+1,'chapter':chapter,'marker':None,'text':note['text'],
                            'parts':[{'pdf_page':number,'text':note['text']}],'references':[]}
                    notes.append(active);issues.append({'pdf_page':number,'kind':'orphan_note_continuation'})
                    continue
                active['parts'].append({'pdf_page':number,'text':note['text']})
                active['text']+='\n'+note['text']
            else:
                key=(chapter,marker)
                if key in by_key:
                    if not available_only:raise ValueError(f'Duplicate note marker in chapter: {key}')
                    issues.append({'pdf_page':number,'kind':'duplicate_note_marker','marker':marker})
                active={'id':len(notes)+1,'chapter':chapter,'marker':marker,'text':note['text'],
                        'parts':[{'pdf_page':number,'text':note['text']}],'references':[]}
                notes.append(active);by_key[key]=active;by_page[(number,marker)]=active
        try:blocks=split_blocks(data['body'],layout,config.get('italics',[]))
        except ValueError as error:
            if not available_only:raise
            issues.append({'pdf_page':number,'kind':'table_layout_unresolved','detail':str(error)})
            blocks=[{'kind':'paragraph','text':data['body'],'first_line_indent_pt':0}]
        for rule in config.get('table_rules',[]):
            matches=[row for block in blocks if block['kind']=='table' for row in block['rows']
                     if normal(row['cells'][0])==normal(rule['row_label'])]
            columns=rule['top_rule_columns']
            if len(matches)!=1 or not columns or any(type(c) is not int or c not in range(3) for c in columns):
                raise ValueError(f'Page {number}: table rule must identify one row and valid columns')
            matches[0]['top_rule_columns']=sorted(set(columns))
        scale=72/layout['dpi'];content=[r for r in layout['regions'] if r['role'] in ('body','table','footnote')]
        left=min(r['bbox_px'][0] for r in content)*scale
        right=max(r['bbox_px'][2] for r in content)*scale
        body_regions=[r for r in content if r['role'] in ('body','table')]
        top=min(r['bbox_px'][1] for r in body_regions)*scale
        bottom=max(r['bbox_px'][3] for r in content)*scale
        rect=layout['source_page_rect'];width=rect[2]-rect[0];height=rect[3]-rect[1]
        pages.append({'pdf_page':number,'chapter':chapter,'printed_page_number':data['printed_page_number'],
                      'running_headers':data['running_headers'],'blocks':blocks,'source_body':raw['body'],
                      'geometry':{'dpi':layout['dpi'],'width_pt':width,'height_pt':height,'left_pt':left,'right_pt':width-right,
                                  'top_pt':top,'bottom_pt':max(25,height-bottom)},
                      'italic_phrases':config.get('italics',[]),'note_italics':config.get('note_italics',[]),
                      'provenance':{'ocr_path':str(source.resolve()),'ocr_sha256':digest(source),
                                    'ocr_model':raw.get('model'),'ocr_run_id':raw.get('run_id'),
                                    'layout_path':str(layout_path.resolve()),'layout_sha256':digest(layout_path),
                                    'source_image_sha256':layout['image_sha256']},
                      'review_required':True,'ocr_warnings':data.get('validation_warnings',[])})
    mark_page_continuations(pages)
    # Resolve actual visible body markers against the logical merged notes.
    for page in pages:
        for block in page['blocks']:
            if block['kind']=='table':continue
            for match in REF_RE.finditer(block['text']):
                marker=marker_number(match.group());key=(page['chapter'],marker)
                linked=by_page.get((page['pdf_page'],marker)) if available_only else by_key.get(key)
                if linked is None:
                    if not available_only:raise ValueError(f"PDF{page['pdf_page']}: body reference {marker} has no note")
                    issues.append({'pdf_page':page['pdf_page'],'kind':'unresolved_body_reference','marker':marker})
                    continue
                if available_only and linked['references']:
                    issues.append({'pdf_page':page['pdf_page'],'kind':'repeated_body_reference_retained_as_text','marker':marker})
                    continue
                linked['references'].append({'pdf_page':page['pdf_page'],'marker':marker})
    unresolved=[n['marker'] for n in notes if not n['references']]
    if unresolved and not available_only:raise ValueError(f'Footnotes without body references: {unresolved}')
    if available_only:
        for note in notes:
            if not note['references']:
                note['anchor_at_page_start']=True
                note['references']=[{'pdf_page':note['parts'][0]['pdf_page'],'marker':note['marker'],'editorial_anchor':True}]
                issues.append({'pdf_page':note['parts'][0]['pdf_page'],'kind':'note_reference_unlocated','marker':note['marker']})
    return {'schema_version':1,'pages':pages,'footnotes':notes,'corrections':audit,
            'style_basis':'Source geometry estimates; italic phrases supplied by reviewed annotations',
            'font_family':'Times New Roman','font_basis':'Approximate substitute; historical typeface not identified',
            'body_size_pt':13.0,'note_size_pt':9.5,'note_line_height_pt':10.5,'text_width_pt':325.0,
            'footnote_numbering':'source_custom' if available_only else 'automatic',
            'available_only':available_only,'requested_range':[start,end],'missing_pages':missing,'review_issues':issues,
            'footnote_line_breaks':'preserved',
            'text_status':'unverified OCR; selected source-backed corrections recorded',
            'limitations':['Italic coverage depends on annotations','Word footnotes may reflow and change pagination',
                           'Table columns inferred from printed rows; review against source',
                           'Footnote line breaks and hyphens preserved; native notes may flow across different pages']}


def source_page_footnotes(model):
    """Create native segments in anchor order while retaining logical-note IDs.

    Unnumbered continuations use a custom zero-width mark, so they do not
    consume a printed footnote number. Keep IDs and XML order in anchor order
    for Word/LibreOffice interoperability.
    """
    segments=[]
    for page in model['pages']:
        for note in model['footnotes']:
            for index,part in enumerate(note['parts']):
                if part['pdf_page']!=page['pdf_page']:continue
                segments.append({**note,'id':len(segments)+1,'logical_id':note['id'],
                                 'text':part['text'],'parts':[part],
                                 'continuation':index>0,
                                 'continues_on_next_page':index<len(note['parts'])-1,
                                 'references':([{'pdf_page':part['pdf_page'],'continuation':True}]
                                               if index else note['references'])})
    if model.get('available_only'):
        page_order={p['pdf_page']:i for i,p in enumerate(model['pages'])}
        positions={}
        for page in model['pages']:
            text='\n'.join(b.get('text','') for b in page['blocks'] if b['kind']!='table')
            for match in REF_RE.finditer(text):
                positions.setdefault((page['pdf_page'],marker_number(match.group())),match.start())
        segments.sort(key=lambda n:(page_order[n['parts'][0]['pdf_page']],
            -1 if n.get('continuation') or n.get('anchor_at_page_start') else
            positions.get((n['parts'][0]['pdf_page'],n['marker']),len(text))))
        for index,note in enumerate(segments,1):note['id']=index
    return segments


def reconstruction_properties(model, run_id):
    first,last=model['pages'][0],model['pages'][-1]
    values={'Project':'BrownRook Newlight','PreparedBy':'Paul La Rocque',
            'SourceAuthors':'Alexander Henry; David Thompson','SourceEditor':'Elliott Coues',
            'SourcePublisher':'Francis P. Harper, New York','SourcePublicationYear':1897,
            'SourceVolume':1,'SourceURL':'https://archive.org/details/newlightonearlyh01henr',
            'SourcePDF':'newlightonearlyh01henr.pdf',
            'PDFPageStart':first['pdf_page'],'PDFPageEnd':last['pdf_page'],
            'PrintedPageStart':str(first['printed_page_number']),'PrintedPageEnd':str(last['printed_page_number']),
            'SourcePageCount':len(model['pages']),'ReconstructionRun':run_id,
            'DocumentLanguage':'en-CA','ReviewRequired':True,
            'TextStatus':model['text_status'],'FontFamily':model['font_family'],
            'BodyFontSizePt':model['body_size_pt'],'FootnoteFontSizePt':model['note_size_pt'],
            'TextWidthPt':model['text_width_pt'],'FootnotePagination':model.get('footnote_pagination','flow'),
            'LogicalFootnoteCount':len(model['footnotes']),
            'NativeFootnoteSegmentCount':len(model.get('footnote_segments',model['footnotes'])),
            'ExporterSHA256':digest(Path(__file__))}
    values['MissingPDFPages']=', '.join(map(str,model.get('missing_pages',[]))) or 'None'
    values['AvailablePagesOnly']=model.get('available_only',False)
    for field,key in [('OCRModels','ocr_model'),('OCRRuns','ocr_run_id')]:
        values[field]='; '.join(sorted({p['provenance'][key] for p in model['pages'] if p['provenance'].get(key)})) or 'Unrecorded'
    values['RendererRuns']='; '.join(sorted({Path(p['provenance']['layout_path']).parent.name for p in model['pages']}))
    for field,key in [('OCRInputSetSHA256','ocr_sha256'),('LayoutInputSetSHA256','layout_sha256')]:
        values[field]=hashlib.sha256(json.dumps([(p['pdf_page'],p['provenance'][key]) for p in model['pages']],separators=(',',':')).encode()).hexdigest()
    source_pdf=ROOT/'data/external'/values['SourcePDF']
    if source_pdf.is_file():values['SourcePDFSHA256']=digest(source_pdf)
    return values


def custom_properties_xml(values):
    """Serialize typed, field-addressable native Word custom properties."""
    cp='http://schemas.openxmlformats.org/officeDocument/2006/custom-properties'
    vt='http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'
    root=ET.Element('{'+cp+'}Properties')
    for pid,(name,value) in enumerate(values.items(),2):
        prop=ET.SubElement(root,'{'+cp+'}property',
                           {'fmtid':'{D5CDD505-2E9C-101B-9397-08002B2CF9AE}','pid':str(pid),'name':name})
        tag='bool' if isinstance(value,bool) else 'i4' if isinstance(value,int) else 'r8' if isinstance(value,float) else 'lpwstr'
        ET.SubElement(prop,'{'+vt+'}'+tag).text=('true' if value else 'false') if tag=='bool' else str(value)
    return ET.tostring(root,encoding='UTF-8',xml_declaration=True)


def justify_footnote_last_line(note):
    """A full stop ends a naturally spaced line, including page segments."""
    return bool(note.get('continues_on_next_page') and not note['text'].rstrip().endswith('.'))


def build_docx(model, output):
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.section import WD_SECTION_START
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.opc.part import Part
    from docx.opc.packuri import PackURI
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from lxml import etree
    document=Document()
    if model.get('custom_properties'):
        custom_part=Part(PackURI('/docProps/custom.xml'),
                         'application/vnd.openxmlformats-officedocument.custom-properties+xml',
                         custom_properties_xml(model['custom_properties']),document.part.package)
        document.part.package.relate_to(custom_part,
            'http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties')
    props=document.core_properties
    props.author=props.last_modified_by='Paul La Rocque'
    props.title='New Light on the Early History of the Greater Northwest — Volume I'
    first,last=model['pages'][0],model['pages'][-1]
    props.subject=f"The Red River of the North — editable reconstruction, PDF pages {first['pdf_page']}–{last['pdf_page']}"
    props.category='Historical book reconstruction'
    props.keywords='Alexander Henry; David Thompson; Elliott Coues; Red River; Canada; historical journals; OCR'
    props.comments=(f"Journals of Alexander Henry and David Thompson; edited by Elliott Coues. "
                    f"Francis P. Harper, New York, 1897. "
                    f"PDF {first['pdf_page']}–{last['pdf_page']}; printed {first['printed_page_number']}–{last['printed_page_number']}. "
                    f"Editable OCR reconstruction for BrownRook; review against source scans required.")
    props.language='en-CA';props.content_status='Draft — OCR review required'
    props.created=props.modified=datetime.now(timezone.utc);props.revision=1
    # Proofing language belongs in Word styles as well as descriptive metadata.
    styles=document.styles.element
    default_rpr=styles.find(qn('w:docDefaults')).find(qn('w:rPrDefault')).find(qn('w:rPr'))
    lang=default_rpr.find(qn('w:lang'))
    if lang is None:lang=OxmlElement('w:lang');default_rpr.append(lang)
    lang.set(qn('w:val'),'en-CA')
    for lang in styles.iter(qn('w:lang')):lang.set(qn('w:val'),'en-CA')
    theme_lang=document.settings.element.find(qn('w:themeFontLang'))
    if theme_lang is not None:theme_lang.set(qn('w:val'),'en-CA')
    app_part=next((part for part in document.part.package.parts if str(part.partname)=='/docProps/app.xml'),None)
    if app_part is not None:
        app=etree.fromstring(app_part.blob)
        app_ns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'
        company=app.find('{'+app_ns+'}Company')
        if company is None:company=etree.SubElement(app,'{'+app_ns+'}Company')
        company.text='BrownRook'
        app_part._blob=etree.tostring(app,xml_declaration=True,encoding='UTF-8',standalone=True)
    for name in ['Normal','Title','Heading 1','Heading 2']:
        style=document.styles[name];style.font.name=model['font_family'];style.font.color.rgb=RGBColor(0,0,0)
        style.font.size=Pt(model['body_size_pt']);style.paragraph_format.space_after=Pt(0)
    for name in ['Heading 1','Heading 2']:
        document.styles[name].font.bold=False
        document.styles[name].font.size=Pt(15 if name=='Heading 1' else 13)
    document.styles['Normal'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    for name in ['Footnote Text','Footnote Reference']:
        if name not in document.styles:document.styles.add_style(name,1 if name=='Footnote Text' else 2)
        document.styles[name].font.name=model['font_family'];document.styles[name].font.size=Pt(model['note_size_pt'])
    document.styles['Footnote Reference'].font.superscript=True
    document.styles['Footnote Text'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    document.styles['Footnote Text'].paragraph_format.first_line_indent=Pt(model['note_size_pt'])
    settings_notes=OxmlElement('w:footnotePr')
    for tag,value in [('pos','pageBottom'),('numFmt','decimal'),
                      ('numStart',model['footnotes'][0]['marker'] if model['footnotes'] else '1'),
                      ('numRestart','continuous')]:
        item=OxmlElement('w:'+tag);item.set(qn('w:val'),value);settings_notes.append(item)
    for separator_id in (-1,0):
        item=OxmlElement('w:footnote');item.set(qn('w:id'),str(separator_id));settings_notes.append(item)
    document.settings.element.insert_element_before(settings_notes,'w:endnotePr','w:compat','w:docVars','w:rsids','m:mathPr')
    # Printed lines use manual breaks; expand their word spaces in justified
    # paragraphs, while each paragraph's final line remains naturally short.
    compat=document.settings.element.find(qn('w:compat'))
    if compat is not None:
        for item in list(compat):
            if item.tag==qn('w:doNotExpandShiftReturn'):compat.remove(item)
    notes_root=etree.Element(qn('w:footnotes'),nsmap={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
    for note_id,kind in [(-1,'separator'),(0,'continuationSeparator')]:
        note=etree.SubElement(notes_root,qn('w:footnote'));note.set(qn('w:id'),str(note_id));note.set(qn('w:type'),kind)
        etree.SubElement(note,qn('w:p'))
    native_notes=model.get('footnote_segments',model['footnotes'])
    by_key={(n['chapter'],n['marker']):n for n in native_notes if not n.get('continuation')}
    by_page={(ref['pdf_page'],n['marker']):n for n in native_notes if not n.get('continuation')
             for ref in n['references'] if not ref.get('editorial_anchor')}
    anchored=set()
    referenced=set()
    def custom_mark(paragraph,run,value):
        # A following run keeps the mark distinct from the native reference.
        # Some readers render same-run custom text twice.
        if model.get('available_only'):
            mark=paragraph.add_run(value);mark.style=document.styles['Footnote Reference']
            mark.font.name=model['font_family'];mark.font.size=run.font.size;mark.font.superscript=True
        else:run.add_text(value)
    def add_continuations(paragraph,page):
        if page['pdf_page'] in anchored:return
        anchored.add(page['pdf_page'])
        for note in native_notes:
            if not (note.get('continuation') or note.get('anchor_at_page_start')) or note['parts'][0]['pdf_page']!=page['pdf_page']:continue
            run=paragraph.add_run();run.style=document.styles['Footnote Reference']
            set_font(run,model['note_size_pt']);run.font.superscript=True
            ref=OxmlElement('w:footnoteReference');ref.set(qn('w:id'),str(note['id']))
            ref.set(qn('w:customMarkFollows'),'1');run._r.append(ref)
            custom_mark(paragraph,run,str(note['marker']) if note.get('marker') and not note.get('continuation') else '\u200b')
    def set_font(run,size,italic=False):
        run.font.name=model['font_family'];run.font.size=Pt(size);run.italic=italic
        run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),model['font_family'])
    def page_field(paragraph,value):
        field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
        r=OxmlElement('w:r');t=OxmlElement('w:t');t.text=str(value);r.append(t);field.append(r);paragraph._p.append(field)
    def add_text(paragraph,text,phrases,size,chapter,refs=True,pdf_page=None):
        for styled in styled_runs(text,phrases):
            cursor=0
            for match in REF_RE.finditer(styled['text']) if refs else []:
                if match.start()>cursor:set_font(paragraph.add_run(styled['text'][cursor:match.start()]),size,styled['italic'])
                marker=marker_number(match.group())
                note=by_page.get((pdf_page,marker)) if model.get('available_only') else by_key.get((chapter,marker))
                if note is None:
                    set_font(paragraph.add_run(match.group()),size,styled['italic']);cursor=match.end();continue
                if model.get('available_only') and note['id'] in referenced:
                    set_font(paragraph.add_run(match.group()),size,styled['italic']);cursor=match.end();continue
                referenced.add(note['id'])
                run=paragraph.add_run();run.style=document.styles['Footnote Reference']
                set_font(run,size);run.font.superscript=True
                ref=OxmlElement('w:footnoteReference');ref.set(qn('w:id'),str(note['id']))
                run._r.append(ref)
                if model.get('footnote_numbering')=='source_custom':
                    ref.set(qn('w:customMarkFollows'),'1');custom_mark(paragraph,run,marker)
                cursor=match.end()
            if cursor<len(styled['text']):set_font(paragraph.add_run(styled['text'][cursor:]),size,styled['italic'])
    if model.get('available_only') and model.get('missing_pages'):
        cover=document.sections[0]
        cover.page_width=Pt(612);cover.page_height=Pt(792)
        cover.top_margin=cover.bottom_margin=cover.left_margin=cover.right_margin=Pt(72)
        p=document.add_paragraph(style='Title')
        set_font(p.add_run('New Light on the Early History of the Greater Northwest'),22)
        p=document.add_paragraph();set_font(p.add_run('Volume I reconstruction draft'),15)
        for text in [
            f"This draft contains {len(model['pages'])} available OCR pages from the requested PDF page range "
            f"{model['requested_range'][0]}–{model['requested_range'][1]}. "
            f"{len(model['missing_pages'])} pages have no available OCR and are omitted. Source page numbers are retained.",
            'Alexander Henry and David Thompson; edited by Elliott Coues. Francis P. Harper, New York, 1897. Reconstruction prepared by Paul La Rocque for BrownRook.',
            'OCR text requires review against the scans. Located footnote references are retained. Notes whose original references are unresolved are attached at the start of their source page for review; unnumbered continuations remain unnumbered. Source-page boundaries are preserved where the recovered text fits.',
            'Missing PDF pages: '+', '.join(map(str,model['missing_pages']))+'.',
        ]:
            p=document.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before=Pt(12)
            for run in styled_runs(text,[]):set_font(p.add_run(run['text']),11)
        document.add_section(WD_SECTION_START.NEW_PAGE)
    for index,page in enumerate(model['pages']):
        section=document.sections[-1] if index==0 else document.add_section(WD_SECTION_START.NEW_PAGE)
        if model.get('available_only') and document.paragraphs:
            # The otherwise empty section-boundary paragraph must not create
            # a blank page after a densely filled source page.
            boundary=document.paragraphs[-1]
            if not boundary.text and boundary._p.find(qn('w:pPr')) is not None:
                previous=boundary._p.getprevious()
                sect=boundary._p.find(qn('w:pPr')).find(qn('w:sectPr'))
                if sect is not None and previous is not None and previous.tag==qn('w:p'):
                    boundary._p.find(qn('w:pPr')).remove(sect)
                    previous.get_or_add_pPr().append(sect)
                    boundary._p.getparent().remove(boundary._p)
                else:
                    boundary.paragraph_format.line_spacing=Pt(1)
                    boundary.paragraph_format.space_before=boundary.paragraph_format.space_after=Pt(0)
                    boundary.paragraph_format.keep_with_next=False
                    boundary.paragraph_format.keep_together=False
                    set_font(boundary.add_run(),1)
        source_g=page['geometry'];g=text_frame(source_g,model.get('text_width_pt',325.0))
        section.page_width=Pt(g['width_pt']);section.page_height=Pt(g['height_pt'])
        section.left_margin=Pt(g['left_pt']);section.right_margin=Pt(g['right_pt'])
        section.top_margin=Pt(max(60,g['top_pt']));section.bottom_margin=Pt(min(115,g['bottom_pt']))
        section.header_distance=Pt(max(25,min(90,g['top_pt']-32)));section.footer_distance=Pt(85)
        section.header.is_linked_to_previous=False;section.footer.is_linked_to_previous=False
        note_properties=OxmlElement('w:footnotePr')
        restart=index>0 and page['chapter']!=model['pages'][index-1]['chapter']
        if restart:
            first=next((n for n in model['footnotes'] if n['chapter']==page['chapter']),None)
            start=OxmlElement('w:numStart');start.set(qn('w:val'),(first['marker'] or '1') if first else '1')
            note_properties.append(start)
        numbering=OxmlElement('w:numRestart');numbering.set(qn('w:val'),'eachSect' if restart else 'continuous')
        note_properties.append(numbering)
        section._sectPr.insert_element_before(note_properties,'w:endnotePr','w:type','w:pgSz','w:pgMar')
        if page['printed_page_number'] is not None:
            pg=OxmlElement('w:pgNumType');pg.set(qn('w:start'),page['printed_page_number']);section._sectPr.append(pg)
        if page['running_headers']:
            p=section.header.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.LEFT
            width=g['width_pt']-g['left_pt']-g['right_pt']
            stops=p.paragraph_format.tab_stops;stops.clear_all()
            stops.add_tab_stop(Pt(width/2),WD_TAB_ALIGNMENT.CENTER)
            stops.add_tab_stop(Pt(width),WD_TAB_ALIGNMENT.RIGHT)
            if page['printed_page_number'] and int(page['printed_page_number'])%2==0:
                page_field(p,page['printed_page_number'])
            p.add_run('\t')
            set_font(p.add_run(' '.join(page['running_headers'])),9.5)
            if page['printed_page_number'] and int(page['printed_page_number'])%2:
                p.add_run('\t');page_field(p,page['printed_page_number'])
        elif page['printed_page_number']:
            p=section.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER;page_field(p,page['printed_page_number'])
        for block_index,block in enumerate(page['blocks']):
            if block['kind']=='table':
                table=document.add_table(rows=0,cols=3);table.autofit=False
                inset=max(0,block['bbox_px'][0]*72/g['dpi']-source_g['left_pt'])
                indent=OxmlElement('w:tblInd');indent.set(qn('w:w'),str(round(inset*20)));indent.set(qn('w:type'),'dxa');table._tbl.tblPr.append(indent)
                width=g['width_pt']-g['left_pt']-g['right_pt']-inset
                for col,fraction in zip(table.columns,[.76,.10,.14]):col.width=Pt(width*fraction)
                for row_index,row_data in enumerate(block['rows']):
                    cells=table.add_row().cells
                    for i,text in enumerate(row_data['cells']):
                        p=cells[i].paragraphs[0];p.paragraph_format.space_after=Pt(0)
                        if row_index==0 and i==0:add_continuations(p,page)
                        p.paragraph_format.line_spacing=Pt(12)
                        p.paragraph_format.keep_with_next=row_index<len(block['rows'])-1
                        p.paragraph_format.keep_together=True
                        if i==1:p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
                        else:p.alignment=WD_ALIGN_PARAGRAPH.LEFT
                        add_text(p,text,page['italic_phrases'],10,page['chapter'],False)
                        tcMar=OxmlElement('w:tcMar')
                        for edge in ['top','bottom','left','right']:
                            padding='80' if ((edge=='left' and i==2) or
                                            (edge=='top' and row_data.get('top_rule_columns'))) else '0'
                            e=OxmlElement('w:'+edge);e.set(qn('w:w'),padding);e.set(qn('w:type'),'dxa');tcMar.append(e)
                        cells[i]._tc.get_or_add_tcPr().append(tcMar)
                        if i in row_data.get('top_rule_columns',[]):
                            borders=OxmlElement('w:tcBorders');top=OxmlElement('w:top')
                            for attr,value in [('val','single'),('sz','6'),('color','000000')]:
                                top.set(qn('w:'+attr),value)
                            borders.append(top);cells[i]._tc.get_or_add_tcPr().append(borders)
                if block is not page['blocks'][-1]:
                    document.add_paragraph().paragraph_format.space_after=Pt(0)
                continue
            style=f"Heading {block['level']}" if block['kind']=='heading' else 'Normal'
            p=document.add_paragraph(style=style);p.paragraph_format.line_spacing=Pt(16)
            add_continuations(p,page)
            p.paragraph_format.widow_control=False
            p.paragraph_format.keep_together=False
            if block['kind']=='heading':
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(16)
            else:
                p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.first_line_indent=Pt(block.get('first_line_indent_pt',0))
                if (block['text'].rstrip().endswith(':') and block_index+1<len(page['blocks'])
                        and page['blocks'][block_index+1]['kind']=='table'):
                    # Preserve the blank printed line introducing an inset table.
                    p.paragraph_format.space_after=Pt(16)
            size=(15 if block.get('level')==1 else 13) if block['kind']=='heading' else model['body_size_pt']
            add_text(p,block['text'],page['italic_phrases'],size,page['chapter'],pdf_page=page['pdf_page'])
            if block.get('continues_on_next_page'):
                # Word otherwise treats the section boundary as a paragraph
                # ending and leaves its last printed line unjustified. The
                # layout-only soft break stretches that continued line too.
                p.add_run().add_break()
    for note in native_notes:
        node=etree.SubElement(notes_root,qn('w:footnote'));node.set(qn('w:id'),str(note['id']))
        phrases=[]
        for part in note['parts']:
            phrases.extend(next(p['note_italics'] for p in model['pages'] if p['pdf_page']==part['pdf_page']))
        paragraphs=note['text'].split('\n\n')
        for index,text in enumerate(paragraphs):
            p=OxmlElement('w:p');pPr=OxmlElement('w:pPr');style=OxmlElement('w:pStyle');style.set(qn('w:val'),'FootnoteText');pPr.append(style)
            spacing=OxmlElement('w:spacing');spacing.set(qn('w:line'),str(round(model.get('note_line_height_pt',10.5)*20)));spacing.set(qn('w:lineRule'),'exact');spacing.set(qn('w:after'),'0');pPr.append(spacing);p.append(pPr)
            from docx.text.paragraph import Paragraph
            wrapper=Paragraph(p,document._body)
            if note.get('continuation'):wrapper.paragraph_format.first_line_indent=Pt(0)
            if index==0:
                mark=wrapper.add_run();set_font(mark,9.5);mark.font.superscript=True
                mark.style=document.styles['Footnote Reference']
                mark._r.append(OxmlElement('w:footnoteRef'))
                if not note.get('continuation'):wrapper.add_run(' ')
            add_text(wrapper,text,phrases,9.5,note['chapter'],False)
            if justify_footnote_last_line(note) and index==len(paragraphs)-1:
                # A source-page segment is not a logical paragraph ending.
                # A layout-only soft break makes Word justify its final line.
                wrapper.add_run().add_break()
            node.append(p)
    part=Part(PackURI('/word/footnotes.xml'),'application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml',etree.tostring(notes_root,xml_declaration=True,encoding='UTF-8',standalone=True),document.part.package)
    document.part.relate_to(part,RT.FOOTNOTES)
    document.save(output)


def audit_docx(path, model):
    """Check the native structures independently of python-docx authoring."""
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    attr='{'+ns['w']+'}id'
    native_notes=model.get('footnote_segments',model['footnotes'])
    continuations={n['id'] for n in native_notes if n.get('continuation')}
    with zipfile.ZipFile(path) as archive:
        if model.get('custom_properties'):
            custom=ET.fromstring(archive.read('docProps/custom.xml'))
            actual={}
            pids=[]
            for prop in custom:
                value=prop[0];kind=value.tag.rsplit('}',1)[-1]
                actual[prop.get('name')]=(value.text=='true' if kind=='bool' else
                                          int(value.text) if kind=='i4' else
                                          float(value.text) if kind=='r8' else value.text or '')
                pids.append(prop.get('pid'))
            if actual!=model['custom_properties'] or len(pids)!=len(set(pids)) or len(actual)!=len(pids):
                raise ValueError('Custom Word properties differ from reconstruction metadata')
            relationships=ET.fromstring(archive.read('_rels/.rels'))
            if not any(r.get('Type','').endswith('/custom-properties') and
                       r.get('Target','').lstrip('/')=='docProps/custom.xml' for r in relationships):
                raise ValueError('Custom Word properties lack their package relationship')
        body=ET.fromstring(archive.read('word/document.xml'))
        notes=ET.fromstring(archive.read('word/footnotes.xml'))
        note_ids={int(n.get(attr)) for n in notes if not n.get('{'+ns['w']+'}type')}
        refs=[int(n.get(attr)) for n in body.findall('.//w:footnoteReference',ns)]
        expected_ids={n['id'] for n in native_notes}
        if note_ids!=expected_ids or set(refs)!=expected_ids:
            raise ValueError('Native Word footnote graph differs from enriched model')
        if len(refs)!=sum(len(n['references']) for n in native_notes):
            raise ValueError('Native Word reference count differs from source')
        if model.get('footnote_numbering')=='automatic':
            for run in body.findall('.//w:r',ns):
                reference=run.find('w:footnoteReference',ns)
                if reference is not None and int(reference.get(attr)) in continuations:
                    if (reference.get('{'+ns['w']+'}customMarkFollows')!='1'
                            or ''.join(t.text or '' for t in run.findall('w:t',ns))!='\u200b'):
                        raise ValueError('Continuation must use a zero-width custom native reference')
                    continue
                if reference is not None and (reference.get('{'+ns['w']+'}customMarkFollows') is not None
                                              or run.find('w:t',ns) is not None):
                    raise ValueError('Automatic Word footnote reference contains a fixed custom mark')
            for note in notes:
                if int(note.get(attr)) in expected_ids and len(note.findall('.//w:footnoteRef',ns))!=1:
                    raise ValueError('Footnote lacks its native automatic reference mark')
        if model.get('footnote_line_breaks')=='preserved':
            originals={n['id']:n['text'] for n in native_notes}
            continued={n['id'] for n in native_notes if justify_footnote_last_line(n)}
            for note in notes:
                if int(note.get(attr)) not in originals:continue
                paragraphs=[]
                for paragraph in note.findall('w:p',ns):
                    text=''.join((item.text or '') if item.tag=='{'+ns['w']+'}t' else '\n'
                                 for item in paragraph.iter() if item.tag in ('{'+ns['w']+'}t','{'+ns['w']+'}br'))
                    paragraphs.append(text)
                # The first space separates Word's automatic mark from text.
                actual='\n\n'.join(paragraphs)
                if int(note.get(attr)) not in continuations:actual=actual.removeprefix(' ')
                if int(note.get(attr)) in continued:
                    if not actual.endswith('\n'):
                        raise ValueError('Continued footnote lacks its final-line justification break')
                    actual=actual[:-1]
                if actual!=originals[int(note.get(attr))]:
                    raise ValueError('Footnote text, line breaks or hyphens differ from source')
        tables=body.findall('.//w:tbl',ns)
        expected_tables=[b for p in model['pages'] for b in p['blocks'] if b['kind']=='table']
        if len(tables)!=len(expected_tables):raise ValueError('Missing native table')
        native_by_id={n['id']:n for n in native_notes}
        def cell_text(cell):
            parts=[]
            for paragraph in cell.findall('.//w:p',ns):
                pending=None
                for run in paragraph.findall('w:r',ns):
                    text=''.join((item.text or '') if item.tag=='{'+ns['w']+'}t' else
                                 '\t' if item.tag=='{'+ns['w']+'}tab' else '\n'
                                 for item in run if item.tag in
                                 ('{'+ns['w']+'}t','{'+ns['w']+'}tab','{'+ns['w']+'}br'))
                    if pending is not None:
                        if text!=pending:raise ValueError('Custom footnote mark differs from source')
                        pending=None
                        continue
                    ref=run.find('w:footnoteReference',ns)
                    if ref is not None and ref.get('{'+ns['w']+'}customMarkFollows')=='1':
                        note=native_by_id[int(ref.get(attr))]
                        mark='\u200b' if note.get('continuation') or note.get('marker') is None else str(note['marker'])
                        if text:
                            if text!=mark:raise ValueError('Custom footnote mark differs from source')
                        else:pending=mark
                    else:parts.append(text)
                if pending is not None:raise ValueError('Custom footnote mark missing')
            return ''.join(parts)
        for native,expected in zip(tables,expected_tables):
            rows=native.findall('w:tr',ns)
            if len(rows)!=len(expected['rows']):raise ValueError('Missing native table row')
            for row,source in zip(rows,expected['rows']):
                cells=[cell_text(c) for c in row.findall('w:tc',ns)]
                if cells!=source['cells']:raise ValueError('Table cell content differs')
                for column,cell in enumerate(row.findall('w:tc',ns)):
                    top=cell.find('w:tcPr/w:tcBorders/w:top',ns)
                    has_rule=top is not None and top.get('{'+ns['w']+'}val')=='single'
                    if has_rule!=(column in source.get('top_rule_columns',[])):
                        raise ValueError('Native table ruling differs from reviewed source annotation')
        fields=sum(len(ET.fromstring(archive.read(name)).findall('.//w:fldSimple',ns))
                   for name in archive.namelist() if re.fullmatch(r'word/(header|footer)\d+\.xml',name))
        if fields!=sum(p['printed_page_number'] is not None for p in model['pages']):
            raise ValueError('Missing page-number field')
    return {'native_notes':len(note_ids),'body_references':len(refs),
            'native_tables':len(tables),'page_number_fields':fields,'status':'passed'}


def audit_footnote_pagination(model, rendered_page_texts):
    """Check continuation starts against physical source pages after rendering.

    Native-note XML integrity alone cannot prove pagination fidelity. Pass one
    extracted text string per rendered page, in the selected source-page order.
    """
    normalize=lambda text: ' '.join(text.split())
    texts=[normalize(text) for text in rendered_page_texts]
    issues=[]
    if len(texts)!=len(model['pages']):
        issues.append({'kind':'page_count','expected':len(model['pages']),'actual':len(texts)})
    positions={page['pdf_page']:i for i,page in enumerate(model['pages'])}
    for note in model['footnotes']:
        for part in note['parts'][1:]:
            anchor=' '.join(part['text'].split()[:12])
            actual=[i for i,text in enumerate(texts) if anchor in text]
            expected=positions[part['pdf_page']]
            if actual!=[expected]:
                issues.append({'kind':'footnote_continuation','marker':note['marker'],
                               'source_pdf_page':part['pdf_page'],'anchor':anchor,
                               'expected_rendered_page':expected+1,
                               'actual_rendered_pages':[i+1 for i in actual]})
    return {'status':'failed' if issues else 'passed','issues':issues}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr-dir',type=Path,required=True)
    parser.add_argument('--renderer-run',type=Path)
    parser.add_argument('--start',type=int,default=35);parser.add_argument('--end',type=int,default=480)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--output',type=Path,
                        help='Output directory (default: data/processed/text for --text-only, otherwise data/processed/word)')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--text-only',action='store_true',
                        help='Combine saved OCR into UTF-8 text and a provenance audit; no renderer or Word dependencies needed')
    modes.add_argument('--both',action='store_true',
                       help='Export TXT and Word together, with stable newlight.txt/newlight.docx under data/processed/book')
    parser.add_argument('--allow-review-issues',action='store_true',
                        help='Require all selected pages, but retain unresolved note links and table layouts for review')
    parser.add_argument('--normalize-with',type=Path,
                        help='LibreOffice soffice executable for Word compatibility normalization')
    parser.add_argument('--text-width',type=float,default=325,
                        help='Maximum Word text-column width in points; original scan geometry is retained (default: 325)')
    parser.add_argument('--enrich-only',action='store_true')
    parser.add_argument('--available-only',action='store_true',
                        help='Export existing OCR pages as an incomplete draft, retaining unresolved notes')
    parser.add_argument('--footnote-pagination',choices=['source','flow'],default='source',
                        help='Anchor continuation segments to source pages (default), or let whole native notes flow')
    args=parser.parse_args(argv)
    if args.start>args.end:parser.error('Require start <= end')
    if args.both and args.enrich_only:parser.error('--both cannot be combined with --enrich-only')
    if args.text_only:
        if args.annotations or args.enrich_only:
            parser.error('--text-only cannot be combined with --annotations or --enrich-only; exports preserve raw OCR')
        output=export_plain_text(args.ocr_dir,args.output or ROOT/'data/processed/text',
                                 args.start,args.end,args.available_only)
        publish_export(output,output.parent.parent.parent/'newlight.txt')
        print(output,flush=True)
        return
    if args.renderer_run is None:parser.error('--renderer-run is required unless --text-only is used')
    if args.output is None:args.output=ROOT/'data/processed'/('book' if args.both else 'word')
    if args.allow_review_issues and not args.available_only:
        missing=[n for n in range(args.start,args.end+1) if not (args.ocr_dir/f'page_{n:04d}.json').is_file()]
        if missing:parser.error('Missing OCR pages: '+', '.join(map(str,missing)))
    if not math.isfinite(args.text_width) or args.text_width<=0:parser.error('--text-width must be finite and positive')
    annotations=json.loads(args.annotations.read_text()) if args.annotations else {}
    model=enrich(args.ocr_dir,args.renderer_run,args.start,args.end,annotations,
                 args.available_only or args.allow_review_issues)
    if not model['pages']:parser.error('No OCR pages available in selected range')
    if model['available_only']:model['body_size_pt']=12.0
    model['text_width_pt']=args.text_width
    model['footnote_pagination']=args.footnote_pagination
    if args.footnote_pagination=='source':
        model['footnote_segments']=source_page_footnotes(model)
        model['limitations'].append('Source-page footnote continuations are separate unnumbered native segments linked by logical_id')
    run=create_run_directory(args.output)
    model['custom_properties']=reconstruction_properties(model,run.name)
    if args.annotations:model['custom_properties']['AnnotationsSHA256']=digest(args.annotations)
    (run/'book.enriched.json').write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
    if args.annotations:
        (run/'annotations.json').write_bytes(args.annotations.read_bytes())
    audit={'source_range':[args.start,args.end],'source_pages':len(model['pages']),'logical_footnotes':len(model['footnotes']),
           'tables':sum(b['kind']=='table' for p in model['pages'] for b in p['blocks']),
           'source_ocr_status':'unverified; selected checks only','limitations':model['limitations'],
           'corrections':model['corrections'],'text_width_pt':model['text_width_pt'],
           'code_sha256':digest(Path(__file__))}
    if args.annotations:audit['annotations_sha256']=digest(args.annotations)
    if not args.enrich_only:
        output=run/f'newlight-pages-{args.start}-{args.end}-{run.name}.docx';build_docx(model,output)
        audit['structure']=audit_docx(output,model)
        if args.normalize_with:audit['normalization']=normalize_docx(output,args.normalize_with)
        audit['docx_sha256']=digest(output);print(output,flush=True)
        publish_export(output,args.output/'newlight.docx')
    if args.both:
        text_output=export_plain_text(args.ocr_dir,args.output,args.start,args.end,args.available_only)
        publish_export(text_output,args.output/'newlight.txt')
        audit['text_output']=str(text_output.resolve())
        audit['text_sha256']=digest(text_output)
        print(args.output/'newlight.txt',flush=True)
    (run/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print('Enriched model:',run/'book.enriched.json',flush=True)


if __name__=='__main__':main()
