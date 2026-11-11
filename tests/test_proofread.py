"""Proofreader checks use synthetic responses; no live model calls in pytest."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from newlight import proofread as proof


def review(**changes):
    value={'coverage':'complete','checks_performed':sorted(proof.CHECKS),
           'findings':[],'uncertainties':[]}
    value.update(changes)
    return value


def fixture_run(tmp_path):
    candidate=tmp_path/'candidate.json';candidate.write_text('{"text":"Original text."}')
    image=tmp_path/'word.png';image.write_bytes(b'synthetic image')
    page={'pdf_page':480,'source_image':'unused','source_sha256':'unused',
          'width':100,'height':100,'bands':[[0,100]],
          'candidate':str(candidate),'candidate_sha256':proof.sha(candidate),
          'word_images':[{'path':str(image),'sha256':proof.sha(image)}],
          'mapping_status':'test mapping'}
    manifest={'pages':[page],'passes':1,'jobs':[{'id':'p0480-b01-r1','pdf_page':480,
              'band':0,'word_image':0,'pass':1,'status':'pending','attempts':0}]}
    return manifest


def fake_response(value,reason='stop'):
    data={'message':{'content':json.dumps(value)},'done_reason':reason}
    return SimpleNamespace(message=SimpleNamespace(content=data['message']['content']),
                           done_reason=reason,model_dump=lambda **kwargs:data)


def test_strip_plan_covers_every_pixel_and_bottom_without_gaps():
    plan=proof.bands(2159,800,100)
    assert plan[0][0]==0 and plan[-1][1]==2159
    assert all(a[1]>=b[0] for a,b in zip(plan,plan[1:]))
    assert all(0<bottom-top<=800 for top,bottom in plan)
    with pytest.raises(ValueError):proof.bands(100,100,100)


def test_reads_actual_word_sections_native_notes_and_table_text(tmp_path):
    path=tmp_path/'book.docx';w=proof.NS['w']
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('word/document.xml',f'''<w:document xmlns:w="{w}"><w:body>
          <w:p><w:pPr><w:sectPr/></w:pPr><w:r><w:t>First page.</w:t><w:footnoteReference w:id="7"/></w:r></w:p>
          <w:tbl><w:tr><w:tc><w:p><w:r><w:t>147</w:t><w:tab/><w:t>63</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
          <w:p><w:r><w:t>Second</w:t><w:br/><w:t>page.</w:t></w:r></w:p><w:sectPr/>
          </w:body></w:document>''')
        archive.writestr('word/footnotes.xml',f'<w:footnotes xmlns:w="{w}"><w:footnote w:id="7"><w:p><w:r><w:t>Real note.</w:t></w:r></w:p></w:footnote></w:footnotes>')
        archive.writestr('word/_rels/document.xml.rels','<Relationships/>')
    sections=proof.word_sections(path)
    assert len(sections)==2
    assert sections[0]['native_footnotes']==[{'id':'7','text':'Real note.'}]
    assert 'First page.' in sections[0]['blocks'][0]['text']
    assert sections[1]['blocks'][0]['rows']==[['147\t63']]
    assert sections[1]['blocks'][1]['text']=='Second\npage.'


def test_duplicate_unattended_runner_is_rejected(tmp_path):
    with proof.exclusive_lock(tmp_path):
        with pytest.raises(ValueError,match='already running'):
            with proof.exclusive_lock(tmp_path):pass


def test_complete_model_response_is_checkpointed_with_evidence(tmp_path,monkeypatch):
    manifest=fixture_run(tmp_path)
    monkeypatch.setattr(proof,'source_crop',lambda p,j,d:d.write_bytes(b'crop'))
    client=SimpleNamespace(chat=lambda **kwargs:fake_response(review()))
    proof.run_reviews(tmp_path,manifest,client,1,{'model':'test','num_ctx':16384})
    assert manifest['status']=='completed'
    assert manifest['jobs'][0]['status']=='done'
    assert proof.gate(tmp_path,manifest).startswith('No discrepancies detected')
    assert list(tmp_path.glob('*.request.json'))
    assert list(tmp_path.glob('*.response.json'))
    assert proof.statistics(manifest)['pages_done']==1


@pytest.mark.parametrize('value,reason',[
    (review(coverage='partial'),'stop'),
    (review(checks_performed=['wording']),'stop'),
    (review(),'length'),
])
def test_partial_or_truncated_response_never_counts_as_proofread(tmp_path,monkeypatch,value,reason):
    manifest=fixture_run(tmp_path)
    monkeypatch.setattr(proof,'source_crop',lambda p,j,d:d.write_bytes(b'crop'))
    client=SimpleNamespace(chat=lambda **kwargs:fake_response(value,reason))
    proof.run_reviews(tmp_path,manifest,client,1,{'model':'test','num_ctx':16384},max_attempts=1)
    assert manifest['jobs'][0]['status']=='failed'
    assert proof.statistics(manifest)['pages_done']==0
    with pytest.raises(ValueError,match='incomplete'):proof.gate(tmp_path,manifest)
    assert list(tmp_path.glob('*.error.json'))


@pytest.mark.parametrize('field,value',[
    ('uncertainties',['Unreadable numeral.']),
    ('findings',[{'category':'numbers','source_reading':'1 brisket','word_reading':'brisket',
                  'explanation':'Numeral omitted.','confidence':'high'}]),
])
def test_completed_review_with_unresolved_findings_fails_quality_gate(tmp_path,monkeypatch,field,value):
    manifest=fixture_run(tmp_path)
    monkeypatch.setattr(proof,'source_crop',lambda p,j,d:d.write_bytes(b'crop'))
    client=SimpleNamespace(chat=lambda **kwargs:fake_response(review(**{field:value})))
    proof.run_reviews(tmp_path,manifest,client,1,{'model':'test','num_ctx':16384})
    assert manifest['jobs'][0]['status']=='done'
    with pytest.raises(ValueError,match='Unresolved'):proof.gate(tmp_path,manifest)


def test_finished_reviews_are_not_repeated_and_changed_evidence_fails(tmp_path,monkeypatch):
    manifest=fixture_run(tmp_path)
    monkeypatch.setattr(proof,'source_crop',lambda p,j,d:d.write_bytes(b'crop'))
    calls=[]
    client=SimpleNamespace(chat=lambda **kwargs:(calls.append(kwargs) or fake_response(review())))
    settings={'model':'test','num_ctx':16384}
    proof.run_reviews(tmp_path,manifest,client,1,settings)
    proof.run_reviews(tmp_path,manifest,client,1,settings)
    assert len(calls)==1
    (tmp_path/'p0480-b01-r1.review.json').write_text('{}')
    with pytest.raises(ValueError,match='changed review evidence'):proof.gate(tmp_path,manifest)


def test_deadline_stops_before_starting_another_request(tmp_path,monkeypatch):
    manifest=fixture_run(tmp_path)
    ticks=iter([0,4000]);monkeypatch.setattr(proof.time,'monotonic',lambda:next(ticks))
    proof.run_reviews(tmp_path,manifest,None,1,{'model':'test','num_ctx':16384})
    assert manifest['status']=='time_limit'
    assert manifest['jobs'][0]['attempts']==0


def test_missing_planned_strip_cannot_pass_gate(tmp_path):
    manifest=fixture_run(tmp_path);manifest['jobs']=[]
    with pytest.raises(ValueError,match='No proofreading'):proof.gate(tmp_path,manifest)
    manifest=fixture_run(tmp_path);manifest['passes']=2
    with pytest.raises(ValueError,match='every planned'):proof.gate(tmp_path,manifest)


def test_uncertain_word_mapping_is_not_declared_fully_checked(tmp_path,monkeypatch):
    manifest=fixture_run(tmp_path)
    manifest['pages'][0]['mapping_status']='candidate range; manual mapping required'
    monkeypatch.setattr(proof,'source_crop',lambda p,j,d:d.write_bytes(b'crop'))
    client=SimpleNamespace(chat=lambda **kwargs:fake_response(review()))
    proof.run_reviews(tmp_path,manifest,client,1,{'model':'test','num_ctx':16384})
    with pytest.raises(ValueError,match='mappings still require manual'):proof.gate(tmp_path,manifest)


def test_generated_full_book_proofreading_gate():
    directory=os.environ.get('NEWLIGHT_PROOFREAD_RUN')
    if not directory:pytest.skip('Set NEWLIGHT_PROOFREAD_RUN to check an unattended proofreading run')
    run=Path(directory);manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['source_pages']==446
    assert {p['pdf_page'] for p in manifest['pages']}==set(range(35,481))
    proof.gate(run,manifest)
