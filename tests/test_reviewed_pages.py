"""Source excerpts visually read on PDF230/233/235/255/293, not full references.

These checks protect the specific brackets, table roles, symbols and omissions
found during recovery. Missing pages are covered by the separate completeness gate.
"""
import json
import os
import re
from pathlib import Path

import pytest


def check_reviewed_page(number,page):
    body=page['body'];flat=' '.join(body.split());folded=flat.casefold()
    markers=[note['marker'] for note in page['footnotes']]
    if number==230:
        assert '[one of the Red Lake Ojibways].' in flat
        assert 'Saturday, May 1st. I set fire' in flat
    elif number==233:
        assert markers==[], 'Return-table rows are body text, not editorial notes'
        assert '†' in body and 'not given.' in folded
        assert 'inland packs of 90 lbs.' in folded
    elif number==235:
        assert markers==[], 'Ledger headings and table rows are not editorial footnotes'
        row=next(line for line in body.splitlines() if 'carcajoux' in line.casefold())
        assert '“' in row, 'Ditto glyph lost'
        assert re.search(r'3\s*\|?\s+5\s*\|?\s+0',row), 'Separate £/s/d amounts fused or misread'
    elif number==255:
        assert markers==[None,'23','24'], 'Continuation, note23 and note24 must stay distinct'
        sugar=[line for line in body.splitlines() if 'kegs of sugar' in line.casefold()]
        assert len(sugar)==1, 'Neighboring Sugar row duplicated into another table row'
        assert not any('sugar' in line.casefold() and 'pemmican' in line.casefold() for line in body.splitlines())
    elif number==293:
        assert markers==['10'], 'Table quantities are not editorial footnote markers'
        assert 'kegs of beef' in folded
        assert re.search(r'(?mi)^\s*3\b[^\n]*\bgum\b[^\n]*\b3\b',body), 'Final gum row omitted'
    elif number==315:
        assert markers==['24','25'], 'Table quantities must not become footnotes'
        assert 'keg salt' in folded and 'small bosses' in folded
    elif number==316:
        assert markers==['26'], 'Census rows are not editorial footnotes'
        assert 'fort des prairies' in folded and 'northwest country' in folded
    elif number==456:
        assert markers==['16'], 'Return-table quantities are not footnotes'
        assert '[copy blank]' in folded and 'halifax' in folded
    elif number==474:
        assert markers==[None], 'Only the continued editorial note belongs in footnotes'
        assert 'pemmican' in folded and 'gain' in folded
        assert 'for whom Fort William was named in 1807' in ' '.join(page['footnotes'][0]['text'].split())


@pytest.mark.parametrize('number',[230,233,235,255,293,315,316,456,474])
def test_generated_reviewed_page(number):
    directory=os.environ.get('NEWLIGHT_QA_OUTPUT')
    if not directory:pytest.skip('Set NEWLIGHT_QA_OUTPUT for generated source checks')
    path=Path(directory)/f'page_{number:04d}.json'
    if not path.is_file():pytest.skip(f'PDF{number} still missing; completeness gate reports it')
    check_reviewed_page(number,json.loads(path.read_text()))


def test_rejects_fused_currency_columns():
    with pytest.raises(AssertionError,match='fused'):
        check_reviewed_page(235,{'body':'13 Carcajoux, @5s. “ 35 0','footnotes':[]})


def test_rejects_truncated_footnote_marker():
    with pytest.raises(AssertionError,match='note24'):
        check_reviewed_page(255,{'body':'10 Kegs of Sugar','footnotes':[{'marker':None},{'marker':'23'},{'marker':'4'}]})


def test_rejects_missing_final_table_row():
    with pytest.raises(AssertionError,match='gum row'):
        check_reviewed_page(293,{'body':'14 Kegs of beef . . . 14','footnotes':[{'marker':'10'}]})
