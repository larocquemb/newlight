from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from newlight.run_paths import create_run_directory


def test_daily_sequence_and_existing_evidence_preserved(tmp_path):
    now = datetime(2026, 11, 11, 18, tzinfo=timezone.utc)
    first = create_run_directory(tmp_path, now=now)
    (first / 'evidence.txt').write_text('retain this')
    second = create_run_directory(tmp_path, now=now)
    assert [first.name, second.name] == ['20261111-01', '20261111-02']
    assert (first / 'evidence.txt').read_text() == 'retain this'
    next_day = create_run_directory(tmp_path, now=datetime(2026, 11, 12, 18, tzinfo=timezone.utc))
    assert next_day.name == '20261112-01'


def test_date_uses_winnipeg_not_utc(tmp_path):
    run = create_run_directory(tmp_path, now=datetime(2026, 11, 11, 3, tzinfo=timezone.utc))
    assert run.name == '20261110-01'


def test_concurrent_runs_get_distinct_serials(tmp_path):
    now = datetime(2026, 11, 11, 18, tzinfo=timezone.utc)
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: create_run_directory(tmp_path, now=now), range(12)))
    assert {p.name for p in paths} == {f'20261111-{n:02d}' for n in range(1, 13)}


def test_naive_timestamp_rejected(tmp_path):
    with pytest.raises(ValueError, match='timezone'):
        create_run_directory(tmp_path, now=datetime(2026, 11, 11))
