"""Readable run directories; atomic creation prevents concurrent collisions."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RUN_TIMEZONE = 'America/Winnipeg'


def create_run_directory(output, *, now=None):
    now = now or datetime.now(ZoneInfo(RUN_TIMEZONE))
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Run timestamp must include a timezone')
    stem = now.astimezone(ZoneInfo(RUN_TIMEZONE)).strftime('%Y%m%d')
    parent = Path(output) / 'runs'
    parent.mkdir(parents=True, exist_ok=True)
    serial = 1
    while True:
        name = f'{stem}-{serial:02d}'
        run = parent / name
        try:
            run.mkdir()
        except FileExistsError:
            serial += 1
        else:
            return run
