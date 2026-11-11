# newlight
Convert: New Light on the Early History of the Greater Northwest Vol I to text

## Renderer v2: deterministic layout proposals

`pdf_renderer_v2.py` renders with PyMuPDF and measures word positions with local
Tesseract. It uses relative letter height, inset margins, column spacing and
neighboring lines to propose body, table, footnote, running-header and page-number
regions. It makes no Ollama requests. The original `pdf_renderer.py` is retained.

Use the existing virtual environment and install the system `tesseract` command
if needed (`brew install tesseract` on macOS):

```sh
source .venv/bin/activate
python pdf_renderer_v2.py --start 35 --end 44
python -m pytest tests/test_pdf_renderer_v2.py -q
```

The full book range is `--start 35 --end 480`; `--dpi` defaults to 200.
Every invocation creates a separate run under `data/processed/rendered-v2/runs/` (or
`--output DIRECTORY`). It retains:

- Original full-page PNGs and raw Tesseract TSV/stderr.
- `page_NNNN.layout.png`: region outlines, blue for body, green for tables,
  magenta for footnotes, orange for running headers and red for page numbers.
- `page_NNNN.layout.json`: measured lines, proposed roles, pixel/PDF-point
  coordinates, note-marker candidates and links/hashes for region crops.
- `page_NNNN.region_XX.ROLE.png`: padded image crops for separate transcription.
- `page_NNNN.draft.body.txt`: unverified local OCR for body and table regions,
  with approximate indentation. This is not a literal verified transcription.
- `manifest.json`: source/code hashes, versions, arguments and run status.

Review overlays before accepting regions. Tables and notes can share a font
size; indentation and column structure help distinguish them. Marker recognition
can be wrong, page continuation links are not inferred, and ambiguous sizes are
flagged. The existing PDF OCR layer is not used as ground truth. This version
prepares crops for the region mode in `ocr_parse.py`.

## Transcribe renderer v2 regions

Pass the exact run directory printed by the renderer (not its parent directory):

```sh
python ocr_parse.py --start 40 --end 42 \
  --renderer-run data/processed/rendered-v2/runs/YOUR_RUN_ID \
  --model ministral-3:14b --num-ctx 8192 --num-predict 4096 \
  --region-lines 1 \
  --output data/processed/parsed/regions-pass1

NEWLIGHT_QA_OUTPUT=data/processed/parsed/regions-pass1 \
  python -m pytest tests/test_note_continuations.py -q
```

This mode sends one crop per request. `--region-lines` controls the maximum
prose lines per crop (default 1); single-line crops reduce line omissions but
require more model calls. Table regions always use one crop per row to retain
column associations. Body and table regions are assembled in
reading order into the body; each footnote region becomes one note. Headers and
printed page numbers are transcribed from their own crops. Layout marker
candidates are not copied as text: the model reads the visible marker again.
Numeric marker representations such as `¹` and `^1` are normalized to `1` in the
marker field; the original response and any separated prefix are retained.
Bracket validation checks matching/nesting of `()`, `[]` and `{}` separately in
body and each footnote. Curly braces are flagged as possible OCR confusion.
Mixed pairs and unmatched boundaries produce review warnings, since a genuine
pair can continue onto another page. Text is not silently replaced or balanced.
If a layout suggests a reference marker but the transcription misses it, the
pipeline separately crops and enlarges the leading glyph for another model read.
It does not fill the field from the layout candidate. Marker inspection images,
requests and responses are retained as evidence.
Cross-page note-continuation flags are adjacency-based proposals and remain
unverified; pages outside the selected range leave unresolved flags null.

No `data/processed/pdf_pages` inputs or five-page model context are used in region mode.
The renderer's full images, layouts and crop hashes are checked before inference.
Every crop request and raw response is retained with the page's region evidence.
A failed crop fails that page, while later pages continue. Successful outputs
are checkpointed before the next page starts. A response with fewer text lines
than the measured crop is rejected rather than accepted as complete. Checkpoints
written before continuation linking have `continuation_proposals_pending: true`.
Within-line coverage is checked against independent OCR token counts and
high-confidence anchors. A disagreement triggers new image reads of small word
spans, retaining the original response and every retry crop/request/response.
An incomplete retry is subdivided adaptively down to individual word boxes.
Merged OCR tokens such as `goto` can match adjacent transcribed words `go to`
for diagnostics only; saved spelling and whitespace are not normalized.
Draft words are never inserted into the output. Unresolved disagreement fails
the page for review; the OCR draft can itself be wrong. These checks detect
omissions but do not establish full character/punctuation accuracy.
Pipeline revisions invalidate older cached outputs: choose a fresh output
directory or explicitly use `--overwrite` after an extraction change.
Successful outputs can be resumed with matching layout/model/settings; use a fresh output directory
or `--overwrite` to change these. Resume leaves existing files and continuation
flags untouched; rerun adjoining pages with `--overwrite` to recalculate links.
Compatible neighboring outputs in the same destination can supply continuation
context for a one-page retry; their hashes are recorded and files left untouched.
The older whole-page mode is available by omitting `--renderer-run`.

Regression fixtures for pages 38–42 contain actual measured TSV geometry. They
check table/note separation, distinct notes, short final lines, margin numbers,
chapter titles and coordinate-scale invariance without running OCR or a model.

## Generated-data locations

All pipeline output defaults are under `data/processed/`:

- `pdf_pages/`: original renderer images and whole-page OCR inputs.
- `rendered-v2/runs/`: layout manifests, source renders, region crops and previews.
- `parsed/`: OCR outputs and request/response evidence.
- `legacy-ocr/`: outputs from `src/newlight/ocr.py`.
- `tmp/`: migrated scratch runs. Put new scratch output here too.
- `migrations/`: relocation manifests with old/new paths and content hashes.

The source PDF stays in `data/external/`, and versioned regression fixtures stay
in `tests/fixtures/`. Historical evidence retains its original path strings; use
the migration manifests to locate moved files. New run commands should use the
new paths. Content hashes and original timestamps are preserved during moves.
