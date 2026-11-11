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
Region pipeline v3 excludes tiny table leader dots from word-count diagnostics
and handles numeric `o`/`0` OCR confusion during comparison only. If a single
word still disagrees with the draft, two image-only reads with different seeds
must agree before the draft anchor can be adjudicated. Both responses and the
disagreement remain in evidence and uncertainties.
Pipeline v4 permits one third image read when the first two disagree, requiring
two matching nonempty readings; three different readings still fail. Unicode
and slash fraction encodings (such as `½` and `1/2`) compare equivalently without
normalizing the saved text or conflating `I` with `1`.
It includes separately stored footnote markers in coverage diagnostics, so
correctly separating a marker does not falsely report a missing number.
Repetition-limit server aborts receive one shorter, bounded retry with adjusted
sampling; both errors and retry requests/responses are retained. Other server
errors are not retried. Prompts request brief uncertainty lists without explanations.
An ornamental initial omitted from TSV is read in a separate enlarged crop with
its adjacent word as context. The returned word must match the measured suffix;
the letter is never supplied from a hard-coded vocabulary or page number.
The initial check tests ink only in the indentation, excluding the adjacent
context word, and only on the region's opening line.
Pipeline v5 joins adjacent fragments of one margin page number into an enlarged
image for a new visual read. It does not infer the digits from PDF page order;
the original regions and derived crop/hash are retained. Distant or overlapping
semantic candidates remain errors. An empty measured-line response is retried
using its visible word spans. Declared running headers duplicated at a body
edge are separated with an exact removal audit; interior duplicates still fail.
Conflicting transcribed footnote numbers require an independent enlarged glyph
reading agreeing with one of those transcriptions, never the layout candidate.
Pipeline revisions invalidate older cached outputs: choose a fresh output
directory or explicitly use `--overwrite` after an extraction change.
Successful outputs can be resumed with matching layout/model/settings; use a fresh output directory
or `--overwrite` to change these. Resume leaves existing files and continuation
flags untouched; rerun adjoining pages with `--overwrite` to recalculate links.
Compatible neighboring outputs in the same destination can supply continuation
context for a one-page retry; their hashes are recorded and files left untouched.
The older whole-page mode is available by omitting `--renderer-run`.

The current `data/processed/parsed/full-book/` recovery dataset combines retained
earlier pipeline outputs with v5 missing-page recovery. Its `recovery_manifest.json` records
every source path/hash and all missing pages; it is incomplete and unverified.
Historical runs remain unchanged. Check its generated-page regressions with:

```sh
NEWLIGHT_QA_OUTPUT=data/processed/parsed/full-book python -m pytest -q
```

Passing these selected page checks does not validate every page of the book.
For a fresh full v5 run, choose a new output directory; retained older checkpoints
are intentionally incompatible with ordinary v5 resume.
To recover only absent pages while preserving the consolidated earlier outputs:

```sh
python ocr_parse.py --start 35 --end 480 \
  --renderer-run data/processed/rendered-v2/runs/20261111T034716Z-9c5339b8 \
  --model ministral-3:14b --num-ctx 8192 --num-predict 4096 \
  --region-lines 1 --missing-only --output data/processed/parsed/full-book
```

`--missing-only` explicitly allows earlier pipeline versions after checking page
identity, image/layout hashes and sidecar consistency. It does not revalidate
their OCR accuracy or rewrite their continuation flags. New pages receive the
current pipeline version. The existing recovery index updates at checkpoints
and completion, with source hashes and missing pages. Failed pages remain missing
and the batch continues; another invocation retries them.

Regression fixtures for pages 38–42 contain actual measured TSV geometry. They
check table/note separation, distinct notes, short final lines, margin numbers,
chapter titles and coordinate-scale invariance without running OCR or a model.

## Editable Word reconstruction

`book_reconstruct.py` combines OCR text with renderer-v2 geometry. It produces an
enriched JSON model, an audit, and an editable DOCX with heading styles, separate
running headers, PAGE fields, inset native tables, and native Word footnotes.
Headers use centered/right tab stops with page numbers at the outside margins.
Body text and footnotes are justified; source geometry supplies first-line
paragraph indents while continuations remain flush left. Drop-cap offsets are
excluded from paragraph indentation. Printed line breaks remain in the body;
the final line of a paragraph is left naturally short.
Unfinished prose continuing across a source-page boundary keeps its final
printed line fully justified too, using a layout-only soft break; the enriched
text remains unchanged. A following heading or indented paragraph prevents
this continuation treatment.
The default Word column is capped at 325 points to avoid excessive word spacing
with the substitute font. Both margins move inward equally, keeping the source
column center; original scan margins stay in the enriched JSON. Use
`--text-width POINTS` to adjust this cap. It never expands beyond the measured
source text area.
Adjacent unnumbered note continuations are merged within a chapter. Missing
selected pages, mismatched image hashes and unresolved footnote links stop the
export. This is a reconstruction, not a pixel-identical facsimile or proofread text.
Footnotes use native automatic Word reference marks at both ends, with numbering
restarted at chapter boundaries and continued across other page sections.
The structure audit rejects fixed custom marks and missing native note marks.

The source PDF has an unreliable OCR/font layer. Font sizes and margins are
approximations; Times New Roman is a substitute. Italics require reviewed phrase
annotations, not inferred PDF font flags. The optional annotation JSON has a
`pages` object keyed by PDF page, with `italics`, `note_italics`, and `corrections`
lists. A correction supplies `before`, `after`, optional `all`, and a `scope` of
`body`, `running_headers`, or `footnote` (with a string `marker`, or null for a
continuation). Corrections are audited; original OCR files remain untouched.
Reviewed table rules use `table_rules`, for example
`{"row_label": "Total cost", "top_rule_columns": [1, 2]}` (zero-based columns).
They export as native cell borders and are checked by the structure audit.

Preview the structured model without document dependencies:

```sh
python book_reconstruct.py --start 38 --end 42 \
  --ocr-dir data/processed/word/pilot-inputs \
  --renderer-run data/processed/rendered-v2/runs/20261111T034716Z-9c5339b8 \
  --annotations data/processed/word/pilot-annotations.json --enrich-only
```

For DOCX creation use a Python runtime with `python-docx` and `lxml`; Codex uses
its bundled document runtime. Run the same command without `--enrich-only`.
Outputs go under `data/processed/word/runs/`; filenames include the run ID so
successive versions are distinguishable. Each run retains the enriched model,
annotation snapshot, correction audit, input hashes and native-structure checks.
Footnotes preserve source line breaks and line-end hyphens as native Word line
breaks; the audit compares their complete text and wrapping against the enriched
model. By default, continuations are separate native footnote segments anchored
to their source pages with invisible custom marks; they consume no printed
number. The enriched JSON retains the complete logical note and links each
segment through `logical_id`. Word's Next Footnote navigation also visits these
continuation anchors. Use `--footnote-pagination flow` for a single native note
that Word may paginate differently. `audit_footnote_pagination` checks rendered
continuation starts against their source pages; native XML checks alone cannot
verify physical pagination.
Full-book export uses `--start 35 --end 480` only once every selected OCR page is
present and note links resolve. Review rendered pages before treating an export
as publication-ready. The five-page pilot uses retained earlier OCR snapshots;
it does not depend on or alter the running full-book inference job.

Use `--available-only` to export existing pages before recovery is complete.
This draft includes a cover listing missing PDF pages and retains unresolved
footnotes at editorial page-start anchors instead of dropping their text or
guessing their original reference locations. Source note numbers use native
custom marks so omissions do not silently renumber them. The enriched model
records gaps and review issues. For this volume, PDF35 maps to printed1; this
fixed offset repairs missing or misread page-number labels in available-only
exports. Snapshot OCR inputs before exporting while recovery is still running.

Run deterministic regressions with `python -m pytest tests/test_book_reconstruct.py -q`.

## Generated-data locations

New renderer-v2, OCR and Word runs use `YYYYMMDD-NN`, for example `20261111-02`.
The date is in America/Winnipeg and the sequence starts at `01` each day within
each output's `runs/` folder. Directory creation is atomic, so concurrent runs
cannot overwrite one another. Existing timestamp/UUID paths remain valid;
readers accept either format. Manifest creation timestamps remain in UTC.

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
