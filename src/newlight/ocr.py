#!/usr/bin/env python3
"""One-pass historical OCR: body and footnotes, using Ollama vision.

Examples:
    python src/newlight/ocr.py --page 41 --overwrite
    python src/newlight/ocr.py --start 35 --end 49
"""

import argparse
import json
import os
import time
from pathlib import Path

from ollama import Client

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data/processed/pdf_pages"
OUTPUT = ROOT / "data/processed/legacy-ocr"
DEFAULT_MODEL = "mistral-small3.2:24b"

PROMPT = """You are transcribing one scanned page from the historical book
New Light on the Early History of the Greater Northwest.

Produce a faithful diplomatic transcription. Preserve all original English,
French, Latin and Indigenous wording, accents, historical spelling, grammar,
capitalization, punctuation, names, and paragraph boundaries. Do not translate,
modernize, correct, summarize, or invent text. Do not insert [unclear] or other
editorial annotations. Do not complete sentences cut off by the page boundary.

The only content distinction is BODY versus FOOTNOTES.
- BODY: all narrative, chapter headings, tables, inventories, and any other
  non-footnote content, in original reading order.
- FOOTNOTES: all editorial notes, usually smaller type near the bottom of
  the page, sometimes separated by a rule or whitespace. They can begin
  mid-sentence when continued from the preceding page. Transcribe every
  visible word of each note, including any continuation.
- Never place footnote text in the body or duplicate text across categories.
- Omit running page headers and printed page numbers from the body.
- Keep tables in body, preserving row order, labels, quantities and units.
  You may omit decorative dotted leaders; do not change any words or numbers.
- Preserve visible footnote reference markers in body. Do not invent markers.
- If no footnotes are visible, return an empty list.

Return exactly one JSON object, with no Markdown, using this structure:
{
  "body": "Complete non-footnote transcription as one string",
  "footnotes": [
    {
      "marker": null,
      "text": "Complete visible text of the footnote",
      "continues_from_previous_page": null,
      "continues_on_next_page": null
    }
  ],
  "printed_page_number": null
}
Use true/false for continuation only when evident from the image; otherwise null.
"""


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def validate(result: dict) -> list[str]:
    """Structural and duplicate checks; not proof of transcription accuracy."""
    issues = []
    if not isinstance(result, dict):
        return ["Response must be a JSON object"]
    body = result.get("body")
    notes = result.get("footnotes")
    if not isinstance(body, str) or not body.strip():
        issues.append("Body missing or empty")
    if not isinstance(notes, list):
        issues.append("Footnotes must be a list")
        return issues
    for i, note in enumerate(notes):
        if not isinstance(note, dict):
            issues.append(f"Footnote {i}: must be an object")
            continue
        text = note.get("text")
        if not isinstance(text, str) or not text.strip():
            issues.append(f"Footnote {i}: missing text")
            continue
        marker = note.get("marker")
        if marker is not None and not isinstance(marker, (str, int)):
            issues.append(f"Footnote {i}: invalid marker")
        for flag in ("continues_from_previous_page", "continues_on_next_page"):
            if note.get(flag) not in (True, False, None):
                issues.append(f"Footnote {i}: invalid {flag}")
        if isinstance(body, str):
            n = normalize(text)
            if len(n) >= 60 and n[:60] in normalize(body):
                issues.append(f"Footnote {i}: likely duplicated in body")
    return issues


def extract(client: Client, model: str, image: Path, num_ctx: int) -> dict:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT, "images": [image.read_bytes()]}],
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": num_ctx,
            "num_predict": 8192,
            "repeat_penalty": 1.05,
        },
    )
    content = response.message.content
    if not content:
        raise ValueError("Ollama returned empty output")
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Ollama did not return a JSON object")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="One-pass historical OCR")
    parser.add_argument("--page", type=int)
    parser.add_argument("--start", type=int, default=35)
    parser.add_argument("--end", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.page is not None and args.end is not None:
        parser.error("--page and --end cannot be used together")
    start = args.page if args.page is not None else args.start
    end = args.page if args.page is not None else (args.end if args.end is not None else start)
    if not (1 <= start <= end <= 490):
        parser.error("Page range must be 1–490")
    if args.num_ctx < 8192:
        parser.error("--num-ctx must be at least 8192")

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = Client(host=host, timeout=600)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Ollama host: {host}\nModel: {args.model}", flush=True)

    failed = 0
    for number in range(start, end + 1):
        image = IMAGES / f"page_{number:04d}.png"
        output = args.output / f"page_{number:04d}.json"
        if output.exists() and not args.overwrite:
            print(f"Skipping page {number}: exists", flush=True)
            continue
        if not image.is_file():
            print(f"ERROR page {number}: missing image {image}", flush=True)
            failed += 1
            continue
        print(f"Processing page {number}...", flush=True)
        started = time.perf_counter()
        try:
            result = extract(client, args.model, image, args.num_ctx)
            warnings = validate(result)
            # Reject structurally invalid output rather than storing it as success.
            if not isinstance(result.get("body"), str) or not isinstance(result.get("footnotes"), list):
                raise ValueError("Invalid body/footnotes structure: " + "; ".join(warnings))
            result.update({
                "pdf_page": number,
                "model": args.model,
                "extraction_method": "single_pass",
                "validation_warnings": warnings,
                "review_required": bool(warnings),
            })
            temp = output.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(output)
            print(f"Saved {output.name} in {time.perf_counter()-started:.1f}s; warnings={len(warnings)}", flush=True)
            for warning in warnings:
                print(f"  WARNING: {warning}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"ERROR page {number}: {exc}", flush=True)
    if failed:
        raise SystemExit(f"{failed} page(s) failed")


if __name__ == "__main__":
    main()
