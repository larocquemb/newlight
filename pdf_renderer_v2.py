#!/usr/bin/env python3
"""Render scans and propose text regions using measured geometry, without an LLM.

Tesseract supplies word boxes and fallible draft text. PyMuPDF renders originals,
region crops and an overlay. Roles are reviewable proposals, never verified OCR.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
from run_paths import create_run_directory

import pymupdf

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data/external/newlightonearlyh01henr.pdf"
COLORS = {"body": (0, 120, 255), "table": (0, 170, 80),
          "footnote": (210, 60, 170), "running_header": (255, 140, 0),
          "page_number": (210, 60, 0)}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def box_union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def parse_tsv(tsv):
    groups = {}
    for word in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if word["level"] != "5" or not word["text"].strip():
            continue
        x, y, w, h = (int(word[k]) for k in ("left", "top", "width", "height"))
        key = tuple(word[k] for k in ("page_num", "block_num", "par_num", "line_num"))
        groups.setdefault(key, []).append({"text": word["text"], "bbox": [x, y, x+w, y+h],
                                          "confidence": float(word["conf"])})
    lines = []
    for words in groups.values():
        words.sort(key=lambda w: w["bbox"][0])
        letters = [w for w in words if sum(c.isalpha() for c in w["text"]) >= 3] or words
        lines.append({"bbox": box_union([w["bbox"] for w in words]),
                      "letter_height": statistics.median(w["bbox"][3]-w["bbox"][1] for w in letters),
                      "text": " ".join(w["text"] for w in words), "words": words})
    return sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))


def is_capital_heading(text):
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) > .85


def split_margins(lines, height):
    """Only split actual numeric margin words, preserving the remaining words."""
    result = []
    for line in lines:
        top = line["bbox"][1] < height * .16
        bottom = line["bbox"][1] > height * .82
        header = top and is_capital_heading(line["text"])
        words = line["words"]
        numbers = [w for w in words if re.fullmatch(r"\d{1,4}", w["text"])]
        if (header or ((top or bottom) and len(numbers) == len(words))) and numbers:
            for word in numbers:
                result.append({**line, "text": word["text"], "bbox": word["bbox"],
                               "words": [word], "role": "page_number"})
            words = [w for w in words if w not in numbers]
            if not words:
                continue
            line = {**line, "words": words, "text": " ".join(w["text"] for w in words),
                    "bbox": box_union([w["bbox"] for w in words])}
        if header:
            line = {**line, "role": "running_header"}
        result.append(line)
    return sorted(result, key=lambda line: (line["bbox"][1], line["bbox"][0]))


def note_marker(line, left_margin, body_height):
    # A numeric OCR prefix is only a marker candidate in an indented note line.
    if line["bbox"][0] < left_margin + body_height * .45:
        return None
    match = re.match(r"^(\d{1,2})(?=\s|[A-Za-z])", line["text"])
    return match.group(1) if match else None


def classify_lines(lines, width, height):
    """Book-specific heuristics using page-relative measurements, not page IDs."""
    lines = split_margins(lines, height)
    content = [line for line in lines if "role" not in line]
    if not content:
        return lines, {}, ["No narrative lines detected"]
    prose = [line for line in content if len(line["text"].split()) >= 5] or content
    opening = [line for line in prose if not is_capital_heading(line["text"])][:5]
    body_height = statistics.median(line["letter_height"] for line in (opening or content))
    lefts = sorted(line["bbox"][0] for line in prose)
    left_margin = lefts[int(.25 * (len(lefts)-1))]
    seen_body = False
    table_left = None
    warnings = []
    previous = None
    for index, line in enumerate(content):
        size = line["letter_height"]
        small = size < body_height * .86
        inset = line["bbox"][0] - left_margin
        numeric = any(c.isdigit() for c in line["text"])
        numeric_words = sum(any(c.isdigit() for c in word["text"]) for word in line["words"])
        gaps = sum(b["bbox"][0]-a["bbox"][2] > body_height
                   for a,b in zip(line["words"], line["words"][1:]))
        table_seed = inset > body_height * 1.5 and numeric and (small or numeric_words >= 2 or gaps >= 2)
        table_continues = (previous and previous["role"] == "table" and table_left is not None
                           and line["bbox"][0] >= table_left-body_height*.3)
        note_continues = (previous and previous["role"] == "footnote"
                          and size < body_height * 1.12
                          and line["bbox"][1]-previous["bbox"][3] < body_height * 1.5)
        next_small = (index+1 < len(content) and content[index+1]["letter_height"] < body_height * .86)
        marker = note_marker(line, left_margin, body_height)
        if seen_body and (table_seed or table_continues) and not note_continues:
            line["role"] = "table"
            table_left = min(table_left, line["bbox"][0]) if table_continues else line["bbox"][0]
        elif seen_body and (note_continues or (small and (next_small or marker))):
            line["role"] = "footnote"
            line["marker_candidate"] = marker
        else:
            line["role"] = "body"
        if line["role"] == "body" and len(line["text"].split()) >= 5 and not is_capital_heading(line["text"]):
            seen_body = True
        ratio = size / body_height if body_height else 1
        if .83 < ratio < .89:
            warnings.append(f"Ambiguous type-size boundary near y={line['bbox'][1]}")
        previous = line
    metrics = {"body_letter_height_px": body_height, "prose_left_px": left_margin,
               "small_type_ratio": .86, "table_inset_body_heights": 1.5}
    return lines, metrics, warnings


def group_regions(lines):
    regions = []
    for line in lines:
        role = line["role"]
        marker = line.get("marker_candidate")
        new = not regions or regions[-1]["role"] != role or (role == "footnote" and marker is not None)
        if new:
            regions.append({"role": role, "marker_candidate": marker, "lines": []})
        regions[-1]["lines"].append(line)
    for region in regions:
        region["bbox_px"] = box_union([line["bbox"] for line in region["lines"]])
        region["draft_text"] = "\n".join(line["text"] for line in region["lines"])
        region["review_required"] = True
    return regions


def render_page(page, number, run, dpi):
    stem = f"page_{number:04d}"
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    full = run / f"{stem}.png"
    pix.save(full)
    proc = subprocess.run(["tesseract", str(full), "stdout", "--psm", "3", "tsv"],
                          capture_output=True, text=True, timeout=180)
    (run / f"{stem}.tsv").write_text(proc.stdout, encoding="utf-8")
    (run / f"{stem}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    proc.check_returncode()
    lines, metrics, warnings = classify_lines(parse_tsv(proc.stdout), pix.width, pix.height)
    regions = group_regions(lines)
    overlay = pymupdf.Pixmap(pix, 0)
    sx, sy = page.rect.width / pix.width, page.rect.height / pix.height
    for i, region in enumerate(regions, 1):
        x0, y0, x1, y1 = region["bbox_px"]
        padded = [max(0, x0-4), max(0, y0-4), min(pix.width, x1+4), min(pix.height, y1+4)]
        rect = pymupdf.Rect(padded[0]*sx, padded[1]*sy, padded[2]*sx, padded[3]*sy)
        rect += (page.rect.x0, page.rect.y0, page.rect.x0, page.rect.y0)
        crop = run / f"{stem}.region_{i:02d}.{region['role']}.png"
        page.get_pixmap(dpi=dpi, alpha=False, clip=rect).save(crop)
        region.update({"id": i, "bbox_pdf_points": list(rect), "crop": crop.name,
                       "crop_sha256": digest(crop)})
        color = COLORS[region["role"]]
        for box in ([x0, y0, x1, min(y0+2, y1)], [x0, max(y0,y1-2), x1,y1],
                    [x0,y0,min(x0+2,x1),y1], [max(x0,x1-2),y0,x1,y1]):
            overlay.set_rect(pymupdf.IRect(box), color)
    overlay_path = run / f"{stem}.layout.png"
    overlay.save(overlay_path)
    # Draft text remains explicitly unverified, separate from model OCR outputs.
    body = [line for line in lines if line["role"] in ("body", "table")]
    draft = "\n".join(" " * max(0, round((line["bbox"][0]-metrics.get("prose_left_px",0)) /
                                       max(1, metrics.get("body_letter_height_px",1)*.5))) + line["text"]
                      for line in body)
    (run / f"{stem}.draft.body.txt").write_text(draft, encoding="utf-8")
    layout = {"pdf_page": number, "dpi": dpi, "width_px": pix.width, "height_px": pix.height,
              "source_page_rect": list(page.rect), "full_image": full.name,
              "image_sha256": digest(full), "overlay": overlay_path.name,
              "metrics": metrics, "warnings": warnings, "regions": regions,
              "status": "proposed_layout_unverified", "review_required": True,
              "method": "Tesseract word geometry; deterministic size, inset, margin and column heuristics",
              "limitations": ["OCR text and marker candidates may be wrong",
                              "Region classes and completeness require visual review",
                              "Draft spaces approximate indentation; original scan is authoritative"]}
    write_json(run / f"{stem}.layout.json", layout)
    return {"pdf_page": number, "status": "proposed_layout_unverified",
            "regions": [{"role": r["role"], "marker_candidate": r["marker_candidate"]} for r in regions],
            "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/rendered-v2")
    parser.add_argument("--start", type=int, default=35)
    parser.add_argument("--end", type=int, default=480)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if not args.input.is_file():
        parser.error(f"PDF not found: {args.input}")
    if not shutil.which("tesseract"):
        parser.error("Tesseract is required for word geometry (macOS: brew install tesseract)")
    with pymupdf.open(args.input) as pdf:
        if not 1 <= args.start <= args.end <= len(pdf):
            parser.error(f"Require 1 <= start <= end <= {len(pdf)}")
        run = create_run_directory(args.output)
        run_id = run.name
        manifest = {"renderer_version": 2, "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source": str(args.input.resolve()), "source_sha256": digest(args.input),
                    "code_sha256": digest(Path(__file__)), "pymupdf_version": pymupdf.VersionBind,
                    "tesseract_version": subprocess.run(["tesseract", "--version"], capture_output=True,
                                                         text=True, check=True).stdout.splitlines()[0],
                    "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    "overlay_colors": COLORS, "status": "running", "pages": []}
        write_json(run / "manifest.json", manifest)
        print(f"Renderer v2 evidence: {run}", flush=True)
        try:
            for number in range(args.start, args.end+1):
                entry = render_page(pdf[number-1], number, run, args.dpi)
                manifest["pages"].append(entry)
                write_json(run / "manifest.json", manifest)
                print(f"Page {number}: " + ", ".join(r["role"] +
                      (f"({r['marker_candidate']})" if r["marker_candidate"] else "")
                      for r in entry["regions"]), flush=True)
            manifest["status"] = "complete_unverified"
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["error"] = str(exc)
            raise
        finally:
            write_json(run / "manifest.json", manifest)


if __name__ == "__main__":
    main()
