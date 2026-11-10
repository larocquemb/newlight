
#!/usr/bin/env python3
"""Render PDF pages to PNG images using PyMuPDF."""

import argparse
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = (
    PROJECT_ROOT / "data/external/newlightonearlyh01henr.pdf"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "pdf_pages"


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    overwrite: bool = False,
) -> None:
    """Render each PDF page to a numbered PNG file."""

    output_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as pdf:
        total = len(pdf)

        print(f"Rendering {total} pages at {dpi} DPI")

        for index, page in enumerate(pdf):
            output_file = output_dir / f"page_{index + 1:04d}.png"

            if output_file.exists() and not overwrite:
                print(f"Skipping {output_file.name}")
                continue

            pix = page.get_pixmap(dpi=dpi, alpha=False)
            pix.save(output_file)

            print(
                f"[{index + 1}/{total}] "
                f"{output_file.name} "
                f"({pix.width} x {pix.height})"
            )

    print("Rendering complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render PDF pages as PNG images."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input PDF file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering resolution (default: 200 DPI)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files",
    )

    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    if not args.input.is_file():
        parser.error(f"PDF not found: {args.input}")

    render_pdf(
        pdf_path=args.input,
        output_dir=args.output,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
