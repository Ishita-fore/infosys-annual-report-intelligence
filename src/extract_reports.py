"""Extract page-by-page text from the annual-report PDFs in data/raw."""

from pathlib import Path
import sys

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> int:
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if len(pdf_files) != 5:
        print(
            f"Expected 5 PDF files in {RAW_DIR}, but found {len(pdf_files)}.",
            file=sys.stderr,
        )
        return 1

    output_files = [PROCESSED_DIR / f"{pdf_path.stem}.txt" for pdf_path in pdf_files]
    existing_outputs = [path for path in output_files if path.exists()]
    if existing_outputs:
        print("Will not overwrite existing output file(s):", file=sys.stderr)
        for path in existing_outputs:
            print(f"  {path}", file=sys.stderr)
        return 1

    extracted_reports: list[tuple[Path, str, int]] = []
    for pdf_path, output_path in zip(pdf_files, output_files):
        try:
            reader = PdfReader(pdf_path)
            pages_text = []
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                pages_text.append(f"--- Page {page_number} ---\n{page_text}\n")
            extracted_reports.append(
                (output_path, "\n".join(pages_text), len(reader.pages))
            )
        except Exception as error:
            print(f"Could not extract {pdf_path.name}: {error}", file=sys.stderr)
            return 1

    for output_path, text, _page_count in extracted_reports:
        try:
            with output_path.open("x", encoding="utf-8") as output_file:
                output_file.write(text)
        except FileExistsError:
            print(f"Will not overwrite existing file: {output_path}", file=sys.stderr)
            return 1

    for output_path, _text, page_count in extracted_reports:
        print(f"Created {output_path.relative_to(PROJECT_ROOT)} ({page_count} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
