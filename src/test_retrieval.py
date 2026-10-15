"""Try example financial questions against the local annual-report index."""

from __future__ import annotations

import argparse

from rag import search_chunks


SAMPLE_QUESTIONS = [
    ("business highlights revenue net profit margin cash flow", 2022, "Business highlights"),
    ("business highlights revenue net profit margin cash flow", 2023, "Business highlights"),
    ("business highlights revenue net profit margin cash flow", 2024, "Business highlights"),
    ("business highlights revenue net profit margin cash flow", 2025, "Business highlights"),
    ("business highlights revenue net profit margin cash flow", 2026, "Business highlights"),
    ("major business risks, risk factors, and strategic priorities", None, None),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Your own question")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    questions = [(args.question, None, None)] if args.question else SAMPLE_QUESTIONS

    for question, year, contains_text in questions:
        year_note = f" (FY{year})" if year else ""
        print(f"\nQUESTION: {question}{year_note}")
        for hit in search_chunks(
            question,
            args.top_k,
            financial_year=year,
            contains_text=contains_text,
        ):
            metadata = hit["metadata"]
            print(
                f"[{metadata['source_filename']}, page {metadata['page_number']}; "
                f"distance={hit['distance']:.3f}]"
            )
            print(" ".join(hit["text"].split())[:650])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())