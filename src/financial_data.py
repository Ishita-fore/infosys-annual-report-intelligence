"""Read the comparable financial highlights from the extracted annual reports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag import report_pages


ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = ROOT / "data" / "processed"
YEARS = range(2022, 2027)


def find_highlights_page(year: int) -> tuple[int, str]:
    """Find the page with the current year's revenue, profit, and margin overview."""
    path = TEXT_DIR / f"FY{year}.txt"
    for page_number, body in report_pages(path):
        if (
            "Business highlights" in body
            and "Revenues" in body
            and ("Operating margin" in body or "operating margin" in body)
        ):
            return page_number, re.sub(r"\s+", " ", body).strip()
    raise ValueError(f"Could not find the financial highlights page in {path.name}")


def first_amount(pattern: str, text: str, label: str) -> int:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not reliably read {label} from the highlights page")
    return int(match.group(1).replace(",", ""))


def get_financial_metrics() -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for year in YEARS:
        page_number, page = find_highlights_page(year)
        revenue = first_amount(
            r"Revenues(?:\s+\([^)]*\))?\s+(?:₹\s*)?([\d,]+)",
            page,
            "revenue",
        )
        net_profit = first_amount(
            r"Net profit(?:\s*\([^)]*\)|[*#])*\s+([\d,]+)",
            page,
            "net profit",
        )

        margin_start = re.search(r"(?:Adjusted\s+)?operating margin", page, re.IGNORECASE)
        if not margin_start:
            raise ValueError(f"Could not locate operating margin for FY{year}")
        margin_candidates = re.findall(
            r"(\d{2}(?:\.\d+)?)\s*%",
            page[margin_start.start() : margin_start.start() + 1200],
        )
        margin_values = [float(value) for value in margin_candidates if 15 <= float(value) <= 30]
        if not margin_values:
            direct = re.search(r"Operating margin\s+(\d{2}\.\d+)%", page, re.IGNORECASE)
            if not direct:
                raise ValueError(f"Could not reliably read operating margin for FY{year}")
            margin_values = [float(direct.group(1))]

        # The highlights label and amount can be separated by explanatory words.
        cash_label = re.search(r"free cash", page, re.IGNORECASE)
        cash_candidates: list[int] = []
        if year == 2026:
            # FY2026 lays its highlight values out after a row of metric labels.
            growth_label = re.search(r"Free cashflow growth", page, re.IGNORECASE)
            if growth_label:
                cash_area = page[growth_label.start() : growth_label.start() + 900]
                amounts = [
                    int(token.replace(",", ""))
                    for token in re.findall(r"\d[\d,]*", cash_area)
                    if len(token.replace(",", "")) >= 5
                ]
                # In this report's tile sequence, revenue precedes free cash flow.
                if len(amounts) >= 2:
                    cash_candidates = [amounts[1]]
        elif cash_label:
            cash_area = page[cash_label.start() : cash_label.start() + 500]
            for token in re.findall(r"\d[\d,]*", cash_area):
                digits = token.replace(",", "")
                if len(digits) >= 5:
                    value = int(digits)
                    if value < 1_000_000:
                        cash_candidates.append(value)
        if not cash_candidates:
            raise ValueError(f"Could not reliably read free cash flow for FY{year}")

        metrics.append(
            {
                "financial_year": year,
                "revenue_crore": revenue,
                "net_profit_crore": net_profit,
                "operating_margin_percent": margin_values[0],
                "free_cash_flow_crore": cash_candidates[0],
                "source_filename": f"FY{year}.pdf",
                "page_number": page_number,
                "margin_is_adjusted": year == 2026,
            }
        )
    return metrics


def format_metrics_table(metrics: list[dict[str, Any]]) -> str:
    lines = [
        "| Financial year | Revenue (₹ crore) | Net profit (₹ crore) | Operating margin | Free cash flow (₹ crore) | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        margin_label = f"{item['operating_margin_percent']:.1f}%"
        if item["margin_is_adjusted"]:
            margin_label += " (adjusted)"
        citation = f"{item['source_filename']}, PDF p. {item['page_number']}"
        lines.append(
            f"| FY{item['financial_year']} | {item['revenue_crore']:,} | "
            f"{item['net_profit_crore']:,} | {margin_label} | "
            f"{item['free_cash_flow_crore']:,} | {citation} |"
        )
    return "\n".join(lines)
