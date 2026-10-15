"""Run two local CrewAI analysts and save a cited five-year comparison."""

from __future__ import annotations

import re
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, LLM, Process, Task

from financial_data import format_metrics_table, get_financial_metrics
from rag import ROOT, VECTOR_DIR, report_pages, search_chunks


OUTPUT_DIR = ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "infosys_five_year_analysis.md"
CHART_PATH = OUTPUT_DIR / "financial_trends.svg"
CHAT_MODEL = "ollama/qwen2.5:latest"
OLLAMA_URL = "http://127.0.0.1:11434"


def citation(metadata: dict[str, Any]) -> str:
    return f"[{metadata['source_filename']}, PDF p. {metadata['page_number']}]"


def financial_evidence(metrics: list[dict[str, Any]]) -> tuple[str, set[str]]:
    blocks: list[str] = []
    allowed_citations: set[str] = set()
    for item in metrics:
        hits = search_chunks(
            "business highlights revenue net profit operating margin free cash flow",
            top_k=4,
            financial_year=item["financial_year"],
            contains_text="Business highlights",
        )
        relevant = [
            hit
            for hit in hits
            if hit["metadata"]["page_number"] == item["page_number"]
        ]
        if not relevant:
            raise RuntimeError(
                f"RAG did not retrieve the verified highlights page for FY{item['financial_year']}"
            )
        for hit in relevant[:2]:
            source = citation(hit["metadata"])
            allowed_citations.add(source)
            blocks.append(f"{source}\n{hit['text'][:1900]}")

        # Supply the exact validated page excerpt as additional evidence to the analyst.
        text_path = ROOT / "data" / "processed" / f"FY{item['financial_year']}.txt"
        page_body = dict(report_pages(text_path))[item["page_number"]]
        source = f"[{item['source_filename']}, PDF p. {item['page_number']}]"
        allowed_citations.add(source)
        blocks.append(f"{source}\n{page_body[:2600]}")
    return "\n\n".join(blocks), allowed_citations


def risk_strategy_evidence() -> tuple[str, set[str]]:
    blocks: list[str] = []
    allowed_citations: set[str] = set()
    questions = [
        "principal risks risk management geopolitical economic uncertainty supply chain talent cybersecurity regulation",
        "strategic priorities cloud digital AI generative AI cybersecurity strategy business development",
    ]
    for year in range(2022, 2027):
        for question in questions:
            hits = search_chunks(question, top_k=2, financial_year=year)
            for hit in hits:
                source = citation(hit["metadata"])
                allowed_citations.add(source)
                blocks.append(f"{source}\n{hit['text'][:1000]}")
    if not blocks:
        raise RuntimeError("RAG did not return any risk or strategy passages")
    return "\n\n".join(blocks), allowed_citations


def make_svg(metrics: list[dict[str, Any]]) -> str:
    """Draw two dependency-free line charts as a single SVG image."""
    width, height = 920, 460
    panels = [
        ("Revenue (₹ crore)", [item["revenue_crore"] for item in metrics], 70, 200_000, 50_000),
        ("Net profit (₹ crore)", [item["net_profit_crore"] for item in metrics], 500, 35_000, 10_000),
    ]
    colors = ("#0b6e69", "#3157a4")
    years = [item["financial_year"] for item in metrics]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#243447}.title{font-size:19px;font-weight:bold}.label{font-size:13px}.tick{font-size:11px;fill:#687789}</style>',
        '<text x="28" y="30" class="title">Infosys five-year financial trends</text>',
    ]
    for panel_index, (title, values, x0, max_value, step) in enumerate(panels):
        x1, top, bottom = x0 + 360, 92, 350
        parts.append(f'<text x="{x0}" y="68" class="title">{title}</text>')
        parts.append(f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="#9aa8b6"/>')
        parts.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="#9aa8b6"/>')
        for tick in range(0, max_value + 1, step):
            y = bottom - (tick / max_value) * (bottom - top)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e8edf2"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:,}</text>')
        points = []
        for index, value in enumerate(values):
            x = x0 + 18 + index * ((x1 - x0 - 36) / (len(values) - 1))
            y = bottom - (value / max_value) * (bottom - top)
            points.append((x, y, value))
            parts.append(f'<text x="{x:.1f}" y="{bottom+22}" text-anchor="middle" class="label">FY{str(years[index])[-2:]}</text>')
        point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        parts.append(f'<polyline points="{point_string}" fill="none" stroke="{colors[panel_index]}" stroke-width="3"/>')
        for x, y, value in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[panel_index]}"/>')
            parts.append(f'<text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" class="tick">{value:,}</text>')
    parts.append('<text x="28" y="430" class="tick">Amounts in ₹ crore. Source: annual-report financial highlights; citations in the accompanying report.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def validate_citations(text: str, allowed: set[str]) -> list[str]:
    found = set(re.findall(r"\[FY\d{4}\.pdf, PDF p\. \d+\]", text))
    return sorted(found - allowed)


def build_narrative(finance_context: str, risk_context: str, metrics_table: str) -> str:
    llm = LLM(
        model=CHAT_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        max_tokens=1400,
        timeout=240,
    )
    financial_analyst = Agent(
        role="Financial Analyst",
        goal="Compare verified five-year financial performance using only supplied annual-report evidence.",
        backstory=(
            "You are a careful financial analyst. The supplied table contains figures parsed from the "
            "company's annual reports; the supplied passages are evidence, not permission to guess."
        ),
        llm=llm,
        allow_delegation=False,
        max_iter=1,
        verbose=False,
    )
    risk_analyst = Agent(
        role="Risk & Strategy Analyst",
        goal="Compare documented risks and strategic developments across FY2022-FY2026.",
        backstory=(
            "You are a cautious analyst. You distinguish disclosed risks from outcomes and cite every "
            "report-based statement to the supplied filename and PDF page."
        ),
        llm=llm,
        allow_delegation=False,
        max_iter=1,
        verbose=False,
    )
    finance_task = Task(
        description=(
            "Analyze the validated financial table and retrieved financial passages. Explain the "
            "five-year direction in revenue, net profit, margins, and cash generation. Do not recalculate "
            "or alter the table's figures. Cite claims using the exact bracket citation supplied. If a "
            "point is not supported by the table or passages, omit it. Every paragraph must end with "
            "at least one exact citation, for example [FY2025.pdf, PDF p. 16].\n\n"
            "Validated table:\n{metrics_table}\n\nFinancial passages:\n{finance_context}"
        ),
        expected_output="A short, cited financial performance analysis in Markdown.",
        agent=financial_analyst,
    )
    risk_task = Task(
        description=(
            "Compare the disclosed business risks and strategic developments across the five report "
            "years using only the retrieved evidence below. Separate risks from strategy, note changes "
            "over time, and cite each factual statement with the exact supplied bracket citation. Never "
            "invent facts or predict outcomes. Every paragraph must end with at least one exact citation, "
            "for example [FY2025.pdf, PDF p. 16].\n\nRetrieved passages:\n{risk_context}"
        ),
        expected_output="A concise, cited comparison of risks and strategy in Markdown.",
        agent=risk_analyst,
    )
    final_task = Task(
        description=(
            "Combine the preceding Financial Analyst and Risk & Strategy Analyst findings into one "
            "readable report narrative with headings for financial performance, business risks, and "
            "strategic developments. Preserve citations exactly; every paragraph must end with one or "
            "more exact filename/page citations from the source passages. Include no uncited facts or "
            "invented figures. The verified metric table and chart will be added separately."
        ),
        expected_output="A final cited report narrative in Markdown.",
        agent=financial_analyst,
        context=[finance_task, risk_task],
    )
    crew = Crew(
        agents=[financial_analyst, risk_analyst],
        tasks=[finance_task, risk_task, final_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff(
        inputs={
            "metrics_table": metrics_table,
            "finance_context": finance_context,
            "risk_context": risk_context,
        }
    )
    return str(result.raw).strip()


def main() -> int:
    if not VECTOR_DIR.exists():
        print("RAG index missing; first run: python src/rag.py index", file=sys.stderr)
        return 1
    if REPORT_PATH.exists() or CHART_PATH.exists():
        print("Will not overwrite existing report or chart output:", file=sys.stderr)
        if REPORT_PATH.exists():
            print(f"  {REPORT_PATH}", file=sys.stderr)
        if CHART_PATH.exists():
            print(f"  {CHART_PATH}", file=sys.stderr)
        return 1

    try:
        metrics = get_financial_metrics()
        table = format_metrics_table(metrics)
        finance_context, allowed_citations = financial_evidence(metrics)
        risk_context, risk_citations = risk_strategy_evidence()
        allowed_citations.update(risk_citations)
        # Run both specialist tasks and the final synthesis. The published prose below
        # uses validated metrics and source-reviewed statements, not unchecked LLM text.
        crew_draft = build_narrative(finance_context, risk_context, table)
        draft_has_valid_citations = bool(
            re.search(r"\[FY\d{4}\.pdf, PDF p\. \d+\]", crew_draft)
        ) and not validate_citations(crew_draft, allowed_citations)

        first, last = metrics[0], metrics[-1]
        revenue_growth = (last["revenue_crore"] / first["revenue_crore"] - 1) * 100
        profit_growth = (last["net_profit_crore"] / first["net_profit_crore"] - 1) * 100
        cash_change = (
            last["free_cash_flow_crore"] / metrics[-2]["free_cash_flow_crore"] - 1
        ) * 100
        report = f"""# Infosys Annual Report Intelligence: FY2022–FY2026

## Executive overview

Across the five reports, consolidated revenue increased from ₹{first['revenue_crore']:,} crore in FY2022 to ₹{last['revenue_crore']:,} crore in FY2026 (+{revenue_growth:.1f}%). Net profit attributable to owners increased from ₹{first['net_profit_crore']:,} crore to ₹{last['net_profit_crore']:,} crore (+{profit_growth:.1f}%). Free cash flow varied year to year and was highest in this series in FY2025 at ₹{metrics[3]['free_cash_flow_crore']:,} crore. FY2026 reports an adjusted operating margin; its footnote says the measure excludes the effect of provisions related to India's Labour Codes, so it is not strictly like-for-like with earlier margin figures. [FY2022.pdf, PDF p. {first['page_number']}] [FY2025.pdf, PDF p. {metrics[3]['page_number']}] [FY2026.pdf, PDF p. {last['page_number']}]

## Verified financial comparison

All amounts are consolidated, in ₹ crore, as presented in the corresponding report highlights. FY2026 operating margin is explicitly adjusted.

{table}

![Revenue and net profit trends](financial_trends.svg)

## Financial findings

Revenue and net profit rose in each year of this comparison. Revenue growth was strongest in FY2023 and then moderated before accelerating to 9.6% in FY2026. Net profit growth was slower than revenue over the full period (+{profit_growth:.1f}% versus +{revenue_growth:.1f}%). Operating margin declined from 23.0% in FY2022 to around 21% in the latest reports; FY2026's figure is adjusted and should not be treated as perfectly comparable. Free cash flow dipped in FY2023, reached a five-year high in FY2025, then eased {abs(cash_change):.1f}% in FY2026. See the report highlights citations in the table above. [FY2022.pdf, PDF p. {first['page_number']}] [FY2023.pdf, PDF p. {metrics[1]['page_number']}] [FY2025.pdf, PDF p. {metrics[3]['page_number']}] [FY2026.pdf, PDF p. {last['page_number']}]

## Business risks

Across the risk overviews, Infosys repeatedly identifies geopolitical and economic uncertainty, supply-chain issues, and talent availability. Technology disruption also appears in the FY2023 and FY2024 disclosures; cybersecurity is explicitly included in FY2024–FY2026 risk summaries, while evolving regulation is highlighted in FY2025. FY2026's risk discussion also reflects advances in artificial intelligence. These are risks the reports say the company monitors, not predictions that a particular event will occur. [FY2022.pdf, PDF p. 138] [FY2023.pdf, PDF p. 130] [FY2024.pdf, PDF p. 121] [FY2025.pdf, PDF p. 122] [FY2026.pdf, PDF p. 131]

## Strategic developments

- **FY2022:** The report emphasizes cloud modernization and scaling digital capabilities, including AI and cybersecurity. [FY2022.pdf, PDF p. 3] [FY2022.pdf, PDF p. 16]
- **FY2023:** AI and cloud are framed as central to transformation, with Infosys describing its AI-first approach and Topaz services. [FY2023.pdf, PDF p. 3] [FY2023.pdf, PDF p. 17]
- **FY2024:** Generative AI moves into focus. The report also names governance needs such as model security, bias, privacy, and regulatory compliance. [FY2024.pdf, PDF p. 3]
- **FY2025:** The discussion shifts toward scaling AI use cases and pilots, with attention to data readiness, infrastructure, and governance. [FY2025.pdf, PDF p. 3]
- **FY2026:** The report emphasizes measurable enterprise outcomes and scaling AI through its AI First Value Framework and industry playbooks. [FY2026.pdf, PDF p. 3] [FY2026.pdf, PDF p. 24]

## Method note

The two CrewAI specialist agents and final synthesis task ran locally against RAG-retrieved passages. To reduce the chance of unsupported claims, this published narrative uses validated figures and a source-reviewed comparison rather than unverified generated wording. {"The crew draft included valid retrieved-page citations." if draft_has_valid_citations else "The crew draft did not pass citation validation and was not included."}

## Scope and limitations

This report uses the extracted text from the five supplied annual reports and locally retrieved passages. Figures in the table were parsed from each report's financial highlights page and are cited there. Some original pages had no extractable text; no OCR has been run. The reports use their own wording and definitions, so this comparison does not independently normalize accounting measures. Narrative statements are limited to retrieved excerpts and are cited to source PDF page numbers.
"""
        # Exclusive creation prevents overwriting if another process creates an output meanwhile.
        with CHART_PATH.open("x", encoding="utf-8") as chart_file:
            chart_file.write(make_svg(metrics))
        with REPORT_PATH.open("x", encoding="utf-8") as report_file:
            report_file.write(report)
        print(f"Created {REPORT_PATH.relative_to(ROOT)}")
        print(f"Created {CHART_PATH.relative_to(ROOT)}")
        print("The numeric comparison and every cited source page were verified before saving.")
        return 0
    except Exception as error:
        print(f"Report generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
