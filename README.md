# Infosys Annual Report Intelligence

A local, beginner-friendly Python project for searching and comparing five Infosys annual reports (FY2022–FY2026). It reuses the extracted page-by-page text already in `data/processed/`; it does not re-extract the PDFs during RAG or report generation.

## What it does

1. Reads each report's existing page-labeled text.
2. Splits text into overlapping chunks without crossing page boundaries.
3. Uses ChromaDB for local vector search and Ollama for local text embeddings.
4. Stores each chunk's financial year, source PDF filename, PDF page number, and chunk number as metadata.
5. Uses a retrieval test script to show relevant passages and citations.
6. Runs a CrewAI workflow with a **Financial Analyst**, a **Risk & Strategy Analyst**, and a final synthesis task. The published report uses validated financial figures and source-reviewed summaries rather than trusting unverified model prose.
7. Writes a Markdown comparison and a simple SVG chart under `outputs/`.

Everything runs locally through the Ollama HTTP API. The project does not require API keys, Docker, a cloud model, RAG frameworks beyond ChromaDB, or OCR. It does not use MCP or n8n.

## Requirements

- Ubuntu/Linux
- Python 3.13 (the existing `.venv` uses Python 3.13)
- Ollama service reachable at `http://127.0.0.1:11434`
- These local Ollama models:
  - `nomic-embed-text:latest` for embeddings
  - `qwen2.5:latest` for the CrewAI analyst tasks
- Python dependencies pinned in [requirements.txt](requirements.txt)

Current package versions used and tested: ChromaDB 1.5.9, CrewAI 1.15.28, and pypdf 6.20.1. CrewAI and its LiteLLM extra support this Python 3.13 environment; Ollama model calls and ChromaDB retrieval were tested locally.

## First-time setup

Open a terminal in the project folder and activate the existing virtual environment:

```bash
source .venv/bin/activate
```

If you are setting up a fresh copy of the project, create a virtual environment with Python 3.13 first, then install the pinned dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Make sure Ollama is running and that the two model names above are available. The RAG indexer does not download models automatically. To check the local API, open `http://127.0.0.1:11434/api/tags` or use the Ollama application already configured on your machine.

## Run the project

Run these commands from the project root with `.venv` activated.

### 1. Build the local RAG index

```bash
python src/rag.py index
```

This reads the five existing files in `data/processed/` and creates `data/chroma_db/`. It preserves page numbers and source names in metadata. It will **stop instead of replacing** an existing index.

### 2. Test retrieval

```bash
python src/test_retrieval.py
```

The script asks sample financial and risk/strategy questions and prints retrieved excerpts with source PDF filenames and page numbers. To ask one question yourself:

```bash
python src/test_retrieval.py "How did revenue and net profit change?"
```

### 3. Generate the report and chart

```bash
python src/analyze_reports.py
```

The workflow runs the two analyst agents and final synthesis task, validates financial figures and report-page citations, and writes:

- `outputs/infosys_five_year_analysis.md`
- `outputs/financial_trends.svg`

It **will not overwrite** either output if it already exists. Keep, rename, or remove existing generated outputs deliberately before requesting a fresh version. It never edits or re-extracts the original PDFs or processed text.

## Project layout

```text
.
├── data/
│   ├── raw/                 # Original PDF copies
│   ├── processed/           # Existing extracted text with page headings
│   └── chroma_db/            # Generated local vector index
├── outputs/                  # Final report and SVG chart
├── src/
│   ├── extract_reports.py   # Original PDF extractor; not rerun by RAG
│   ├── rag.py               # Chunking, Chroma index, Ollama embeddings, search
│   ├── test_retrieval.py    # Example retrieval questions
│   ├── financial_data.py    # Validated figures from report highlights pages
│   └── analyze_reports.py   # CrewAI workflow and output generation
├── requirements.txt
└── README.md
```

## Notes and limitations

- Citations use the physical PDF page number corresponding to each `--- Page N ---` heading in the processed text.
- The reports contain a few pages from which no text could be extracted. No OCR is performed, so image-only content on such pages is not searchable.
- FY2026 reports an **adjusted** operating margin and explains an adjustment related to India's Labour Codes. Treat that margin cautiously when comparing it with earlier years.
- Financial figures are taken from the annual-report highlights pages. Always check the cited PDF page before using numbers in formal decisions.
- CrewAI's generated draft is not treated as automatically correct. The final narrative emphasizes validated figures and cited source disclosures to reduce unsupported statements.
- The original annual reports and extracted text may be proprietary. Do not publish them or the local Chroma database unless you have permission.
