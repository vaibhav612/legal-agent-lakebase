"""Table-aware answers (fix 011, Stage C, 2026-10-30).

A request like 'Can you show me Arizona assigned patents?' targets a
STRUCTURED table (Schedule 1.1(e) etc.).  UC chunks_v2 preserves the exact
table markup in table_html (column headers, cell boundaries, row order),
and fix 011 syncs that markup into Lakebase document_chunks_v2.

Rendering such a request through the prose-only LLM answer template loses
the structure (the flattened chunk_text glues values together) — so this
module builds a DETERMINISTIC structured-table answer:

  * detect the table request from the query phrasing,
  * merge the same-section table chunks (multi-page continuations) in
    original chunk order,
  * emit {columns, rows} with EXACT source cell values (never numerically
    coerced, never inferred — a missing cell stays empty),
  * produce a grouped SourceRef with the merged PDF page range so the UI
    can open the original PDF.

No LLM is used for cell values.  Dependency-free (stdlib html.parser)."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional

from .state import EvidenceItem, SourceRef


# ── Table-request detection ─────────────────────────────────────────

# 'Show me / list / display' phrasings that ask to SEE the content of
# something (as opposed to asking a question about it).
_SHOW_PATTERNS: list[str] = [
    "show me", "can you show", "can i see", "show the", "show all",
    "list the", "list all", "display the", "lay out", "give me the table",
    "give me a table", "print the table", "output the table",
]

# Subject words that signal the requested content is a structured table
# (patent/trademark/domain schedules, tables, inventories).
_TABLE_SUBJECTS: list[str] = [
    "table", "patents", "patent", "trademarks", "trademark",
    "domain names", "domains", "schedule", "schedules", "inventory",
    "list of", "rows",
]


def detect_table_request(query: str) -> bool:
    """True when the query asks to SEE structured tabular content.

    Requires BOTH a show/list phrasing and a table-ish subject word, so a
    question like 'what are the insurance requirements?' (prose answer)
    never takes the table path, and 'show me the patents' does.
    """
    q = (query or "").lower()
    has_show = any(p in q for p in _SHOW_PATTERNS)
    has_subject = any(s in q for s in _TABLE_SUBJECTS)
    return has_show and has_subject


# ── HTML table parsing (stdlib only) ────────────────────────────────

class _TableParser(HTMLParser):
    """Extract header cells and data rows from simple <table> markup.

    Cell text is accumulated exactly as present (convert_charrefs=True
    unescapes entities); whitespace inside cells is collapsed.  The first
    row's <th> cells are the column headers.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: Optional[list[str]] = None
        self._cell: Optional[list[str]] = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("th", "td") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_table_html(table_html: str) -> Optional[dict]:
    """Parse stored <table> markup into {columns, rows}.

    columns come from the first row's header cells; rows are lists of exact
    source strings.  Returns None when the markup is empty or unparsable
    (never a guessed structure).
    """
    if not table_html or "<table" not in table_html.lower():
        return None
    parser = _TableParser()
    try:
        parser.feed(table_html)
        parser.close()
    except Exception:
        return None
    rows = parser.rows
    if not rows:
        return None
    return {"columns": rows[0], "rows": rows[1:]}


def html_table_to_markdown(table_html: str) -> str:
    """Render stored table markup as a Markdown pipe table (LLM context).

    Every value is the exact source string; empty cells stay empty.
    Returns "" when the markup does not parse.
    """
    parsed = parse_table_html(table_html)
    if not parsed:
        return ""
    cols = parsed["columns"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in parsed["rows"]:
        lines.append("| " + " | ".join(c if c else "" for c in row) + " |")
    return "\n".join(lines)


# ── Multi-chunk table assembly ──────────────────────────────────────

def _chunk_sort_key(e: EvidenceItem):
    """Order table chunks by original chunk_index (continuation order).

    chunk ids are '<doc_id>_<index>'; fall back to string order otherwise.
    """
    m = re.search(r"_(\d+)$", e.chunk_id or "")
    return (int(m.group(1)) if m else 10**9, e.chunk_id or "")


def merge_table_chunks(evidence: list[EvidenceItem]) -> Optional[dict]:
    """Merge same-section table chunks into ONE structured table.

    A long schedule continues across PDF pages as multiple chunks sharing
    the same (doc_id, section_path); they are ordered by chunk_index and
    their rows concatenated (repeated header rows from continuation pages
    are dropped).  Column headers come from the first chunk.  Never invents
    or reorders cells; a missing cell stays empty.
    """
    table_items = [e for e in evidence if (e.table_html or "").strip()]
    if not table_items:
        return None
    first = table_items[0]
    group_key = (first.doc_id, first.section_path)
    members = sorted(
        [e for e in table_items if (e.doc_id, e.section_path) == group_key],
        key=_chunk_sort_key,
    )
    if not members:
        return None

    columns: list[str] = []
    rows: list[list[str]] = []
    for member in members:
        parsed = parse_table_html(member.table_html)
        if not parsed:
            continue
        if not columns:
            columns = parsed["columns"]
        for row in parsed["rows"]:
            # Continuation pages often repeat the header row — drop an exact
            # repeat of the canonical columns, keep everything else verbatim.
            if row == columns:
                continue
            rows.append(row)
    if not columns:
        return None

    pages = [e for e in members if e.page_start is not None]
    page_start = min((e.page_start for e in pages), default=None)
    page_end = max((e.page_end for e in members if e.page_end is not None),
                   default=page_start)

    return {
        "columns": columns,
        "rows": rows,
        "doc_id": first.doc_id,
        "doc_name": first.doc_name,
        "section_path": first.section_path,
        "page_start": page_start,
        "page_end": page_end,
        "chunk_ids": [e.chunk_id for e in members],
        "row_count": len(rows),
        "continuation_note": (
            "Table continues across multiple PDF pages; rows are shown in "
            "original order."
            if len(members) > 1 else None
        ),
    }


# ── Deterministic table answer ─────────────────────────────────────

def build_table_answer(query: str, evidence: list[EvidenceItem]) -> Optional[dict]:
    """Build the full table-answer payload for a detected table request.

    Returns {answer_text, table_result, source_refs} or None when the
    evidence contains no usable table.  The answer text is deterministic
    (no LLM): an honest introduction naming the table, row count, and PDF
    page range — the UI renders the actual table separately.
    """
    if not detect_table_request(query):
        return None
    merged = merge_table_chunks(evidence)
    if not merged:
        return None

    from .display import display_doc_name, physical_page

    doc_title = display_doc_name(merged["doc_name"]) or merged["doc_name"]
    p1 = physical_page(merged["page_start"])
    p2 = physical_page(merged["page_end"])
    if p1 is not None and p2 is not None and p1 != p2:
        loc = f"PDF pages {p1}\u2013{p2}"
    elif p1 is not None:
        loc = f"PDF page {p1}"
    else:
        loc = "the original PDF"
    section = merged["section_path"] or "the requested table"

    lines = [
        f"Here is **{section}** from the {doc_title} ({loc}) \u2014 "
        f"{merged['row_count']} rows, shown below with the original "
        "columns and exact cell values.",
    ]
    if merged["continuation_note"]:
        lines.append(f"*{merged['continuation_note']}*")
    answer_text = "\n\n".join(lines)

    source_ref = SourceRef(
        doc_id=merged["doc_id"],
        doc_name=merged["doc_name"],
        section_path=merged["section_path"],
        page_start=merged["page_start"],
        page_end=merged["page_end"],
        excerpt=(merged["section_path"] or "") + f" \u2014 {merged['row_count']} rows ({loc})",
        chunk_id=merged["chunk_ids"][0] if merged["chunk_ids"] else None,
    )
    return {
        "answer_text": answer_text,
        "table_result": merged,
        "source_refs": [source_ref],
    }
