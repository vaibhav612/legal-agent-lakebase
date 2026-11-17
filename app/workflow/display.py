"""Display-name helpers for citations and source panels (Stage 2/3 fix, 2026-10-29).

Deterministic, LLM-free formatting of stored identifiers for HUMAN display
only. These helpers NEVER modify canonical values: `doc_id`, `file_name`,
`file_path`, `chunk_id`, and database records keep their stored (URL-encoded)
form everywhere outside the presentation layer.

The canonical filename convention in this corpus (verified against all 60
documents, 2026-10-29) is one of:

* EDGAR-style (48 docs):  {Company}_{YYYYMMDD}_{Form}_{Accession}_{Exhibit}_{Title}.pdf
  -- the Title segment carries its spaces as %20 and contains no underscore.
* Legacy EDGAR (9 docs):  {COMPANY}_{MM}_{DD}_{YYYY-EX-x.x-TITLE}.PDF (all caps,
  dash-separated tail).
* Titles with an internal underscore (3 docs): the 8-part variant of the
  first pattern, e.g. "..._License%20Agreement_%20Reseller%20Agreement.pdf".

`display_doc_name` parses ONLY what is present in the stored filename -- it
never invents clauses, parties, or titles.
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import unquote


def _humanize_company(token: str) -> str:
    """Insert spaces at lower->upper-case boundaries ('ArmstrongFlooringInc' ->
    'Armstrong Flooring Inc'). All-caps tokens (e.g. 'ACCELERATEDTECHNOLOGIESHOLDINGCORP')
    have no such boundary and are returned unchanged -- never a guessed split.
    """
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", token)


def display_doc_name(raw) -> str:
    """Human-readable display title for a stored (URL-encoded) file name.

    Deterministic, derived from the stored name only: URL-decode, drop the
    .pdf suffix, split on '_' (EDGAR-style names yield company in part 0 and
    the document title in parts 5+); legacy all-caps names use the
    dash-separated tail segment as title; the company token is camel-case
    spaced only when it is mixed case. Falls back to the decoded base name
    whenever the structure is not recognised. Never raises on odd input.
    """
    if not raw:
        return ""
    s = unquote(str(raw)).strip()
    s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE).strip()
    if not s:
        return ""
    parts = [p.strip() for p in s.split("_") if p.strip()]
    if not parts:
        return s
    if len(parts) >= 7:
        # EDGAR-style: company_date_form_exhibit_accession_EXHIBIT-REPEAT_title(s)
        # (verified across the 48 seven-part names in the corpus: the exhibit
        # token is repeated in part 5 and the human title is parts 6+).
        company = _humanize_company(parts[0])
        title = parts[6] if len(parts) == 7 else " - ".join(parts[6:])
        return f"{company} — {title}" if title else company
    if len(parts) == 6:
        # Six-part variant (none in the current corpus, kept for safety):
        # company_date_form_exhibit_title
        company = _humanize_company(parts[0])
        title = " - ".join(parts[5:])
        return f"{company} — {title}" if title else company
    if len(parts) == 4 and "-" in parts[-1]:
        # Legacy all-caps: COMPANY_MM_DD_YYYY-EX-x.x-TITLE
        company = parts[0]
        segs = [seg.strip() for seg in parts[-1].split("-") if seg.strip()]
        with_space = [seg for seg in segs if " " in seg]
        title = max(with_space, key=len) if with_space else (segs[-1] if segs else "")
        return f"{company} — {title}" if title else company
    # Unrecognised structure: decoded base name with underscores as separators.
    return " ".join(parts)


def physical_page(page_id_zero_based) -> Optional[int]:
    """Convert the corpus 0-indexed page id to the 1-based physical PDF page
    number (what a PDF viewer shows). Verified against the Armstrong PDF:
    governing-law bboxes page_id 12 render on pdf[12] (0-indexed) == viewer
    page 13. Returns None for None/invalid input -- never a guess.
    """
    try:
        p = int(page_id_zero_based)
    except (TypeError, ValueError):
        return None
    return p + 1 if p >= 0 else None
