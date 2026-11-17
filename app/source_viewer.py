"""Source viewer: canonical-PDF page rendering with exact evidence highlighting.

Stage 3 fix (2026-10-29). Implements the spec's source-page navigation
(Acceptance 15: open/preview the PDF at the correct page when technically
supported) and exact chunk highlighting.

Verified facts this module is built on (audit 2026-10-29, live against the
Armstrong PDF):

* Canonical PDFs live in UC volume workspace.legal_assistant.legal_docs_volume
  (documents.file_path, URL-encoded), readable via the Databricks Files API
  (w.files.download(file_path=...).contents).
* chunks_v2.bboxes / evidence.rects are pixel rectangles at the per-page
  scale pages.px_per_pt (array<struct<page_id:int, coord:int[]>>), TOP-LEFT
  origin, page_id 0-indexed == pypdfium2 page index. Verified: the Armstrong
  governing-law rect [203,253,1531,823] px on page_id 12 contains exactly
  the governing-law clause text on pdf[12].
* pages.scale_status records calibration provenance ('assumed_200dpi'); it
  is surfaced to the user rather than hidden.

Honesty rules: a missing/invalid bbox or missing calibration NEVER produces
a guessed highlight - the page renders plain and highlighted=False with an
explicit note. Nothing is fabricated: file paths come from the documents
table; rects from the database; no URLs are invented.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from workflow.display import display_doc_name, physical_page

logger = logging.getLogger(__name__)

# Read-only warehouse lookups (tables granted to the app identity 2026-10-29)
_DOCUMENTS_TABLE = "workspace.legal_assistant_v2.documents"
_PAGES_TABLE = "workspace.legal_assistant_v2.pages"
_CHUNKS_TABLE = "workspace.legal_assistant_v2.chunks_v2"

# Display cap for rendered pages (pure downscale of the rendered image AND
# the overlay together, so alignment is preserved).
MAX_RENDER_WIDTH_PX = 1400


def parse_rects(raw):
    """Normalise a stored rects/bboxes column into [{'page_id': int,
    'coord': [x1, y1, x2, y2]}]. Tolerant of JSON-string or decoded-list
    executor output; anything unrecognised yields [] (never a guess)."""
    import json as _json
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("page_id"))
            coord = [int(c) for c in (item.get("coord") or [])]
        except (TypeError, ValueError):
            continue
        if len(coord) == 4 and pid >= 0:
            out.append({"page_id": pid, "coord": coord})
    return out


def load_pdf_bytes(file_path: str) -> bytes:
    """Read a canonical PDF from its UC volume path via the Files API.

    file_path is the EXACT stored (URL-encoded) path from the documents
    table - never a constructed or guessed path. The Files API accepts the
    human-readable (decoded) form of a /Volumes path.
    """
    from urllib.parse import unquote
    from databricks.sdk import WorkspaceClient

    if not file_path or not str(file_path).startswith("/Volumes/"):
        raise ValueError("No canonical PDF path is available for this source.")
    w = WorkspaceClient()
    resp = w.files.download(file_path=unquote(str(file_path)))
    contents = resp.contents
    if hasattr(contents, "read"):
        return contents.read()
    if isinstance(contents, (bytes, bytearray)):
        return bytes(contents)
    return b"".join(contents)


def get_doc_pdf_info(run_sql: Callable, doc_id: str) -> Optional[dict]:
    """file_name / file_path / page_count for a document (bound :doc_id)."""
    rows = run_sql(
        "SELECT file_name, file_path, page_count FROM " + _DOCUMENTS_TABLE
        + " WHERE doc_id = :doc_id LIMIT 1",
        {"doc_id": str(doc_id)},
    )
    if not rows:
        return None
    row = rows[0] or {}
    return {
        "file_name": row.get("file_name"),
        "file_path": row.get("file_path"),
        "page_count": row.get("page_count"),
    }


def get_page_calibration(run_sql: Callable, doc_id: str, page_id: int) -> Optional[dict]:
    """px_per_pt / width_pt / height_pt / scale_status for one page."""
    rows = run_sql(
        "SELECT px_per_pt, width_pt, height_pt, scale_status FROM " + _PAGES_TABLE
        + " WHERE doc_id = :doc_id AND page_id = :page_id LIMIT 1",
        {"doc_id": str(doc_id), "page_id": int(page_id)},
    )
    if not rows:
        return None
    row = rows[0] or {}
    try:
        return {
            "px_per_pt": float(row.get("px_per_pt")),
            "width_pt": float(row.get("width_pt")),
            "height_pt": float(row.get("height_pt")),
            "scale_status": row.get("scale_status"),
        }
    except (TypeError, ValueError):
        return None


def get_chunk_bboxes(run_sql: Callable, chunk_id: str) -> list:
    """Authoritative chunk bboxes from chunks_v2, keyed by the shared
    chunk_id (the same identity used by Lakebase document_chunks_v2 - the
    two stores are joined by chunk_id, never by re-derived text)."""
    rows = run_sql(
        "SELECT bboxes FROM " + _CHUNKS_TABLE + " WHERE chunk_id = :chunk_id LIMIT 1",
        {"chunk_id": str(chunk_id)},
    )
    if not rows:
        return []
    return parse_rects((rows[0] or {}).get("bboxes"))


def render_page_with_highlight(
    pdf_bytes: bytes,
    page_id: int,
    rects: Optional[list],
    px_per_pt: Optional[float] = None,
    max_width_px: int = MAX_RENDER_WIDTH_PX,
) -> dict:
    """Render the cited page and draw the exact highlight rectangles.

    Coordinate convention (verified 2026-10-29): rect coords are pixels at
    px_per_pt scale, top-left origin; pypdfium2 renders at scale=px_per_pt
    px/pt, so coords map 1:1 onto the rendered bitmap. When px_per_pt is
    unknown or no rect targets this page, the page still renders and the
    result reports highlighted=False with an explicit note - never a
    guessed highlight.

    Returns a dict with keys: image (PIL image), physical_page (1-based),
    highlighted (bool), note (str); or {'error': ...} for an invalid page.
    """
    import pypdfium2 as pdfium

    page_id = int(page_id)
    pdf = pdfium.PdfDocument(pdf_bytes)
    if page_id < 0 or page_id >= len(pdf):
        return {
            "error": "Stored page_id " + str(page_id) + " is outside this PDF "
                     "(1.." + str(len(pdf)) + " pages). No page rendered.",
        }

    scale = float(px_per_pt) if px_per_pt else 1.0
    bitmap = pdf[page_id].render(scale=scale)
    image = bitmap.to_pil().convert("RGB")
    rendered_w, rendered_h = image.size

    target_rects = [
        r for r in (rects or [])
        if int(r.get("page_id", -1)) == page_id and len(r.get("coord") or []) == 4
    ]

    highlighted = False
    note = ""
    if not px_per_pt:
        note = ("Exact highlight unavailable: no page calibration (px_per_pt) "
                "is recorded for this page. Rendering the page without a highlight.")
    elif not target_rects:
        note = ("No stored bounding box targets this page. Rendering the page "
                "without a highlight.")
    else:
        from PIL import ImageDraw
        overlay = ImageDraw.Draw(image, "RGBA")
        clipped = 0
        for r in target_rects:
            x1, y1, x2, y2 = [float(c) for c in r["coord"]]
            if x2 <= x1 or y2 <= y1:
                continue
            if x1 < 0 or y1 < 0 or x2 > rendered_w or y2 > rendered_h:
                clipped += 1
            x1c, y1c = max(0.0, x1), max(0.0, y1)
            x2c = min(float(rendered_w), x2)
            y2c = min(float(rendered_h), y2)
            overlay.rectangle([x1c, y1c, x2c, y2c], fill=(255, 213, 0, 66),
                              outline=(214, 64, 20, 220), width=3)
        highlighted = True
        note = ("Highlighted region = stored evidence rectangle (top-left px "
                "origin, scale px_per_pt).")
        if clipped:
            note += (" " + str(clipped) + " rectangle(s) extend beyond the "
                     "rendered page and were clipped - alignment may be imperfect.")

    if image.width > max_width_px:
        ratio = max_width_px / image.width
        image = image.resize((max_width_px, max(1, int(image.height * ratio))))

    return {
        "image": image,
        "physical_page": page_id + 1,
        "highlighted": highlighted,
        "note": note,
    }


def build_source_view(run_sql: Callable, ref) -> dict:
    """Assemble everything the UI needs to show one source reference.

    ``ref`` is a SourceRef-shaped object (pydantic model or mapping) with
    the canonical identifiers. Missing warehouse data is reported, never
    fabricated. Returns a dict with keys: display_name, file_path,
    page_id, physical_page, page_count, rects, rect_source, calibration,
    error (str or None).
    """
    if isinstance(ref, dict):
        get = ref.get
    else:
        get = lambda k: getattr(ref, k, None)

    doc_id = get("doc_id")
    info = get_doc_pdf_info(run_sql, doc_id) if doc_id else None
    file_path = get("file_path") or (info or {}).get("file_path")
    page_count = get("page_count") or (info or {}).get("page_count")

    page_id = get("page_start")
    rects = get("rects") or []
    rect_source = get("rect_source")

    # Hybrid-route refs carry no rects (Lakebase has none); fetch the
    # authoritative chunk bboxes from chunks_v2 via the shared chunk_id.
    if not rects and get("chunk_id"):
        try:
            rects = get_chunk_bboxes(run_sql, get("chunk_id"))
            rect_source = "chunk_bbox (chunks_v2.bboxes)"
        except Exception as exc:  # honest degradation, never a guess
            logger.warning("chunk bbox lookup failed for %s: %s", get("chunk_id"), exc)
            rects = []

    calibration = None
    if page_id is not None and doc_id:
        try:
            calibration = get_page_calibration(run_sql, doc_id, int(page_id))
        except Exception as exc:
            logger.warning("calibration lookup failed for %s p%s: %s", doc_id, page_id, exc)

    out = {
        "display_name": display_doc_name(get("doc_name")),
        "file_path": file_path,
        "page_id": page_id,
        "physical_page": physical_page(page_id),
        "page_count": page_count,
        "rects": rects,
        "rect_source": rect_source,
        "calibration": calibration,
        "error": None,
    }
    if not file_path:
        out["error"] = ("No canonical PDF path is recorded for this document "
                        "- page preview unavailable.")
    return out
