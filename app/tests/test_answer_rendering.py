"""Stage 2/3 regression tests (2026-10-29): answer rendering, display
names, citation metadata, extraction-status preservation, and the source
viewer (page selection, coordinates, honest missing-bbox).

No network, no warehouse, no LLM: pure functions plus fake executors and a
generated blank PDF.
"""

from __future__ import annotations

import json
import os
import sys

_app_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_app_root))

import pytest

from workflow.display import display_doc_name, physical_page
from workflow.state import SourceRef
from workflow.structured_lookup import _build_answer_text

ARMSTRONG = "ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual%20Property%20Agreement.pdf"
ARMSTRONG_DISPLAY = "Armstrong Flooring Inc — Intellectual Property Agreement"


# -- display_doc_name: real corpus naming patterns (verified 2026-10-29) --

@pytest.mark.parametrize("raw,expected", [
    (ARMSTRONG, ARMSTRONG_DISPLAY),
    ("BellringBrandsInc_20190920_S-1_EX-10.12_11817081_EX-10.12_Manufacturing%20Agreement1.pdf",
     "Bellring Brands Inc — Manufacturing Agreement1"),
    ("EhaveInc_20190515_20-F_EX-4.44_11678816_EX-4.44_License%20Agreement_%20Reseller%20Agreement.pdf",
     "Ehave Inc — License Agreement - Reseller Agreement"),
    ("ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT%20VENTURE%20AGREEMENT.PDF",
     "ACCELERATEDTECHNOLOGIESHOLDINGCORP — JOINT VENTURE AGREEMENT"),
])
def test_display_doc_name_corpus_patterns(raw, expected):
    assert display_doc_name(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("BellringBrands_MSA.pdf", "BellringBrands MSA"),
    ("plain.pdf", "plain"),
    ("", ""),
    (None, ""),
])
def test_display_doc_name_fallbacks(raw, expected):
    assert display_doc_name(raw) == expected


def test_display_doc_name_decodes_only_for_display():
    out = display_doc_name(ARMSTRONG)
    assert "%20" not in out and ".pdf" not in out
    # the canonical stored name is untouched
    assert ARMSTRONG.endswith("Intellectual%20Property%20Agreement.pdf")


# -- physical_page: 0-indexed stored id -> 1-based viewer page -----------

@pytest.mark.parametrize("stored,expected", [(0, 1), (12, 13), (105, 106)])
def test_physical_page_mapping(stored, expected):
    assert physical_page(stored) == expected


@pytest.mark.parametrize("bad", [None, "", "x", -1])
def test_physical_page_invalid_is_none_not_a_guess(bad):
    assert physical_page(bad) is None


# -- _build_answer_text: deterministic natural-language rendering --------

def _doc(**kw):
    base = {
        "doc_id": "d1", "doc_name": ARMSTRONG, "is_canonical": True,
        "document_class": "contract", "file_path": "/Volumes/x/y.pdf",
        "page_count": 40, "field": "governing_law",
        "record_status": "has_record",
        "values": [{"value_raw": "Delaware", "value_normalized": "Delaware",
                    "value_status": "found_explicit", "value_index": 1,
                    "extraction_method": "llm", "run_id": "r1", "field_id": "f1"}],
        "evidence": [{"evidence_id": "e1", "chunk_id": "c21",
                      "quote": "This Agreement shall be governed by the laws of the State of Delaware.",
                      "page_start": 12, "page_end": 12,
                      "evidence_type": "direct_quote",
                      "section_path": "13. MISCELLANEOUS",
                      "rects": [{"page_id": 12, "coord": [203, 253, 1531, 823]}],
                      "rect_source": "element_box"}],
    }
    base.update(kw)
    return base


def _lookup(docs, mode="lookup", **kw):
    lk = {"target_field": "governing_law", "mode": mode,
          "results": docs, "status_counts": {}}
    lk.update(kw)
    return lk


def test_lead_sentence_found_explicit():
    text = _build_answer_text(_lookup([_doc()]))
    first = text.splitlines()[0]
    assert first == ("The governing law of the " + ARMSTRONG_DISPLAY
                     + " is 'Delaware' (explicitly stated in the document).")


@pytest.mark.parametrize("status,phrase", [
    ("inferred", "inferred from the document - not explicitly stated"),
    ("conflicting", "recorded with conflicting values"),
    ("redacted", "redacted in the source document"),
])
def test_lead_sentence_preserves_extraction_status_semantics(status, phrase):
    doc = _doc(values=[{"value_raw": "X", "value_normalized": "X",
                        "value_status": status, "value_index": 1,
                        "extraction_method": "llm", "run_id": "r1",
                        "field_id": "f1"}])
    text = _build_answer_text(_lookup([doc]))
    assert phrase in text.splitlines()[0]
    # the exact status is still reported verbatim in the detail line
    assert "extracted (" + status + ")" in text


def test_not_found_and_no_record_lead_sentences_and_caveat():
    nf = _doc(values=[{"value_raw": None, "value_normalized": None,
                       "value_status": "not_found", "value_index": 1,
                       "extraction_method": "llm", "run_id": "r",
                       "field_id": "f"}])
    text = _build_answer_text(_lookup([nf], status_counts={"not_found": 1}))
    first = text.splitlines()[0]
    assert "has no extracted value available (value_status=not_found)" in first
    assert "is not proof" in text  # MISSING_CAVEAT present

    nr = _doc(record_status="no_record", values=[], evidence=[])
    text2 = _build_answer_text(_lookup([nr], status_counts={"no_record": 1}))
    assert "has no extraction record for governing law" in text2.splitlines()[0]


def test_evidence_line_carries_section_and_physical_page_of_total():
    text = _build_answer_text(_lookup([_doc()]))
    assert "Section 13. MISCELLANEOUS" in text
    assert "page 13 of 40" in text
    assert "governed by the laws of the State of Delaware" in text


def test_evidence_line_never_invents_page_or_section_when_absent():
    doc = _doc(evidence=[{"evidence_id": "e1", "chunk_id": "c1",
                          "quote": "some quote", "page_start": None,
                          "page_end": None, "evidence_type": "direct_quote",
                          "section_path": None}])
    text = _build_answer_text(_lookup([doc]))
    ev = [l for l in text.splitlines() if l.strip().startswith("Evidence")]
    assert len(ev) == 1
    assert "Section" not in ev[0] and "page" not in ev[0]


def test_multi_doc_and_filter_modes_keep_structured_header():
    docs = [_doc(), _doc(doc_id="d2", doc_name="CerenceInc_20191002_8-K_EX-10.4_11827494_EX-10.4_Intellectual%20Property%20Agreement.pdf")]
    text = _build_answer_text(_lookup(docs, mode="value_filter", filter_value="Delaware"))
    assert "Structured lookup - documents where governing_law matches" in text
    # only the two scripted docs appear (Bellring is in neither)
    assert "Bellring" not in text
    assert "Cerence Inc — Intellectual Property Agreement" in text


def test_display_names_in_doc_lines_and_no_encoded_chars():
    text = _build_answer_text(_lookup([_doc()]))
    assert ARMSTRONG_DISPLAY in text
    assert "%20" not in text
    assert ".pdf" not in text


# -- source_viewer: page selection, coords, honest missing data ----------

from source_viewer import (build_source_view, get_chunk_bboxes,
                           load_pdf_bytes, parse_rects,
                           render_page_with_highlight)


def _blank_pdf():
    """4-page A4 blank PDF generated in-memory (no fixture file needed)."""
    import pypdfium2 as pdfium
    out = pdfium.PdfDocument.new()
    for _ in range(4):
        out.new_page(595, 842)
    import io
    buf = io.BytesIO()
    out.save(buf)
    return buf.getvalue()


@pytest.mark.parametrize("raw,expected", [
    (None, []),
    ("not json", []),
    ("42", []),
    ([{"page_id": 1, "coord": [1, 2, 3, 4]}], [{"page_id": 1, "coord": [1, 2, 3, 4]}]),
    ('[{"page_id": 12, "coord": [203, 253, 1531, 823]}]',
     [{"page_id": 12, "coord": [203, 253, 1531, 823]}]),
    ([{"page_id": -1, "coord": [1, 2, 3, 4]}, {"page_id": 1, "coord": [1, 2]},
      {"page_id": 1, "coord": None}, "junk", {"page_id": 1, "coord": [0, 0, 9, 9]}],
     [{"page_id": 1, "coord": [0, 0, 9, 9]}]),
])
def test_parse_rects_tolerant_and_honest(raw, expected):
    assert parse_rects(raw) == expected


class FakeViewerSql:
    """Fake executor keyed by table name; records calls."""

    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append({"sql": sql, "params": dict(params or {})})
        if "FROM workspace.legal_assistant_v2.documents" in sql:
            return [{"file_name": "x.pdf", "file_path": "/Volumes/a/b.pdf",
                     "page_count": 40}]
        if "FROM workspace.legal_assistant_v2.pages" in sql:
            return [{"px_per_pt": 2.7777777777777777, "width_pt": 595.0,
                     "height_pt": 842.0, "scale_status": "assumed_200dpi"}]
        if "FROM workspace.legal_assistant_v2.chunks_v2" in sql:
            return [{"bboxes": json.dumps([{"page_id": 1, "coord": [10, 20, 400, 700]}])}]
        raise AssertionError("unexpected sql: " + sql)


def test_build_source_view_assembles_and_enriches():
    ref = SourceRef(doc_id="d1", doc_name=ARMSTRONG, chunk_id="c21",
                    page_start=1, page_end=1)
    view = build_source_view(FakeViewerSql().execute, ref)
    assert view["error"] is None
    assert view["display_name"] == ARMSTRONG_DISPLAY
    assert view["file_path"] == "/Volumes/a/b.pdf"
    assert view["physical_page"] == 2 and view["page_count"] == 40
    assert view["rects"] == [{"page_id": 1, "coord": [10, 20, 400, 700]}]
    assert view["rect_source"] == "chunk_bbox (chunks_v2.bboxes)"
    assert view["calibration"]["px_per_pt"] == pytest.approx(2.77778, rel=1e-3)
    assert view["calibration"]["scale_status"] == "assumed_200dpi"


def test_build_source_view_missing_pdf_path_is_honest():
    class NoRow(FakeViewerSql):
        def execute(self, sql, params=None):
            if "documents" in sql:
                return []
            return FakeViewerSql.execute(self, sql, params)
    view = build_source_view(NoRow().execute,
                            SourceRef(doc_id="missing", doc_name="Unknown.pdf"))
    assert view["file_path"] is None
    assert "page preview unavailable" in (view["error"] or "")


def test_get_chunk_bboxes_passes_bound_params():
    ex = FakeViewerSql()
    get_chunk_bboxes(ex.execute, "c21")
    assert ex.calls[-1]["params"] == {"chunk_id": "c21"}


def test_render_page_rejects_out_of_range_page_id():
    res = render_page_with_highlight(_blank_pdf(), 7, [], px_per_pt=2.78)
    assert res["error"] and "outside this PDF" in res["error"]


def test_render_page_no_calibration_renders_plain_with_note():
    res = render_page_with_highlight(_blank_pdf(), 1,
                                     [{"page_id": 1, "coord": [10, 20, 400, 700]}],
                                     px_per_pt=None)
    assert res["highlighted"] is False
    assert "no page calibration" in res["note"].lower()
    assert res["image"].size[0] > 0


def test_render_page_no_target_rect_on_page_renders_plain_with_note():
    res = render_page_with_highlight(_blank_pdf(), 0,
                                     [{"page_id": 1, "coord": [10, 20, 400, 700]}],
                                     px_per_pt=2.78)
    assert res["highlighted"] is False
    assert "no stored bounding box targets this page" in res["note"].lower()


def test_render_page_highlight_localized_exactly():
    pdf = _blank_pdf()
    rects = [{"page_id": 1, "coord": [50, 60, 900, 1500]}]
    with_hl = render_page_with_highlight(pdf, 1, rects, px_per_pt=2.78)
    plain = render_page_with_highlight(pdf, 1, [], px_per_pt=2.78)
    assert with_hl["highlighted"] is True
    a, b = with_hl["image"].convert("RGB"), plain["image"].convert("RGB")
    sx = a.width / (595.0 * 2.78)
    x1, y1, x2, y2 = [int(c * sx) for c in [50, 60, 900, 1500]]
    inside = sum(1 for x in range(x1 + 5, x2 - 5, 20) for y in range(y1 + 5, y2 - 5, 20)
                 if a.getpixel((x, y)) != b.getpixel((x, y)))
    outside = sum(1 for x in range(5, x1 - 5, 20) for y in range(5, y1 - 5, 20)
                  if a.getpixel((x, y)) != b.getpixel((x, y)))
    assert inside > 50
    assert outside == 0


def test_render_page_clips_rect_extending_beyond_page():
    rects = [{"page_id": 1, "coord": [-100, -100, 5000, 5000]}]
    res = render_page_with_highlight(_blank_pdf(), 1, rects, px_per_pt=2.78)
    assert res["highlighted"] is True
    assert "clipped" in res["note"]


def test_render_page_downscales_preserving_alignment():
    rects = [{"page_id": 1, "coord": [50, 60, 900, 1500]}]
    res = render_page_with_highlight(_blank_pdf(), 1, rects, px_per_pt=2.78,
                                     max_width_px=500)
    assert res["image"].size[0] == 500
    assert abs(res["image"].size[1] / res["image"].size[0] - 842.0 / 595.0) < 0.02


def test_load_pdf_bytes_rejects_non_volume_path():
    with pytest.raises(ValueError):
        load_pdf_bytes("https://invented.example/doc.pdf")
    with pytest.raises(ValueError):
        load_pdf_bytes("")
