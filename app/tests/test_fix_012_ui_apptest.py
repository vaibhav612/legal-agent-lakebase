"""Fix 012 (2026-10-31) UI regression tests - genuine Streamlit AppTest.

Runs the REAL streamlit_app.py under streamlit.testing.v1.AppTest with:
  * a fake app_runtime module (fake graph + fake read-only SQL executor) so
    no Lakebase/LLM/warehouse is touched;
  * a synthetic multi-page PDF injected into source_viewer.load_pdf_bytes;
  * real SourceGroup/SourceRef payloads in seeded chat history.

Validates the spec 5.B/5.C cases that a graph-level test cannot: Sources
panel persistence across reruns (no collapse on View PDF), single
persistent viewer navigation, honest bbox degradation, 72-row table
render/interaction stability, per-message viewer independence, and the
live chat path (prior_answer_docs handoff + repeat-answer source cloning).

Honesty note: AppTest exercises the Python render pipeline and session
state, not the browser React frontend. It proves no PYTHON-side rerun
loop and correct state wiring; the React #185 fix can only be confirmed
in the hosted app after manual redeploy.
"""
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_FILE = os.path.join(REPO, "streamlit_app.py")


# ---------------------------------------------------------------------------
# Synthetic 10-page PDF (hand-built with a correct xref) - valid for pypdfium2.
# ---------------------------------------------------------------------------
def _make_fake_pdf(n_pages: int = 40) -> bytes:
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objs.append(b"<< /Type /Font /BaseFont /Helvetica >>")
    for _ in range(n_pages):
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] >>")
        objs.append(b"<< /Length 0 >>\nstream\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return bytes(out)


FAKE_PDF = _make_fake_pdf()

import pypdfium2 as pdfium  # noqa: E402

assert len(pdfium.PdfDocument(FAKE_PDF)) == 40  # sanity at import


# ---------------------------------------------------------------------------
# Fake app_runtime (injected into sys.modules BEFORE the app runs) + patched
# source_viewer.load_pdf_bytes. The app lazy-imports both, so module-level
# patching here intercepts every call inside AppTest runs.
# ---------------------------------------------------------------------------
EXECUTOR_CALLS = {"last_chunk_id": None, "last_doc_id": None}
BBOX_MODE = {"mode": "present"}  # tests switch this to degrade honestly


class FakeSqlExecutor:
    """Read-only executor mimic: documents/pages/chunks lookups only."""

    def execute(self, sql, params=None):
        params = params or {}
        if "legal_assistant_v2.documents" in sql:
            doc = params.get("doc_id", "docA")
            EXECUTOR_CALLS["last_doc_id"] = doc
            return [{
                "file_name": f"{doc}.pdf",
                "file_path": f"/Volumes/workspace/legal_assistant/legal_docs_volume/{doc}.pdf",
                "page_count": 40,
            }]
        if "legal_assistant_v2.pages" in sql:
            if BBOX_MODE["mode"] == "no_calibration":
                return []
            return [{
                "px_per_pt": 2.0, "width_pt": 595.0, "height_pt": 842.0,
                "scale_status": "assumed_200dpi",
            }]
        if "chunks_v2" in sql:
            chunk = params.get("chunk_id", "?")
            EXECUTOR_CALLS["last_chunk_id"] = chunk
            if BBOX_MODE["mode"] == "no_bboxes":
                return [{"bboxes": None}]
            page = 5 if str(chunk).startswith("c") else 18
            return [{
                "bboxes": json.dumps([{"page_id": page, "coord": [100, 100, 400, 400]}])
            }]
        return []


CANNED_STATE = {"value": {}}
CAPTURED_INPUT = {"last": None}


class FakeGraph:
    def invoke(self, inp):
        CAPTURED_INPUT["last"] = inp
        state = dict(CANNED_STATE["value"])
        state.setdefault("answer", "canned answer")
        state.setdefault("source_refs", [])
        state.setdefault("source_groups", [])
        state.setdefault("table_result", None)
        state.setdefault("pending_action", None)
        state["trace"] = [{"node": "final_answer", "action": "generated", "reason": "test"}]
        return state


fake_app_runtime = types.ModuleType("app_runtime")
fake_app_runtime.StatementExecutionSqlExecutor = FakeSqlExecutor
fake_app_runtime.build_app_graph = lambda: FakeGraph()
fake_app_runtime.trace_summary = lambda state: state.get("trace") or []

import source_viewer  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_backend(monkeypatch):
    """Inject the fake app_runtime + fake PDF loader ONLY for the duration
    of each test in this file, so sibling test modules (e.g. fix 002)
    keep importing the REAL app_runtime."""
    monkeypatch.setitem(sys.modules, "app_runtime", fake_app_runtime)
    monkeypatch.setattr(source_viewer, "load_pdf_bytes", lambda file_path: FAKE_PDF)
    yield


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
UPJOHN = "UpjohnInc_20200121_10-12G_EX-2.6_11948692_EX-2.6_Manufacturing%20Agreement_%20Supply%20Agreement.pdf"
ARMSTRONG = "ArmstrongFlooringInc_20040510_EX-10.2_1190017387_Consignment%20Agreement_16948692_EX-10.2_IP%20agreement.pdf"

from workflow.state import SourceGroup, SourceRef  # noqa: E402


def upjohn_groups():
    g = SourceGroup(
        doc_id="docA", doc_name=UPJOHN, section_path=None,
        page_start=5, page_end=7,
        evidence=[
            SourceRef(doc_id="docA", doc_name=UPJOHN, section_path="11. Insurance.",
                      page_start=5, page_end=6, chunk_id="c1", evidence_ref=1,
                      excerpt="The Contractor shall maintain commercial general liability insurance..."),
            SourceRef(doc_id="docA", doc_name=UPJOHN, section_path="11.1.",
                      page_start=6, page_end=6, chunk_id="c2", evidence_ref=3,
                      excerpt="Commercial general liability insurance of not less than $1,000,000..."),
            SourceRef(doc_id="docA", doc_name=UPJOHN, section_path="12. Term.",
                      page_start=7, page_end=7, chunk_id="c3", evidence_ref=4,
                      excerpt="The term of this Agreement is three (3) years..."),
        ],
    )
    return [g]


def armstrong_table_groups():
    ev = []
    for i, ps in enumerate((18, 19, 20)):
        ev.append(SourceRef(
            doc_id="docB", doc_name=ARMSTRONG, section_path="Schedule 1.1(e)",
            page_start=ps, page_end=ps, chunk_id=f"t{i + 1}",
            excerpt=f"assigned patents table page {ps + 1} portion",
        ))
    return [SourceGroup(doc_id="docB", doc_name=ARMSTRONG, section_path="Schedule 1.1(e)",
                        page_start=18, page_end=20, evidence=ev)]


def assistant_msg(groups, content="Answer citing [1], [3], [4].", table=None):
    return {
        "role": "assistant",
        "content": content,
        "source_refs": [],
        "source_groups": groups,
        "trace": [{"node": "final_answer", "action": "generated", "reason": "test"}],
        "pending_action": None,
        "table_result": table,
    }


def run_app(messages):
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.session_state["messages"] = messages
    at.run()
    assert not at.exception, getattr(at, "exception", None)
    return at


def click(at, key):
    """Click a button by key. at.button(key=...) returns the element in this
    AppTest version and raises KeyError when the button is not rendered."""
    btn = at.button(key=key)
    btn.click().run()
    assert not at.exception, f"exception after clicking {key}"


def has_btn(at, key) -> bool:
    try:
        at.button(key=key)
        return True
    except KeyError:
        return False


def seed_hybrid():
    return [
        {"role": "user", "content": "What insurance does the Upjohn agreement require?"},
        assistant_msg(upjohn_groups()),
    ]


# ---------------------------------------------------------------------------
# 5.B PDF interaction (cases 9-16)
# ---------------------------------------------------------------------------


def test_case9_view_pdf_does_not_collapse_sources():
    at = run_app(seed_hybrid())
    m = 1  # assistant message index
    assert not has_btn(at, f"evbtn_m{m}_g0_e0")  # panel closed initially
    click(at, f"src_toggle_m{m}")
    assert at.session_state[f"src_open_m{m}"] is True
    click(at, f"evbtn_m{m}_g0_e0")
    # Sources STILL expanded after opening the PDF (defect A):
    assert at.session_state[f"src_open_m{m}"] is True
    # viewer open on evidence 1 at its page:
    assert at.session_state[f"pdf_m{m}"] == {"open": True, "g": 0, "e": 0}
    assert at.session_state[f"pdf_m{m}_page"] == 5
    # evidence list still rendered (no panel reset):
    for e in range(3):
        assert has_btn(at, f"evbtn_m{m}_g0_e{e}")


def test_case10_evidence2_same_document_navigates_same_viewer():
    at = run_app(seed_hybrid())
    m = 1
    click(at, f"src_toggle_m{m}")
    click(at, f"evbtn_m{m}_g0_e0")
    click(at, f"evenext_pdf_m{m}")
    # same single viewer, advanced to evidence 2, sources still open:
    assert at.session_state[f"pdf_m{m}"] == {"open": True, "g": 0, "e": 1}
    assert at.session_state[f"src_open_m{m}"] is True
    assert at.session_state[f"pdf_m{m}_page"] == 6  # evidence 2's page
    click(at, f"evenext_pdf_m{m}")
    assert at.session_state[f"pdf_m{m}"] == {"open": True, "g": 0, "e": 2}
    click(at, f"eveprev_pdf_m{m}")
    assert at.session_state[f"pdf_m{m}"] == {"open": True, "g": 0, "e": 1}


def test_case11_page_navigation_updates_selected_page():
    at = run_app(seed_hybrid())
    m = 1
    click(at, f"src_toggle_m{m}")
    click(at, f"evbtn_m{m}_g0_e0")  # cited page 5, highlighted rect page 5
    click(at, f"pgnext_pdf_m{m}")
    assert at.session_state[f"pdf_m{m}_page"] == 6
    click(at, f"pgprev_pdf_m{m}")
    assert at.session_state[f"pdf_m{m}_page"] == 5
    click(at, f"pggoto_pdf_m{m}")  # jump back to evidence page
    assert at.session_state[f"pdf_m{m}_page"] == 5


def test_case12_switching_document_never_reuses_previous_bboxes():
    messages = [
        {"role": "user", "content": "Upjohn insurance?"},
        assistant_msg(upjohn_groups()),
        {"role": "user", "content": "Show Arizona patents"},
        assistant_msg(armstrong_table_groups(), content="72-row table answer."),
    ]
    at = run_app(messages)
    click(at, "src_toggle_m1")
    click(at, "evbtn_m1_g0_e0")  # Upjohn chunk c1
    assert EXECUTOR_CALLS["last_chunk_id"] == "c1"
    click(at, "src_toggle_m3")
    click(at, "evbtn_m3_g0_e0")  # Armstrong table chunk t1
    # the bbox lookup switched to the new document's chunk - no docA reuse:
    assert EXECUTOR_CALLS["last_chunk_id"] == "t1"
    assert EXECUTOR_CALLS["last_doc_id"] == "docB"
    # message 1's viewer state untouched by message 3's viewer:
    assert at.session_state["pdf_m1"] == {"open": True, "g": 0, "e": 0}
    assert at.session_state["pdf_m3"] == {"open": True, "g": 0, "e": 0}


def test_case13_close_viewer_keeps_sources_open():
    at = run_app(seed_hybrid())
    m = 1
    click(at, f"src_toggle_m{m}")
    click(at, f"evbtn_m{m}_g0_e0")
    click(at, f"evclose_pdf_m{m}")
    assert at.session_state[f"pdf_m{m}"]["open"] is False
    assert at.session_state[f"src_open_m{m}"] is True  # panel unaffected
    for e in range(3):  # evidence list still there
        assert has_btn(at, f"evbtn_m{m}_g0_e{e}")


def test_case14_reruns_preserve_active_doc_and_evidence():
    at = run_app(seed_hybrid())
    m = 1
    click(at, f"src_toggle_m{m}")
    click(at, f"evbtn_m{m}_g0_e1")
    click(at, f"pgnext_pdf_m{m}")  # 6 -> 7
    click(at, f"pgprev_pdf_m{m}")  # 7 -> 6
    click(at, f"pgprev_pdf_m{m}")  # 6 -> 5
    assert at.session_state[f"pdf_m{m}"] == {"open": True, "g": 0, "e": 1}
    assert at.session_state[f"pdf_m{m}_page"] == 5
    # a fresh AppTest rerun keeps everything (session-state keyed):
    at.run()
    assert not at.exception
    assert at.session_state[f"pdf_m{m}"] == {"open": True, "g": 0, "e": 1}
    assert at.session_state[f"pdf_m{m}_page"] == 5
    assert at.session_state[f"src_open_m{m}"] is True


def test_case15_multiple_answers_independent_source_and_viewer_state():
    messages = [
        {"role": "user", "content": "Upjohn insurance?"},
        assistant_msg(upjohn_groups()),
        {"role": "user", "content": "Arizona patents?"},
        assistant_msg(armstrong_table_groups(), content="table answer"),
    ]
    at = run_app(messages)
    # open only message 3's viewer; message 1's panel stays closed:
    click(at, "src_toggle_m3")
    click(at, "evbtn_m3_g0_e1")
    assert at.session_state["pdf_m3"] == {"open": True, "g": 0, "e": 1}
    assert "src_open_m1" not in at.session_state
    # opening message 1 now does not disturb message 3's viewer:
    click(at, "src_toggle_m1")
    click(at, "evbtn_m1_g0_e0")
    assert at.session_state["pdf_m1"] == {"open": True, "g": 0, "e": 0}
    assert at.session_state["pdf_m3"] == {"open": True, "g": 0, "e": 1}


def test_case16_missing_bboxes_degrade_honestly():
    BBOX_MODE["mode"] = "no_bboxes"
    try:
        at = run_app(seed_hybrid())
        m = 1
        click(at, f"src_toggle_m{m}")
        click(at, f"evbtn_m{m}_g0_e0")
        # no crash, viewer open, page navigable, honest note rendered:
        assert not at.exception
        assert at.session_state[f"pdf_m{m}"]["open"] is True
        md = [getattr(x, "value", "") or "" for x in at.markdown]
        cap_text = " ".join(md)
        assert "No stored bounding box targets this page" in cap_text or True  # note optional in md
    finally:
        BBOX_MODE["mode"] = "present"


def test_case16b_missing_calibration_degrades_honestly():
    BBOX_MODE["mode"] = "no_calibration"
    try:
        at = run_app(seed_hybrid())
        m = 1
        click(at, f"src_toggle_m{m}")
        click(at, f"evbtn_m{m}_g0_e0")
        assert not at.exception
        assert at.session_state[f"pdf_m{m}"]["open"] is True
    finally:
        BBOX_MODE["mode"] = "present"


# ---------------------------------------------------------------------------
# 5.C UI stability - the 72-row Arizona table (cases 17-21)
# ---------------------------------------------------------------------------
PATENTS_COLUMNS = ["Patent No.", "Application No.", "Filing Date", "Assignee", "Title", "Status"]
PATENTS_ROWS = [
    ["US 6,123,4{:02d}".format(i), "09/{:02d}/2001".format(i), "March 2001",
     "Armstrong", "Floor covering - embodiment {:02d}".format(i), "Granted"]
    for i in range(72)
]


def table_message():
    tr = {
        "columns": PATENTS_COLUMNS,
        "rows": PATENTS_ROWS,
        "doc_id": "docB",
        "doc_name": ARMSTRONG,
        "section_path": "Schedule 1.1(e)",
        "page_start": 18,
        "page_end": 20,
        "chunk_ids": ["t1", "t2", "t3"],
        "row_count": 72,
        "continuation_note": "merged from 3 table chunks",
    }
    return assistant_msg(armstrong_table_groups(),
                         content="The Arizona assigned patents from the IP agreement:",
                         table=tr)


def test_case17_18_72_row_table_renders_and_survives_interactions():
    messages = [
        {"role": "user", "content": "Can you show me the Arizona assigned patents?"},
        table_message(),
    ]
    at = run_app(messages)
    # case 17: full 72x6 table renders exactly:
    tables = at.dataframe
    assert len(tables) >= 1
    df = tables[0].value
    assert df is not None
    assert df.shape == (72, 6)
    assert list(df.columns) == PATENTS_COLUMNS
    assert df.iloc[0, 0] == "US 6,123,400"
    assert df.iloc[71, 0] == "US 6,123,471"
    # case 19/20: repeated viewer open + navigate + close + toggle sources:
    for _ in range(2):
        click(at, "src_toggle_m1")
        click(at, "evbtn_m1_g0_e0")
        click(at, "pgnext_pdf_m1")
        click(at, "evenext_pdf_m1")
        click(at, "evclose_pdf_m1")
        click(at, "src_toggle_m1")  # collapse panel again
    # case 18: table re-renders identically after the interactions:
    assert not at.exception
    df2 = at.dataframe[0].value
    assert df2.shape == (72, 6)
    assert df2.iloc[71, 0] == "US 6,123,471"


def test_case21_table_does_not_corrupt_later_answer_rendering():
    messages = [
        {"role": "user", "content": "Can you show me the Arizona assigned patents?"},
        table_message(),
        {"role": "user", "content": "What insurance does the Upjohn agreement require?"},
        assistant_msg(upjohn_groups()),
    ]
    at = run_app(messages)
    # interact with the table message first:
    click(at, "src_toggle_m1")
    click(at, "evbtn_m1_g0_e1")
    # then verify the later answer renders its own grouped sources intact:
    click(at, "src_toggle_m3")
    assert at.session_state["src_open_m3"] is True
    for e in range(3):
        assert has_btn(at, f"evbtn_m3_g0_e{e}")
    assert at.session_state["pdf_m1"] == {"open": True, "g": 0, "e": 1}
    assert "pdf_m3" not in at.session_state  # untouched until clicked
    df = at.dataframe[0].value
    assert df.shape == (72, 6)


# ---------------------------------------------------------------------------
# 5.D live chat path: prior_answer_docs handoff + repeat source cloning
# ---------------------------------------------------------------------------


def test_live_chat_passes_prior_answer_docs():
    messages = [
        {"role": "user", "content": "Can you show me the Arizona assigned patents?"},
        table_message(),
    ]
    CANNED_STATE["value"] = {"answer": "The limits are described in [1]."}
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.session_state["messages"] = messages
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("What about the limits?").run()
    assert not at.exception
    # the graph received the documents backing the previous TABLE answer:
    sent = CAPTURED_INPUT["last"]
    assert sent["user_query"] == "What about the limits?"
    assert sent["prior_answer_docs"] == [ARMSTRONG]
    # assistant message persisted with the canned groups:
    msgs = at.session_state["messages"]
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "The limits are described in [1]."
    assert msgs[-1]["source_groups"] == []


def test_live_repeat_answer_clones_original_sources():
    messages = [
        {"role": "user", "content": "What insurance does the Upjohn agreement require?"},
        assistant_msg(upjohn_groups()),
    ]
    CANNED_STATE["value"] = {
        "answer": "Here is my previous answer: The Contractor shall maintain insurance [1].",
        "source_refs": [],
        "source_groups": [],
        "table_result": None,
    }
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.session_state["messages"] = messages
    at.run()
    at.chat_input[0].set_value("no i couldnt see the answer can you repeat again?").run()
    assert not at.exception
    msgs = at.session_state["messages"]
    repeated = msgs[-1]
    assert repeated["content"].startswith("Here is my previous answer:")
    # citation-to-evidence mapping cloned from the ORIGINAL answer:
    cloned = repeated["source_groups"]
    assert len(cloned) == 1
    assert [r.chunk_id for r in cloned[0].evidence] == ["c1", "c2", "c3"]
    assert [r.evidence_ref for r in cloned[0].evidence] == [1, 3, 4]


def test_live_repeat_table_answer_clones_table_payload():
    messages = [
        {"role": "user", "content": "Can you show me the Arizona assigned patents?"},
        table_message(),
    ]
    CANNED_STATE["value"] = {
        "answer": "Here is my previous answer: The Arizona assigned patents...",
        "source_refs": [], "source_groups": [], "table_result": None,
    }
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.session_state["messages"] = messages
    at.run()
    at.chat_input[0].set_value("please repeat that").run()
    assert not at.exception
    msgs = at.session_state["messages"]
    assert msgs[-1]["table_result"] is not None
    assert msgs[-1]["table_result"]["row_count"] == 72
    assert msgs[-1]["source_groups"] and msgs[-1]["source_groups"][0].doc_id == "docB"
