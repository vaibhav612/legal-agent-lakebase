"""Unit tests for the hybrid retrieval fixes in lakebase_retriever.py.

Covers the four corrected defect areas — no network, no Lakebase, no LLM:
  A. BM25 cache invalidation (unified, text-sensitive corpus fingerprint)
  B. RRF channel separation (independent semantic / BM25 rankings)
  C. Filenames and provenance for BM25-only hits
  D. Context limit accuracy (separators counted, skips recorded)
"""
import os
import sys

_app_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_app_root))

from lakebase_retriever import (
    BM25Index,
    LakebaseRetriever,
    MAX_CONTEXT_CHARS,
    RRF_K,
    _md5_chunk_checksum,
    corpus_fingerprint,
)


CHUNKS = [
    {"chunk_id": "c1", "doc_id": "d1", "chunk_text": "termination of the agreement upon material breach",
     "chunk_index": 0, "section_path": "9. Termination", "page_start": 3, "page_end": 4,
     "file_name": "Alpha_Distribution_Agreement.pdf"},
    {"chunk_id": "c2", "doc_id": "d2", "chunk_text": "governing law and jurisdiction venue arbitration",
     "chunk_index": 0, "section_path": "12. Governing Law", "page_start": 7, "page_end": 7,
     "file_name": "Beta_MSA.pdf"},
    {"chunk_id": "c3", "doc_id": "d2", "chunk_text": "payment terms net forty five days invoice",
     "chunk_index": 2, "section_path": "4. Payment", "page_start": 5, "page_end": 5,
     "file_name": "Beta_MSA.pdf"},
]


class StubConn:
    """Stub pg8000 connection that routes SQL by content."""

    def __init__(self, probe=None, corpus_rows=None, semantic_rows=None):
        self.probe = probe
        self.corpus_rows = corpus_rows
        self.semantic_rows = semantic_rows
        self.sql_log = []

    def run(self, sql, **params):
        self.sql_log.append(sql)
        if "md5(chunk_id" in sql:
            return self.probe
        if "query_embedding" in sql:
            return self.semantic_rows
        return self.corpus_rows


def make_retriever(conn=None, bm25_index=None):
    """Build a LakebaseRetriever without any network initialization."""
    r = LakebaseRetriever.__new__(LakebaseRetriever)
    r.conn = conn
    r.bm25_index = bm25_index
    r.bm25_corpus_hash = None
    r.last_context_assembly = None
    r.retriever_mode = "hybrid_v2"
    return r


def probe_for(chunks):
    """DB-side probe result with the same semantics as corpus_fingerprint."""
    return [(len(chunks),
             sum(_md5_chunk_checksum(c["chunk_id"], c.get("chunk_text") or "") for c in chunks))]


def corpus_rows_for(chunks):
    # Fix 011: the BM25 corpus SELECT is now 10-wide — table_html, has_table
    # ride between page_end and file_name.
    return [(c["chunk_id"], c["chunk_text"], c["doc_id"], c["chunk_index"],
             c["section_path"], c["page_start"], c["page_end"],
             None, False, c["file_name"])
            for c in chunks]


# ---------- A. BM25 cache invalidation ----------

def test_fingerprint_stable_and_order_independent():
    """Unchanged corpus (any row order) must produce one stable fingerprint."""
    fp1 = corpus_fingerprint(CHUNKS)
    fp2 = corpus_fingerprint(list(reversed(CHUNKS)))
    fp3 = corpus_fingerprint(CHUNKS)
    assert fp1 == fp2 == fp3


def test_fingerprint_detects_new_chunk():
    extra = dict(CHUNKS[0])
    extra["chunk_id"] = "c4"
    assert corpus_fingerprint(CHUNKS) != corpus_fingerprint(CHUNKS + [extra])


def test_fingerprint_detects_removed_chunk():
    assert corpus_fingerprint(CHUNKS) != corpus_fingerprint(CHUNKS[:2])


def test_fingerprint_detects_modified_text_same_chunk_id():
    """Same count and same chunk_ids, but edited chunk_text must be detected."""
    modified = [dict(c) for c in CHUNKS]
    modified[1]["chunk_text"] = modified[1]["chunk_text"] + " as amended"
    assert corpus_fingerprint(CHUNKS) != corpus_fingerprint(modified)


def test_bm25_index_hash_matches_corpus_fingerprint():
    """BM25Index.build and corpus_fingerprint must use ONE versioning strategy."""
    idx = BM25Index()
    idx.build(CHUNKS)
    assert idx.corpus_hash == corpus_fingerprint(CHUNKS)


def test_ensure_bm25_no_rebuild_on_unchanged_corpus():
    """An unchanged corpus must not trigger a full BM25 reload/rebuild."""
    idx = BM25Index()
    idx.build(CHUNKS)
    conn = StubConn(probe=probe_for(CHUNKS))
    r = make_retriever(conn=conn, bm25_index=idx)
    r._ensure_bm25()
    assert len(conn.sql_log) == 1, "expected only the cheap fingerprint probe, no corpus reload"
    assert "md5(chunk_id" in conn.sql_log[0]


def test_ensure_bm25_rebuilds_on_modified_text():
    """Modified chunk_text under an existing chunk_id must force a rebuild."""
    idx = BM25Index()
    idx.build(CHUNKS)
    modified = [dict(c) for c in CHUNKS]
    modified[2]["chunk_text"] = "payment terms net thirty days"
    conn = StubConn(probe=probe_for(modified), corpus_rows=corpus_rows_for(modified))
    r = make_retriever(conn=conn, bm25_index=idx)
    r._ensure_bm25()
    assert len(conn.sql_log) == 2, "expected probe + full corpus reload"
    assert r.bm25_index.corpus_hash == corpus_fingerprint(modified)


# ---------- B. RRF channel separation ----------

def _semantic(cid, distance=0.1):
    return {"chunk_id": cid, "chunk_text": f"text {cid}", "doc_id": "d1",
            "doc_name": "Alpha_Distribution_Agreement.pdf", "chunk_index": 0,
            "section_path": "9. Termination", "page_start": 3, "page_end": 4,
            "distance": distance}


def test_rrf_empty_channels():
    r = make_retriever()
    assert r._rrf_fuse([], [], num_results=5) == []


def test_rrf_semantic_only_contribution():
    """A chunk found only by vector search gets only its semantic contribution."""
    r = make_retriever()
    fused = r._rrf_fuse([_semantic("c1"), _semantic("c2")], [], num_results=5)
    assert [f["chunk_id"] for f in fused] == ["c1", "c2"]
    assert fused[0]["rrf_score"] == 1.0 / (RRF_K + 0 + 1)
    assert fused[0]["semantic_rank"] == 0
    assert fused[0]["bm25_rank"] is None and fused[0]["bm25_score"] is None
    assert fused[0]["distance"] == 0.1


def test_rrf_bm25_only_contribution_and_provenance():
    """A chunk found only by BM25 keeps only its BM25 rank and real metadata."""
    idx = BM25Index()
    idx.build(CHUNKS)
    r = make_retriever(bm25_index=idx)
    bm25 = [{"rank": 0, "chunk_id": "c2", "bm25_score": 3.5}]
    fused = r._rrf_fuse([], bm25, num_results=5)
    assert len(fused) == 1
    row = fused[0]
    assert row["chunk_id"] == "c2"
    assert row["rrf_score"] == 1.0 / (RRF_K + 0 + 1)
    # No artificial semantic values assigned to a BM25-only candidate
    assert row["semantic_rank"] is None
    assert row["distance"] is None
    # Provenance from the BM25 corpus record — real file_name, not a doc_id stand-in
    assert row["doc_name"] == "Beta_MSA.pdf"
    assert row["section_path"] == "12. Governing Law"
    assert row["page_start"] == 7 and row["page_end"] == 7
    assert row["doc_id"] == "d2" and row["chunk_id"] == "c2"
    assert row["bm25_score"] == 3.5 and row["bm25_rank"] == 0


def test_rrf_overlap_sums_channel_contributions():
    """An overlapping chunk sums 1/(k+sem_rank+1) + 1/(k+bm25_rank+1) exactly."""
    r = make_retriever()
    sem = [_semantic("c1"), _semantic("c2"), _semantic("c3")]
    bm25 = [{"rank": 0, "chunk_id": "c3", "bm25_score": 2.0},
            {"rank": 1, "chunk_id": "c1", "bm25_score": 1.0}]
    fused = r._rrf_fuse(sem, bm25, num_results=5)
    by_id = {f["chunk_id"]: f for f in fused}
    assert by_id["c3"]["rrf_score"] == 1.0 / (RRF_K + 2 + 1) + 1.0 / (RRF_K + 0 + 1)
    assert by_id["c1"]["rrf_score"] == 1.0 / (RRF_K + 0 + 1) + 1.0 / (RRF_K + 1 + 1)
    assert by_id["c2"]["rrf_score"] == 1.0 / (RRF_K + 1 + 1)
    # Final order follows pure RRF: c1 (0.03252) > c3 (0.03226) > c2 (0.01613)
    assert [f["chunk_id"] for f in fused] == ["c1", "c3", "c2"]
    # Each channel's rank reflects that channel's own ranking only
    assert by_id["c3"]["semantic_rank"] == 2 and by_id["c3"]["bm25_rank"] == 0
    assert by_id["c1"]["semantic_rank"] == 0 and by_id["c1"]["bm25_rank"] == 1
    assert by_id["c2"]["bm25_rank"] is None


def test_rrf_num_results_truncation():
    r = make_retriever()
    sem = [_semantic(f"c{i}") for i in range(10)]
    assert len(r._rrf_fuse(sem, [], num_results=3)) == 3


def test_search_hybrid_v2_stubbed_channels_independent():
    """End-to-end hybrid search with stubbed DB: BM25-only hit keeps file_name.

    The semantic channel returns only c1; the lexical query matches only c2.
    The fused result must contain both, each with its own channel's values.
    """
    idx = BM25Index()
    idx.build(CHUNKS)
    semantic_rows = [("c1", CHUNKS[0]["chunk_text"], "d1",
                      "Alpha_Distribution_Agreement.pdf", 0, "9. Termination", 3, 4,
                      None, False, 0.05)]
    conn = StubConn(probe=probe_for(CHUNKS), corpus_rows=corpus_rows_for(CHUNKS),
                    semantic_rows=semantic_rows)
    r = make_retriever(conn=conn, bm25_index=idx)
    r.embed_query = lambda q: [0.0] * 8  # no network
    fused = r.search_hybrid_v2("governing law jurisdiction venue arbitration", num_results=5)
    by_id = {f["chunk_id"]: f for f in fused}
    assert set(by_id) == {"c1", "c2"}
    c2 = by_id["c2"]
    assert c2["doc_name"] == "Beta_MSA.pdf", "BM25-only hit must carry the real file_name"
    assert c2["semantic_rank"] is None and c2["bm25_rank"] is not None
    assert c2["section_path"] == "12. Governing Law"
    c1 = by_id["c1"]
    assert c1["doc_name"] == "Alpha_Distribution_Agreement.pdf"
    assert c1["distance"] is not None and c1["bm25_rank"] is None


# ---------- C/D. Context assembly and the 24,000-char budget ----------

def test_context_empty_chunks():
    r = make_retriever()
    text, stats = r._assemble_context([])
    assert text == ""
    assert stats["packed_chunks"] == 0 and stats["skipped"] == []


def test_context_missing_metadata():
    """A chunk with only doc_id and text still produces a valid citation header."""
    r = make_retriever()
    chunks = [{"chunk_id": "c9", "doc_id": "doc_9", "chunk_text": "no metadata here"}]
    text, stats = r._assemble_context(chunks)
    assert text == "[Document 1] doc_9\nno metadata here"
    assert stats["packed_chunks"] == 1 and stats["skipped"] == []


def test_context_missing_text_recorded():
    r = make_retriever()
    chunks = [{"chunk_id": "c1", "doc_id": "d", "doc_name": "D", "chunk_text": "   "},
              {"chunk_id": "c2", "doc_id": "d", "doc_name": "D", "chunk_text": "valid text"}]
    text, stats = r._assemble_context(chunks)
    assert stats["packed_chunks"] == 1
    assert stats["skipped"] == [{"chunk_id": "c1", "reason": "missing_or_empty_chunk_text"}]


def test_context_limit_counts_separators_and_records_skips():
    """Packing must stay within MAX_CONTEXT_CHARS including separators, and
    every skipped chunk must be recorded with a reason."""
    r = make_retriever()
    header = "[Document 1] Alpha.pdf"
    t1 = "a" * (20000 - len(header) - 1)   # block1 = 20,000 chars
    chunks = [{"chunk_id": "c1", "doc_id": "d", "doc_name": "Alpha.pdf", "chunk_text": t1},
              {"chunk_id": "c2", "doc_id": "d", "doc_name": "Alpha.pdf", "chunk_text": "b" * 5000}]
    text, stats = r._assemble_context(chunks)
    assert stats["total_chars"] == len(text) <= MAX_CONTEXT_CHARS
    assert stats["packed_chunks"] == 1
    assert len(stats["skipped"]) == 1
    assert stats["skipped"][0]["chunk_id"] == "c2"
    assert stats["skipped"][0]["reason"].startswith("context_limit")


def test_context_never_exceeds_cap_by_separator():
    """Regression: the old check ignored the +2 separator, allowing the
    assembled context to reach MAX_CONTEXT_CHARS + 2."""
    r = make_retriever()
    h1 = "[Document 1] D"
    t1 = "a" * (23998 - len(h1) - 1)      # block1 = 23,998
    h2 = "[Document 2] D"
    t2 = "bb"                             # block2 = 18; 23998 + 2 + 18 = 24,018 > cap
    chunks = [{"chunk_id": "c1", "doc_id": "d", "doc_name": "D", "chunk_text": t1},
              {"chunk_id": "c2", "doc_id": "d", "doc_name": "D", "chunk_text": t2}]
    text, stats = r._assemble_context(chunks)
    assert stats["packed_chunks"] == 1
    assert len(text) == 23998 <= MAX_CONTEXT_CHARS
    assert stats["skipped"][0]["chunk_id"] == "c2"


def test_context_exact_boundary_included():
    """A chunk that lands exactly on the 24,000-char cap must be included."""
    r = make_retriever()
    h1 = "[Document 1] D"
    t1 = "a" * (23981 - len(h1) - 1)      # block1 = 23,981
    h2 = "[Document 2] D"
    t2 = "bb"                             # block2 = 18; 23,980 + 2 + 18 = 24,000
    chunks = [{"chunk_id": "c1", "doc_id": "d", "doc_name": "D", "chunk_text": t1},
              {"chunk_id": "c2", "doc_id": "d", "doc_name": "D", "chunk_text": t2}]
    text, stats = r._assemble_context(chunks)
    assert stats["packed_chunks"] == 2
    assert stats["total_chars"] == MAX_CONTEXT_CHARS
    assert stats["skipped"] == []
    assert len(text) == MAX_CONTEXT_CHARS


# ---------- E. Document-scoped hybrid retrieval (D1 regression) ----------
# D1 (2026-10-29): search() forwarded doc_ids to search_hybrid_v2, which did
# not accept the parameter — every hybrid_v2 retrieval raised TypeError and
# document-scoped capability nodes swallowed it into zero-evidence results.
# These tests pin the fixed contract: doc_ids scopes BOTH channels before RRF
# fusion, doc_ids=None keeps corpus-wide retrieval, values are bound as SQL
# parameters (never interpolated), and the return schema is unchanged.


def _sem_row(chunk_id, doc_id, doc_name, text, distance=0.10,
             table_html=None, has_table=False):
    """Tuple row in the search_hybrid_v2 SQL SELECT shape.

    Fix 011 (2026-10-30): the semantic SELECT now carries table_html and
    has_table (positions 8/9) before the distance score, so table chunks
    keep their structure through retrieval.
    """
    return (chunk_id, text, doc_id, doc_name, 0, "1. Section", 1, 2,
            table_html, has_table, distance)


SEM_ROWS = [
    _sem_row("c1", "d1", "Alpha_Distribution_Agreement.pdf",
             "termination of the agreement upon material breach"),
    _sem_row("c2", "d2", "Beta_MSA.pdf",
             "governing law and jurisdiction venue arbitration"),
    _sem_row("c3", "d2", "Beta_MSA.pdf",
             "payment terms net forty five days invoice"),
]


class ScopingConn:
    """Stub pg8000 connection that honors the doc_ids WHERE clause exactly as
    Postgres would (ANY(:doc_ids) filters the semantic candidate rows), and
    records every SQL statement and bound parameter set for inspection."""

    def __init__(self, semantic_rows):
        self.semantic_rows = semantic_rows
        self.sql_log = []
        self.param_log = []

    def run(self, sql, **params):
        self.sql_log.append(sql)
        self.param_log.append(params)
        if "query_embedding" in sql:
            rows = self.semantic_rows
            if "ANY(:doc_ids)" in sql and "doc_ids" in params:
                wanted = set(params["doc_ids"])
                rows = [r for r in rows if r[2] in wanted]
            return rows
        return []


def make_scoped_retriever(semantic_rows=SEM_ROWS):
    """Retriever wired for search_hybrid_v2 without any network access:
    stubbed embedding + BM25 corpus prebuilt from CHUNKS, ScopingConn as DB."""
    conn = ScopingConn(semantic_rows)
    idx = BM25Index()
    idx.build(CHUNKS)
    r = make_retriever(conn=conn, bm25_index=idx)
    r._ensure_bm25 = lambda: None           # corpus already built; no probe
    r.embed_query = lambda query: [0.0, 0.0]  # no embedding endpoint
    return r, conn


def _semantic_call(conn):
    idx = [i for i, sql in enumerate(conn.sql_log) if "query_embedding" in sql]
    assert idx, "expected a semantic SQL call"
    return idx[-1]


def test_search_none_doc_ids_corpus_wide_no_typeerror():
    """D1 core regression: plain corpus-wide hybrid search must not raise and
    must not add a doc_ids WHERE clause."""
    r, conn = make_scoped_retriever()
    results = r.search("termination", num_results=5)            # doc_ids omitted
    assert isinstance(results, list) and results
    results = r.search("termination", num_results=5, doc_ids=None)
    i = _semantic_call(conn)
    assert "ANY(:doc_ids)" not in conn.sql_log[i]
    assert "doc_ids" not in conn.param_log[i]


def test_search_dispatches_doc_ids_in_hybrid_v2_mode():
    """D1 regression: search() must forward doc_ids to search_hybrid_v2
    without TypeError and the scope must hold end to end."""
    r, conn = make_scoped_retriever()
    results = r.search("governing law", num_results=3, doc_ids=["d2"])
    assert results, "expected fused results for the scoped document"
    assert all(f["doc_id"] == "d2" for f in results)


def test_search_doc_ids_adds_where_and_binds_parameter():
    """The scope must be a bound SQL parameter inside a WHERE ANY(:doc_ids)
    clause — never interpolated into the statement text."""
    r, conn = make_scoped_retriever()
    r.search("governing law", num_results=5, doc_ids=["d2"])
    i = _semantic_call(conn)
    assert "WHERE dc.doc_id = ANY(:doc_ids)" in conn.sql_log[i]
    assert conn.param_log[i]["doc_ids"] == ["d2"]


def test_search_doc_ids_semantic_channel_excludes_other_docs():
    """No semantic evidence may be returned for a document outside doc_ids."""
    r, conn = make_scoped_retriever()
    results = r.search("termination", num_results=5, doc_ids=["d2"])
    assert all(f["doc_id"] == "d2" for f in results)
    assert "c1" not in {f["chunk_id"] for f in results}  # c1 belongs to d1


def test_search_doc_ids_filters_bm25_before_fusion():
    """BM25-only lexical hits from out-of-scope documents must be removed
    BEFORE RRF fusion — never fused and never surfaced."""
    # 'payment terms' matches only c3 (doc d2) lexically; scope is d1 only.
    r, conn = make_scoped_retriever(
        [_sem_row("c1", "d1", "Alpha_Distribution_Agreement.pdf",
                  "termination of the agreement upon material breach")])
    results = r.search("payment terms net forty five days", num_results=5, doc_ids=["d1"])
    assert "c3" not in {f["chunk_id"] for f in results}
    assert all(f["doc_id"] == "d1" for f in results)


def test_search_doc_ids_blocks_out_of_scope_lexical_channel():
    """Mirror of the above: a lexically strong hit in another document
    ('termination' in c1/d1) must not enter a d2-scoped result."""
    r, conn = make_scoped_retriever()
    results = r.search("termination of the agreement", num_results=5, doc_ids=["d2"])
    assert "c1" not in {f["chunk_id"] for f in results}
    assert all(f["doc_id"] == "d2" for f in results)


def test_search_doc_ids_multiple_documents():
    """A multi-document scope must allow every named document and no others."""
    r, conn = make_scoped_retriever()
    results = r.search("termination payment terms", num_results=5, doc_ids=["d1", "d2"])
    assert len(results) >= 2
    assert {f["doc_id"] for f in results} <= {"d1", "d2"}


def test_search_doc_ids_empty_list_returns_no_evidence():
    """An empty doc scope is a closed scope (no documents), NOT corpus-wide."""
    r, conn = make_scoped_retriever()
    results = r.search("governing law", num_results=5, doc_ids=[])
    assert results == []


def test_search_doc_ids_injection_safe():
    """A hostile doc_ids value is bound as a parameter value and never appears
    inside the SQL statement text."""
    r, conn = make_scoped_retriever()
    malicious = "d1' OR '1'='1"
    r.search("governing law", num_results=5, doc_ids=[malicious])
    i = _semantic_call(conn)
    assert malicious not in conn.sql_log[i]
    assert conn.param_log[i]["doc_ids"] == [malicious]
    assert "WHERE dc.doc_id = ANY(:doc_ids)" in conn.sql_log[i]


def test_search_doc_ids_return_schema_unchanged():
    """The fused result schema (consumed by every capability node) is
    unchanged by the document-scoping fix."""
    r, conn = make_scoped_retriever()
    results = r.search("governing law", num_results=5, doc_ids=["d2"])
    expected_keys = {
        "chunk_id", "chunk_text", "doc_id", "doc_name", "chunk_index",
        "section_path", "page_start", "page_end", "distance",
        "bm25_score", "bm25_rank", "semantic_rank", "rrf_score", "rrf_rank",
    }
    for row in results:
        assert expected_keys <= set(row.keys())
