"""Lakebase Postgres + pgvector retrieval for Legal Assistant RAG.

Supports two retrieval modes via the RETRIEVER_MODE environment variable:
  - "legacy"   : pgvector cosine search on the old document_chunks table
  - "hybrid_v2": pgvector semantic + in-memory BM25 + RRF fusion on document_chunks_v2
"""

import hashlib
import json
import logging
import os
import math
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# pg8000 is imported lazily inside __init__ to avoid ModuleNotFoundError
# when this file is imported in environments without pg8000 installed

# Lakebase connection config
LAKEBASE_PROJECT = "legal-intelligence-db"
LAKEBASE_DATABASE = "databricks_postgres"
LAKEBASE_PORT = 5432

# LLM config (model name for serving endpoint)
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"

# Retrieval mode configuration
RETRIEVER_MODE = os.environ.get("RETRIEVER_MODE", "hybrid_v2")


def validate_retriever_mode(mode: str) -> None:
    """Stage C2 (2026-10-29): fail fast on an UNRECOGNIZED retrieval mode.

    Prevents search() from silently falling back to the legacy path when the
    configured mode is a typo or an unsupported value.  'legacy' remains
    accepted for explicit, deliberate use (e.g. comparison runs); anything
    else is a configuration error and must stop startup with a clear
    message.
    """
    if mode not in {"hybrid_v2", "legacy"}:
        raise ValueError(
            f"Unknown RETRIEVER_MODE {mode!r}: expected 'hybrid_v2' "
            "(production) or 'legacy' (explicit comparison)."
        )

# Hybrid retrieval parameters
SEMANTIC_CANDIDATE_POOL = 30
BM25_CANDIDATE_POOL = 30
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
MAX_CONTEXT_CHARS = 24000


def _md5_chunk_checksum(chunk_id: str, chunk_text: str) -> int:
    """Signed int64 of the first 16 hex chars of md5(chunk_id + ':' + chunk_text).

    Mirrors the Postgres expression used by the corpus-change probe in
    _ensure_bm25() so the Python-side fingerprint and the DB-side probe
    agree exactly. The ':' separator is safe because chunk_ids are fixed
    hex strings; a NUL separator is NOT usable because Postgres text
    values cannot contain 0x00 bytes ('invalid byte sequence for encoding
    "UTF8": 0x00' — verified against the live database).
    """
    digest = hashlib.md5((chunk_id + ":" + (chunk_text or "")).encode("utf-8")).hexdigest()
    v = int(digest[:16], 16)
    return v - (1 << 64) if v >= (1 << 64) // 2 else v


def corpus_fingerprint(chunks: List[Dict]) -> str:
    """Deterministic, text-sensitive corpus version: '<count>:<checksum>'.

    Detects added chunks, removed chunks, and modified chunk_text under an
    unchanged chunk_id. Order-independent (sum), so it is stable regardless
    of the row order returned by the database.
    """
    checksum = sum(_md5_chunk_checksum(c['chunk_id'], c.get('chunk_text') or '') for c in chunks)
    return f"{len(chunks)}:{checksum}"


class BM25Index:
    """In-memory BM25 index over chunk text for lexical retrieval."""

    def __init__(self, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.by_chunk_id = {}
        self.inverted_index = defaultdict(dict)
        self.doc_freq = defaultdict(int)
        self.avgdl = 0.0
        self.N = 0
        self._corpus_hash = None

    @staticmethod
    def tokenize(text):
        return [t for t in re.findall(r'[a-z0-9]+', text.lower()) if len(t) > 1]

    def build(self, chunks):
        self.docs = []
        self.inverted_index = defaultdict(dict)
        self.doc_freq = defaultdict(int)
        total_len = 0
        for i, chunk in enumerate(chunks):
            tokens = self.tokenize(chunk.get('chunk_text', ''))
            doc_len = len(tokens)
            self.docs.append({'chunk_id': chunk['chunk_id'], 'tokens': tokens, 'doc_len': doc_len, 'metadata': chunk})
            total_len += doc_len
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for term, freq in tf.items():
                self.inverted_index[term][i] = freq
                self.doc_freq[term] += 1
        self.N = len(self.docs)
        self.avgdl = total_len / self.N if self.N > 0 else 0.0
        self.by_chunk_id = {d['chunk_id']: d for d in self.docs}
        self._corpus_hash = corpus_fingerprint(chunks)

    @property
    def corpus_hash(self):
        return self._corpus_hash

    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict]:
        """Return a copy of the stored corpus record for a chunk, or None."""
        doc = self.by_chunk_id.get(chunk_id)
        return dict(doc['metadata']) if doc else None

    def search(self, query, k=BM25_CANDIDATE_POOL):
        query_terms = self.tokenize(query)
        if not query_terms:
            return []
        scores = defaultdict(float)
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            idf = math.log((self.N - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5) + 1.0)
            for doc_idx, tf in self.inverted_index[term].items():
                doc = self.docs[doc_idx]
                norm = 1 - self.b + self.b * doc['doc_len'] / self.avgdl if self.avgdl > 0 else 1
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{'rank': r, 'chunk_id': self.docs[idx]['chunk_id'], 'bm25_score': s} for r, (idx, s) in enumerate(ranked)]


class LakebaseRetriever:
    """RAG retriever using Lakebase Postgres + pgvector."""

    def __init__(self):
        """Initialize Lakebase connection using OAuth token via REST API."""
        try:
            from databricks.sdk import WorkspaceClient
            import requests
            import pg8000.native

            w = WorkspaceClient()
            auth_header = w.config.authenticate()
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            headers.update(auth_header)
            base_url = w.config.host.rstrip("/")

            # 1. Find the Lakebase project by display name
            resp = requests.get(f"{base_url}/api/2.0/postgres/projects",
                               headers=headers, timeout=30)
            resp.raise_for_status()
            projects_data = resp.json()

            project = None
            for p in projects_data.get("projects", []):
                spec = p.get("spec", {})
                display_name = spec.get("displayName", "")
                pname = p.get("name", "").removeprefix("projects/")
                if display_name == LAKEBASE_PROJECT or pname == LAKEBASE_PROJECT:
                    project = p
                    break

            if not project:
                raise RuntimeError(f"Lakebase project '{LAKEBASE_PROJECT}' not found.")

            project_name = project["name"]  # e.g., "projects/legal-intelligence-db"

            # 2. Find the production branch
            resp = requests.get(f"{base_url}/api/2.0/postgres/{project_name}/branches",
                               headers=headers, timeout=30)
            resp.raise_for_status()
            branches_data = resp.json()

            branch = None
            for b in branches_data.get("branches", []):
                if "production" in b.get("name", ""):
                    branch = b
                    break
            if not branch and branches_data.get("branches"):
                branch = branches_data["branches"][0]

            if not branch:
                raise RuntimeError("No branch found.")

            branch_name = branch["name"]  # e.g., "projects/.../branches/production"

            # 3. Find the read-write endpoint
            resp = requests.get(
                f"{base_url}/api/2.0/postgres/{branch_name}/endpoints",
                headers=headers, timeout=30)
            resp.raise_for_status()
            endpoints_data = resp.json()

            if not endpoints_data.get("endpoints"):
                raise RuntimeError("No endpoint found for this branch.")

            endpoint = endpoints_data["endpoints"][0]
            endpoint_name = endpoint["name"]  # e.g., "projects/.../endpoints/primary"
            host = endpoint["status"]["hosts"]["host"]

            # 4. Get current user (OAuth login role = Databricks identity)
            username = w.current_user.me().user_name

            # 5. Generate Lakebase-scoped OAuth token (valid 1 hour)
            cred_url = f"{base_url}/api/2.0/postgres/credentials"
            resp = requests.post(cred_url, headers=headers,
                                json={"endpoint": endpoint_name}, timeout=30)
            resp.raise_for_status()
            token = resp.json().get("token")

            if not token:
                raise RuntimeError("Failed to generate database credential.")

            # 6. Connect using pg8000 with the OAuth token as password
            self.conn = pg8000.native.Connection(
                host=host,
                port=LAKEBASE_PORT,
                database=LAKEBASE_DATABASE,
                user=username,
                password=token,
                ssl_context=True
            )

            self.databricks_token = None
            self.bm25_index = None
            self.bm25_corpus_hash = None
            self.last_context_assembly = None
            self.retriever_mode = RETRIEVER_MODE
            # Stage C2 (2026-10-29): unrecognized modes must stop startup —
            # see validate_retriever_mode().
            validate_retriever_mode(self.retriever_mode)

            logger.info(f"✅ LakebaseRetriever initialized (host={host}, user={username}, mode={self.retriever_mode})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LakebaseRetriever: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for user query using raw HTTP."""
        try:
            # Use raw HTTP with WorkspaceClient credentials
            from databricks.sdk import WorkspaceClient
            import requests
            
            w = WorkspaceClient()
            
            # Get auth header from workspace client
            auth_header = w.config.authenticate()
            
            # Call embedding endpoint with raw HTTP
            url = f"{w.config.host}/serving-endpoints/databricks-gte-large-en/invocations"
            headers = {"Content-Type": "application/json"}
            headers.update(auth_header)
            
            response = requests.post(url, json={"input": query}, headers=headers, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract embedding from response (raw HTTP returns dict)
            # Foundation Model embedding endpoints return 'data' with 'embedding' key
            if not isinstance(response_data, dict):
                raise ValueError(f"Expected dict response, got {type(response_data)}")
            
            if 'data' not in response_data:
                raise ValueError(f"No 'data' in response. Keys: {list(response_data.keys())}")
            
            data_list = response_data['data']
            if not isinstance(data_list, list) or len(data_list) == 0:
                raise ValueError(f"Invalid or empty data list: {data_list}")
            
            first_item = data_list[0]
            if not isinstance(first_item, dict) or 'embedding' not in first_item:
                raise ValueError(f"No embedding in data item. Keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'not a dict'}")
            
            embedding = first_item['embedding']
            
            logger.info(f"✅ Generated embedding (dim: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise

    def search(self, query: str, num_results: int = 5,
               doc_ids: Optional[List[str]] = None) -> List[Dict]:
        """Search using the configured retrieval mode (legacy or hybrid_v2).

        Args:
            query: natural-language search query.
            num_results: top-k results to return after fusion.
            doc_ids: optional list of doc_id values to restrict retrieval to.
                     When provided, the filter is applied inside the semantic
                     SQL WHERE clause and the BM25 candidate list *before* RRF
                     fusion — never as a post-fusion top-k filter.  When None
                     (default), retrieval is corpus-wide (unchanged behaviour).
        """
        if self.retriever_mode == "hybrid_v2":
            return self.search_hybrid_v2(query, num_results=num_results,
                                         doc_ids=doc_ids)
        return self.search_legacy(query, num_results=num_results,
                                  doc_ids=doc_ids)

    def search_legacy(self, query: str, num_results: int = 5,
                      doc_ids: Optional[List[str]] = None) -> List[Dict]:
        """Search Lakebase using pgvector similarity (legacy document_chunks table).

        When *doc_ids* is provided a WHERE clause restricts retrieval to those
        documents inside the SQL query, before the LIMIT top-k.
        """
        try:
            # Generate query embedding
            query_embedding = self.embed_query(query)
            embedding_json = json.dumps(query_embedding)

            # Execute pgvector search with document names
            search_sql = """
                SELECT
                    dc.chunk_id,
                    dc.chunk_text,
                    dc.doc_id,
                    COALESCE(ld.file_name, dc.doc_id) as doc_name,
                    COALESCE(ld.doc_type, 'Unknown') as doc_type,
                    dc.embedding <=> :query_embedding::vector as distance
                FROM document_chunks dc
                LEFT JOIN legal_documents ld ON dc.doc_id = ld.doc_id
            """
            kwargs: Dict[str, object] = {
                "query_embedding": embedding_json,
                "num_results": num_results,
            }
            if doc_ids is not None:
                search_sql += " WHERE dc.doc_id = ANY(:doc_ids)\n"
                kwargs["doc_ids"] = list(doc_ids)
            search_sql += (
                " ORDER BY dc.embedding <=> :query_embedding::vector\n"
                " LIMIT :num_results"
            )

            results = self.conn.run(search_sql, **kwargs)
            
            # Format results with decoded file names
            from urllib.parse import unquote
            chunks = []
            for row in results:
                chunks.append({
                    "chunk_id": row[0],
                    "chunk_text": row[1],
                    "doc_id": row[2],
                    "doc_name": unquote(row[3]) if row[3] else row[2],
                    "doc_type": row[4] or 'Unknown',
                    "distance": float(row[5])
                })
            
            logger.info(f"✅ Retrieved {len(chunks)} chunks from Lakebase")
            return chunks
        except Exception as e:
            logger.error(f"❌ pgvector search failed: {e}")
            raise

    def _load_bm25_corpus(self) -> List[Dict]:
        """Load all chunks from document_chunks_v2 for BM25 indexing.

        Joins legal_documents_v2 so every BM25 corpus record carries the
        verified file_name (not just doc_id) for provenance on BM25-only hits.
        """
        rows = self.conn.run("""
            SELECT dc.chunk_id, dc.chunk_text, dc.doc_id, dc.chunk_index,
                   dc.section_path, dc.page_start, dc.page_end,
                   dc.table_html, dc.has_table, ld.file_name
            FROM document_chunks_v2 dc
            LEFT JOIN legal_documents_v2 ld ON dc.doc_id = ld.doc_id
            ORDER BY dc.chunk_id
        """)
        from urllib.parse import unquote
        chunks = []
        for r in rows:
            # Fix 011: table structure rides along in BM25 corpus metadata so
            # BM25-only hits keep their table_html for the answer path.
            raw_ht = r[8]
            chunks.append({
                'chunk_id': r[0], 'chunk_text': r[1], 'doc_id': r[2],
                'chunk_index': r[3], 'section_path': r[4],
                'page_start': r[5], 'page_end': r[6],
                'table_html': r[7],
                'has_table': raw_ht if isinstance(raw_ht, bool) else str(raw_ht).lower() == 'true',
                'file_name': unquote(r[9]) if r[9] else None,
            })
        logger.info(f"  BM25 corpus loaded: {len(chunks)} chunks")
        return chunks

    def _ensure_bm25(self):
        """Load or refresh the in-memory BM25 index if the corpus has changed.

        The DB-side probe computes the same fingerprint semantics as
        corpus_fingerprint(): row count plus a signed-int64 md5 checksum over
        (chunk_id, chunk_text) pairs. Unchanged content (same count, IDs, and
        text) does not trigger a rebuild; modified text under an existing
        chunk_id changes the checksum and forces a rebuild.
        """
        if self.bm25_index is not None:
            probe = self.conn.run("""
                SELECT COUNT(*),
                       COALESCE(SUM(('x' || substr(md5(chunk_id || ':' || coalesce(chunk_text, '')), 1, 16))::bit(64)::bigint), 0)
                FROM document_chunks_v2
            """)
            current_fingerprint = f"{probe[0][0]}:{int(probe[0][1])}"
            if current_fingerprint == self.bm25_index.corpus_hash:
                return
            logger.info("  BM25 corpus changed — rebuilding index")
        chunks = self._load_bm25_corpus()
        self.bm25_index = BM25Index()
        self.bm25_index.build(chunks)
        self.bm25_corpus_hash = self.bm25_index.corpus_hash
        logger.info(f"  BM25 index built (N={self.bm25_index.N}, avgdl={self.bm25_index.avgdl:.0f})")

    def _rrf_fuse(self, semantic_results: List[Dict], bm25_results: List[Dict],
                  num_results: int) -> List[Dict]:
        """Fuse independently ranked semantic and BM25 lists using Reciprocal Rank Fusion.

        Each channel contributes 1/(RRF_K + channel_rank + 1) based on its own
        ranking only. BM25-only candidates are never merged into the semantic
        ranking, so they receive no artificial semantic scores or ranks (and
        vice versa). Metadata is joined after fusion: from the semantic
        channel when it retrieved the chunk, otherwise from the BM25 corpus
        record (which carries the verified file_name).
        """
        rrf_scores = defaultdict(float)
        semantic_meta = {s['chunk_id']: s for s in semantic_results}
        bm25_meta = {b['chunk_id']: b for b in bm25_results}
        semantic_ranks = {s['chunk_id']: rank for rank, s in enumerate(semantic_results)}
        bm25_ranks = {b['chunk_id']: rank for rank, b in enumerate(bm25_results)}
        for rank, s in enumerate(semantic_results):
            rrf_scores[s['chunk_id']] += 1.0 / (RRF_K + rank + 1)
        for rank, b in enumerate(bm25_results):
            rrf_scores[b['chunk_id']] += 1.0 / (RRF_K + rank + 1)
        fused = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))[:num_results]
        results = []
        for rank, (cid, score) in enumerate(fused):
            row = {}
            if cid in semantic_meta:
                row.update(semantic_meta[cid])
            else:
                meta = self.bm25_index.get_chunk_metadata(cid) if self.bm25_index else None
                meta = meta or {}
                row.update(meta)
                row['doc_name'] = meta.get('file_name') or meta.get('doc_id') or cid
                row['distance'] = None
            row['bm25_score'] = bm25_meta[cid]['bm25_score'] if cid in bm25_meta else None
            row['bm25_rank'] = bm25_ranks.get(cid)
            row['semantic_rank'] = semantic_ranks.get(cid)
            row['rrf_score'] = score
            row['rrf_rank'] = rank
            results.append(row)
        return results

    def search_hybrid_v2(self, query: str, num_results: int = 5,
                         doc_ids: Optional[List[str]] = None) -> List[Dict]:
        """Hybrid retrieval: pgvector semantic + in-memory BM25 + RRF fusion.

        When *doc_ids* is provided the document filter is applied *inside* both
        retrieval channels before RRF fusion:
          * Semantic channel: a ``WHERE dc.doc_id = ANY(:doc_ids)`` clause is
            added to the pgvector SQL, restricting the candidate pool before
            the ``LIMIT :pool`` top-k.
          * BM25 channel: candidates whose stored metadata ``doc_id`` is not in
            the set are removed from the BM25 result list *before* fusion.
        This ensures the filter is not a post-fusion top-k filter.
        (D1 fix, canonical since 2026-10-29; regression coverage in
        tests/test_hybrid_retrieval.py section E — document-scoped retrieval.)
        """
        try:
            self._ensure_bm25()
            query_embedding = self.embed_query(query)
            embedding_json = json.dumps(query_embedding)
            pool = max(SEMANTIC_CANDIDATE_POOL, num_results * 3)
            semantic_sql = """
                SELECT
                    dc.chunk_id, dc.chunk_text, dc.doc_id,
                    COALESCE(ld.file_name, dc.doc_id) as doc_name,
                    dc.chunk_index, dc.section_path, dc.page_start, dc.page_end,
                    dc.table_html, dc.has_table,
                    dc.embedding <=> :query_embedding::vector as distance
                FROM document_chunks_v2 dc
                LEFT JOIN legal_documents_v2 ld ON dc.doc_id = ld.doc_id
            """
            sem_kwargs: Dict[str, object] = {
                "query_embedding": embedding_json,
                "pool": pool,
            }
            if doc_ids is not None:
                semantic_sql += " WHERE dc.doc_id = ANY(:doc_ids)\n"
                sem_kwargs["doc_ids"] = list(doc_ids)
            semantic_sql += (
                " ORDER BY dc.embedding <=> :query_embedding::vector\n"
                " LIMIT :pool"
            )
            sem_rows = self.conn.run(semantic_sql, **sem_kwargs)
            from urllib.parse import unquote
            semantic_results = []
            for r in sem_rows:
                raw_ht = r[9]
                semantic_results.append({
                    'chunk_id': r[0], 'chunk_text': r[1], 'doc_id': r[2],
                    'doc_name': unquote(r[3]) if r[3] else r[2],
                    'chunk_index': r[4], 'section_path': r[5],
                    'page_start': r[6], 'page_end': r[7],
                    'table_html': r[8],
                    'has_table': raw_ht if isinstance(raw_ht, bool) else str(raw_ht).lower() == 'true',
                    'distance': float(r[10]),
                })
            bm25_results = self.bm25_index.search(query, k=max(BM25_CANDIDATE_POOL, num_results * 3))
            # ── Document filter for BM25 channel (before fusion) ──
            if doc_ids is not None and self.bm25_index is not None:
                doc_id_set = set(doc_ids)
                filtered_bm25 = []
                for r in bm25_results:
                    meta = self.bm25_index.get_chunk_metadata(r['chunk_id'])
                    if meta and meta.get('doc_id') in doc_id_set:
                        filtered_bm25.append(r)
                bm25_results = filtered_bm25
            # Channels stay independent: BM25-only candidates are NOT appended
            # to semantic_results. Fusion and the metadata join happen in
            # _rrf_fuse, using each channel's own ranking.
            fused = self._rrf_fuse(semantic_results, bm25_results, num_results)
            logger.info(f"  Hybrid search: {len(semantic_results)} semantic, {len(bm25_results)} BM25, {len(fused)} fused")
            return fused
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            raise

    def get_chunks_for_expansion(self, doc_id: str,
                                 section_path: Optional[str] = None,
                                 chunk_indexes: Optional[List[int]] = None) -> List[Dict]:
        """Fetch stored chunks of one document — by section and/or by a
        chunk_index list — for evidence refinement (fix 011, Stage A).

        Used to complete a provision split across chunks (the remaining
        chunks of a numbered section) and to pull chunk_index-adjacent
        neighbours.  Returns chunk dicts in chunk_index order, including
        the preserved table_html / has_table structure.
        """
        from urllib.parse import unquote
        sql = """
            SELECT dc.chunk_id, dc.chunk_text, dc.doc_id,
                   COALESCE(ld.file_name, dc.doc_id) as doc_name,
                   dc.chunk_index, dc.section_path, dc.page_start, dc.page_end,
                   dc.table_html, dc.has_table
            FROM document_chunks_v2 dc
            LEFT JOIN legal_documents_v2 ld ON dc.doc_id = ld.doc_id
            WHERE dc.doc_id = :doc_id
        """
        kwargs: Dict[str, object] = {"doc_id": doc_id}
        conds: List[str] = []
        if section_path:
            conds.append("dc.section_path = :section_path")
            kwargs["section_path"] = section_path
        if chunk_indexes:
            conds.append("dc.chunk_index = ANY(:idxs)")
            kwargs["idxs"] = [int(i) for i in chunk_indexes]
        if conds:
            sql += " AND (" + " OR ".join(conds) + ")\n"
        sql += " ORDER BY dc.chunk_index"
        rows = self.conn.run(sql, **kwargs)
        chunks: List[Dict] = []
        for r in rows:
            raw_ht = r[9]
            chunks.append({
                'chunk_id': r[0], 'chunk_text': r[1], 'doc_id': r[2],
                'doc_name': unquote(r[3]) if r[3] else r[2],
                'chunk_index': r[4], 'section_path': r[5],
                'page_start': r[6], 'page_end': r[7],
                'table_html': r[8],
                'has_table': raw_ht if isinstance(raw_ht, bool) else str(raw_ht).lower() == 'true',
            })
        return chunks

    def _assemble_context(self, context_chunks: List[Dict]) -> Tuple[str, Dict]:
        """Pack retrieved chunks into a context string under MAX_CONTEXT_CHARS.

        The budget counts headers, chunk text, and the two-character
        separators between blocks. Chunks that no longer fit (or that have no
        usable text) are recorded with a reason in the returned stats.
        """
        from urllib.parse import unquote
        context_parts = []
        skipped = []
        total_chars = 0
        for i, chunk in enumerate(context_chunks):
            # Fix 011 (2026-10-30, Stage C): for table chunks the structured
            # Markdown table REPLACES the flattened chunk_text — same content,
            # cell boundaries preserved, so the LLM sees an actual table
            # instead of glued-together values.
            text = chunk.get('table_markdown') or chunk.get('chunk_text')
            if not text or not str(text).strip():
                skipped.append({'chunk_id': chunk.get('chunk_id'),
                                'reason': 'missing_or_empty_chunk_text'})
                continue
            doc_name = unquote(chunk.get('doc_name') or chunk.get('file_name')
                               or chunk.get('doc_id') or 'Unknown')
            section = chunk.get('section_path') or ''
            ps = chunk.get('page_start')
            pe = chunk.get('page_end')
            if ps is not None and pe is not None:
                page_info = f" | Page {ps}" if ps == pe else f" | Pages {ps}-{pe}"
            else:
                page_info = ''
            section_info = f" | Section: {section}" if section else ''
            header = f"[Document {i+1}] {doc_name}{section_info}{page_info}"
            block = f"{header}\n{text}"
            sep = 2 if context_parts else 0
            if total_chars + len(block) + sep > MAX_CONTEXT_CHARS:
                skipped.append({'chunk_id': chunk.get('chunk_id'),
                                'reason': (f"context_limit: would add {len(block) + sep} chars "
                                           f"at {total_chars}/{MAX_CONTEXT_CHARS}")})
                continue
            context_parts.append(block)
            total_chars += len(block) + sep
        context_text = "\n\n".join(context_parts)
        stats = {'packed_chunks': len(context_parts), 'total_chars': total_chars, 'skipped': skipped}
        self.last_context_assembly = stats
        return context_text, stats

    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate LLM response using retrieved context."""
        try:
            # Build context from chunks with document names, section paths, and page ranges
            context_text, ctx_stats = self._assemble_context(context_chunks)
            if ctx_stats['skipped']:
                logger.info(f"  Context packing: {ctx_stats['packed_chunks']} packed, "
                            f"{len(ctx_stats['skipped'])} skipped: {ctx_stats['skipped']}")
            
            # Fix 011 (2026-10-30, Stage B): answer-synthesis prompt rewritten.
            # Hosted failures verified 2026-10-30: the old "Be concise and
            # direct" prompt produced a summary that never extracted the exact
            # amounts/limits, and unstructured prose. The new prompt enforces
            # (a) a fixed answer structure, (b) verbatim extraction of
            # figures/terms, and (c) positional [n] citations that the
            # final-answer source selector parses (_cited_indices) to group
            # sources per claim.
            prompt = f"""You are a legal assistant helping a user understand legal documents. Answer based ONLY on the provided document excerpts.

**Document Excerpts:**
{context_text}

**User Question:** {query}

**Guidelines:**
- Structure the answer in Markdown:
  **Direct answer** - one or two sentences answering the question directly.
  **Key requirements** (or **Key points**) - a bulleted list of the substantive rules, obligations, amounts, dates, or definitions found in the excerpts.
  **Qualifications** - only if the excerpts impose conditions, exceptions, or the question is only partially covered.
- Quote EXACT figures, defined terms, and deadlines as they appear in the excerpts (amounts, periods, dates, party names). Never round or re-word numbers.
- Mark each claim with its source position as [n], where n is the Document number shown in the excerpt headers (for example: employer's liability of $1,000,000 [1]). Use ONLY numbers that appear in the headers.
- For broad/general questions (e.g., "what types of documents exist?"), give a concise high-level summary instead of a bulleted extraction.
- If the excerpts do not contain enough information for part of the answer, state the evidence gap explicitly for that part. If nothing in the excerpts answers the question, say "The available documents don't contain enough information to answer this question."
- Do NOT invent page numbers, section references, or citation numbers. Only cite what appears in the excerpt headers.
- Do NOT start with phrases like "Based on the provided legal document excerpts...".
- Use professional but accessible language.

**Answer:**"""

            # Call Databricks Foundation Model API with raw HTTP
            from databricks.sdk import WorkspaceClient
            import requests
            
            w = WorkspaceClient()
            
            # Get auth header from workspace client
            auth_header = w.config.authenticate()
            
            # Call LLM endpoint with raw HTTP
            url = f"{w.config.host}/serving-endpoints/{LLM_MODEL}/invocations"
            headers = {"Content-Type": "application/json"}
            headers.update(auth_header)
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful legal assistant."},
                    {"role": "user", "content": prompt}
                ],
                # Fix 011: 1000 tokens truncated structured answers with
                # quoted requirements; 2000 gives room for extraction + citations.
                "max_tokens": 2000,
                "temperature": 0.1
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract answer from response (raw HTTP returns dict)
            if not isinstance(response_data, dict):
                raise ValueError(f"Expected dict response, got {type(response_data)}")
            
            if 'choices' not in response_data:
                raise ValueError(f"No 'choices' in response. Keys: {list(response_data.keys())}")
            
            choices = response_data['choices']
            if not choices or len(choices) == 0:
                raise ValueError(f"Empty choices list")
            
            first_choice = choices[0]
            if not isinstance(first_choice, dict) or 'message' not in first_choice:
                raise ValueError(f"Invalid choice format: {first_choice}")
            
            message = first_choice['message']
            if not isinstance(message, dict) or 'content' not in message:
                raise ValueError(f"Invalid message format: {message}")
            
            answer = message['content']
            if not answer:
                raise ValueError(f"Empty content in message")
            
            logger.info("✅ Generated LLM response")
            return answer
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            raise

    def invoke(self, query: str, num_results: int = 5) -> Dict:
        """Main RAG pipeline: robust, production-grade legal agent."""
        try:
            # 1. Greet or handle edge-case queries
            greetings = {"hi", "hello", "hey", "help", "test", "start", "begin", "greetings"}
            q_clean = query.strip().lower()
            if not q_clean or q_clean in greetings:
                welcome = "Welcome to the Legal Intelligence Assistant. Ask a legal question or reference a document, and I'll help!"
                logger.info("✅ Greeted user for generic query")
                return {
                    "content": welcome,
                    "chunks": [],
                    "query": query
                }
            # 2. Retrieve relevant chunks only for real legal questions
            chunks = self.search(query, num_results=num_results)
            # 3. Generate response
            answer = self.generate_response(query, chunks)
            logger.info("✅ RAG pipeline completed successfully")
            return {
                "content": answer,
                "chunks": chunks,
                "query": query
            }
        except Exception as e:
            logger.error(f"❌ RAG pipeline failed: {e}")
            return {
                "content": f"I encountered an error: {str(e)}",
                "chunks": [],
                "query": query
            }


# Singleton instance
_retriever_instance = None

def get_lakebase_retriever() -> LakebaseRetriever:
    """Get or create singleton LakebaseRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("Creating new LakebaseRetriever instance...")
        _retriever_instance = LakebaseRetriever()
    return _retriever_instance
