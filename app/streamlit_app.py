"""Legal Intelligence Assistant — LangGraph capability workflow app.

Milestone 5 (Stage C1, 2026-10-29): the chat is wired to the LangGraph
workflow built by app_runtime.build_app_graph() with REAL capability nodes
only — hybrid_v2 retrieval (pgvector + BM25 + RRF fusion), read-only
warehouse SQL via named-parameter statement execution, and grounded answer
generation from validated evidence.  The legacy single-step RAG path
(retriever.invoke) is fully replaced.  Fail-fast: the workflow never falls
back silently to a legacy retrieval path.
"""
import os
import traceback

import streamlit as st

# Page config
st.set_page_config(page_title="Legal Intelligence Assistant", layout="wide")

st.title("⚖️ Legal Intelligence Assistant")
st.caption("Powered by Lakebase Postgres, pgvector & the LangGraph capability workflow")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource(show_spinner=False)
def get_workflow():
    """Build the LangGraph workflow once per app process (fail-fast)."""
    from app_runtime import build_app_graph
    return build_app_graph()


@st.cache_resource(show_spinner=False)
def get_sql_executor():
    """Read-only warehouse executor for source-viewer lookups (Stage 3)."""
    from app_runtime import StatementExecutionSqlExecutor
    return StatementExecutionSqlExecutor()


@st.cache_data(show_spinner=False)
def _load_pdf_cached(file_path: str) -> bytes:
    """Fetch a canonical PDF once per file path (UC volume via Files API)."""
    from source_viewer import load_pdf_bytes
    return load_pdf_bytes(file_path)


def _get(obj, key, default=None):
    """Attribute-or-key access tolerant of pydantic objects and plain dicts
    (history messages carry SourceGroup/SourceRef objects; AppTest fixtures
    may carry dicts)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _groups_from_msg(msg: dict) -> list:
    """Fix 012 (Defect B): the message's document-level source groups.

    Prefers source_groups (new payload); a legacy message that carries only
    source_refs is grouped on the fly by canonical doc_id so old history
    keeps rendering correctly (group_refs_by_doc, additive, no data change).
    """
    groups = msg.get("source_groups") or []
    if groups:
        return groups
    refs = msg.get("source_refs") or []
    if refs:
        from workflow.answers import group_refs_by_doc
        return group_refs_by_doc(refs)
    return []


def _render_pdf_viewer(m_idx: int, groups: list) -> None:
    """Fix 012 (2026-10-31): persistent, single PDF viewer per assistant
    message (Defects A + viewer requirements).

    All viewer state lives in session_state under explicit keys, so it
    survives every Streamlit rerun and never depends on widget auto-state:

    * pdf_m{m_idx}          {"open": bool, "g": group idx, "e": evidence idx}
    * pdf_m{m_idx}_page     current 0-based PDF page id

    Behaviour required by fix 012 and implemented here:
    - one viewer per message (clicking another evidence passage NAVIGATES
      the same viewer; no duplicate viewers, no panel reset);
    - prev/next evidence controls (across document groups the active PDF
      switches with them; no bounding boxes from the previous document are
      ever reused - the highlight set is rebuilt from the ACTIVE evidence
      chunk's stored bboxes);
    - prev/next page plus jump-to-evidence-page;
    - explicit Close leaves the Sources panel untouched;
    - No st.rerun() in the sources/viewer/table interaction paths: every
      control updates session_state and re-reads it within the same run, so
      render-time state-mutation loops are impossible by construction (the
      only st.rerun in the app is the one-shot Clear Chat button).
    """
    from source_viewer import build_source_view, render_page_with_highlight

    state_key = f"pdf_m{m_idx}"
    page_key = f"{state_key}_page"
    view_state = st.session_state.get(state_key) or {}
    if not view_state.get("open"):
        return

    flat = []
    for g_idx, g in enumerate(groups):
        for e_idx, ref in enumerate(_get(g, "evidence", []) or []):
            flat.append((g_idx, e_idx, ref))
    if not flat:
        return

    g_idx = int(view_state.get("g", 0))
    e_idx = int(view_state.get("e", 0))
    try:
        pos = next(i for i, (gg, ee, _) in enumerate(flat) if gg == g_idx and ee == e_idx)
    except StopIteration:
        pos = 0
        g_idx, e_idx, _ = flat[0]

    # ── evidence navigation + close ────────────────────────────────────
    nav_e1, nav_e2, nav_c = st.columns(3)
    if pos > 0 and nav_e1.button("⏮ Previous evidence", key=f"eveprev_{state_key}"):
        pos -= 1
        st.session_state[state_key] = {"open": True, "g": flat[pos][0], "e": flat[pos][1]}
        st.session_state[page_key] = _get(flat[pos][2], "page_start")
    if pos < len(flat) - 1 and nav_e2.button("Next evidence ⏭", key=f"evenext_{state_key}"):
        pos += 1
        st.session_state[state_key] = {"open": True, "g": flat[pos][0], "e": flat[pos][1]}
        st.session_state[page_key] = _get(flat[pos][2], "page_start")
    if nav_c.button("✕ Close viewer", key=f"evclose_{state_key}"):
        st.session_state[state_key] = {"open": False, "g": g_idx, "e": e_idx}
        return
    # re-read after any evidence navigation this run
    view_state = st.session_state.get(state_key) or {}
    if not view_state.get("open"):
        return
    g_idx = int(view_state.get("g", 0))
    e_idx = int(view_state.get("e", 0))
    try:
        pos = next(i for i, (gg, ee, _) in enumerate(flat) if gg == g_idx and ee == e_idx)
    except StopIteration:
        pos = 0
    active = flat[pos][2]

    # ── build the source view for the ACTIVE evidence (cached per chunk) ─
    chunk_id = _get(active, "chunk_id")
    cache_key = f"view_m{m_idx}_{chunk_id if chunk_id else f'g{g_idx}_e{e_idx}'}"
    view = st.session_state.get(cache_key)
    if view is None:
        try:
            view = build_source_view(get_sql_executor().execute, active)
        except Exception as exc:
            st.caption(f"📄 Source page preview unavailable in this environment: {exc}")
            return
        st.session_state[cache_key] = view
    if view.get("error"):
        st.caption("📄 " + view["error"])
        return
    try:
        pdf_bytes = _load_pdf_cached(view["file_path"])
    except Exception as exc:
        st.warning(f"Could not load the source PDF: {exc}")
        return

    cited = view.get("page_id")
    if cited is None:
        cited = 0
    total = view.get("page_count") or 0

    # ── page navigation ─────────────────────────────────────────────────
    nav1, nav2, nav3 = st.columns(3)
    cur = st.session_state.get(page_key, cited)
    if cur is None:
        cur = cited
    if nav1.button("◀ Previous page", key=f"pgprev_{state_key}"):
        st.session_state[page_key] = max(0, int(cur) - 1)
    if nav2.button("🎯 Evidence page", key=f"pggoto_{state_key}"):
        st.session_state[page_key] = cited
    if (not total or int(cur) < int(total) - 1) and nav3.button("Next page ▶", key=f"pgnext_{state_key}"):
        st.session_state[page_key] = int(cur) + 1
    cur = st.session_state.get(page_key, cited)
    if cur is None:
        cur = cited

    try:
        cal = view.get("calibration") or {}
        res = render_page_with_highlight(
            pdf_bytes, int(cur), view["rects"],
            px_per_pt=cal.get("px_per_pt"),
        )
    except Exception as exc:
        st.warning(f"Could not render the source PDF page: {exc}")
        return
    if res.get("error"):
        st.warning(res["error"])
        return
    cap = (view["display_name"] + " — PDF page " + str(res["physical_page"]))
    if total:
        cap += f" of {total}"
    if res["highlighted"]:
        cap += " — exact evidence highlight"
    elif int(cur) != int(cited) and view.get("physical_page"):
        cap += f" — evidence page is PDF page {view['physical_page']}"
    st.image(res["image"], caption=cap)
    st.caption(res["note"])
    if cal.get("scale_status"):
        st.caption(f"Page calibration: {cal['scale_status']}.")
    st.download_button(
        "Download full PDF",
        data=pdf_bytes,
        file_name=(view["display_name"] or "document") + ".pdf",
        key=f"dl_{state_key}",
    )


def _render_assistant_payload(msg: dict, m_idx: int) -> None:
    """Sources panel + workflow trace for ONE assistant message.

    Fix 012 (2026-10-31) — defects A/B/C remediation:

    * Defect A: the Sources panel open/closed state lives in session_state
      (src_open_m{m_idx}) behind a toggle button, NOT in an st.expander.
      st.expander's open state is widget-internal and auto-keyed by
      element-tree position; the panel rendered through two different
      script paths (live submit path vs history replay loop), so the
      expander instance the user opened was never the one rendered after
      the rerun triggered by 'View PDF' — the panel collapsed. A
      session-state key is path-independent and survives every rerun.
    * Defect B: sources are grouped ONE top-level entry per canonical
      document (doc_id); every evidence passage keeps its own [n]
      evidence_ref, chunk_id, section, page range, and excerpt, each with
      its own 'View in PDF' action. Document titles fall back explicitly
      (never a section number in place of a title).
    * One persistent PDF viewer per message (see _render_pdf_viewer).
    * The table payload renders with a stable element key and an explicit
      string dtype (stable Arrow schema across reruns — defect C hardening
      against React error #185).
    """
    table_result = msg.get("table_result")
    if table_result:
        import pandas as pd
        from workflow.display import display_doc_name, physical_page
        columns = table_result.get("columns") or []
        rows = table_result.get("rows") or []
        if columns:
            # Object dtype leaves the Arrow schema to per-serialization
            # type inference; an explicit string dtype pins it. Cells stay
            # the EXACT source strings (no numeric coercion, leading zeros
            # preserved). The stable key pins widget identity across
            # reruns (React update-loop hardening, defect C).
            df = pd.DataFrame(rows, columns=columns).astype("string")
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                key=f"tbl_m{m_idx}",
            )
        caption_parts = []
        if table_result.get("section_path"):
            caption_parts.append(str(table_result["section_path"]))
        caption_parts.append(f"{table_result.get('row_count', len(rows))} rows — exact values from the source document")
        p1 = physical_page(table_result.get("page_start"))
        p2 = physical_page(table_result.get("page_end"))
        if p1 is not None and p2 is not None and p1 != p2:
            caption_parts.append(f"PDF pages {p1}-{p2}")
        elif p1 is not None:
            caption_parts.append(f"PDF page {p1}")
        if table_result.get("doc_name"):
            caption_parts.append(display_doc_name(table_result["doc_name"]) or "")
        st.caption(" · ".join(part for part in caption_parts if part))
        if table_result.get("continuation_note"):
            st.caption(table_result["continuation_note"])

    groups = _groups_from_msg(msg)
    if groups:
        from workflow.display import display_doc_name, physical_page
        n_docs = len(groups)
        n_ev = sum(len(_get(g, "evidence", []) or []) for g in groups)
        doc_word = "document" + ("s" if n_docs != 1 else "")
        ev_word = "evidence passage" + ("s" if n_ev != 1 else "")
        open_key = f"src_open_m{m_idx}"
        is_open = bool(st.session_state.get(open_key))
        # Fix 012 stability rule: the label must stay CONSTANT across
        # open/close states. Streamlit derives widget identity from the
        # element's full args (label included) even with a user key, so a
        # label that changes with state (e.g. a ▸/▾ arrow) recreates the
        # widget on every toggle — churn that both breaks test automation
        # and feeds the React re-render instability (defect C). The open
        # state is conveyed by the panel contents themselves.
        if st.button(
            f"📄 Sources ({n_docs} {doc_word} · {n_ev} {ev_word})",
            key=f"src_toggle_m{m_idx}",
        ):
            st.session_state[open_key] = not is_open
            is_open = not is_open
        if not is_open:
            return
        for g_idx, g in enumerate(groups):
            doc_title = display_doc_name(_get(g, "doc_name")) or ""
            if not doc_title:
                did = _get(g, "doc_id")
                # Explicit fallback — never an empty title, never a bare
                # section number standing in for the document name.
                doc_title = (f"Document {str(did)[:12]}…") if did else "Untitled document"
            meta = []
            g_section = _get(g, "section_path")
            if g_section:
                meta.append(f"Section {g_section}")
            gp1 = physical_page(_get(g, "page_start"))
            gp2 = physical_page(_get(g, "page_end"))
            if gp1 is not None and gp2 is not None and gp1 != gp2:
                meta.append(f"PDF pp. {gp1}–{gp2}")
            elif gp1 is not None:
                meta.append(f"PDF page {gp1}")
            header = f"**{doc_title}**" + (f" — {', '.join(meta)}" if meta else "")
            st.markdown(header)
            for e_idx, ref in enumerate(_get(g, "evidence", []) or []):
                er = _get(ref, "evidence_ref")
                badge = f"[{er}]" if er else f"E{e_idx + 1}"
                line = f"Evidence {badge}"
                sec = _get(ref, "section_path")
                if sec:
                    line += f" — Section {sec}"
                rp1 = physical_page(_get(ref, "page_start"))
                rp2 = physical_page(_get(ref, "page_end"))
                if rp1 is not None and rp2 is not None and rp1 != rp2:
                    line += f" — PDF pp. {rp1}–{rp2}"
                elif rp1 is not None:
                    line += f" — PDF page {rp1}"
                col_info, col_btn = st.columns([0.82, 0.18])
                col_info.markdown(f"**{line}**")
                excerpt = _get(ref, "excerpt", "") or ""
                if excerpt:
                    shown = excerpt[:300]
                    col_info.caption(shown + ("…" if len(excerpt) > 300 else ""))
                if col_btn.button(
                    "View in PDF",
                    key=f"evbtn_m{m_idx}_g{g_idx}_e{e_idx}",
                    use_container_width=True,
                ):
                    # Navigate the single persistent viewer — the Sources
                    # panel state is never touched by this click.
                    st.session_state[f"pdf_m{m_idx}"] = {
                        "open": True, "g": g_idx, "e": e_idx,
                    }
                    st.session_state[f"pdf_m{m_idx}_page"] = _get(ref, "page_start")
        _render_pdf_viewer(m_idx, groups)
        st.caption("⚠️ *AI-generated response. For legal advice, consult a qualified attorney.*")

    # Workflow diagnostics: node/action/reason only — no evidence
    # payloads, SQL parameters, or credentials (app_runtime contract).
    summary = msg.get("trace") or []
    if summary:
        with st.expander("🔎 Workflow trace", expanded=False):
            for entry in summary:
                line = f"`{entry.get('node')}` → **{entry.get('action')}**"
                if entry.get("reason"):
                    line += f" — {entry['reason']}"
                st.markdown(f"- {line}")

# Display chat messages from history (fix 006: assistant messages replay
# their Sources panel + trace + viewer too, not just the answer text)
for m_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("role") == "assistant":
            _render_assistant_payload(message, m_idx)

# Chat input — routed through the LangGraph capability workflow
if prompt := st.chat_input("Ask about legal documents..."):
    # Add user message and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Follow-up context: prior turns only (the current question is user_query).
    # The workflow's query planner resolves follow-ups against this history
    # (spec Section 2.1).
    conversation_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing your question through the workflow..."):
                graph = get_workflow()
                # Fix 010: pass back the pending recommended action (e.g.
                # the comparison-candidate shortlist) from the last
                # assistant turn so the planner can resolve acceptance
                # phrasings like 'sure compare them then' to the exact
                # recommended documents.
                invoke_input = {
                    "user_query": prompt,
                    "conversation_history": conversation_history,
                }
                pending_action = next(
                    (
                        m.get("pending_action")
                        for m in reversed(st.session_state.messages[:-1])
                        if m.get("role") == "assistant" and m.get("pending_action")
                    ),
                    None,
                )
                if pending_action:
                    invoke_input["pending_action"] = pending_action
                # Fix 012 (2026-10-31, Defect D): the canonical documents
                # that backed the PREVIOUS assistant answer, so a follow-up
                # that names no agreement ('What about the limits?') is
                # scoped to the agreement the previous answer actually used
                # (or, when several, the planner asks one focused
                # clarification instead of silently searching the corpus).
                prior_docs = None
                for m in reversed(st.session_state.messages[:-1]):
                    if m.get("role") == "assistant":
                        docs = []
                        for g in (m.get("source_groups") or []):
                            dn = _get(g, "doc_name")
                            if dn and dn not in docs:
                                docs.append(dn)
                        if not docs:
                            tr = m.get("table_result") or {}
                            if tr.get("doc_name"):
                                docs = [tr["doc_name"]]
                        if not docs:
                            for r in (m.get("source_refs") or []):
                                dn = _get(r, "doc_name")
                                if dn and dn not in docs:
                                    docs.append(dn)
                        prior_docs = docs or None
                        break
                if prior_docs:
                    invoke_input["prior_answer_docs"] = prior_docs
                state = graph.invoke(invoke_input)
        except Exception as e:
            # Startup or graph-level operational failure — clear message,
            # credential-free diagnostics (Stage A transparency requirement).
            st.error(f"⚠️ The assistant could not process your question: {e}")
            with st.expander("🔍 Diagnostics"):
                st.code(traceback.format_exc())
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {e}",
            })
        else:
            answer = state.get("answer") or "(no answer produced)"
            st.markdown(answer)

            # Citations (spec Section 1.7): readable document title, section,
            # physical page, and a supporting excerpt - and (Stage 3 fix,
            # 2026-10-29) an in-app PDF page preview with the exact stored
            # evidence highlight when the metadata is available.
            # Fix 006: the rich payload (source_refs + trace summary) is
            # stored ON the assistant history message so the Sources panel,
            # trace, and viewer survive every rerun (button clicks, new
            # questions) via _render_assistant_payload in the replay loop.
            from app_runtime import trace_summary
            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "source_refs": state.get("source_refs") or [],
                # Fix 012 (Defect B): document-grouped sources — one
                # top-level entry per canonical document; every evidence
                # passage keeps its chunk/section/page/[n] identity.
                "source_groups": state.get("source_groups") or [],
                "trace": trace_summary(state) or [],
                # Fix 010: persist the pending recommended action (if any)
                # so a follow-up acceptance ('sure compare them then') can
                # resolve the exact documents on the next turn.
                "pending_action": state.get("pending_action"),
                # Fix 011 (Stage C): persist the structured table payload so
                # history replay re-renders the table, not flattened text.
                "table_result": state.get("table_result"),
            }
            # Fix 012 (repeat integrity): 'Here is my previous answer:' is
            # the repeat branch's lead — that branch returns the prior text
            # WITHOUT rebuilding evidence, so a fresh sources payload would
            # be empty.  Clone the ORIGINAL message's citation-to-evidence
            # mapping so the repeated answer keeps its exact sources (and
            # its table, when the original was a table answer).
            if answer.startswith("Here is my previous answer:"):
                for m in reversed(st.session_state.messages):
                    if m.get("role") == "assistant" and not str(m.get("content", "")).startswith("⚠️"):
                        assistant_msg["source_refs"] = m.get("source_refs") or []
                        assistant_msg["source_groups"] = m.get("source_groups") or []
                        assistant_msg["table_result"] = m.get("table_result")
                        break
            # Index this message will occupy in history (appended below)
            _render_assistant_payload(assistant_msg, len(st.session_state.messages))
            st.session_state.messages.append(assistant_msg)

# Sidebar
with st.sidebar:
    st.header("⚖️ Legal Assistant")
    st.write("AI-powered legal document search with Lakebase Postgres & pgvector")
    
    st.divider()
    
    msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Questions Asked", msg_count)
    
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        # Fix 006/009: purge per-source viewer state (views, latches, page
        # positions) so cleared sessions never reuse stale state at recycled
        # message indexes.
        # Fix 012: also purge the new panel/viewer keys (src_open_* toggles,
        # pdf_m* viewer state incl. pdf_m*_page, view_m* view cache).
        for k in [k for k in list(st.session_state) if k.startswith(("view_", "show_view_", "page_", "src_open_", "pdf_m"))]:
            del st.session_state[k]
        st.rerun()
    
    st.divider()
    
    st.subheader("🤖 Agent Info")
    st.caption("**Workflow:** LangGraph capability graph (corpus overview, clause discovery, structured lookup, comparison, hybrid Q&A)")
    st.caption("**Retrieval:** hybrid_v2 — pgvector semantic + BM25 + RRF fusion")
    st.caption("**Model:** Meta Llama 3.3 70B Instruct")
    st.caption("**Embedding:** databricks-gte-large-en")
    st.caption("**Backend:** Lakebase Postgres + SQL warehouse (read-only)")
    
    st.divider()
    
    # Debug toggle
    if st.checkbox("🔧 Show Debug Info", value=False):
        st.subheader("Environment")
        st.code(f"DATABRICKS_HOST: {'Set' if os.environ.get('DATABRICKS_HOST') else 'Not set'}")
        st.code(f"DATABRICKS_TOKEN: {'Set' if os.environ.get('DATABRICKS_TOKEN') else 'Not set'}")
