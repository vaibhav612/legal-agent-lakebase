"""Intent classification for the Legal Assistant LangGraph (spec Section 2.3).

Provides three classifier implementations:
  - IntentClassifier: abstract base for dependency injection.
  - RuleBasedIntentClassifier: deterministic, pattern-based — NO LLM calls.
    Used for unit tests and as a safe fallback.
  - LLMIntentClassifier: calls the LLM endpoint with schema validation
    and bounded retries.  Falls back to 'unsupported' on failure — never
    silently falls back to 'hybrid_qa'.

Design note: The behavior spec warns not to assume constrained JSON-schema
generation is supported by the existing LLM endpoint.  The LLMIntentClassifier
therefore prompts for JSON, parses it, and validates against the Intent
pydantic model.  If parsing or validation fails after max_retries, it
returns an 'unsupported' intent with a clarification message.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional

from pydantic import ValidationError

from .state import Intent, ALLOWED_FIELDS, ALLOWED_NODES, MAX_PLAN_STEPS
from .traces import make_trace

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────

class IntentClassifier(ABC):
    """Abstract intent classifier — inject a subclass for testing or production."""

    @abstractmethod
    def classify(self, query: str, conversation_history: list[dict] | None = None) -> Intent:
        """Classify user query into an Intent.  Must never return None."""
        ...


# ── Rule-based classifier (deterministic, no LLM) ──────────────────────

class RuleBasedIntentClassifier(IntentClassifier):
    """Deterministic, pattern-based intent classifier for testing and fallback.

    Uses keyword/regex matching to classify queries into the 7 intent labels.
    Does NOT call any LLM, database, or external service.
    """

    # Map natural-language field phrases to allowlisted field names.
    FIELD_PATTERNS: list[tuple[str, str]] = [
        ("governing law", "governing_law"),
        ("agreement date", "agreement_date"),
        ("effective date", "effective_date"),
        ("expiration date", "expiration_or_term"),
        ("expiration", "expiration_or_term"),
        ("termination for convenience", "termination_for_convenience"),
        ("payment terms", "payment_terms"),
        ("renewal term", "renewal_term"),
        ("notice to terminate", "notice_to_terminate_renewal"),
        ("contract title", "contract_title"),
        ("parties", "parties"),
    ]

    # Value-filter questions (Milestone 3): phrase -> field whose value the
    # remainder of the query filters on (bound as a SQL parameter downstream).
    VALUE_FILTER_PATTERNS: list[tuple[str, str]] = [
        ("governed by", "governing_law"),
    ]

    # Status-filter questions (Milestone 3): which documents have missing or
    # uncertain values for a field.  Marker phrase -> status_filter value.
    STATUS_FILTER_MARKERS: list[tuple[str, str]] = [
        ("missing or uncertain", "missing_or_uncertain"),
        ("missing value", "missing"),
        ("missing values", "missing"),
        ("not extracted", "missing"),
        ("no extracted value", "missing"),
        ("uncertain value", "uncertain"),
        ("uncertain values", "uncertain"),
        ("marked uncertain", "uncertain"),
    ]

    CORPUS_PATTERNS: list[str] = [
        "how many documents", "how many contracts", "how many docs",
        "what types of documents", "list all documents",
        "show me all documents", "corpus composition",
        "document distribution", "what documents do you have",
        "count of documents",
    ]

    # Fix 013 (2026-11-03, corpus-misrouting investigation): corpus-level
    # detection for phrasings the exact-substring CORPUS_PATTERNS miss
    # ('How many canonical documents ...', 'What kind of documents ...',
    # 'distribution by document_class').  These are consulted ONLY when no
    # document name, value filter, discovery phrasing or comparison
    # phrasing is present -- see _is_corpus_query -- so specific field
    # questions keep their existing routes.
    CORPUS_COUNT_RE = re.compile(
        r"\bhow\s+many\s+"
        r"(?:(?:total|canonical|registered|distinct|duplicate|qualified)\s+)*"
        r"(?:documents|docs|contracts|agreements)\b"
    )
    # 'How many agreements contain indemnification provisions?' is a
    # clause-shaped question, not a corpus metadata count: a counted noun
    # followed by a content verb stays out of corpus_overview.
    CORPUS_COUNT_VERB_EXCLUSION_RE = re.compile(
        r"\bhow\s+many\s+[^.?]{0,40}?(?:documents|docs|contracts|agreements)\s+"
        r"(?:contain|mention|include|discuss|address|cover)\b"
    )
    CORPUS_KIND_RE = re.compile(
        r"\bwhat\s+(?:kind|kinds|type|types)\s+of\s+documents\b"
    )
    CORPUS_CLASS_PHRASES: list[str] = [
        "document class", "document classes", "document_class",
        "document categories", "document category",
        "class distribution", "how are documents classified",
        "classification of documents",
    ]
    CORPUS_AGGREGATE_MARKERS: list[str] = [
        "total document count", "document count", "canonical document count",
        "registered in the corpus", "registered in corpus",
        "across the corpus", "across the whole corpus",
        "in the entire collection", "in the whole collection",
        "the entire corpus", "the whole corpus",
        "excluding duplicates", "group by document class",
        "grouped by document class", "by the document_class field",
        "distribution of documents", "distribution of the documents",
    ]

    # Fix 010 (2026-10-29, hosted smoke finding): 'What documents can be
    # compared?' used to fall through to the hybrid_qa default (branch 8)
    # because branch 5 requires >= 2 extracted document names — the result
    # was a generic semantic-search answer built from irrelevant top-k
    # chunks (land-use reports, investor reports, hosting monitoring)
    # displayed as sources. These phrasings are instead a COMPARISON
    # CANDIDATE DISCOVERY request (aspect='candidates'): the comparison
    # node answers from the canonical document inventory and verified
    # metadata — never from retrieval chunks. Anchored on 'compared'/'
    # comparable' + a documents/agreements/pairs token so genuine legal
    # questions ('compare the governing law of X and Y') never match.
    COMPARE_CANDIDATES_PATTERNS: list[str] = [
        "what documents can be compared",
        "which documents can be compared",
        "what agreements can be compared",
        "which agreements can be compared",
        "what documents are comparable",
        "which documents are comparable",
        "what agreements are comparable",
        "which agreements are comparable",
        "what can be compared",
        "what could be compared",
        "what can i compare",
        "what should i compare",
        "documents for comparison",
        "candidates for comparison",
        "what pairs of documents",
        "which pairs of documents",
        "what pairs of agreements",
        "which pairs of agreements",
        "what comparisons are possible",
        "what comparisons can you",
        "comparison options",
    ]

    DISCOVERY_PATTERNS: list[str] = [
        "find all", "search all contracts for",
        "across all contracts", "across all documents",
        "all contracts that have", "exhaustive search",
        "every contract",
        # Milestone 3 (clause discovery): broader discovery phrases
        "which agreements contain", "which contracts contain",
        "which documents contain",
        "agreements that contain", "contracts that contain",
        "contain limitation of liability",
        "find indemnification provisions",
        "find indemnification clauses",
        "find limitation of liability",
        "find force majeure",
        "provisions in these selected",
        "provisions in the selected",
        "clauses in these selected",
        "clauses in the selected",
    ]

    DUMP_PATTERNS: list[str] = [
        "show me all text from all documents",
        "show all text from all",
        "dump all documents",
        "give me all the text",
        "display all document content",
        "all text from every document",
    ]

    MULTI_STEP_MARKERS: list[str] = [
        "and tell me", "and which", "and then",
        "and also", "first...then",
    ]

    FOLLOWUP_MARKERS: list[str] = [
        "what about", "how about", "and for",
        "same for", "also check",
    ]

    AMBIGUOUS_MARKERS: list[str] = [
        "what about it", "tell me more about that",
        "tell me more", "continue",
        "more details", "elaborate",
    ]

    def classify(self, query: str, conversation_history: list[dict] | None = None) -> Intent:
        q = query.lower().strip()
        history = conversation_history or []

        # ── 1. Corpus dump → unsupported ──
        if any(p in q for p in self.DUMP_PATTERNS):
            return Intent(
                label="unsupported",
                scope="all",
                clarification_needed=(
                    "I cannot display all text from all documents. "
                    "I can search for specific clauses, look up extracted "
                    "fields, or summarize the corpus composition."
                ),
            )

        # ── 2. Follow-up with conversation context ──
        if self._is_followup(q, history):
            resolved = self._resolve_followup(query, q, history)
            if resolved is not None:
                return resolved

        # ── 3. Corpus overview ──
        # Fix 013: corpus detection is now _is_corpus_query() -- the
        # original substring CORPUS_PATTERNS plus guarded count /
        # classification / aggregate phrasings, so 'How many canonical
        # documents ...' and 'What kind of documents ...' reach this
        # branch instead of falling through to the hybrid_qa default.
        if self._is_corpus_query(q, query):
            # Could be a filtered listing by field value.  A field phrase
            # mentioned only inside a negated grouping instruction
            # ('do not group by contract title') never activates the
            # filtered variant.
            for pattern, field_name in self.FIELD_PATTERNS:
                if (pattern in q
                        and not self._field_mention_is_negated(q, pattern)
                        and ("where" in q or "that have" in q or "with" in q or " have " in q)):
                    return Intent(
                        label="corpus_overview",
                        scope="filtered",
                        target_field=field_name,
                        needs_evidence=False,
                    )
            return Intent(label="corpus_overview", scope="all", needs_evidence=False)

        # ── 4. Clause discovery ──
        if any(p in q for p in self.DISCOVERY_PATTERNS):
            clause = self._extract_clause_pattern(q)
            doc_names = self._extract_discovery_doc_names(query)
            return Intent(
                label="clause_discovery",
                scope="specific_docs" if doc_names else "all",
                target_doc_names=doc_names,
                clause_pattern=clause or "unspecified",
                coverage_requirement="exhaustive",
                needs_evidence=True,
            )

        # ── 4b. Comparison-candidate discovery (fix 010) ──
        # 'What documents can be compared?' — must run BEFORE branch 5
        # because the phrasing contains 'compare'.
        if any(p in q for p in self.COMPARE_CANDIDATES_PATTERNS):
            return Intent(
                label="comparison",
                scope="all",
                comparison_aspect="candidates",
                target_doc_names=[],
                needs_evidence=False,
            )

        # ── 5. Comparison ──
        if "compare" in q or " vs " in q or " versus " in q or "difference between" in q:
            docs = self._extract_doc_names(query)
            aspect = self._extract_comparison_aspect(q)

            # Multi-step: comparison + follow-up action
            if any(m in q for m in self.MULTI_STEP_MARKERS):
                return Intent(
                    label="multi_step",
                    scope="specific_docs",
                    target_doc_names=docs,
                    comparison_aspect=aspect or "unspecified",
                    needs_evidence=True,
                )

            if len(docs) >= 2:
                return Intent(
                    label="comparison",
                    scope="specific_docs",
                    target_doc_names=docs,
                    comparison_aspect=aspect or "unspecified",
                    needs_evidence=True,
                )
            # Fall through if we can't identify 2 docs — might be ambiguous

            # ── 5b. Corpus-wide (oversized) comparison ──
            # A 'compare' request with no resolvable document names that
            # targets the whole corpus ('all/every/each N contracts') is an
            # oversized request: reject with a reason and supported
            # alternatives instead of falling through to hybrid_qa
            # (spec Section 1.5; benchmark M4-H02, defect D9(e)).
            if re.search(r"\b(?:all|every|each)\b", q):
                return Intent(
                    label="unsupported",
                    scope="all",
                    clarification_needed=(
                        "Comparing the whole corpus at once is too large a "
                        "request. I can compare a small number of named "
                        "documents, or look up a single field across the "
                        "corpus."
                    ),
                )

        # ── 5c. Multi-step structured questions ──
        # A field or value-filter question joined to another question by a
        # multi-step marker ('and which ... governed by ...') must not be
        # silently collapsed into a single structured_lookup.  The query
        # planner derives a per-clause plan for it (benchmark M4-G01,
        # defect D9(d)).
        if any(m in q for m in self.MULTI_STEP_MARKERS):
            # Fix 013: a field phrase mentioned only in a negated grouping
            # instruction does not count as an active field here either.
            has_field = any(
                p in q and not self._field_mention_is_negated(q, p)
                for p, _ in self.FIELD_PATTERNS
            )
            has_value_filter = any(p in q for p, _ in self.VALUE_FILTER_PATTERNS)
            if has_field or has_value_filter:
                return Intent(
                    label="multi_step",
                    scope="all",
                    needs_evidence=True,
                )

        # ── 6. Structured lookup ──
        # 6a. Value-filter questions ("which agreements are governed by X?")
        vf = self._extract_value_filter(query, q)
        if vf is not None:
            field_name, value = vf
            if value:
                doc = self._extract_single_doc(query)
                return Intent(
                    label="structured_lookup",
                    scope="specific_docs" if doc else "filtered",
                    target_doc_names=[doc] if doc else [],
                    target_field=field_name,
                    target_value=value,
                    needs_evidence=False,
                )

        for pattern, field_name in self.FIELD_PATTERNS:
            # Fix 013: a field phrase mentioned only inside a negated
            # grouping/filtering instruction ('do not group documents by
            # contract title') must not activate a structured field lookup.
            if pattern in q and not self._field_mention_is_negated(q, pattern):
                # 6b. Status-filter questions ("which agreements have missing
                # or uncertain values for X?" where X matches a field phrase)
                status = self._extract_status_filter(q)
                if status is not None:
                    return Intent(
                        label="structured_lookup",
                        scope="filtered",
                        target_field=field_name,
                        status_filter=status,
                        needs_evidence=False,
                    )
                doc = self._extract_single_doc(query)
                if doc:
                    return Intent(
                        label="structured_lookup",
                        scope="specific_docs",
                        target_doc_names=[doc],
                        target_field=field_name,
                        needs_evidence=False,
                    )
                # Field mentioned but no specific document → corpus overview
                return Intent(
                    label="corpus_overview",
                    scope="filtered",
                    target_field=field_name,
                    needs_evidence=False,
                )

        # ── 6c. Pure greetings (deployment fix 002) ──
        # 'hi' / 'hello' alone used to fall through to the <3-word ambiguity
        # branch below and got the confused "not sure what you're asking"
        # clarification. A pure greeting now gets a welcome + capability
        # summary through the same clarification route — NO corpus or
        # warehouse query is executed. The intent stays 'unsupported' (the
        # frozen 7-label taxonomy is unchanged); only the message differs.
        if self._is_greeting(q):
            return Intent(
                label="unsupported",
                scope="all",
                clarification_needed=(
                    "Hello! I'm the Legal Intelligence Assistant. I can search "
                    "specific clauses in named agreements (e.g. 'What is the "
                    "governing law of the Armstrong Flooring intellectual "
                    "property agreement?'), look up extracted fields such as "
                    "dates, parties or payment terms, discover which documents "
                    "contain a clause, or summarize the corpus composition."
                ),
            )

        # ── 6d. Repeat-the-previous-answer requests (fix 008, hosted smoke
        # 2026-10-29) ──
        # 'i couldn't see the pdf, can you repeat the answer again?' is a
        # CONVERSATIONAL request about the previous reply, not a new legal
        # question — previously it fell through to hybrid_qa, retrieved
        # irrelevant chunks, and abstained ('available documents don't
        # contain enough information'), which is wrong agent behavior.
        # Re-display the last assistant answer verbatim via the same
        # clarification route as the greeting fix 002: no retrieval, no
        # SQL, no LLM. Frozen 7-label taxonomy unchanged. Requires a prior
        # assistant answer in history; a first-turn 'repeat' has nothing
        # to repeat and falls through to normal classification.
        if any(m in q for m in self.REPEAT_ANSWER_MARKERS):
            prev_answer = next(
                (msg.get("content", "") for msg in reversed(history)
                 if msg.get("role") == "assistant" and msg.get("content")),
                None,
            )
            if prev_answer:
                # Fix 011 (2026-10-30, Stage E): be HONEST about what the
                # previous reply actually was before repeating it.  Fix 008
                # repeated ANY previous reply verbatim with 'Here is my
                # previous answer' — including error messages and
                # abstentions, which misrepresents the actual situation.
                stripped = (prev_answer or "").strip()
                if stripped.startswith("Error:") or stripped.startswith("I encountered an error"):
                    lead = (
                        "My previous reply was an error message rather than an "
                        "answer, so repeating it verbatim will not help. The "
                        "error was: " + chr(10) + chr(10) + stripped + chr(10) + chr(10) +
                        "Please try rephrasing your question — the underlying "
                        "problem should then be diagnosed rather than repeated."
                    )
                    return Intent(
                        label="unsupported",
                        scope="all",
                        clarification_needed=lead,
                    )
                from .answers import _is_generated_abstention
                if (_is_generated_abstention(stripped)
                        or "do not contain enough information" in stripped.lower()
                        or "don't contain enough information" in stripped.lower()):
                    lead = (
                        "My previous reply was not an answer: I examined the "
                        "available documents and they did not contain enough "
                        "information to answer your question, so there is no "
                        "answer to repeat. Please try rephrasing the question "
                        "or naming a different agreement — I can also show "
                        "which documents the corpus contains."
                    )
                    return Intent(
                        label="unsupported",
                        scope="all",
                        clarification_needed=lead,
                    )
                lead = "Here is my previous answer: " + chr(10) + chr(10)
                return Intent(
                    label="unsupported",
                    scope="all",
                    clarification_needed=lead + prev_answer,
                )

        # ── 7. Ambiguous queries ──
        if len(q.split()) < 3 or any(p in q for p in self.AMBIGUOUS_MARKERS):
            return Intent(
                label="unsupported",
                scope="all",
                clarification_needed=(
                    "I'm not sure what you're asking. Could you rephrase your question? "
                    "I can search for specific clauses, look up extracted fields, "
                    "or summarize the corpus composition."
                ),
            )

        # ── 8. Default: hybrid_qa (legal question requiring retrieval) ──
        # D2 completion (2026-10-29): a hybrid question that names a document
        # ('... under the Upjohn manufacturing and supply agreement') carries
        # the resolved name so the hybrid_qa node can scope retrieval to that
        # document before fusion — evidence must not leak from other
        # documents. Generic phrasings ('this contract', no name) correctly
        # resolve to no document and stay corpus-wide.
        doc = self._extract_single_doc(query)
        return Intent(
            label="hybrid_qa",
            scope="specific_docs" if doc else "all",
            target_doc_names=[doc] if doc else [],
            needs_evidence=True,
        )

    # ── Helper methods ──

    # Pure greetings (deployment fix 002): matched after punctuation is
    # stripped — 'hi!', 'Hello there.' etc. Never matches longer questions
    # ('hi, what is the governing law of X?' still routes by content).
    GREETINGS: frozenset[str] = frozenset({
        "hi", "hello", "hey", "hiya", "howdy", "greetings", "yo",
        "hi there", "hello there", "hey there", "good morning",
        "good afternoon", "good evening", "good day",
    })

    @classmethod
    def _is_greeting(cls, q: str) -> bool:
        """True only for a PURE greeting (punctuation stripped, nothing else)."""
        cleaned = re.sub(r"[!?.,;:\"']", " ", q).strip()
        return bool(cleaned) and cleaned in cls.GREETINGS

    # Repeat-previous-answer phrasings (fix 008, hosted smoke 2026-10-29).
    # Matched as substrings against the lowercased query; anchored on the
    # word 'answer'/'that' so genuine legal questions are never hijacked
    # ('What is the governing law of X?' never matches; 'repeat the
    # termination clause obligations' also does not — it names content,
    # not the previous ANSWER).
    REPEAT_ANSWER_MARKERS: list[str] = [
        "repeat the answer", "repeat your answer", "repeat that answer",
        "repeat the previous answer", "repeat the previous response",
        "repeat your previous", "repeat it again",
        "show the answer again", "show that again", "say that again",
        "give me the answer again", "the answer again", "your answer again",
        "answer that again", "post the answer again",
        # Fix 011 (2026-10-30, Stage E): hosted phrasings that escaped fix
        # 008 ('no i couldnt see the answer can you repeat again?') and
        # fell through to corpus-wide hybrid_qa -> abstention.  Still
        # anchored on answer/that visibility phrasings so genuine legal
        # questions are never hijacked.
        "repeat again", "can you repeat again", "repeat the last answer",
        "show the last answer", "give me the last answer",
        "last answer again", "show me again", "post that again",
        "share the answer again",
        "couldn't see the answer", "couldnt see the answer",
        "can't see the answer", "cant see the answer",
        "didn't see the answer", "didnt see the answer",
        "couldn't see your answer", "couldnt see your answer",
        "couldn't see it, can you", "couldnt see it, can you",
        "couldn't read the answer", "couldnt read the answer",
        "couldn't see the pdf, can you", "couldnt see the pdf, can you",
        "didn't catch the answer", "didnt catch the answer",
        "missed the answer",
    ]

    @staticmethod
    def _extract_value_filter(original_query: str, q: str) -> tuple[str, str] | None:
        """Detect a value-filter question and extract (field, value fragment).

        e.g. "Which agreements are governed by Delaware?" ->
             ("governing_law", "Delaware").  The fragment is capped at 100
             chars and is bound as a SQL parameter by the lookup node —
        it never becomes part of the SQL text.
        """
        for pattern, field_name in RuleBasedIntentClassifier.VALUE_FILTER_PATTERNS:
            if pattern in q:
                m = re.search(
                    re.escape(pattern) + r"\s+(.+?)\s*[?.]*\s*$",
                    original_query,
                    re.IGNORECASE,
                )
                if not m:
                    return (field_name, "")
                value = m.group(1).strip().strip("'\";,.!?")[:100]
                return (field_name, value)
        return None

    @staticmethod
    def _extract_status_filter(q: str) -> str | None:
        """Detect a missing/uncertain-value question and return the
        status_filter value, or None if no marker matches."""
        for marker, status in RuleBasedIntentClassifier.STATUS_FILTER_MARKERS:
            if marker in q:
                return status
        return None

    # ── Fix 013 helpers ─────────────────────────────────────────────────

    # A field phrase mentioned only as an EXCLUDED grouping/filtering
    # dimension must not activate a structured lookup.  Both the negation
    # instruction and its connector must sit inside the ~60 characters of
    # text immediately before the field phrase.
    _NEGATED_FIELD_PREFIX_RES: tuple = (
        # 'do not group documents by ...', "don't sort by ...",
        # 'never filter on ...'
        re.compile(
            r"\b(?:do\s+not|don'?t|never)\b[^.?!,;]{0,60}?\b(?:by|on|using|against)\s*$"
        ),
        # 'exclude ...', 'excluding ...', 'without ...', 'except ...'
        re.compile(
            r"\b(?:exclude|excluding|except|without)\s+(?:the\s+|any\s+|all\s+)?$"
        ),
    )

    @staticmethod
    def _field_mention_is_negated(q: str, pattern: str) -> bool:
        """True when the field phrase occurs only inside a negated
        grouping/filtering instruction ('do not group by contract title',
        'exclude governing law from the distribution')."""
        idx = q.find(pattern)
        if idx < 0:
            return False
        prefix = q[max(0, idx - 60):idx]
        return any(
            r.search(prefix)
            for r in RuleBasedIntentClassifier._NEGATED_FIELD_PREFIX_RES
        )

    def _is_corpus_query(self, q: str, original_query: str) -> bool:
        """True when the query is a corpus-level metadata question.

        Tier 1: the original exact-substring CORPUS_PATTERNS -- unchanged
        behavior, consulted first.

        Tier 2 (Fix 013): count phrasings with modifiers ('how many
        canonical/total/registered documents'), 'what kind(s)/type(s) of
        documents', document-class/category phrasings and corpus
        aggregate markers.  Tier 2 is consulted only after the guards
        below, so queries naming a document, carrying a value filter
        ('governed by'), a discovery phrasing or a comparison phrasing
        keep their existing routes and are never hijacked.
        """
        if any(p in q for p in self.CORPUS_PATTERNS):
            return True

        # Guards (tier 2 only) -- never hijack another capability's query.
        if self._extract_value_filter(original_query, q) is not None:
            return False
        if any(p in q for p in self.DISCOVERY_PATTERNS):
            return False
        if ("compare" in q or " vs " in q or " versus " in q
                or "difference between" in q):
            return False
        if self.CORPUS_COUNT_VERB_EXCLUSION_RE.search(q):
            return False
        if self._extract_single_doc(original_query):
            return False

        if self.CORPUS_COUNT_RE.search(q) or self.CORPUS_KIND_RE.search(q):
            return True
        if any(p in q for p in self.CORPUS_CLASS_PHRASES):
            return True
        # Aggregate markers additionally require no active (non-negated)
        # field phrase: 'distribution of documents by governing law' keeps
        # its existing field-coverage route instead of a plain summary.
        if any(p in q for p in self.CORPUS_AGGREGATE_MARKERS):
            has_field = any(
                p in q and not self._field_mention_is_negated(q, p)
                for p, _ in self.FIELD_PATTERNS
            )
            if not has_field:
                return True
        return False

    @staticmethod
    def _is_followup(q: str, history: list[dict]) -> bool:
        """Detect whether this query is a follow-up requiring context.

        Fix 011 (2026-10-30, Stage E): extended beyond 'what about X' with
        explain/breakdown phrasings about the PREVIOUS reply ('explain the
        previous answer', 'give me the breakdown') — these need the previous
        turn's context to resolve, and previously fell through to a corpus-
        wide hybrid_qa search on the follow-up text alone.
        """
        if not history:
            return False
        followup_markers = [
            "what about", "how about", "same for", "and for",
            # Fix 011: explain/breakdown phrasings about the previous answer
            "explain the previous", "explain your previous", "explain your answer",
            "explain that answer", "explain the answer", "explain that",
            "break it down", "give me the breakdown", "the breakdown of",
            "breakdown of", "more details on that", "tell me more about that",
            "elaborate on that", "elaborate on your", "elaborate on the previous",
        ]
        return any(m in q for m in followup_markers)

    def _resolve_followup(
        self, original_query: str, q: str, history: list[dict]
    ) -> Intent | None:
        """Resolve a follow-up using conversation history.

        Carries over the previous intent's key attributes when the follow-up
        references a new document or asks 'what about X?'.
        """
        # Find the last user message in history that had a substantive query
        prev_query = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                prev_query = msg.get("content", "")
                break
        if not prev_query:
            return None

        # Re-classify the previous query to get its intent
        prev_intent = self.classify(prev_query, None)

        # Extract the new document name from the follow-up
        new_doc = self._extract_single_doc(original_query)

        if prev_intent.label == "comparison":
            docs = list(prev_intent.target_doc_names) + ([new_doc] if new_doc else [])
            # Keep it as comparison with the same aspect
            return Intent(
                label="comparison",
                scope="specific_docs",
                target_doc_names=docs,
                comparison_aspect=prev_intent.comparison_aspect,
                needs_evidence=True,
            )
        elif prev_intent.label == "structured_lookup":
            if new_doc:
                return Intent(
                    label="structured_lookup",
                    scope="specific_docs",
                    target_doc_names=[new_doc],
                    target_field=prev_intent.target_field,
                    needs_evidence=False,
                )
        elif prev_intent.label == "hybrid_qa":
            # Fix 011 (2026-10-30, Stage E): a follow-up on the previous
            # hybrid question KEEPS the previous document scope (the
            # follow-up phrasing rarely re-names the agreement, so dropping
            # the scope made 'what about the limits?' search the whole
            # corpus and lose the context) and carries a COMBINED search
            # query (previous question + follow-up phrasing) so retrieval
            # sees the full intent — the raw follow-up text alone ('the
            # limits') matches nothing.
            prev_docs = list(prev_intent.target_doc_names or [])
            docs = prev_docs + ([new_doc] if new_doc and new_doc not in prev_docs else [])
            combined = (prev_query.strip() + " " + original_query.strip()).strip()
            return Intent(
                label="hybrid_qa",
                scope="specific_docs" if docs else "all",
                target_doc_names=docs,
                search_query=combined,
                needs_evidence=True,
                # Fix 012 (2026-10-31, Defect D): mark this as a follow-up
                # resolution so the planner can scope it to the document
                # that backed the previous ANSWER (prior_answer_docs) when
                # neither the follow-up nor the previous QUESTION names an
                # extractable agreement (e.g. a table answer like 'Can you
                # show me the Arizona assigned patents?' - the scope lives
                # in the ANSWER's sources, not in the question text).
                resolved_from_followup=True,
            )
        elif prev_intent.label == "clause_discovery":
            return Intent(
                label="clause_discovery",
                scope="all",
                clause_pattern=prev_intent.clause_pattern,
                coverage_requirement="exhaustive",
                needs_evidence=True,
            )

        return None

    @staticmethod
    def _extract_clause_pattern(q: str) -> str | None:
        """Extract a clause pattern from a discovery query."""
        # "find all force majeure clauses across all contracts"
        m = re.search(r"find all (.+?) clauses", q)
        if m:
            return m.group(1).strip()
        m = re.search(r"all (.+?) clauses across", q)
        if m:
            return m.group(1).strip()
        m = re.search(r"all contracts that have (.+)", q)
        if m:
            return m.group(1).strip()
        # "which agreements contain limitation-of-liability provisions?"
        m = re.search(r"(?:which|what)\s+(?:agreements?|contracts?|documents?)\s+contain\s+(.+?)\s+provisions?", q)
        if m:
            return m.group(1).strip()
        # "find indemnification provisions in these selected agreements"
        m = re.search(r"find\s+(.+?)\s+(?:provisions?|clauses?)\s+in", q)
        if m:
            return m.group(1).strip()
        # "find indemnification provisions"
        m = re.search(r"find\s+(.+?)\s+(?:provisions?|clauses?)", q)
        if m:
            return m.group(1).strip()
        # "agreements that contain limitation of liability"
        m = re.search(r"(?:agreements?|contracts?)\s+that\s+contain\s+(.+)", q)
        if m:
            return m.group(1).strip()
        # "contain limitation of liability"
        m = re.search(r"contain\s+(.+?)\s*(?:provisions?|clauses?)?\s*$", q)
        if m:
            return m.group(1).strip()
        # "find force majeure"
        m = re.search(r"find\s+(.+?)\s*$", q)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _extract_discovery_doc_names(query: str) -> list[str]:
        """Extract document names from a clause-discovery query.

        Detects patterns like:
          - "in these selected agreements" → []  (scope is specific but no names)
          - "in the X ... agreement(s)" → [core]  (multi-word titles supported;
            'in the Armstrong Flooring intellectual property agreement' →
            ['ArmstrongFlooring'], see _name_from_phrase)
          - "in the X and Y agreements" → [core(X), core(Y)]
        Returns an empty list when no specific document names are found,
        meaning the search is corpus-wide.
        """
        generic = r"(?:[Aa]greements?|[Cc]ontracts?|[Dd]ocuments?)"

        # "in the X and Y agreements" — two named documents, tried first
        m = re.search(
            r"\bin\s+(?:the\s+|these\s+|a\s+|an\s+)?(.+?)\s+and\s+(.+?)\s+" + generic + r"\b",
            query,
        )
        if m:
            names = []
            for raw in (m.group(1), m.group(2)):
                core = RuleBasedIntentClassifier._name_from_phrase(raw)
                if core:
                    names.append(core)
            if len(names) == 2:
                return names

        # "in the X ... agreement" — single named document (multi-word title)
        m = re.search(
            r"\bin\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+" + generic + r"\b",
            query,
        )
        if m:
            phrase = m.group(1).strip()
            # "in these selected agreements" / "in the selected agreements"
            # — scope is specific but no names extracted; the node will
            # need to resolve from context or return a clarification.
            # Return empty so scope stays "all" (the user said "selected"
            # but didn't name them).
            if re.fullmatch(r"(?:these\s+|the\s+)?selected", phrase, flags=re.IGNORECASE):
                return []
            core = RuleBasedIntentClassifier._name_from_phrase(phrase)
            if core:
                return [core]
        return []

    @staticmethod
    def _strip_leading_article(name: str) -> str:
        """Drop a leading English article ('the/a/an Armstrong ...').

        Baseline defect D9(b)/E01: 'the Armstrong Flooring' failed to match
        any file name because the article was kept in the extracted name.
        """
        return re.sub(r"^(?:the|a|an)\s+", "", name.strip(), flags=re.IGNORECASE)

    @staticmethod
    def _name_core(name: str) -> str | None:
        """Reduce a raw name phrase to its filename-matchable core.

        Keeps the leading run of capitalized words and removes inner
        whitespace so the result works as a case-insensitive substring of
        canonical EDGAR-style file names ('Armstrong Flooring intellectual
        property' → 'ArmstrongFlooring', matching
        'ArmstrongFlooringInc_..._Intellectual Property Agreement.pdf').
        Lowercase descriptors after the run ('intellectual property',
        'manufacturing and supply') are dropped because they are not
        contiguous inside the file names.  Returns None when no capitalized
        token exists — nothing safe to match on.
        """
        m = re.search(r"[A-Z][A-Za-z0-9&.']*(?:\s+[A-Z][A-Za-z0-9&.']*)*", name)
        if not m:
            return None
        return re.sub(r"\s+", "", m.group(0)) or None

    @staticmethod
    def _name_from_phrase(phrase: str) -> str | None:
        """Extract a filename-matchable name core from a raw query phrase.

        Drops 'compare/versus' boilerplate and leading articles, takes the
        tail after a structural connector ('of/in/between') when present,
        then reduces to the leading capitalized-word run via _name_core.
        """
        s = phrase.strip().strip(",.")
        s = re.sub(r"^(?:compare|versus)\s+", "", s, flags=re.IGNORECASE)
        s = RuleBasedIntentClassifier._strip_leading_article(s)
        m = re.search(
            r"\b(?:of|in|between)\s+(?:the\s+|a\s+|an\s+)?(.+)$",
            s, flags=re.IGNORECASE,
        )
        if m:
            s = m.group(1).strip()
        return RuleBasedIntentClassifier._name_core(s)

    # Common leading question/generic words that must never be treated as a
    # document name (baseline: 'Which agreements are governed by ...?'
    # extracted 'Which' as a document name).
    _NON_NAME_WORDS: set[str] = {
        "which", "what", "where", "when", "who", "why", "how",
        "compare", "find", "show", "list", "tell",
        "the", "a", "an", "all", "any", "every", "each",
        "agreement", "contract", "document", "agreements", "contracts",
        "documents",
        # Fix 013: generic instruction/aggregate words that must never be
        # extracted as document names ('Show the distribution ...', 'Include
        # every document class', 'Do not group ...', 'Sort by ...',
        # 'Total document count.', 'Canonical document count.').
        "include", "including", "do", "does", "did", "don't", "dont",
        "please", "exclude", "excluding", "except", "sort", "group",
        "filter", "give", "provide", "summarize", "summarise",
        "total", "canonical", "registered", "distinct", "duplicate",
        "count",
    }

    @staticmethod
    def _extract_doc_names(query: str) -> list[str]:
        """Extract document names from a comparison query.

        Looks for 'between X and Y' / 'of X and Y' / 'X vs Y' /
        'Compare X and Y'.  Each captured phrase is normalized to its
        filename-matchable name core: leading articles ('the Armstrong
        Flooring' → 'ArmstrongFlooring', baseline defect D9(b)/E01) and
        trailing generic words are stripped, then the leading
        capitalized-word run is kept (see _name_from_phrase) so that
        multi-word names never degrade to a trailing fragment ('Services',
        baseline defect D9(c)/D04) that substring-matches unrelated
        documents.  Returns [] when fewer than two usable cores are found
        — the caller then decides between clarification and corpus-wide
        handling, never guessing a document.
        """
        def _pair_to_cores(a: str, b: str) -> list[str]:
            cores = []
            for raw in (a, b):
                raw = RuleBasedIntentClassifier._strip_leading_article(raw)
                for suffix in (" agreement", " contract", " document",
                               " agreements", " contracts", " documents", " and"):
                    raw = raw.removesuffix(suffix)
                core = RuleBasedIntentClassifier._name_from_phrase(raw)
                if core:
                    cores.append(core)
            return cores if len(cores) == 2 else []

        # Pattern: "between X and Y" / "of X and Y"
        m = re.search(
            r"\b(?:between|of)\s+(.+?)\s+and\s+(.+?)(?:\s+(?:agreements?|contracts?|documents?)|$)",
            query, re.IGNORECASE,
        )
        if m:
            names = _pair_to_cores(m.group(1), m.group(2))
            if names:
                return names

        # Pattern: "X vs Y" / "X versus Y"
        m = re.search(
            r"\b(.+?)\s+(?:vs|versus)\s+(.+?)(?:\s+(?:agreements?|contracts?|documents?)|$)",
            query, re.IGNORECASE,
        )
        if m:
            names = _pair_to_cores(m.group(1), m.group(2))
            if names:
                return names

        # Pattern: "Compare X and Y"
        m = re.search(
            r"\bcompare\s+(.+?)\s+and\s+(.+?)(?:\s+(?:agreements?|contracts?|documents?)|$)",
            query, re.IGNORECASE,
        )
        if m:
            names = _pair_to_cores(m.group(1), m.group(2))
            if names:
                return names

        return []

    @staticmethod
    def _extract_single_doc(query: str) -> str | None:
        """Extract a single document name from a query.

        Multi-word document titles are supported (baseline defect
        D9(a)/D01): the name phrase before the trailing generic noun is
        reduced to its filename-matchable core via _name_from_phrase
        ('of the Armstrong Flooring intellectual property agreement' →
        'ArmstrongFlooring'; 'of the West Pharmaceutical Services
        agreement' → 'WestPharmaceuticalServices').  Question and generic
        words ('Which agreements are governed by ...?') are never
        extracted as names.
        """
        generic = r"(?:[Aa]greements?|[Cc]ontracts?|[Dd]ocuments?)"

        # Primary: preposition + optional article + name phrase + generic
        # noun ('of the X ... agreement', 'under the X contract', ...).
        m = re.search(
            r"\b(?:of|under|in|for|from|about)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+" + generic + r"\b",
            query,
        )
        if m:
            name = RuleBasedIntentClassifier._name_from_phrase(m.group(1))
            # Fix 013: apply the same question/instruction-word guard as
            # the secondary and fallback patterns below -- a capitalized
            # instruction word ('Show the distribution of the canonical
            # documents') must never become a target document name.
            if (name and name.lower()
                    not in RuleBasedIntentClassifier._NON_NAME_WORDS):
                return name

        # Secondary: capitalized run directly before the generic noun
        # ('West Pharmaceutical Services agreement'), skipping a leading
        # article ('The Bellring Agreement' → 'Bellring').
        m = re.search(
            r"\b(?:[Tt]he\s+)?([A-Z][A-Za-z0-9&.']*(?:\s+[A-Z][A-Za-z0-9&.']*)*)\s+" + generic + r"\b",
            query,
        )
        if m:
            candidate = re.sub(r"\s+", "", m.group(1))
            if candidate.lower() not in RuleBasedIntentClassifier._NON_NAME_WORDS:
                return candidate

        # Fallback (hosted smoke finding 2026-10-29): prepositional name
        # phrase WITHOUT a trailing generic noun — 'What is the governing
        # law of the Armstrong Flooring intellectual property' (or a
        # truncated '...intellectual proper', or '...of the Armstrong
        # Flooring?').  Both patterns above anchor on the trailing noun, so
        # these phrasings previously extracted NO name and a document-
        # specific field question misrouted to corpus_overview field
        # coverage in the hosted app.  The phrase is reduced by
        # _name_from_phrase to its leading capitalized run, so lowercase
        # descriptors ('intellectual property') are dropped and pure-generic
        # fragments ('of the agreement' → no capitalized run) yield nothing —
        # a name is never guessed.  Runs AFTER the noun-anchored patterns so
        # behavior with the noun present is unchanged.
        m = re.search(
            r"\b(?:of|under|in|for|from|about)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s*[?.!]*\s*$",
            query,
        )
        if m:
            name = RuleBasedIntentClassifier._name_from_phrase(m.group(1))
            if name and name.lower() not in RuleBasedIntentClassifier._NON_NAME_WORDS:
                return name

        # Follow-up: "What about X?" where X is a CAPITALIZED name run.
        # Fix 011 (2026-10-30, Stage E): re.IGNORECASE made [A-Z] match
        # lowercase words, so 'what about the limits?' extracted 'the' as a
        # document name — follow-up turns then carried a junk scope entry.
        # The pattern now requires a genuinely capitalized X and rejects
        # question/generic words (same guard as the fallback above).
        m = re.search(r"(?:what about|how about) ([A-Z][a-zA-Z]+)", query)
        if m and m.group(1).lower() not in RuleBasedIntentClassifier._NON_NAME_WORDS:
            return m.group(1)
        return None

    @staticmethod
    def _extract_comparison_aspect(q: str) -> str | None:
        """Extract what aspect is being compared."""
        # "compare termination clauses between..." → "termination clauses"
        m = re.search(r"compare (.+?) (?:between|of|across|vs)", q)
        if m:
            aspect = m.group(1).strip()
            # Remove leading 'the ' if present
            aspect = re.sub(r"^the ", "", aspect)
            return aspect
        m = re.search(r"difference between.*?in (.+?)", q)
        if m:
            return m.group(1).strip()
        return None


# ── LLM-based classifier (production, with retries) ──────────────────────

class LLMIntentClassifier(IntentClassifier):
    """LLM-based intent classifier with schema validation and bounded retries.

    Calls the LLM endpoint, parses the JSON response, validates against
    the Intent pydantic model, and retries on failure.

    On exhaustion of retries, returns an 'unsupported' intent with a
    clarification message — NEVER falls back to 'hybrid_qa'.
    """

    def __init__(
        self,
        llm_caller: Callable[[str], str],
        max_retries: int = 3,
    ):
        """Args:
            llm_caller: function that takes a prompt and returns the LLM's
                       text response.  In production this wraps the raw HTTP
                       call to the serving endpoint.
            max_retries: maximum number of classification attempts.
        """
        self._llm_caller = llm_caller
        self._max_retries = max_retries

    def classify(self, query: str, conversation_history: list[dict] | None = None) -> Intent:
        prompt = self._build_prompt(query, conversation_history or [])

        for attempt in range(1, self._max_retries + 1):
            try:
                raw = self._llm_caller(prompt)
                intent = self._parse_and_validate(raw)
                logger.info("LLM classifier succeeded on attempt %d: %s", attempt, intent.label)
                return intent
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                logger.warning("LLM classifier attempt %d failed: %s", attempt, e)
                # Adjust prompt for retry
                if attempt < self._max_retries:
                    prompt = self._build_retry_prompt(query, conversation_history or [], str(e))

        # All retries exhausted — safe fallback (NOT hybrid_qa)
        logger.error("LLM classifier failed after %d attempts", self._max_retries)
        return Intent(
            label="unsupported",
            scope="all",
            clarification_needed=(
                "I was unable to classify your request after multiple attempts. "
                "Please rephrase your question. I can search for specific clauses, "
                "look up extracted fields, or summarize the corpus composition."
            ),
        )

    def _build_prompt(self, query: str, history: list[dict]) -> str:
        history_str = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-3:]
        ) if history else "(none)"

        return (
            "You are an intent classifier for a legal document assistant.\n"
            "Classify the user's query into exactly one intent label.\n\n"
            f"Allowed labels: {', '.join(Intent.model_fields['label'].annotation.__args__)}\n"
            f"Allowed fields: {', '.join(ALLOWED_FIELDS)}\n\n"
            "Respond with a JSON object with these fields:\n"
            '  label, scope (all|specific_docs|filtered), target_doc_names (list),\n'
            '  target_field (null or one of the allowed fields),\n'
            '  target_value (null or string: value fragment to filter field values by,\n'
            '               e.g. a jurisdiction for governing_law),\n'
            '  status_filter (null or one of: missing, uncertain, missing_or_uncertain\n'
            '                for structured_lookup data-quality questions),\n'
            '  clause_pattern (null or string),\n'
            '  comparison_aspect (null or string), coverage_requirement (exhaustive|best_effort|top_k),\n'
            '  needs_evidence (bool), clarification_needed (null or string)\n\n'
            f"Conversation history:\n{history_str}\n\n"
            f"Query: {query}\n"
        )

    def _build_retry_prompt(self, query: str, history: list[dict], error: str) -> str:
        base = self._build_prompt(query, history)
        return (
            base
            + f"\n\nYour previous response was invalid: {error}\n"
            "Please respond with a valid JSON object only."
        )

    @staticmethod
    def _parse_and_validate(raw_response: str) -> Intent:
        """Parse raw LLM JSON response and validate against Intent schema."""
        # Strip markdown code fences if present
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        data = json.loads(text)
        return Intent(**data)


# ── Query planner node for LangGraph ──────────────────────────────────────

# ── Fix 010 (2026-10-29, hosted smoke finding): pending-action ──────────────
# acceptance resolution.  When the app passes back a pending_action from a
# previous turn (a recommended comparison-candidate shortlist) and the user
# accepts it ('sure compare them then'), the planner resolves the EXACT
# previously recommended documents instead of re-classifying the acceptance
# phrasing — which contains no document names and would otherwise fall to
# the hybrid_qa default, lose the candidate context, and ask the user to
# repeat details the assistant already established.

# Acceptance phrasings.  Anchored on compare/go-ahead/do-that/yes words;
# only consulted when a pending action exists AND the turn is short (<= 10
# words), so a long genuine legal question is never hijacked.
PENDING_ACCEPTANCE_MARKERS: list[str] = [
    "sure compare", "yes compare", "ok compare", "okay compare",
    "compare them", "compare those", "compare it", "compare then",
    "compare both", "compare the pair", "compare the suggested",
    "compare the recommended", "compare the first", "compare the second",
    "compare the third", "compare the ones you", "compare those agreements",
    "compare those two", "compare them both", "compare them for me",
    "go ahead", "do that", "do the comparison", "run the comparison",
    "please compare", "do it then", "yes do that", "sure do that",
    "do the compare", "make the comparison",
]

# Ordinal words that select WHICH recommended pair the user accepted.
_PAIR_ORDINALS: dict[str, int] = {
    "first": 0, "one": 0, "1st": 0, "second": 1, "two": 1, "2nd": 1,
    "third": 2, "three": 2, "3rd": 2,
}


def resolve_pending_action(query: str, pending: dict) -> Optional[Intent]:
    """Resolve an accepted pending recommendation into a concrete Intent.

    Returns None (fall through to normal classification) when:
      - pending is not a comparison-candidates recommendation, or
      - the query does not match an acceptance phrasing, or
      - the turn is too long to be a conversational acceptance, or
      - the pending structure is malformed (never a guessed intent).

    The returned intent names the EXACT canonical file names carried in the
    pending action; the comparison node re-resolves them against the
    canonical inventory (the pending names are hints, never trusted ids).
    aspect='default' triggers the multi-dimension matrix comparison with
    sensible default dimensions from the extracted contract fields.
    """
    if not isinstance(pending, dict) or pending.get("type") != "comparison_candidates":
        return None
    q = (query or "").lower().strip()
    if not any(m in q for m in PENDING_ACCEPTANCE_MARKERS):
        return None
    if len(q.split()) > 10:
        return None
    pairs = pending.get("pairs") or []
    if not pairs:
        return None
    idx = 0
    for word in q.split():
        if word in _PAIR_ORDINALS:
            idx = _PAIR_ORDINALS[word]
            break
    if idx >= len(pairs):
        idx = 0
    pair = pairs[idx]
    names = [str(d.get("file_name") or "") for d in (pair.get("docs") or [])]
    names = [n for n in names if n]
    if len(names) < 2:
        return None
    return Intent(
        label="comparison",
        scope="specific_docs",
        target_doc_names=names,
        comparison_aspect="default",
        needs_evidence=True,
    )


def make_query_planner_node(classifier: IntentClassifier):
    """Factory: create a LangGraph query-planner node bound to a classifier.

    This allows tests to inject a RuleBasedIntentClassifier and production
    to inject an LLMIntentClassifier.
    """

    def query_planner_node(state: dict) -> dict:
        query = state.get("user_query", "")
        history = state.get("conversation_history", [])
        # Fix 010: an ACCEPTED pending recommendation resolves to the exact
        # previously recommended documents BEFORE normal classification —
        # the acceptance phrasing alone would fall through to hybrid_qa.
        intent = None
        pending = state.get("pending_action")
        if pending:
            intent = resolve_pending_action(query, pending)
        if intent is None:
            try:
                intent = classifier.classify(query, history)
            except Exception as e:
                logger.error("Query planner classifier failed: %s", e)
                intent = Intent(
                    label="unsupported",
                    scope="all",
                    clarification_needed=(
                        "I was unable to process your request. "
                        "Please rephrase your question. I can search for "
                        "specific clauses, look up extracted fields, or "
                        "summarize the corpus composition."
                    ),
                )

        # Fix 012 (2026-10-31, Defect D): follow-up document scope from the
        # previous ANSWER.  When a follow-up resolved to hybrid_qa without
        # naming any agreement (neither the follow-up nor the previous
        # QUESTION carried an extractable document name), the app-provided
        # prior_answer_docs - the canonical documents that backed the
        # previous assistant answer - define the scope.  Exactly one
        # document: scope the search to it (never a silent corpus-wide
        # search).  More than one: ask ONE focused clarification instead of
        # guessing (ambiguity is surfaced, not resolved by assumption).
        if (
            getattr(intent, "resolved_from_followup", False)
            and intent.label == "hybrid_qa"
            and not intent.target_doc_names
        ):
            prior_docs = [d for d in (state.get("prior_answer_docs") or []) if d]
            if len(prior_docs) == 1:
                intent = intent.model_copy(update={
                    "target_doc_names": list(prior_docs),
                    "scope": "specific_docs",
                })
            elif len(prior_docs) > 1:
                try:
                    from .display import display_doc_name
                    pretty_docs = [display_doc_name(d) or d for d in prior_docs]
                except Exception:
                    pretty_docs = list(prior_docs)
                intent = Intent(
                    label="unsupported",
                    scope="all",
                    clarification_needed=(
                        "Your question could refer to more than one agreement "
                        "from the previous answer. Which one do you mean - "
                        + " or ".join(pretty_docs)
                        + "?"
                    ),
                )

        trace_entry = make_trace(
            "query_planner",
            action="classified",
            route=("clarification" if intent.clarification_needed
                   else "plan_validation" if intent.label == "multi_step"
                   else "guard_node" if intent.label == "unsupported"
                   else intent.label),
            reason=intent.clarification_needed or f"label={intent.label}, scope={intent.scope}",
            target_field=intent.target_field,
            clause_pattern=intent.clause_pattern,
            comparison_aspect=intent.comparison_aspect,
        )

        update: dict = {
            "intent": intent,
            "trace": [trace_entry],
        }

        # Carry over target_doc_names from intent
        if intent.target_doc_names:
            update["target_doc_names"] = intent.target_doc_names

        # Generate a plan for multi_step intents (stub plan for foundation)
        if intent.label == "multi_step":
            update["plan"] = _generate_stub_plan(intent, query)

        return update

    return query_planner_node


def _split_multi_step_query(query: str) -> list[str]:
    """Split a multi-step query into its clause sub-questions.

    Splits on the classifier's multi-step markers ('and tell me',
    'and which', 'and then', 'and also', 'first...then') and returns the
    non-empty clause texts stripped of joining punctuation.  A query
    without markers returns a single element (the whole query), so plans
    are only derived for genuinely multi-step phrasings.
    """
    parts = re.split(
        r",?\s*\band also\b|\band then\b|\band tell me\b|\band which\b|\bfirst\.\.\.then\b",
        query,
        flags=re.IGNORECASE,
    )
    clauses = []
    for part in parts:
        part = part.strip().strip(",.")
        if part:
            clauses.append(part)
    return clauses


def _describe_step(clause_intent: Intent, clause: str) -> str:
    """Human-readable description for a plan step (shown in traces)."""
    label = clause_intent.label
    if label == "structured_lookup":
        parts = [clause_intent.target_field or "field"]
        if clause_intent.target_doc_names:
            parts.append("for " + ", ".join(clause_intent.target_doc_names))
        if clause_intent.target_value:
            parts.append(f"filtered by '{clause_intent.target_value}'")
        if clause_intent.status_filter:
            parts.append(f"status {clause_intent.status_filter}")
        return "Look up " + " ".join(parts)
    if label == "clause_discovery":
        return f"Discover '{clause_intent.clause_pattern or 'unspecified'}' clauses"
    if label == "comparison":
        return (
            f"Compare {clause_intent.comparison_aspect or 'unspecified'} across "
            + ", ".join(clause_intent.target_doc_names)
        )
    if label == "corpus_overview":
        return "Summarize corpus metadata"
    return f"Answer: {clause[:80]}"


def _generate_stub_plan(intent: Intent, query: str | None = None) -> list[dict]:
    """Generate a multi-step plan from the intent.

    Two sources, in order:
      1. Comparison intents with an aspect and named documents keep the
         existing two-step lookup + comparison plan (unchanged behavior).
      2. Otherwise, when the original query is available, split it on
         multi-step markers and classify each clause with the rule-based
         classifier; each capability clause becomes one plan step
         (baseline defect D9(d)/G01: 'What is the governing law of X, and
         which agreements are governed by Y?' → structured_lookup(lookup)
         + structured_lookup(value_filter)).  A query without markers
         yields no steps, so plan validation rejects it rather than
         silently converting to hybrid_qa.
    """
    steps: list[dict] = []
    step_id = 1

    if intent.comparison_aspect and intent.target_doc_names:
        # Step 1: Structured lookup for the comparison aspect
        field = _guess_field_from_aspect(intent.comparison_aspect)
        steps.append({
            "step_id": step_id,
            "node": "structured_lookup",
            "description": f"Look up {field or intent.comparison_aspect} for each document",
            "target_doc_names": intent.target_doc_names,
            "target_field": field,
        })
        step_id += 1

        # Step 2: Comparison
        steps.append({
            "step_id": step_id,
            "node": "comparison",
            "description": f"Compare {intent.comparison_aspect} across documents",
            "target_doc_names": intent.target_doc_names,
            "comparison_aspect": intent.comparison_aspect,
        })
        step_id += 1
        return steps

    # Clause-based plan for non-comparison multi-step queries.
    if query:
        rule = RuleBasedIntentClassifier()
        for clause in _split_multi_step_query(query):
            if len(steps) >= MAX_PLAN_STEPS:
                break
            clause_intent = rule.classify(clause)
            # Plan steps are capability steps only — never hybrid_qa (that
            # would be the silent fallback plan_validation must reject),
            # never nested multi_step, never unsupported/clarification.
            if (clause_intent.label in ("unsupported", "multi_step", "hybrid_qa")
                    or clause_intent.clarification_needed
                    or clause_intent.label not in ALLOWED_NODES):
                continue
            steps.append({
                "step_id": step_id,
                "node": clause_intent.label,
                "description": _describe_step(clause_intent, clause),
                "target_doc_names": clause_intent.target_doc_names,
                "target_field": clause_intent.target_field,
                "target_value": clause_intent.target_value,
                "status_filter": clause_intent.status_filter,
                "clause_pattern": clause_intent.clause_pattern,
                "comparison_aspect": clause_intent.comparison_aspect,
            })
            step_id += 1

    # If no steps could be generated, return empty list.
    # plan_validation will reject it rather than silently converting to hybrid_qa.
    return steps


def _guess_field_from_aspect(aspect: str) -> str | None:
    """Guess the allowlisted field name from a comparison aspect string."""
    aspect_lower = aspect.lower()
    for pattern, field_name in RuleBasedIntentClassifier.FIELD_PATTERNS:
        if pattern in aspect_lower:
            return field_name
    return None
