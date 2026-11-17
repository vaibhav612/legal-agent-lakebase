# Fix 013 (2026-11-03, corpus-misrouting investigation) regression tests.
#
# Root cause: CORPUS_PATTERNS' narrow substring matching failed to
# recognize corpus phrasings like 'how many canonical documents', 'what
# kind of documents' and 'document_class distribution', routing them to
# hybrid_qa instead of corpus_overview.  FIELD_PATTERNS matched negated
# field mentions ('Do not group by contract title'), activating
# structured_lookup on grouping instructions.  The primary doc-name
# extraction pattern lacked the _NON_NAME_WORDS guard, extracting
# instruction words ('Show') as document names.
#
# These tests pin the fixed behavior AND the preserved routes of every
# neighboring capability (value filters, discovery, comparison, clause
# questions, multi-step) so the broadened corpus detection can never
# hijack another intent.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from workflow.classifier import RuleBasedIntentClassifier
from workflow.state import Intent


@pytest.fixture(scope="module")
def clf() -> RuleBasedIntentClassifier:
    return RuleBasedIntentClassifier()


# The three user-reported misrouting cases, verbatim.
CASE_A = (
    "How many canonical documents do we have in the corpus, and how are "
    "they distributed across document types?"
)
CASE_B = (
    "How many total documents are registered in the corpus, and how many "
    "of those are canonical (excluding duplicates)? Show the distribution "
    "of the canonical documents by the document_class field. Include every "
    "document class, its count, and the total. Do not group documents by "
    "contract title or contract subtype."
)
CASE_C = "What kind of documents are there in corpus?"


class TestReportedCases:
    """The three reproduced symptoms must all route to corpus_overview."""

    @pytest.mark.parametrize("query", [CASE_A, CASE_B, CASE_C], ids=["A", "B", "C"])
    def test_routes_to_corpus_overview(self, clf, query):
        intent = clf.classify(query, None)
        assert intent.label == "corpus_overview"
        assert intent.scope == "all"
        assert intent.target_field is None
        assert intent.target_doc_names == []
        assert not intent.clarification_needed

    def test_stable_across_sequenced_conversation(self, clf):
        history = [
            {"role": "user", "content": CASE_A},
            {"role": "assistant", "content": "previous answer"},
        ]
        intent = clf.classify(CASE_B, history)
        assert intent.label == "corpus_overview"
        assert intent.scope == "all"
        assert intent.target_field is None
        assert intent.target_doc_names == []

        history += [
            {"role": "user", "content": CASE_B},
            {"role": "assistant", "content": "previous answer"},
        ]
        intent = clf.classify(CASE_C, history)
        assert intent.label == "corpus_overview"


class TestBroadenedCorpusPhrasings:
    """Corpus-question variations that tier-2 detection must cover."""

    @pytest.mark.parametrize(
        "query",
        [
            "How many total documents are registered in the corpus?",
            "How many canonical documents are there?",
            "How many registered documents?",
            "How many distinct documents are in the corpus?",
            "What kinds of documents are in the corpus?",
            "What document categories exist?",
            "What document classes are in the corpus?",
            "List all document classes.",
            "Show the distribution of documents by document_class.",
            "Show document class distribution.",
            "How are documents classified?",
            "Total document count.",
            "Canonical document count.",
            "How many duplicate documents are registered?",
            "How many contracts or agreements are in the corpus?",
        ],
    )
    def test_corpus_phrasing(self, clf, query):
        intent = clf.classify(query, None)
        assert intent.label == "corpus_overview", (
            f"{query!r} routed to {intent.label}"
        )


class TestPreservedRoutes:
    """Every neighboring capability's route must be unchanged.

    The tier-2 corpus guards must never hijack: original exact corpus
    phrasings, document-specific field lookups, value filters, discovery,
    clause-shaped questions, hybrid questions, multi-step questions,
    comparisons and comparison-candidate discovery.
    """

    @pytest.mark.parametrize(
        "query,expected_label",
        [
            # Tier 1: original corpus phrasings keep their route.
            ("How many documents are in the corpus?", "corpus_overview"),
            # Document-specific field lookups.
            ("What is the governing law of the Armstrong agreement?",
             "structured_lookup"),
            ("What is the effective date of the Upjohn agreement?",
             "structured_lookup"),
            # Value filters ('governed by' guard).
            ("Which agreements are governed by Delaware law?",
             "structured_lookup"),
            ("How many agreements are governed by Delaware law?",
             "structured_lookup"),
            # Field-filtered corpus coverage (pre-fix behavior preserved).
            ("How many contracts have a termination for convenience clause?",
             "corpus_overview"),
            # Clause discovery.
            ("Find indemnification provisions in the Armstrong Flooring "
             "intellectual property agreement.", "clause_discovery"),
            # Clause-shaped 'how many ... contain' stays hybrid (count-verb
            # exclusion guard).
            ("How many agreements contain indemnification provisions?",
             "hybrid_qa"),
            # Hybrid retrieval questions.
            ("What are the termination conditions in the Upjohn agreement?",
             "hybrid_qa"),
            # Multi-step structured questions.
            ("What is the governing law of the Armstrong Flooring "
             "intellectual property agreement, and which agreements are "
             "governed by Delaware law?", "multi_step"),
            # Comparison (single-doc limitation preserved).
            ("Compare the Armstrong agreement with the Cerence agreement.",
             "hybrid_qa"),
            # Comparison-candidate discovery (fix 010).
            ("What documents can be compared?", "comparison"),
            # Pre-fix filtered field-coverage behavior preserved.
            ("What is the contract title of this document?", "corpus_overview"),
        ],
    )
    def test_route_preserved(self, clf, query, expected_label):
        intent = clf.classify(query, None)
        assert intent.label == expected_label, (
            f"{query!r} expected {expected_label}, got {intent.label}"
        )


class TestNegatedFieldMentions:
    """Negated grouping/filtering instructions must not activate a field
    lookup (previously 'Do not group by contract title' became a
    structured_lookup on contract_title)."""

    @pytest.mark.parametrize(
        "query",
        [
            "Do not group by contract title.",
            "Do not group documents by contract title.",
            "Do not filter by governing law.",
            "Do not sort by effective date.",
            "Exclude contract subtype from the distribution.",
            "Exclude governing law from the distribution.",
        ],
    )
    def test_negated_not_structured_lookup(self, clf, query):
        intent = clf.classify(query, None)
        assert intent.label != "structured_lookup", (
            f"{query!r} misrouted to structured_lookup"
        )
        assert intent.target_field is None

    def test_positive_field_question_keeps_field_route(self, clf):
        # The negation guard must not swallow a genuine field question.
        for query, field in [
            ("What is the contract title?", "contract_title"),
            ("What is the governing law?", "governing_law"),
        ]:
            intent = clf.classify(query, None)
            assert intent.target_field == field


class TestDocNameExtractionGuard:
    """The primary extraction pattern must apply _NON_NAME_WORDS so
    instruction/aggregate words are never extracted as document names."""

    @pytest.mark.parametrize(
        "query",
        [
            "Show the distribution of canonical documents.",
            "Include every document class.",
            "Do not group documents by contract title.",
            "List all document classes.",
            "Sort the results by contract title.",
            "Total document count.",
            "How many total documents are registered?",
        ],
    )
    def test_instruction_words_not_names(self, clf, query):
        assert clf._extract_single_doc(query) is None, (
            f"{query!r} extracted a spurious document name"
        )

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("What is the governing law of the Armstrong agreement?",
             "Armstrong"),
            ("What is the effective date of the Upjohn manufacturing and "
             "supply agreement?", "Upjohn"),
            ("What is the governing law of the Armstrong Flooring "
             "intellectual property agreement?", "ArmstrongFlooring"),
        ],
    )
    def test_genuine_names_still_extracted(self, clf, query, expected):
        assert clf._extract_single_doc(query) == expected

    def test_comparison_pairs_still_extracted(self, clf):
        names = clf._extract_doc_names(
            "Compare the Armstrong Flooring intellectual property agreement "
            "and the Cerence agreement."
        )
        assert names == ["ArmstrongFlooring", "Cerence"]


class TestTaxonomyAndSchemaUnchanged:
    """Fix 013 is localized: the 7-label taxonomy, scopes and the Intent
    schema are untouched."""

    def test_intent_labels_frozen(self):
        labels = Intent.model_fields["label"].annotation.__args__
        assert set(labels) == {
            "hybrid_qa", "structured_lookup", "clause_discovery",
            "corpus_overview", "comparison", "multi_step", "unsupported",
        }

    def test_corpus_overview_intent_schema(self, clf):
        intent = clf.classify(CASE_A, None)
        assert isinstance(intent, Intent)
        assert intent.needs_evidence is False
        assert intent.target_value is None
        assert intent.status_filter is None
