# ⚖️ Legal Document Intelligence

**A production-grade, agentic RAG system for legal contract analysis — built end-to-end on Databricks.**

Ask questions in plain English about a corpus of contracts and get **cited, page-referenced answers** — from clause extraction to cross-document comparison. Powered by **hybrid retrieval (Vector + BM25 + RRF)** on **Lakebase Postgres (pgvector)**, orchestrated by a **LangGraph** multi-capability agent, and served through a **Streamlit** app on Databricks Apps.

![status](https://img.shields.io/badge/status-production-blue) ![tests](https://img.shields.io/badge/unit_tests-268%2F268_passing-brightgreen) ![python](https://img.shields.io/badge/python-3.11-blue)

---

## 🖼️ How a Question Flows Through the System

![Legal Document Intelligence — User Query Flow](docs/images/query-flow.png)

![Legal Document Intelligence — End-to-End Architecture & Data Flow](docs/images/data-flow.png)

---

## ✨ What It Does

| Capability | What you can ask |
| --- | --- |
| 🔍 **Legal Q&A** | *"What indemnification obligations do these agreements contain?"* |
| 📋 **Structured Lookup** | *"What is the effective date and governing law of the Cerence agreement?"* |
| 🧭 **Clause Discovery** | *"Find every termination-for-convenience clause across the corpus."* |
| 🆚 **Document Comparison** | *"Compare liability caps between Agreement X and Agreement Y."* |
| 📊 **Corpus Overview** | *"How many NDAs were signed in 2025, and with which parties?"* |

Every answer is **validated against retrieved evidence** and returned with **document + page + passage citations** — no hallucinated legalese.

## 🏗️ Architecture Highlights

- **Agentic pipeline (LangGraph):** intent classification (7 intents) → plan validation → capability routing → hybrid retrieval → evidence validation → grounded generation → cited response
- **Hybrid retrieval:** Vector search (pgvector HNSW) **fused in parallel with BM25 keyword search via Reciprocal Rank Fusion (RRF)** — catching exact legal terms *and* semantic paraphrases
- **Lakebase Postgres** as the vector store (`VECTOR(1024)` + HNSW cosine index) — no separate vector-DB service, **~50–70% infra cost savings** vs. a dedicated endpoint
- **Answer generation:** Meta-Llama 3.3 70B Instruct (Databricks Serverless)
- **Structured field extraction:** 10+ contract fields (parties, dates, governing law, liability caps…) stored in Delta and queryable via allowlisted SQL
- **Fail-fast, read-only runtime:** parameterized, credential-free warehouse SQL; explicit hybrid_v2 retrieval mode; abstention when evidence is insufficient

## 🔄 Ingestion & Data Pipeline

```
Contract PDFs → Auto Loader → ai_parse_document (OCR + structure)
    → Delta Lake (documents · elements · pages)
    → Semantic chunking (400–500 tokens, boundary-aware)
    → databricks-gte-large-en embeddings (1024-dim)
    → Lakebase Postgres · pgvector (document_chunks_v2 + HNSW index)
    → Serves hybrid retrieval for the agent
```

Fully **incremental and idempotent** — re-runs skip already-processed documents; a fresh run takes ~18 min, subsequent runs ~5–8 min.

## 🛠️ Tech Stack

| Layer | Technology |
| --- | --- |
| Platform | Databricks · Unity Catalog · Delta Lake · Lakeflow Jobs |
| Ingestion | Auto Loader · `ai_parse_document` · PySpark |
| Retrieval | Lakebase Postgres + pgvector · BM25 · RRF |
| Embeddings | `databricks-gte-large-en` (1024-dim) |
| Agent | LangGraph · 7-intent router · plan validator |
| LLM | Meta-Llama 3.3 70B Instruct (Serverless) |
| UI | Streamlit (Databricks Apps) |
| Testing & Deploy | pytest (268 tests) · MLflow · Databricks Apps |

## 📂 Repository Layout

```
legal-agent-lakebase/
├── app/                  # Streamlit UI, app_runtime (fail-fast hybrid_v2), retriever
│   ├── workflow/         # LangGraph agent: state, nodes, graph, retrieval, validation
│   └── tests/            # pytest suite (routing, retrieval, capabilities, E2E)
├── lakebase_pipeline/    # Databricks notebooks: bronze ingest → chunk → embed
├── docs/
│   └── images/           # architecture diagrams
└── README.md
```

## 🚀 Getting Started

1. **Ingest** — drop contract PDFs into the Unity Catalog volume; the Auto Loader pipeline parses and chunks them.
2. **Embed & sync** — run the gold notebook to generate embeddings and upsert into Lakebase Postgres.
3. **Serve** — register the LangGraph agent via MLflow / Model Serving.
4. **Run** — deploy the Streamlit app on Databricks Apps and start asking questions.

## 📈 Project Status

- ✅ **268/268 unit tests passing** across routing, retrieval, and all 5 capabilities
- ✅ 15-case live acceptance run — 13 PASS, 1 partial-by-design, 1 behavior recorded
- ✅ 66-case benchmark suite built (owner-reviewed batches in progress)
- ✅ **Deployed to production** on Databricks Apps

## 📜 License

All rights reserved © 2026
