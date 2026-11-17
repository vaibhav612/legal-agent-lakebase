# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Silver Layer - Parse and Chunk (Lakebase Pipeline)
# MAGIC %md
# MAGIC # 🥈 Silver Layer - Parse PDFs & Semantic Chunking (Lakebase Pipeline)
# MAGIC
# MAGIC ## Purpose
# MAGIC Transform raw PDFs into semantically meaningful text chunks optimized for Lakebase vector search.
# MAGIC
# MAGIC ## Processing Steps
# MAGIC 1. **Parse PDFs**: Extract text with `ai_parse_document()` v2.0
# MAGIC 2. **Semantic Chunking**: Split into 400-500 token chunks
# MAGIC 3. **Add Overlap**: 50-100 tokens between chunks for context
# MAGIC 4. **Metadata Tracking**: doc_id, chunk_id, page references
# MAGIC
# MAGIC ## Input
# MAGIC * Table: `workspace.legal_assistant_lakebase.legal_docs_bronze`
# MAGIC * Source: Binary PDF content
# MAGIC
# MAGIC ## Output
# MAGIC * Table: `workspace.legal_assistant_lakebase.legal_docs_chunks`
# MAGIC * Format: Text chunks with rich metadata
# MAGIC
# MAGIC ## Execution
# MAGIC * Run in Workflow Job (Task 2, after Bronze)
# MAGIC * Processes only new documents since last run

# COMMAND ----------

# DBTITLE 1,Define Semantic Chunking UDF
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType
import re

print("🛠️  Registering semantic chunking UDF...")

# Schema for chunked output
chunk_schema = ArrayType(
    StructType([
        StructField("chunk_text", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("char_count", IntegerType(), False),
        StructField("token_estimate", IntegerType(), False),
        StructField("page_references", StringType(), True)
    ])
)

@udf(returnType=chunk_schema)
def semantic_chunk_text(full_text: str, pages_info: str):
    """
    Enterprise semantic chunking with overlap.
    Target: 400-500 tokens (~1800 chars) per chunk
    Overlap: 50-100 tokens (~300 chars) between chunks
    Boundaries: Paragraphs and sentences preserved
    """
    if not full_text or len(full_text.strip()) < 50:
        return []
    
    TARGET_TOKENS = 450
    CHARS_PER_TOKEN = 4
    TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN  # ~1800
    OVERLAP_TOKENS = 75
    OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN  # ~300
    
    # Split on paragraph boundaries
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
    if not paragraphs:
        paragraphs = [full_text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0
    
    for paragraph in paragraphs:
        para_length = len(paragraph)
        
        if current_length + para_length <= TARGET_CHARS:
            current_chunk.append(paragraph)
            current_length += para_length + 2
        else:
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "chunk_text": chunk_text,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text),
                    "token_estimate": len(chunk_text) // CHARS_PER_TOKEN,
                    "page_references": pages_info
                })
                chunk_index += 1
                
                # Add overlap from previous chunk
                overlap_text = chunk_text[-OVERLAP_CHARS:] if len(chunk_text) > OVERLAP_CHARS else chunk_text
                current_chunk = [overlap_text, paragraph]
                current_length = len(overlap_text) + para_length + 2
            else:
                # Single large paragraph - split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = []
                temp_length = 0
                
                for sentence in sentences:
                    if temp_length + len(sentence) <= TARGET_CHARS:
                        temp_chunk.append(sentence)
                        temp_length += len(sentence) + 1
                    else:
                        if temp_chunk:
                            chunk_text = " ".join(temp_chunk)
                            chunks.append({
                                "chunk_text": chunk_text,
                                "chunk_index": chunk_index,
                                "char_count": len(chunk_text),
                                "token_estimate": len(chunk_text) // CHARS_PER_TOKEN,
                                "page_references": pages_info
                            })
                            chunk_index += 1
                        temp_chunk = [sentence]
                        temp_length = len(sentence)
                
                current_chunk = temp_chunk
                current_length = temp_length
    
    # Last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append({
            "chunk_text": chunk_text,
            "chunk_index": chunk_index,
            "char_count": len(chunk_text),
            "token_estimate": len(chunk_text) // CHARS_PER_TOKEN,
            "page_references": pages_info
        })
    
    return chunks

print("✅ Semantic chunking UDF registered")
print("   Target: 400-500 tokens/chunk, 50-100 token overlap")

# COMMAND ----------

# DBTITLE 1,Create Silver chunks table
# MAGIC %sql
# MAGIC -- Create Silver table in NEW schema
# MAGIC CREATE TABLE IF NOT EXISTS workspace.legal_assistant_lakebase.legal_docs_chunks (
# MAGIC   chunk_id STRING NOT NULL COMMENT 'Unique chunk ID (doc_id + chunk_index)',
# MAGIC   doc_id STRING NOT NULL COMMENT 'Parent document ID from Bronze',
# MAGIC   source_file STRING COMMENT 'Original filename',
# MAGIC   file_path STRING COMMENT 'Full path to source file',
# MAGIC   chunk_text STRING COMMENT 'Semantic text chunk (400-500 tokens)',
# MAGIC   chunk_index INT COMMENT 'Chunk order within document (0, 1, 2...)',
# MAGIC   total_chunks INT COMMENT 'Total chunks in this document',
# MAGIC   char_count INT COMMENT 'Character count',
# MAGIC   token_estimate INT COMMENT 'Estimated token count',
# MAGIC   page_references STRING COMMENT 'Page numbers spanned',
# MAGIC   parsed_at TIMESTAMP COMMENT 'Parsing timestamp'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Silver: Parsed and semantically chunked legal documents (Lakebase pipeline)';
# MAGIC
# MAGIC SELECT 'Silver table ready' AS status;

# COMMAND ----------

# DBTITLE 1,Parse and Chunk PDFs
from pyspark.sql.functions import sha2, concat_ws, col, current_timestamp, explode, monotonically_increasing_id

print("🔍 Starting Silver layer processing (Lakebase pipeline)...\n")

# Get list of already-processed documents (Serverless-compatible)
processed_docs_df = spark.table("workspace.legal_assistant_lakebase.legal_docs_chunks") \
    .select("doc_id").distinct()

# Convert to list (no RDD operations - Serverless compatible)
processed_docs = [row.doc_id for row in processed_docs_df.collect()]

print(f"✅ Already processed: {len(processed_docs)} documents")

# Read NEW PDFs from Bronze (not yet in Silver)
print("\n📂 Reading new PDFs from Bronze...")

new_pdfs = spark.table("workspace.legal_assistant_lakebase.legal_docs_bronze")

if processed_docs:
    new_pdfs = new_pdfs.filter(~col("doc_id").isin(processed_docs))

new_count = new_pdfs.count()
print(f"   Found {new_count} new PDFs to process")

if new_count == 0:
    print("\n✅ No new documents to process. Exiting.")
    dbutils.notebook.exit("No new documents")

# Parse PDFs with ai_parse_document
print("\n🧠 Parsing PDFs with ai_parse_document()...")

parsed_pdfs = new_pdfs.selectExpr(
    "doc_id",
    "file_name as source_file",
    "file_path",
    "CAST(ai_parse_document(content) AS STRING) as parsed_text",
    "'Pages: Multiple' as page_info"  # Simplified - extract from parser in prod
)

# Apply semantic chunking
print("✂️  Applying semantic chunking...")

chunked_df = parsed_pdfs \
    .withColumn("chunks", semantic_chunk_text(col("parsed_text"), col("page_info"))) \
    .select(
        "doc_id",
        "source_file",
        "file_path",
        explode("chunks").alias("chunk")
    ) \
    .select(
        concat_ws("_", col("doc_id"), col("chunk.chunk_index")).alias("chunk_id"),
        "doc_id",
        "source_file",
        "file_path",
        col("chunk.chunk_text").alias("chunk_text"),
        col("chunk.chunk_index").alias("chunk_index"),
        col("chunk.char_count").alias("char_count"),
        col("chunk.token_estimate").alias("token_estimate"),
        col("chunk.page_references").alias("page_references"),
        current_timestamp().alias("parsed_at")
    )

# Add total_chunks per document
from pyspark.sql.window import Window
from pyspark.sql.functions import max as spark_max

window_spec = Window.partitionBy("doc_id")

final_df = chunked_df.withColumn(
    "total_chunks",
    spark_max("chunk_index").over(window_spec) + 1
)

# Write to Silver table (append mode - incremental)
print("\n💾 Writing chunks to Silver table...")

final_df.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("workspace.legal_assistant_lakebase.legal_docs_chunks")

chunk_count = final_df.count()

print("\n" + "="*60)
print("✅ Silver layer complete (Lakebase pipeline)!")
print("="*60)
print(f"Documents processed: {new_count}")
print(f"Chunks created: {chunk_count}")
print(f"Avg chunks/doc: {chunk_count / new_count:.1f}")

# Show summary
display(spark.sql("""
    SELECT
        COUNT(DISTINCT doc_id) AS total_docs,
        COUNT(*) AS total_chunks,
        ROUND(AVG(total_chunks), 1) AS avg_chunks_per_doc,
        ROUND(AVG(token_estimate), 0) AS avg_tokens_per_chunk,
        MIN(parsed_at) AS earliest_parse,
        MAX(parsed_at) AS latest_parse
    FROM workspace.legal_assistant_lakebase.legal_docs_chunks
"""))

# COMMAND ----------

