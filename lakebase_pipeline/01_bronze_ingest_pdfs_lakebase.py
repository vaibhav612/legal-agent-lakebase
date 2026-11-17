# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Bronze Layer - PDF Ingestion (Lakebase Pipeline)
# MAGIC %md
# MAGIC # 🥉 Bronze Layer - Incremental PDF Ingestion (Lakebase Pipeline)
# MAGIC
# MAGIC ## Purpose
# MAGIC Ingest PDF files from Unity Catalog volume into Bronze Delta table using Auto Loader.
# MAGIC
# MAGIC ## Features
# MAGIC * **Incremental**: Only processes new/changed files
# MAGIC * **Auto Loader**: Tracks processed files automatically
# MAGIC * **Idempotent**: Safe to rerun multiple times
# MAGIC * **Shared Volume**: Uses `/Volumes/workspace/legal_assistant/legal_docs_volume/`
# MAGIC
# MAGIC ## Input
# MAGIC * Source: `/Volumes/workspace/legal_assistant/legal_docs_volume/`
# MAGIC * Format: PDF files (binary)
# MAGIC
# MAGIC ## Output
# MAGIC * Table: `workspace.legal_assistant_lakebase.legal_docs_bronze`
# MAGIC * Contains: Binary PDFs + metadata
# MAGIC
# MAGIC ## Execution
# MAGIC * Run in Workflow Job (Task 1)
# MAGIC * Schedule: Every 6 hours

# COMMAND ----------

# DBTITLE 1,Create Bronze table
# MAGIC %sql
# MAGIC -- Create Bronze table in NEW schema
# MAGIC CREATE TABLE IF NOT EXISTS workspace.legal_assistant_lakebase.legal_docs_bronze (
# MAGIC   doc_id STRING NOT NULL COMMENT 'Unique document ID (SHA256 of file path)',
# MAGIC   file_path STRING NOT NULL COMMENT 'Source file path in volume',
# MAGIC   file_name STRING NOT NULL COMMENT 'Original filename',
# MAGIC   content BINARY COMMENT 'Raw PDF binary content',
# MAGIC   file_size BIGINT COMMENT 'File size in bytes',
# MAGIC   modification_time TIMESTAMP COMMENT 'Last modified timestamp',
# MAGIC   ingestion_time TIMESTAMP COMMENT 'When ingested into Bronze',
# MAGIC   _metadata STRUCT<
# MAGIC     file_path: STRING,
# MAGIC     file_name: STRING,
# MAGIC     file_size: BIGINT,
# MAGIC     file_modification_time: TIMESTAMP
# MAGIC   > COMMENT 'Auto Loader metadata'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Bronze: Raw PDF documents for Lakebase pipeline (Auto Loader ingestion)';
# MAGIC
# MAGIC SELECT 'Bronze table ready' AS status;

# COMMAND ----------

# DBTITLE 1,Ingest PDFs with Auto Loader
from pyspark.sql.functions import sha2, col, current_timestamp

print("📂 Starting Bronze ingestion (Lakebase pipeline)...")

# SHARED volume path (reusing existing)
volume_path = "/Volumes/workspace/legal_assistant/legal_docs_volume/"
schema_location = "/Volumes/workspace/legal_assistant/legal_docs_volume/_schema/bronze_lakebase"
checkpoint_location = "/Volumes/workspace/legal_assistant/legal_docs_volume/_checkpoint/bronze_lakebase"

# Count existing records in NEW schema
before_count = spark.table("workspace.legal_assistant_lakebase.legal_docs_bronze").count()
print(f"   Existing records: {before_count}")

# Auto Loader: Read new PDFs (STREAMING)
print(f"\n🔍 Scanning for new PDFs in {volume_path}...")

new_pdfs_stream = spark.readStream \
    .format("cloudFiles") \
    .option("cloudFiles.format", "binaryFile") \
    .option("cloudFiles.schemaLocation", schema_location) \
    .load(volume_path) \
    .filter("lower(_metadata.file_name) LIKE '%.pdf'") \
    .select(
        sha2(col("_metadata.file_path").cast("string"), 256).alias("doc_id"),
        col("_metadata.file_path").alias("file_path"),
        col("_metadata.file_name").alias("file_name"),
        col("content"),
        col("_metadata.file_size").alias("file_size"),
        col("_metadata.file_modification_time").alias("modification_time"),
        current_timestamp().alias("ingestion_time"),
        col("_metadata")
    )

# Define merge logic for streaming
def merge_to_bronze(batch_df, batch_id):
    if batch_df.count() > 0:
        print(f"\n💾 Writing batch {batch_id} to Bronze table (MERGE for deduplication)...")
        batch_df.createOrReplaceTempView("new_pdfs_temp")
        
        spark.sql("""
            MERGE INTO workspace.legal_assistant_lakebase.legal_docs_bronze AS target
            USING new_pdfs_temp AS source
            ON target.doc_id = source.doc_id
            WHEN MATCHED AND source.modification_time > target.modification_time THEN
                UPDATE SET
                    target.file_path = source.file_path,
                    target.file_name = source.file_name,
                    target.content = source.content,
                    target.file_size = source.file_size,
                    target.modification_time = source.modification_time,
                    target.ingestion_time = source.ingestion_time,
                    target._metadata = named_struct(
                        'file_path', source._metadata.file_path,
                        'file_name', source._metadata.file_name,
                        'file_size', source._metadata.file_size,
                        'file_modification_time', source._metadata.file_modification_time
                    )
            WHEN NOT MATCHED THEN
                INSERT (doc_id, file_path, file_name, content, file_size, modification_time, ingestion_time, _metadata)
                VALUES (
                    source.doc_id,
                    source.file_path,
                    source.file_name,
                    source.content,
                    source.file_size,
                    source.modification_time,
                    source.ingestion_time,
                    named_struct(
                        'file_path', source._metadata.file_path,
                        'file_name', source._metadata.file_name,
                        'file_size', source._metadata.file_size,
                        'file_modification_time', source._metadata.file_modification_time
                    )
                )
        """)
        print(f"   ✅ Batch {batch_id} merged successfully")
    else:
        print(f"   ℹ️  Batch {batch_id} is empty (no new files)")

# Write stream with foreachBatch and availableNow trigger
print("\n💾 Processing new files with Auto Loader streaming...")

query = new_pdfs_stream.writeStream \
    .foreachBatch(merge_to_bronze) \
    .option("checkpointLocation", checkpoint_location) \
    .trigger(availableNow=True) \
    .start()

print("⏳ Waiting for stream processing to complete...")
query.awaitTermination()
print("✅ Stream processing complete")

# Count after ingestion
after_count = spark.table("workspace.legal_assistant_lakebase.legal_docs_bronze").count()
new_files = after_count - before_count

print("\n" + "="*60)
print("✅ Bronze ingestion complete (Lakebase pipeline)!")
print("="*60)
print(f"New files ingested: {new_files}")
print(f"Total documents in Bronze: {after_count}")

# Show summary
if after_count > 0:
    display(spark.sql("""
        SELECT
            COUNT(*) AS total_docs,
            ROUND(SUM(file_size) / 1024 / 1024, 2) AS total_size_mb,
            MIN(modification_time) AS oldest_doc,
            MAX(modification_time) AS newest_doc,
            MAX(ingestion_time) AS last_ingestion
        FROM workspace.legal_assistant_lakebase.legal_docs_bronze
    """))

# COMMAND ----------

