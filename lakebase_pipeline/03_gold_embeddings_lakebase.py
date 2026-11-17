# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Gold Layer - Embeddings to Lakebase
# MAGIC %md
# MAGIC # 🥇 Gold Layer - Generate Embeddings & Write to Lakebase
# MAGIC
# MAGIC ## Purpose
# MAGIC Generate vector embeddings and store them in Lakebase Postgres with pgvector.
# MAGIC
# MAGIC ## Processing Steps
# MAGIC 1. **Read new chunks**: From `workspace.legal_assistant_lakebase.legal_docs_chunks`
# MAGIC 2. **Check Lakebase**: Which chunks are already embedded
# MAGIC 3. **Generate embeddings**: Using `ai_query('databricks-gte-large-en')`
# MAGIC 4. **Insert to Lakebase**: Using `pg8000` with `ON CONFLICT DO NOTHING`
# MAGIC
# MAGIC ## Input
# MAGIC * Table: `workspace.legal_assistant_lakebase.legal_docs_chunks`
# MAGIC
# MAGIC ## Output
# MAGIC * Lakebase Database: `legal-intelligence-db`
# MAGIC * Table: `document_chunks` (with VECTOR(1024) embeddings)
# MAGIC
# MAGIC ## Execution
# MAGIC * Run in Workflow Job (Task 3, after Silver)
# MAGIC * Uses service principal PAT for auth
# MAGIC * Idempotent: safe to rerun

# COMMAND ----------

# DBTITLE 1,Install pg8000 driver
# MAGIC %pip install pg8000==1.30.5
# MAGIC %pip install -U databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import databricks.sdk.service as svc

services = sorted(
    name for name in dir(svc)
    if not name.startswith("_")
)

print("\n".join(services))

# COMMAND ----------

# DBTITLE 1,Connect to Lakebase (OAuth via SDK)
import itertools
from databricks.sdk import WorkspaceClient
import pg8000.native

# Pre-authenticated on attached compute — no token/profile needed
w = WorkspaceClient()

# 1. Find the Lakebase project by display name
PROJECT_DISPLAY_NAME = "legal-intelligence-db"
project = None
for p in w.postgres.list_projects(page_size=50):
    pname = p.spec.display_name if p.spec else p.name.removeprefix("projects/")
    if pname == PROJECT_DISPLAY_NAME:
        project = p
        break

if not project:
    raise RuntimeError(f"Lakebase project '{PROJECT_DISPLAY_NAME}' not found. "
                       f"Check the project name in the Lakebase UI.")
print(f"✅ Found project: {project.name}")

# 2. Find the production branch
branch = None
for b in w.postgres.list_branches(parent=project.name):
    if "production" in b.name:
        branch = b
        break

if not branch:
    # Fall back to first available branch
    branch = next(iter(w.postgres.list_branches(parent=project.name)))

print(f"✅ Branch: {branch.name}  (state: {branch.status.current_state})")

# 3. Find the read-write endpoint (has the host we need)
endpoint = None
for ep in w.postgres.list_endpoints(parent=branch.name):
    endpoint = ep  # Take the first endpoint (primary read-write)
    break

if not endpoint:
    raise RuntimeError("No endpoint found for this branch. Create one in the Lakebase UI.")

host = endpoint.status.hosts.host
print(f"✅ Endpoint: {endpoint.name}")
print(f"   Host: {host}")
print(f"   State: {endpoint.status.current_state}")

# 4. Get current user (OAuth login role = Databricks email)
username = w.current_user.me().user_name
print(f"✅ Login user: {username}")

# 5. Generate Lakebase-scoped OAuth token (valid 1 hour)
credential = w.postgres.generate_database_credential(endpoint=endpoint.name)
token = credential.token

# 6. Connect using pg8000 with the OAuth token as password
database_name = "databricks_postgres"  # Default Lakebase database name

conn = pg8000.native.Connection(
    host=host,
    port=5432,
    database=database_name,
    user=username,
    password=token,
    ssl_context=True
)

# Quick validation
result = conn.run("SELECT 1 AS test")
assert result[0][0] == 1

print(f"\n✅ Connected to Lakebase using OAuth token (SDK-generated)")
print(f"   Host: {host}")
print(f"   Database: {database_name}")
print(f"   Token expires in ~1 hour — re-run this cell to refresh")

# COMMAND ----------

# DBTITLE 1,Check which chunks need embeddings
from pyspark.sql.functions import col

print("\n🔍 Checking which chunks are already in Lakebase...")


# Get all chunk_ids from Silver
silver_chunks = spark.table("workspace.legal_assistant_lakebase.legal_docs_chunks") \
    .select("chunk_id", "chunk_text", "doc_id", "source_file", "page_references")

total_chunks = silver_chunks.count()
print(f"   Total chunks in Silver: {total_chunks}")

# Get chunk_ids already in Lakebase
try:
    existing_ids = conn.run("""
        SELECT chunk_id FROM document_chunks
    """)
    existing_set = {row[0] for row in existing_ids}
    print(f"   Already in Lakebase: {len(existing_set)}")
except Exception as e:
    print(f"   ⚠️  Could not read from Lakebase: {e}")
    print("   Assuming Lakebase is empty (first run)")
    existing_set = set()

# Filter to only NEW chunks
new_chunks = silver_chunks.filter(~col("chunk_id").isin(list(existing_set))) if existing_set else silver_chunks

new_count = new_chunks.count()
print(f"   🎯 Need to embed: {new_count} new chunks")

if new_count == 0:
    print("\n✅ All chunks already embedded. Exiting.")
    dbutils.notebook.exit("No new chunks to embed")

# COMMAND ----------

# DBTITLE 1,Generate embeddings with ai_query
from pyspark.sql.functions import col

print("\n🧠 Generating embeddings with ai_query...")

# Register temp view for SQL
new_chunks.createOrReplaceTempView("chunks_to_embed")

# Generate embeddings using ai_query
embeddings_df = spark.sql("""
    SELECT
        chunk_id,
        chunk_text,
        doc_id,
        source_file,
        page_references,
        ai_query(
            'databricks-gte-large-en',
            chunk_text,
            returnType => 'array<float>'
        ) as embedding
    FROM chunks_to_embed
""")

# Collect to driver (OK for batches of <10K chunks)
print(f"   Collecting {new_count} embeddings to driver...")
embeddings_list = embeddings_df.collect()

print(f"✅ Generated {len(embeddings_list)} embeddings")
print(f"   Embedding dimension: {len(embeddings_list[0]['embedding'])} (should be 1024)")

# COMMAND ----------

# DBTITLE 1,Insert embeddings into Lakebase
import json

print("\n💾 Inserting embeddings into Lakebase...")

insert_sql = """
    INSERT INTO document_chunks (chunk_id, doc_id, chunk_text, embedding)
    VALUES (:chunk_id, :doc_id, :chunk_text, :embedding::vector)
    ON CONFLICT (chunk_id) DO NOTHING
"""

inserted_count = 0
error_count = 0

for i, row in enumerate(embeddings_list, 1):
    try:
        # Convert embedding to JSON string (pgvector parses JSON arrays)
        embedding_json = json.dumps(row['embedding'])
        
        conn.run(
            insert_sql,
            chunk_id=row['chunk_id'],
            doc_id=row['doc_id'],
            chunk_text=row['chunk_text'],
            embedding=embedding_json
        )
        inserted_count += 1
        
        if i % 100 == 0:
            print(f"   Progress: {i}/{new_count} chunks inserted...")
    
    except Exception as e:
        error_count += 1
        print(f"   ⚠️  Error on chunk {i}: {e}")
        if error_count > 10:  # Abort if too many errors
            print(f"   ❌ Too many errors ({error_count}), aborting!")
            raise

print("\n" + "="*60)
print("✅ Gold layer complete (Lakebase pipeline)!")
print("="*60)
print(f"Chunks inserted: {inserted_count}")
print(f"Errors: {error_count}")
print(f"Skipped (duplicates): {new_count - inserted_count - error_count}")

# Verify final count in Lakebase
final_count = conn.run("SELECT COUNT(*) FROM document_chunks")[0][0]
print(f"\n🎯 Total chunks in Lakebase: {final_count}")

# COMMAND ----------

# DBTITLE 1,Close connection
conn.close()
print("✅ Lakebase connection closed")

# COMMAND ----------

