# 🚀 Legal Agent Lakebase - Deployment Guide

**App Name:** `legal-agent-lakebase`  
**Service Principal:** `b56f8f6a-9514-4fa0-9b63-5547e21a7f52`  
**URL:** https://legal-agent-lakebase-7474653882108611.aws.databricksapps.com

---

## ✅ **Pre-Deployment Checklist**

### 1. **Source Files Status**
- ✅ `streamlit_app.py` - Main app with debug mode (7.1 KB)
- ✅ `lakebase_retriever.py` - Fixed token handling (7.9 KB)
- ✅ `chat_history.py` - Unity Catalog integration (4.8 KB)
- ✅ `debug_info.py` - **NEW** Diagnostic dashboard (8.8 KB)
- ✅ `requirements.txt` - All dependencies (110 bytes)
- ✅ `app.yaml` - Resource configuration (478 bytes)

### 2. **Code Fixes Applied**
- ✅ Removed `w.dbutils.secrets` (not available in Apps)
- ✅ Removed `w.config.token` fallback (unreliable in Apps)
- ✅ Using `os.environ.get("DATABRICKS_TOKEN")` correctly
- ✅ Added comprehensive debug dashboard

---

## ⚙️ **Resources Configuration (In UI)**

### Required Resources:

| Resource Type | Configuration | Status |
|---------------|---------------|--------|
| **Secret** | Scope: `lakebase`<br>Key: `db_password`<br>Permission: `Can read` | ✅ |
| **Database** | Project: `legal-intelligence-db`<br>Database: `databricks_postgres`<br>Permission: `Can connect` | ✅ |
| **SQL Warehouse** | Serverless Starter Warehouse<br>Permission: `Can use` | ✅ |
| **UC Table** | `workspace.legal_assistant_lakebase.chat_history`<br>Permission: `Can modify` ⚠️ | **Fix needed** |

### ⚠️ **CRITICAL FIX NEEDED:**
The UC Table permission must be **Can modify** (not just "Can select") because the app needs to INSERT chat history.

**How to fix:**
1. Click the pencil/edit icon on the UC table row
2. Change permission to `Can modify`
3. Click **Save**

---

## 🔑 **User Authorization Scopes**

All correct - no changes needed:

- ✅ `catalog.catalogs:read` - Unity Catalog access
- ✅ `catalog.schemas:read` - Schema browsing
- ✅ `postgres` - **Lakebase database connection**
- ✅ `model-serving` - **LLM and embedding endpoints**
- ✅ `catalog.tables:read` - Table metadata

---

## 🚀 **Deployment Steps**

1. ✅ **Fix UC table permission** (see above)
2. ✅ **Click "Save"** in Resources section
3. ✅ **Navigate to "Deployments" tab** (left sidebar)
4. ✅ **Click "Deploy"** button
5. ⏳ **Wait 2-3 minutes** for:
   - Python dependencies installation (pg8000, streamlit, databricks-sdk, scramp, requests)
   - App initialization
   - Lakebase connection establishment
6. 🌐 **Open app URL** (see top of document)
7. 🔧 **Enable Debug Mode** (see below)

---

## 🔍 **Debug Dashboard - How to Use**

After deployment, you can access the diagnostic dashboard to troubleshoot issues:

### **Steps:**
1. Open the app: https://legal-agent-lakebase-7474653882108611.aws.databricksapps.com
2. In the left sidebar, find **"🔧 Debug Mode"**
3. Check the box: **"Show Diagnostic Dashboard"**
4. The dashboard will display:

### **What the Debug Dashboard Shows:**

#### 1️⃣ **Environment Variables**
- `DATABRICKS_TOKEN` availability (masked)
- `DATABRICKS_HOST`
- Python paths
- Home directory

#### 2️⃣ **Python Dependencies**
- Installed packages and versions
- Missing packages (if any)

#### 3️⃣ **Databricks SDK Connection**
- WorkspaceClient initialization status
- Current user identity
- Token availability

#### 4️⃣ **Secret Access Test**
- Tests `lakebase/db_password` secret access
- Shows if secret is readable

#### 5️⃣ **Lakebase Postgres Connection**
- Tests connection to Lakebase database
- Counts document chunks (should show 2,720)

#### 6️⃣ **SQL Warehouse Access**
- Lists accessible warehouses
- Tests chat_history table access

#### 7️⃣ **LLM Endpoint Access**
- Tests embedding endpoint (`databricks-gte-large-en`)
- Verifies token works for model serving

#### 8️⃣ **File System & Imports**
- Shows current working directory
- Tests if `lakebase_retriever.py` and `chat_history.py` can be imported

---

## 🧪 **Expected Behavior After Deployment**

### **✅ Success Indicators:**
- App loads without 502 error
- Main chat interface appears
- "Initializing Lakebase connection..." completes
- Can type a question and get a response

### **🔴 If App Crashes:**
1. Enable **Debug Mode** (sidebar checkbox)
2. Look for **red ❌ icons** in the diagnostic dashboard
3. Take a screenshot of the **exact error messages**
4. Share with me for troubleshooting

### **Common Issues & Solutions:**

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `DATABRICKS_TOKEN not available` | Token not injected by app runtime | Check app service principal permissions |
| `Cannot access secret` | Secret permission not set | Verify lakebase secret resource is configured |
| `Lakebase connection failed` | Database permission issue | Verify database resource has "Can connect" |
| `ModuleNotFoundError: pg8000` | Dependencies not installed | Wait for full deployment (2-3 min) |
| `Cannot query chat_history` | Table permission insufficient | Change to "Can modify" |

---

## 📊 **Architecture Summary**

```
User Question → Streamlit UI (streamlit_app.py)
                    ↓
             lakebase_retriever.py
                    ↓
   ┌────────────────┴────────────────┐
   ↓                                 ↓
Lakebase Postgres              LLM Endpoints
(pgvector search)          (embedding + generation)
- Document chunks (2,720)   - databricks-gte-large-en (embeddings)
- legal-intelligence-db     - databricks-meta-llama-3-3-70b-instruct
   ↓                                 ↓
Retrieved context            Generated response
                    ↓
            chat_history.py
                    ↓
    Unity Catalog Delta Table
    (workspace.legal_assistant_lakebase.chat_history)
```

---

## 📝 **Final Checklist Before Deploy**

- ✅ All source files updated and saved
- ⚠️ **UC table permission changed to "Can modify"**
- ✅ All resources configured in UI
- ✅ User authorization scopes correct
- ✅ Debug dashboard available for troubleshooting
- ✅ Service principal has secret access

---

## 🎯 **Next Steps After Successful Deployment**

1. **Test basic functionality:**
   - Ask: "What are the key provisions in employment contracts?"
   - Verify response comes from Lakebase chunks
   
2. **Test chat history:**
   - Enable "Save conversation history"
   - Ask a few questions
   - Click "Load Previous Conversation"
   - Verify messages persist

3. **Monitor performance:**
   - Check response times (should be < 5 seconds)
   - Verify correct document chunks retrieved
   - Ensure LLM responses are relevant

4. **Production readiness:**
   - Add more legal documents to volume
   - Run daily ingestion pipeline
   - Monitor Lakebase database size
   - Set up cost alerts ($47-140/month expected)

---

**Created:** 2026-06-25  
**Last Updated:** 2026-06-25  
**Author:** Genie Code Assistant
