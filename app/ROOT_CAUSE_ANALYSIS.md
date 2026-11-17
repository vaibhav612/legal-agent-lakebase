# 🔴 Root Cause Analysis: 502 Bad Gateway Error

**Date:** 2026-06-25  
**Issue:** App showing "App Not Available" / 502 error, no UI loading at all

---

## 🎯 **Root Cause Identified**

The app was **crashing during startup, BEFORE Streamlit could render any UI**.

### **The Problem Code:**

**OLD `streamlit_app.py` (Lines 30-32):**
```python
# This runs IMMEDIATELY when app starts!
if 'retriever' not in st.session_state:
    with st.spinner("Initializing Lakebase connection..."):
        st.session_state.retriever = LakebaseRetriever()
```

**Why this caused 502 error:**

1. `LakebaseRetriever()` tries to:
   - Read secret: `lakebase/db_password`
   - Get token: `os.environ.get("DATABRICKS_TOKEN")`
   - Connect to Lakebase Postgres
   - Test database connection

2. If **ANY** of these fail (secret permission, token missing, network issue):
   - App crashes **before** Streamlit renders anything
   - No sidebar, no debug mode, no error messages
   - Just "502 Bad Gateway" from the proxy

3. **Why debug mode couldn't help:**
   - Debug mode checkbox is in the sidebar
   - Sidebar never loads because app crashes first
   - User is stuck with no diagnostics

---

## ✅ **The Fix**

### **NEW `streamlit_app.py` - Key Changes:**

#### 1. **No Initialization at Startup**
```python
# Initialize session state (SAFE - no connections!)
if 'retriever' not in st.session_state:
    st.session_state.retriever = None  # Lazy initialization
if 'retriever_error' not in st.session_state:
    st.session_state.retriever_error = None
```

#### 2. **Debug Mode Always Accessible**
```python
# Sidebar loads FIRST, before any connections
with st.sidebar:
    debug_mode = st.checkbox("Show Diagnostic Dashboard", ...)
```

#### 3. **Lazy Initialization (Only When Needed)**
```python
# Chat input handler
if prompt := st.chat_input("Ask a legal question..."):
    # NOW we initialize (only when user sends first message)
    if st.session_state.retriever is None:
        with st.spinner("🔄 Initializing Lakebase connection..."):
            try:
                from lakebase_retriever import LakebaseRetriever
                st.session_state.retriever = LakebaseRetriever()
            except Exception as e:
                st.session_state.retriever_error = str(e)
                st.error(f"❌ Failed: {e}")
                st.info("💡 Enable Debug Mode to diagnose")
                st.stop()
```

#### 4. **Graceful Error Handling**
```python
# Show error in UI, don't crash
if st.session_state.retriever_error:
    st.error(f"⚠️ Retriever failed: {st.session_state.retriever_error}")
    st.info("💡 Try enabling Debug Mode in sidebar")
    st.stop()  # Stop execution but keep UI visible
```

---

## 🧪 **Expected Behavior After Fix**

### **✅ What Should Happen Now:**

1. **App loads immediately**
   - Streamlit UI appears
   - Sidebar shows with all controls
   - No connection attempts yet

2. **Debug mode is accessible**
   - User can check the box
   - Diagnostic dashboard loads
   - Can see exactly what's broken

3. **Lazy connection on first use**
   - Only connects when user types a message
   - If connection fails, shows error in UI
   - User can enable debug mode to diagnose

4. **No more 502 errors**
   - App never crashes on startup
   - Always shows UI, even if backend is broken
   - Errors are displayed, not hidden

---

## 📊 **Before vs. After**

| Aspect | BEFORE (Broken) | AFTER (Fixed) |
|--------|----------------|---------------|
| **Startup** | Connects to Lakebase immediately | No connections |
| **If connection fails** | 502 error, no UI | UI loads, shows error |
| **Debug access** | Impossible (no UI) | Always available |
| **Error visibility** | Hidden (proxy error) | Displayed in Streamlit |
| **User experience** | Stuck, can't diagnose | Can see exactly what's wrong |

---

## 🚀 **Deployment Instructions**

### **1. Deploy the Fixed Code**
- Code already updated in your workspace
- Go to Deployments tab in Apps UI
- Click "Deploy"
- Wait 2-3 minutes

### **2. After Deployment - Two Paths:**

#### **Path A: App Works**
- UI loads immediately
- You see the sidebar
- Try asking a question
- App connects to Lakebase on first message

#### **Path B: App Still Has Issues**
- UI **should still load** (unlike before!)
- Enable "🔧 Debug Mode" checkbox in sidebar
- Take screenshot of diagnostic dashboard
- Share with me to identify the exact issue

---

## 🎓 **Lesson Learned**

**Golden Rule for Databricks Apps:**

> **Never initialize expensive resources (database connections, API calls, secret reads) during Streamlit's initial run. Always use lazy initialization with proper error handling.**

**Best Practice:**
```python
# ✅ GOOD - Lazy initialization
if user_action_happens:
    if resource is None:
        try:
            resource = initialize_resource()
        except Exception as e:
            show_error_in_ui(e)
            return

# ❌ BAD - Eager initialization
resource = initialize_resource()  # Crashes if fails
```

---

## 📝 **Checklist for Next Deployment**

- ✅ Code updated with lazy initialization
- ✅ Debug mode always accessible
- ✅ Error handling prevents crashes
- ⏳ Deploy via UI (Deployments tab)
- ⏳ Wait 2-3 minutes for deployment
- ⏳ Open app URL
- ⏳ **You should see the UI this time!**
- ⏳ If issues persist, enable debug mode and share results

---

**Status:** Ready to deploy  
**Expected Outcome:** UI will load, debug mode will be accessible  
**Next Step:** Deploy through UI and test
