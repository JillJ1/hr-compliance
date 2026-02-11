import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import difflib
import json
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple

# -------------------- ADVANCED DEPENDENCIES (graceful fallbacks) --------------------
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from bs4 import BeautifulSoup
    HTML_SUPPORT = True
except ImportError:
    HTML_SUPPORT = False

# Semantic similarity – try sentence-transformers, fallback to TF-IDF, fallback to simple diff
SEMANTIC_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except ImportError:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        SEMANTIC_AVAILABLE = False   # will use TF-IDF
    except ImportError:
        SEMANTIC_AVAILABLE = None    # no semantic capability

# -------------------- CONFIGURATION --------------------
class ComplianceConfig:
    IRS_SEC_127_LIMIT = 5250.00
    MIN_TENURE_YEARS = 1.0
    OPENSTATES_BASE_URL = "https://v3.openstates.org/bills"
    OPENSTATES_JURISDICTION = "Ohio"
    API_TIMEOUT = 15
    MAX_PDF_PAGES = 20
    MAX_EXTRACT_CHARS = 15000
    PAGE_TITLE = "Ohio Health System | Enterprise Compliance Suite"
    LAYOUT = "wide"
    ENABLE_SEMANTIC = st.secrets.get("ENABLE_SEMANTIC", True) if hasattr(st, 'secrets') else True

# -------------------- CORE HELPER FUNCTIONS (from your original code) --------------------
def validate_schema(df: pd.DataFrame, required_columns: dict) -> list:
    """Validate DataFrame columns and types."""
    errors = []
    missing_cols = [col for col in required_columns.keys() if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    for col, dtype in required_columns.items():
        if col in df.columns:
            try:
                if dtype == 'float':
                    pd.to_numeric(df[col], errors='raise')
                elif dtype == 'datetime':
                    pd.to_datetime(df[col], errors='raise')
            except Exception:
                errors.append(f"Column '{col}' contains invalid data types (expected {dtype}).")
    return errors

def tokenize_sentences(text: str) -> list:
    """Split text into sentences."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# -------------------- OPEN STATES API & TEXT EXTRACTION --------------------
@st.cache_data(ttl=3600, show_spinner="Fetching bill metadata...")
def fetch_bill_metadata(query: str, api_key: str) -> Dict:
    """Query Open States v3 API for bill metadata."""
    if not api_key:
        return {"error": "API key missing. Please configure OPENSTATES_API_KEY in secrets."}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    params = {
        "jurisdiction": ComplianceConfig.OPENSTATES_JURISDICTION,
        "q": query,
        "sort": "updated_desc",
        "page": 1,
        "per_page": 1
    }
    try:
        resp = requests.get(ComplianceConfig.OPENSTATES_BASE_URL, headers=headers, params=params, timeout=ComplianceConfig.API_TIMEOUT)
        if resp.status_code == 401:
            return {"error": "401 Unauthorized – invalid Open States API key."}
        if resp.status_code != 200:
            return {"error": f"API error {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        if not data.get("results"):
            return {"error": f"No bills found for '{query}' in Ohio."}
        bill = data["results"][0]
        text_url = None
        versions = bill.get("versions", [])
        if versions:
            for v in versions:
                links = v.get("links", [])
                for link in links:
                    url = link.get("url")
                    if url:
                        text_url = url
                        break
                if text_url:
                    break
        return {
            "identifier": bill.get("identifier"),
            "title": bill.get("title"),
            "session": bill.get("session"),
            "updated_at": bill.get("updated_at"),
            "text_url": text_url,
            "abstract": bill.get("abstract") or bill.get("title"),
            "status": bill.get("status"),
            "classification": bill.get("classification", []),
            "subjects": bill.get("subjects", [])
        }
    except Exception as e:
        return {"error": f"Network or parsing error: {str(e)}"}

@st.cache_data(ttl=3600, show_spinner="Downloading & extracting bill text...")
def extract_bill_text(url: str) -> str:
    """Download PDF/HTML and extract text."""
    if not url:
        return "[No public text URL available for this bill.]"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            if not PDF_SUPPORT:
                return "[PDF extraction requires 'pdfplumber'. Install with: pip install pdfplumber]"
            with pdfplumber.open(BytesIO(resp.content)) as pdf:
                pages = pdf.pages[:ComplianceConfig.MAX_PDF_PAGES]
                text = "\n".join(p.extract_text() or "" for p in pages)
                return text[:ComplianceConfig.MAX_EXTRACT_CHARS] + ("…" if len(text) > ComplianceConfig.MAX_EXTRACT_CHARS else "")
        elif 'text/html' in content_type or url.endswith(('.htm', '.html')):
            if not HTML_SUPPORT:
                return "[HTML extraction requires 'beautifulsoup4'. Install with: pip install beautifulsoup4]"
            soup = BeautifulSoup(resp.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text[:ComplianceConfig.MAX_EXTRACT_CHARS] + ("…" if len(text) > ComplianceConfig.MAX_EXTRACT_CHARS else "")
        else:
            return f"[Unsupported content type: {content_type}. Direct link: {url}]"
    except Exception as e:
        return f"[Failed to extract text: {str(e)}]\n\nDirect URL: {url}"

# -------------------- SEMANTIC COMPARISON ENGINE --------------------
def compute_semantic_differences(policy_text: str, bill_text: str) -> Dict:
    """Compare policy and bill text using embeddings or TF-IDF."""
    if not policy_text or not bill_text:
        return {"error": "Missing text for comparison."}
    policy_sentences = tokenize_sentences(policy_text)[:100]
    bill_sentences = tokenize_sentences(bill_text)[:100]
    if not policy_sentences or not bill_sentences:
        return {"error": "No sentences to compare."}
    
    # Use Sentence Transformers if available
    if SEMANTIC_AVAILABLE and ComplianceConfig.ENABLE_SEMANTIC:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            policy_emb = model.encode(policy_sentences, convert_to_tensor=True)
            bill_emb = model.encode(bill_sentences, convert_to_tensor=True)
            cos_scores = util.cos_sim(policy_emb, bill_emb)
            similarities = []
            for i, p_sent in enumerate(policy_sentences):
                best_idx = int(cos_scores[i].argmax())
                best_score = float(cos_scores[i][best_idx])
                similarities.append({
                    "policy_sentence": p_sent,
                    "bill_sentence": bill_sentences[best_idx],
                    "similarity": best_score,
                    "risk": "HIGH" if best_score < 0.6 else "MEDIUM" if best_score < 0.8 else "LOW"
                })
            overall = float(cos_scores.mean())
            return {
                "method": "sentence-transformers",
                "overall_similarity": overall,
                "compliance_risk": "LOW" if overall > 0.8 else "MEDIUM" if overall > 0.6 else "HIGH",
                "sentence_analysis": similarities[:20]
            }
        except Exception as e:
            st.warning(f"Semantic model failed, falling back to TF‑IDF: {e}")
    
    # Fallback: TF-IDF Cosine Similarity
    try:
        all_sentences = policy_sentences + bill_sentences
        vectorizer = TfidfVectorizer().fit_transform(all_sentences)
        vectors = vectorizer.toarray()
        policy_vecs = vectors[:len(policy_sentences)]
        bill_vecs = vectors[len(policy_sentences):]
        similarities = []
        for i, p_vec in enumerate(policy_vecs):
            if bill_vecs.shape[0] == 0:
                continue
            sims = cosine_similarity([p_vec], bill_vecs)[0]
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            similarities.append({
                "policy_sentence": policy_sentences[i],
                "bill_sentence": bill_sentences[best_idx],
                "similarity": best_score,
                "risk": "HIGH" if best_score < 0.2 else "MEDIUM" if best_score < 0.4 else "LOW"
            })
        overall = np.mean([s["similarity"] for s in similarities]) if similarities else 0.0
        return {
            "method": "TF-IDF",
            "overall_similarity": overall,
            "compliance_risk": "LOW" if overall > 0.4 else "MEDIUM" if overall > 0.2 else "HIGH",
            "sentence_analysis": similarities[:20]
        }
    except Exception as e:
        return {"error": f"Semantic analysis failed: {e}"}

# -------------------- DIFF VISUALIZATION --------------------
def render_side_by_side_diff(policy_sentences, bill_sentences):
    """Enhanced diff with colored blocks."""
    differ = difflib.SequenceMatcher(None, policy_sentences, bill_sentences)
    for tag, i1, i2, j1, j2 in differ.get_opcodes():
        if tag == 'equal':
            for sent in policy_sentences[i1:i2]:
                with st.expander(f"✅ {sent[:60]}...", expanded=False):
                    st.caption(sent)
        elif tag == 'replace':
            st.markdown("**:orange[🔄 MODIFIED SECTION]**")
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Policy (old)**")
                for sent in policy_sentences[i1:i2]:
                    st.error(sent)
            with col_right:
                st.markdown("**Statute (new)**")
                for sent in bill_sentences[j1:j2]:
                    st.success(sent)
        elif tag == 'delete':
            st.markdown("**:red[❌ REMOVED FROM POLICY]**")
            for sent in policy_sentences[i1:i2]:
                st.error(sent)
        elif tag == 'insert':
            st.markdown("**:green[➕ NEW REQUIREMENT]**")
            for sent in bill_sentences[j1:j2]:
                st.success(sent)

# -------------------- LEGISLATIVE INTELLIGENCE MODULE --------------------
def render_legislative_redliner():
    st.header("⚖️ Legislative Intelligence Engine")
    st.markdown("##### Real‑time Bill Analysis & Compliance Gap Detection")
    st.markdown("---")
    
    api_key = st.secrets.get("OPENSTATES_API_KEY") if hasattr(st, 'secrets') else None
    if not api_key:
        st.error("🚨 **Open States API key not found.** Please add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
        st.stop()
    
    # Session state
    if "bill_data" not in st.session_state:
        st.session_state.bill_data = None
    if "bill_text" not in st.session_state:
        st.session_state.bill_text = None
    if "policy_text" not in st.session_state:
        st.session_state.policy_text = ""
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None
    
    col1, col2 = st.columns([3, 1])
    with col1:
        bill_query = st.text_input("🔎 **Search Ohio Bill** (e.g., HB33, SB1, 'minimum wage')", value="HB33", key="bill_query_input")
    with col2:
        search_clicked = st.button("📥 Fetch Bill", type="primary", use_container_width=True)
    
    if search_clicked and bill_query:
        with st.status("🔄 Connecting to Open States API...", expanded=True) as status:
            st.write("Querying legislative database...")
            bill_data = fetch_bill_metadata(bill_query, api_key)
            if "error" in bill_data:
                status.update(label="❌ Fetch failed", state="error")
                st.error(bill_data["error"])
                st.session_state.bill_data = None
            else:
                status.update(label="✅ Bill found", state="complete")
                st.session_state.bill_data = bill_data
                st.session_state.bill_text = None
                st.session_state.comparison_result = None
    
    if st.session_state.bill_data and "error" not in st.session_state.bill_data:
        bill = st.session_state.bill_data
        
        # Header with metrics – make sure 'cols' is defined here
        cols = st.columns([2,1,1,1])
        cols[0].success(f"**{bill['identifier']}** – {bill['title'][:120]}…")
        cols[1].metric("Session", bill.get('session', 'N/A'))
        cols[2].metric("Status", str(bill.get('status', 'N/A')).capitalize())
        cols[3].metric("Updated", bill.get('updated_at', '')[:10] if bill.get('updated_at') else 'N/A')
        
        with st.expander("📄 View official source & abstract"):
            if bill.get('text_url'):
                st.markdown(f"**Full text:** [{bill['text_url']}]({bill['text_url']})")
            st.markdown(f"**Abstract:** {bill.get('abstract', 'No abstract available.')}")
            if bill.get('subjects'):
                st.markdown(f"**Subjects:** {', '.join(bill['subjects'][:5])}")
        
        if bill.get('text_url'):
            if st.button("📄 Extract Full Bill Text", use_container_width=True):
                with st.spinner("Downloading and parsing document..."):
                    text = extract_bill_text(bill['text_url'])
                    st.session_state.bill_text = text
            if st.session_state.bill_text:
                with st.expander("📜 Extracted Bill Text (preview)", expanded=False):
                    st.text_area("Bill content", st.session_state.bill_text[:5000], height=200, disabled=True)
                    if len(st.session_state.bill_text) > 5000:
                        st.caption(f"*Showing first 5,000 of {len(st.session_state.bill_text):,} characters*")
        
        st.markdown("---")
        st.subheader("📋 Compare with Internal Policy")
        policy_source = st.radio(
            "Policy source:",
            ["Paste current handbook text", "Use sample policy"],
            horizontal=True,
            key="policy_source_radio"
        )
        if policy_source == "Use sample policy":
            sample = "Employees are entitled to a 30-minute unpaid meal break if they work more than 6 consecutive hours."
            st.session_state.policy_text = st.text_area(
                "Edit sample policy:",
                value=sample,
                height=150,
                key="policy_text_area_sample"
            )
        else:
            st.session_state.policy_text = st.text_area(
                "Paste your policy text:",
                value=st.session_state.policy_text,
                height=200,
                key="policy_text_area",
                placeholder="Copy from employee handbook..."
            )
        
        analyze_disabled = not (st.session_state.policy_text and st.session_state.bill_text)
        if st.button("🔬 Run Compliance Gap Analysis", disabled=analyze_disabled, type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing semantic alignment..."):
                policy_sentences = tokenize_sentences(st.session_state.policy_text)
                bill_sentences = tokenize_sentences(st.session_state.bill_text)
                semantic = compute_semantic_differences(st.session_state.policy_text, st.session_state.bill_text)
                st.session_state.comparison_result = {
                    "diff": {"policy": policy_sentences, "bill": bill_sentences},
                    "semantic": semantic,
                    "timestamp": datetime.now().isoformat()
                }
        
        if st.session_state.comparison_result:
            res = st.session_state.comparison_result
            st.markdown("---")
            st.subheader("📊 Compliance Gap Analysis")
            if "semantic" in res and "overall_similarity" in res["semantic"]:
                sem = res["semantic"]
                sim = sem["overall_similarity"]
                risk = sem.get("compliance_risk", "UNKNOWN")
                risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(risk, "gray")
                cols = st.columns([1,1,2])
                cols[0].metric("Overall Similarity", f"{sim:.1%}")
                cols[1].markdown(f"**Compliance Risk**  \n:<span style='color:{risk_color};font-size:1.8rem;font-weight:bold'>{risk}</span>", unsafe_allow_html=True)
                with cols[2]:
                    st.info(f"**Method:** {sem.get('method', 'N/A')}  \n*Semantic similarity measures conceptual alignment, not just exact wording.*")
                if sem.get("sentence_analysis"):
                    with st.expander("⚠️ High‑Risk Policy Sentences", expanded=True):
                        risky = [s for s in sem["sentence_analysis"] if s.get("risk") == "HIGH"][:5]
                        if risky:
                            for r in risky[:5]:
                                st.markdown(f"- **Policy:** {r['policy_sentence']}")
                                st.markdown(f"  → **Bill:** {r['bill_sentence'][:150]}...")
                                st.markdown(f"  *Similarity: {r['similarity']:.1%}*")
                                st.divider()
                        else:
                            st.success("No high‑risk sentences detected.")
            with st.expander("📝 Character‑level Difference (exact changes)", expanded=False):
                if "diff" in res:
                    render_side_by_side_diff(res["diff"]["policy"], res["diff"]["bill"])
            export_data = {
                "bill": st.session_state.bill_data,
                "policy": st.session_state.policy_text,
                "analysis": res,
                "generated": datetime.now().isoformat()
            }
            export_json = json.dumps(export_data, indent=2)
            st.download_button(
                "📥 Export Full Analysis (JSON)",
                data=export_json,
                file_name=f"compliance_gap_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        if not st.session_state.bill_data:
            st.info("👆 Enter a bill number and click 'Fetch Bill' to begin.")

# -------------------- TUITION REIMBURSEMENT MODULE (upgraded) --------------------
def render_tuition_module():
    st.header("💰 Tuition Reimbursement Auditor")
    st.markdown("##### IRS §127 & Tenure Eligibility Engine")
    st.caption(f"Current IRS limit: **${ComplianceConfig.IRS_SEC_127_LIMIT:,.2f}** | Min tenure: **{ComplianceConfig.MIN_TENURE_YEARS} year**")
    
    with st.expander("📥 Download CSV Template", expanded=False):
        template_df = pd.DataFrame(columns=['EmployeeID', 'TenureYears', 'RequestAmount', 'DegreeProgram'])
        st.download_button(
            "Download Template",
            data=template_df.to_csv(index=False),
            file_name="tuition_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    uploaded = st.file_uploader("Upload employee requests (CSV)", type=['csv'])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"CSV parsing error: {e}")
            return
        
        required = {'EmployeeID': 'str', 'TenureYears': 'float', 'RequestAmount': 'float'}
        errors = validate_schema(df, required)
        if errors:
            for e in errors:
                st.error(e)
            st.stop()
        
        df['TenureYears'] = pd.to_numeric(df['TenureYears'], errors='coerce')
        df['RequestAmount'] = pd.to_numeric(df['RequestAmount'], errors='coerce')
        df.dropna(subset=['TenureYears', 'RequestAmount'], inplace=True)
        
        conditions = [
            (df['TenureYears'] < ComplianceConfig.MIN_TENURE_YEARS),
            (df['RequestAmount'] > ComplianceConfig.IRS_SEC_127_LIMIT)
        ]
        status_choices = ['Ineligible', 'Eligible (Taxable)']
        basis_choices = [
            f"Tenure < {ComplianceConfig.MIN_TENURE_YEARS}yr",
            f"Exceeds IRS ${ComplianceConfig.IRS_SEC_127_LIMIT:,.0f}"
        ]
        
        df['Decision'] = np.select(conditions, status_choices, default='Eligible (Tax‑Free)')
        df['Basis'] = np.select(conditions, basis_choices, default='Meets criteria')
        df['Taxable_Amount'] = np.where(df['Decision'] == 'Eligible (Taxable)',
                                         df['RequestAmount'] - ComplianceConfig.IRS_SEC_127_LIMIT,
                                         0.0)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Processed", len(df))
        col2.metric("Tax‑Free Eligible", len(df[df['Decision'] == 'Eligible (Tax‑Free)']))
        col3.metric("Taxable", len(df[df['Decision'] == 'Eligible (Taxable)']))
        col4.metric("Total Exposure", f"${df['Taxable_Amount'].sum():,.2f}")
        
        st.dataframe(
            df[['EmployeeID', 'RequestAmount', 'Decision', 'Taxable_Amount', 'Basis']],
            use_container_width=True,
            hide_index=True
        )
        
        st.download_button(
            "📄 Download Audit Report (CSV)",
            data=df.to_csv(index=False),
            file_name=f"tuition_audit_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# -------------------- POLICY REDLINER (classic, upgraded) --------------------
def render_redliner_module():
    st.header("📝 Policy Redliner (Classic)")
    st.markdown("##### Sentence‑level statutory comparator")
    col1, col2 = st.columns(2)
    with col1:
        policy = st.text_area("Current policy", height=250)
    with col2:
        statute = st.text_area("New statutory text", height=250)
    if st.button("Run Classic Diff"):
        if policy and statute:
            render_side_by_side_diff(tokenize_sentences(policy), tokenize_sentences(statute))
        else:
            st.warning("Both fields required.")

# -------------------- TIMELINE PROJECTOR (unchanged, solid) --------------------
def render_projector_module():
    st.header("📅 Compliance Work‑Back Schedule")
    eff_date = st.date_input("Effective date of new law", value=datetime.today() + timedelta(days=90))
    if st.button("Generate Timeline"):
        milestones = [
            ("Audit of Current Policies", 90),
            ("Draft Policy Updates", 60),
            ("Legal Review", 45),
            ("Executive Approval", 30),
            ("Manager Training", 15),
            ("Employee Notification", 7),
            ("Go‑Live", 0)
        ]
        data = []
        for task, days in milestones:
            due = eff_date - timedelta(days=days)
            data.append({"Milestone": task, "Due Date": due.strftime("%Y-%m-%d"), "Lead Days": days})
        st.table(pd.DataFrame(data))
        csv = pd.DataFrame(data).to_csv(index=False)
        st.download_button("Export Timeline", csv, "compliance_timeline.csv")

# -------------------- MAIN APP SHELL --------------------
def main():
    st.set_page_config(
        page_title=ComplianceConfig.PAGE_TITLE,
        layout=ComplianceConfig.LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        .stApp { background-color: #f5f7fb; }
        .stButton>button { border-radius: 8px; font-weight: 500; }
        .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<h1 style='color:#0b3b5c;'>Ohio Health</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("##### Enterprise Compliance OS")
    st.sidebar.divider()
    
    module = st.sidebar.radio(
        "Select Module",
        ["🏥 Legislative Intelligence", "💰 Tuition Auditor", "📝 Policy Redliner", "📅 Timeline Projector"],
        index=0
    )
    
    st.sidebar.divider()
    with st.sidebar.expander("🔧 System Status", expanded=False):
        st.write(f"**API:** {'✅ Live' if st.secrets.get('OPENSTATES_API_KEY') else '❌ No key'}")
        st.write(f"**PDF extraction:** {'✅' if PDF_SUPPORT else '❌ pdfplumber missing'}")
        st.write(f"**Semantic AI:** {'✅' if SEMANTIC_AVAILABLE else '⚠️ TF‑IDF fallback'}")
    
    if module == "🏥 Legislative Intelligence":
        render_legislative_redliner()
    elif module == "💰 Tuition Auditor":
        render_tuition_module()
    elif module == "📝 Policy Redliner":
        render_redliner_module()
    elif module == "📅 Timeline Projector":
        render_projector_module()

if __name__ == "__main__":
    main()
