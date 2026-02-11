import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import difflib
import hashlib
import json
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple

# -------------------- ADVANCED DEPENDENCIES --------------------
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

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except ImportError:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        SEMANTIC_AVAILABLE = False
    except ImportError:
        SEMANTIC_AVAILABLE = None

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

# -------------------- SUPABASE CLIENT --------------------
@st.cache_resource
def init_supabase() -> Optional[Client]:
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
        return None

# -------------------- HANDBOOK PERSISTENCE --------------------
def save_handbook(content: str):
    supabase = init_supabase()
    if not supabase:
        return False
    try:
        result = supabase.table("handbooks").select("*").execute()
        if result.data:
            supabase.table("handbooks").update({
                "content": content,
                "updated_at": "now()"
            }).eq("id", result.data[0]["id"]).execute()
        else:
            supabase.table("handbooks").insert({"content": content}).execute()
        return True
    except Exception as e:
        st.error(f"Failed to save handbook: {e}")
        return False

def load_handbook() -> str:
    supabase = init_supabase()
    if not supabase:
        return ""
    try:
        result = supabase.table("handbooks").select("*").execute()
        if result.data:
            return result.data[0]["content"]
    except Exception as e:
        st.error(f"Failed to load handbook: {e}")
    return ""

# -------------------- BILL MONITORING --------------------
def add_monitored_bill(bill_id: str, bill_title: str, text_hash: str):
    supabase = init_supabase()
    if not supabase:
        return False
    try:
        existing = supabase.table("monitored_bills").select("*").eq("bill_id", bill_id).execute()
        if existing.data:
            supabase.table("monitored_bills").update({
                "last_text_hash": text_hash,
                "last_checked": "now()",
                "bill_title": bill_title
            }).eq("bill_id", bill_id).execute()
        else:
            supabase.table("monitored_bills").insert({
                "bill_id": bill_id,
                "bill_title": bill_title,
                "last_text_hash": text_hash,
                "last_checked": "now()"
            }).execute()
        return True
    except Exception as e:
        st.error(f"Failed to add monitored bill: {e}")
        return False

def get_monitored_bills() -> List[Dict]:
    supabase = init_supabase()
    if not supabase:
        return []
    try:
        result = supabase.table("monitored_bills").select("*").execute()
        return result.data
    except Exception as e:
        st.error(f"Failed to load monitored bills: {e}")
        return []

def check_bill_updates(api_key: str) -> List[Dict]:
    monitored = get_monitored_bills()
    changed = []
    for bill in monitored:
        bill_data = fetch_bill_metadata(bill["bill_id"], api_key)
        if "error" in bill_data:
            continue
        if not bill_data.get("text_url"):
            continue
        current_text = extract_bill_text(bill_data["text_url"])
        current_hash = hashlib.md5(current_text.encode()).hexdigest()
        if current_hash != bill.get("last_text_hash"):
            changed.append({
                "bill_id": bill["bill_id"],
                "bill_title": bill_data.get("title", bill["bill_id"]),
                "old_hash": bill.get("last_text_hash"),
                "new_hash": current_hash,
                "last_checked": bill.get("last_checked")
            })
            supabase = init_supabase()
            if supabase:
                supabase.table("monitored_bills").update({
                    "last_text_hash": current_hash,
                    "last_checked": "now()"
                }).eq("bill_id", bill["bill_id"]).execute()
    return changed

# -------------------- CORE HELPER FUNCTIONS (YOUR ORIGINAL) --------------------
def validate_schema(df: pd.DataFrame, required_columns: dict) -> list:
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
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# -------------------- OPEN STATES API (IMPROVED SEARCH) --------------------
@st.cache_data(ttl=300)
def search_bills(keyword: str, api_key: str) -> List[Dict]:
    """Return up to 10 bills matching keyword, excluding resolutions/commemoratives."""
    if not api_key:
        return []
    headers = {"X-API-KEY": api_key}
    params = {
        "jurisdiction": ComplianceConfig.OPENSTATES_JURISDICTION,
        "q": keyword,
        "sort": "updated_desc",
        "page": 1,
        "per_page": 10
    }
    try:
        resp = requests.get(ComplianceConfig.OPENSTATES_BASE_URL, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("results", [])
        # Filter out resolutions and commemorative bills
        filtered = []
        for bill in results:
            classifications = bill.get("classification", [])
            if not any(c in ["resolution", "commemorative", "memorial"] for c in classifications):
                filtered.append(bill)
        return filtered[:10]
    except:
        return []

@st.cache_data(ttl=3600)
def get_bill_by_number(bill_number: str, api_key: str) -> Dict:
    """Fetch a specific bill by its identifier (e.g., 'HB33')."""
    if not api_key:
        return {"error": "API key missing."}
    headers = {"X-API-KEY": api_key}
    params = {
        "jurisdiction": ComplianceConfig.OPENSTATES_JURISDICTION,
        "bill_id": bill_number.upper(),
        "page": 1,
        "per_page": 1
    }
    try:
        resp = requests.get(ComplianceConfig.OPENSTATES_BASE_URL, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return {"error": f"API error {resp.status_code}"}
        data = resp.json()
        if not data.get("results"):
            return {"error": f"Bill '{bill_number}' not found."}
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
        return {"error": f"Network error: {str(e)}"}

@st.cache_data(ttl=3600)
def extract_bill_text(url: str) -> str:
    if not url:
        return "[No text URL available]"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            if not PDF_SUPPORT:
                return "[PDF extraction requires pdfplumber]"
            with pdfplumber.open(BytesIO(resp.content)) as pdf:
                pages = pdf.pages[:ComplianceConfig.MAX_PDF_PAGES]
                text = "\n".join(p.extract_text() or "" for p in pages)
                return text[:ComplianceConfig.MAX_EXTRACT_CHARS]
        elif 'text/html' in content_type or url.endswith(('.htm', '.html')):
            if not HTML_SUPPORT:
                return "[HTML extraction requires beautifulsoup4]"
            soup = BeautifulSoup(resp.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text[:ComplianceConfig.MAX_EXTRACT_CHARS]
        else:
            return f"[Unsupported content type: {content_type}]"
    except Exception as e:
        return f"[Extraction failed: {str(e)}]"

# -------------------- SEMANTIC COMPARISON --------------------
def compute_semantic_differences(policy_text: str, bill_text: str) -> Dict:
    if not policy_text or not bill_text:
        return {"error": "Missing text."}
    policy_sentences = tokenize_sentences(policy_text)[:100]
    bill_sentences = tokenize_sentences(bill_text)[:100]
    if not policy_sentences or not bill_sentences:
        return {"error": "No sentences to compare."}
    
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
        except:
            pass
    
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

# -------------------- 🏥 LEGISLATIVE INTELLIGENCE MODULE --------------------
def render_legislative_redliner():
    st.header("⚖️ Legislative Intelligence Engine")
    st.markdown("##### Persistent Handbook + Bill Monitoring")
    st.markdown("---")
    
    api_key = st.secrets.get("OPENSTATES_API_KEY") if hasattr(st, 'secrets') else None
    if not api_key:
        st.error("🚨 Open States API key missing. Add to Streamlit Secrets.")
        st.stop()
    
    # Load saved handbook
    if "handbook_loaded" not in st.session_state:
        saved = load_handbook()
        st.session_state.handbook_content = saved
        st.session_state.handbook_loaded = True
    
    # Sidebar: Monitored Bills
    with st.sidebar.expander("🔔 Monitored Bills", expanded=True):
        if st.button("🔄 Check for Updates"):
            with st.spinner("Checking monitored bills..."):
                changed = check_bill_updates(api_key)
                if changed:
                    for b in changed:
                        st.error(f"🚨 **{b['bill_id']}** – {b['bill_title'][:80]}...\n\n*Changed since {b['last_checked']}*")
                else:
                    st.success("All monitored bills are up to date.")
        monitored = get_monitored_bills()
        if monitored:
            for m in monitored:
                st.markdown(f"- **{m['bill_id']}**  \n  {m.get('bill_title', '')[:50]}...")
        else:
            st.info("No bills being monitored.")
    
    # ---------- Two Ways to Find a Bill ----------
    st.subheader("🔎 Find Relevant Legislation")
    
    method = st.radio("Search method:", ["By Bill Number (e.g., HB33)", "By Keyword"], horizontal=True)
    
    if method == "By Bill Number (e.g., HB33)":
        bill_number = st.text_input("Enter bill number", value="HB33").strip()
        if st.button("Fetch Bill", type="primary"):
            with st.spinner("Fetching bill..."):
                bill_data = get_bill_by_number(bill_number, api_key)
                if "error" in bill_data:
                    st.error(bill_data["error"])
                    st.session_state.current_bill = None
                else:
                    st.session_state.current_bill = bill_data
                    st.session_state.bill_text = None
    else:
        keyword = st.text_input("Search by keyword (e.g., 'minimum wage', 'healthcare')")
        if keyword:
            with st.spinner("Searching..."):
                results = search_bills(keyword, api_key)
                if results:
                    bill_titles = [f"{r['identifier']}: {r['title'][:80]}..." for r in results]
                    selected_idx = st.selectbox("Select a bill to analyze", range(len(bill_titles)), format_func=lambda i: bill_titles[i])
                    selected_bill = results[selected_idx]
                    with st.spinner("Fetching details..."):
                        bill_data = get_bill_by_number(selected_bill["identifier"], api_key)
                        if "error" not in bill_data:
                            st.session_state.current_bill = bill_data
                            st.session_state.bill_text = None
                        else:
                            st.error(bill_data["error"])
                else:
                    st.warning("No matching bills found. Try different keywords.")
    
    # ---------- Display Current Bill ----------
    if st.session_state.get("current_bill"):
        bill = st.session_state.current_bill
        st.markdown("---")
        cols = st.columns([2,1,1,1])
        cols[0].success(f"**{bill['identifier']}** – {bill['title'][:120]}…")
        cols[1].metric("Session", bill.get('session', 'N/A'))
        cols[2].metric("Status", str(bill.get('status', 'N/A')).capitalize())
        cols[3].metric("Updated", bill.get('updated_at', '')[:10] if bill.get('updated_at') else 'N/A')
        
        with st.expander("📄 Bill Abstract & Source"):
            if bill.get('text_url'):
                st.markdown(f"**Full text:** [{bill['text_url']}]({bill['text_url']})")
            st.markdown(f"**Abstract:** {bill.get('abstract', 'No abstract.')}")
        
        if bill.get('text_url'):
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("📄 Extract Full Bill Text", use_container_width=True):
                    with st.spinner("Downloading..."):
                        st.session_state.bill_text = extract_bill_text(bill['text_url'])
            with col2:
                if st.button("🔔 Monitor This Bill", use_container_width=True):
                    if st.session_state.get("bill_text"):
                        text_hash = hashlib.md5(st.session_state.bill_text.encode()).hexdigest()
                        if add_monitored_bill(bill['identifier'], bill['title'], text_hash):
                            st.success("Now monitoring this bill for changes.")
                    else:
                        st.warning("Please extract the bill text first.")
        
        if st.session_state.get("bill_text"):
            with st.expander("📜 Full Bill Text (preview)", expanded=False):
                st.text_area("Bill content", st.session_state.bill_text[:5000], height=200, disabled=True)
    
    # ---------- Handbook Section ----------
    st.markdown("---")
    st.subheader("📘 Employee Handbook (Saved in Cloud)")
    
    handbook = st.text_area(
        "Paste your employee handbook or policy text:",
        value=st.session_state.get("handbook_content", ""),
        height=300,
        key="handbook_editor"
    )
    
    if st.button("💾 Save Handbook to Cloud", use_container_width=True):
        if handbook.strip():
            if save_handbook(handbook):
                st.success("Handbook saved permanently.")
                st.session_state.handbook_content = handbook
            else:
                st.error("Failed to save handbook.")
    
    # ---------- Compliance Gap Analysis ----------
    if st.session_state.get("bill_text") and st.session_state.get("handbook_content"):
        st.markdown("---")
        st.subheader("🔬 Compliance Gap Analysis")
        if st.button("Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                semantic = compute_semantic_differences(st.session_state.handbook_content, st.session_state.bill_text)
                if "overall_similarity" in semantic:
                    sim = semantic["overall_similarity"]
                    risk = semantic.get("compliance_risk", "UNKNOWN")
                    risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(risk, "gray")
                    cols = st.columns([1,1,2])
                    cols[0].metric("Overall Similarity", f"{sim:.1%}")
                    cols[1].markdown(f"**Compliance Risk**  \n:<span style='color:{risk_color};font-size:1.8rem;font-weight:bold'>{risk}</span>", unsafe_allow_html=True)
                    with cols[2]:
                        st.info(f"**Method:** {semantic.get('method', 'N/A')}")
                
                if semantic.get("sentence_analysis"):
                    with st.expander("⚠️ High‑Risk Policy Sentences", expanded=True):
                        risky = [s for s in semantic["sentence_analysis"] if s.get("risk") == "HIGH"][:5]
                        if risky:
                            for r in risky[:5]:
                                st.markdown(f"- **Policy:** {r['policy_sentence']}")
                                st.markdown(f"  → **Bill:** {r['bill_sentence'][:150]}...")
                                st.markdown(f"  *Similarity: {r['similarity']:.1%}*")
                                st.divider()
                        else:
                            st.success("No high‑risk sentences detected.")
                
                with st.expander("📝 Character‑level Difference"):
                    render_side_by_side_diff(
                        tokenize_sentences(st.session_state.handbook_content),
                        tokenize_sentences(st.session_state.bill_text)
                    )
                
                export_data = {
                    "bill": st.session_state.current_bill,
                    "handbook": st.session_state.handbook_content,
                    "analysis": semantic,
                    "generated": datetime.now().isoformat()
                }
                st.download_button(
                    "📥 Export Analysis (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

# -------------------- 💰 TUITION REIMBURSEMENT MODULE (YOUR ORIGINAL, RESTORED) --------------------
def render_tuition_module():
    st.header("💰 Tuition Reimbursement Audit")
    st.markdown("### IRS Section 127 & Tenure Eligibility Engine")
    st.markdown(
        """
        **Purpose:** Batch process scholarship requests to determine eligibility and tax liability.
        **Statutory Reference:** 26 U.S. Code § 127 - Educational assistance programs.
        """
    )

    with st.expander("📥 Download Batch Template", expanded=False):
        st.caption("Use this schema for upload. Do not include sensitive PII in non-secure environments.")
        template_df = pd.DataFrame(columns=['EmployeeID', 'TenureYears', 'RequestAmount', 'DegreeProgram'])
        st.download_button(
            label="Download Empty CSV Schema",
            data=template_df.to_csv(index=False),
            file_name="tuition_audit_template.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("Upload Request CSV", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            return

        required_schema = {
            'EmployeeID': 'str',
            'TenureYears': 'float',
            'RequestAmount': 'float'
        }
        
        validation_errors = validate_schema(df, required_schema)
        if validation_errors:
            for err in validation_errors:
                st.error(f"Schema Validation Error: {err}")
            st.stop()

        df['TenureYears'] = pd.to_numeric(df['TenureYears'], errors='coerce')
        df['RequestAmount'] = pd.to_numeric(df['RequestAmount'], errors='coerce')
        
        if df.isna().any().any():
            st.warning("Rows with invalid numeric data were excluded from analysis.")
            df.dropna(subset=['TenureYears', 'RequestAmount'], inplace=True)

        conditions = [
            (df['TenureYears'] < ComplianceConfig.MIN_TENURE_YEARS),
            (df['RequestAmount'] > ComplianceConfig.IRS_SEC_127_LIMIT)
        ]
        
        choices_status = ['Ineligible', 'Eligible (Taxable)']
        choices_basis = [
            f"Tenure below minimum ({ComplianceConfig.MIN_TENURE_YEARS} year)",
            f"Exceeds IRS Sec. 127 Limit (${ComplianceConfig.IRS_SEC_127_LIMIT:,.2f})"
        ]
        
        df['Decision_Status'] = np.select(conditions, choices_status, default='Eligible (Tax-Free)')
        df['Decision_Basis'] = np.select(conditions, choices_basis, default='Meets Tenure & IRS Criteria')
        
        df['Taxable_Amount'] = np.where(
            df['Decision_Status'] == 'Eligible (Taxable)',
            df['RequestAmount'] - ComplianceConfig.IRS_SEC_127_LIMIT,
            0.0
        )

        st.divider()
        st.subheader("Audit Results")
        
        total_exposure = df['Taxable_Amount'].sum()
        ineligible_count = len(df[df['Decision_Status'] == 'Ineligible'])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records Processed", len(df))
        m2.metric("Total Taxable Exposure", f"${total_exposure:,.2f}")
        m3.metric("Ineligible Requests", ineligible_count)

        st.dataframe(
            df[['EmployeeID', 'RequestAmount', 'Decision_Status', 'Taxable_Amount', 'Decision_Basis']],
            use_container_width=True,
            hide_index=True
        )
        
        output_csv = df.to_csv(index=False)
        st.download_button(
            "📄 Download Audit Report",
            data=output_csv,
            file_name=f"tuition_audit_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# -------------------- 📝 POLICY REDLINER MODULE (YOUR ORIGINAL, RESTORED) --------------------
def render_redliner_module():
    st.header("📝 Policy Gap Analysis")
    st.markdown("### Statutory vs. Internal Policy Comparator")
    st.markdown(
        """
        **Purpose:** Compare internal policy text against statutory text at the sentence level to identify compliance gaps.
        **Method:** Tokenizes text into sentences and performs a sequence match diff.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Internal Policy Text")
        policy_text = st.text_area("Paste current handbook text here:", height=300, key="redliner_policy")
    with col2:
        st.markdown("#### New Statutory Text")
        statute_text = st.text_area("Paste new legislative text here:", height=300, key="redliner_statute")

    if st.button("Run Gap Analysis"):
        if not policy_text or not statute_text:
            st.warning("Both text fields are required for analysis.")
            return

        st.divider()
        st.subheader("Analysis Output")

        policy_sentences = tokenize_sentences(policy_text)
        statute_sentences = tokenize_sentences(statute_text)

        render_side_by_side_diff(policy_sentences, statute_sentences)

# -------------------- 📅 TIMELINE PROJECTOR MODULE (YOUR ORIGINAL, RESTORED) --------------------
def render_projector_module():
    st.header("📅 Compliance Work-Back Schedule")
    st.markdown("### Implementation Timeline Generator")
    st.markdown("**Purpose:** Generate operational milestones backwards from a statutory effective date.")

    effective_date = st.date_input("Statutory Effective Date")
    
    if st.button("Generate Timeline"):
        milestones = [
            ("Audit of Current Policies", 90),
            ("Drafting of Policy Updates", 60),
            ("Legal Counsel Review", 45),
            ("Executive Sign-off", 30),
            ("Manager Training", 15),
            ("Employee Notification", 7),
            ("Go-Live / Effective Date", 0)
        ]

        st.subheader("Operational Milestones")
        
        timeline_data = []
        for task, days in milestones:
            due_date = effective_date - timedelta(days=days)
            timeline_data.append({
                "Milestone": task,
                "Due Date": due_date.strftime('%Y-%m-%d'),
                "Days Until Effective": days,
                "Phase": "Preparation" if days > 0 else "Execution"
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        st.table(timeline_df)
        
        csv = timeline_df.to_csv(index=False)
        st.download_button(
            "📥 Export Timeline (CSV)",
            data=csv,
            file_name=f"compliance_timeline_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

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
        st.write(f"**Supabase:** {'✅ Connected' if init_supabase() else '❌ Not configured'}")
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
