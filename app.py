#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Sentinel – Enterprise Workforce Compliance System
Multi‑tenant, production‑ready, fully filtered by company_id.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import difflib
import hashlib
import json
import uuid
from datetime import datetime, timedelta, date
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

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

SEMANTIC_AVAILABLE = False
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

try:
    from fpdf import FPDF
    PDF_REPORT_AVAILABLE = True
except ImportError:
    PDF_REPORT_AVAILABLE = False

# -------------------- CONFIGURATION --------------------
class Config:
    IRS_SEC_127_LIMIT = 5250.00
    MIN_TENURE_YEARS = 1.0
    OPENSTATES_BASE_URL = "https://v3.openstates.org/bills"
    OPENSTATES_JURISDICTION = "Ohio"
    API_TIMEOUT = 15
    MAX_PDF_PAGES = 30
    MAX_EXTRACT_CHARS = 20000
    PAGE_TITLE = "HR Sentinel • Workforce Compliance"
    LAYOUT = "wide"
    ENABLE_SEMANTIC = st.secrets.get("ENABLE_SEMANTIC", True) if hasattr(st, 'secrets') else True
    RISK_THRESHOLDS = {"LOW": 0.8, "MEDIUM": 0.6}

# -------------------- CACHED RESOURCES --------------------
@st.cache_resource
def get_supabase_client() -> Optional[Client]:
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
        return None

@st.cache_resource
def load_semantic_model():
    if not SEMANTIC_AVAILABLE:
        return None
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.warning(f"Could not load semantic model: {e}")
        return None

# -------------------- SUPABASE HELPERS --------------------
def get_or_create_default_company(supabase: Client) -> str:
    try:
        resp = supabase.table("companies").select("id").limit(1).execute()
        if resp.data:
            return resp.data[0]["id"]
        else:
            company_id = str(uuid.uuid4())
            supabase.table("companies").insert({
                "id": company_id,
                "name": "Founding Company"
            }).execute()
            return company_id
    except Exception as e:
        st.error(f"Failed to initialize company: {e}")
        return ""

def log_action(supabase: Client, action: str, details: dict = None):
    try:
        supabase.table("audit_log").insert({
            "action": action,
            "details": json.dumps(details) if details else None,
            "user_id": "system"
        }).execute()
    except Exception:
        pass

# -------------------- CORE HELPERS --------------------
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

def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()

def risk_level_from_score(score: float) -> str:
    if score >= Config.RISK_THRESHOLDS["LOW"]:
        return "LOW"
    elif score >= Config.RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    else:
        return "HIGH"

# -------------------- OPEN STATES API --------------------
@st.cache_data(ttl=300, show_spinner="Searching Ohio legislation...")
def search_bills(keyword: str, api_key: str, year: int = 0) -> List[Dict]:
    if not api_key or not keyword:
        return []
    headers = {"X-API-KEY": api_key}
    params = {
        "jurisdiction": Config.OPENSTATES_JURISDICTION,
        "q": keyword,
        "sort": "updated_desc",
        "page": 1,
        "per_page": 50
    }
    try:
        resp = requests.get(Config.OPENSTATES_BASE_URL, headers=headers, params=params, timeout=Config.API_TIMEOUT)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("results", [])
        filtered = []
        for bill in results:
            bill_id = bill.get("identifier", "").upper()
            if (bill_id.startswith("HB") or bill_id.startswith("SB") or
                bill_id.startswith("HJR") or bill_id.startswith("SJR")):
                if year > 2000:
                    session = bill.get("session", "")
                    if str(year) not in session:
                        continue
                filtered.append(bill)
        return filtered[:20]
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

@st.cache_data(ttl=3600, show_spinner="Fetching bill details...")
def get_bill_by_number(bill_number: str, api_key: str) -> Dict:
    if not api_key:
        return {"error": "API key missing"}
    bill_id = bill_number.strip().upper().replace(" ", "")
    headers = {"X-API-KEY": api_key}
    params = {
        "jurisdiction": Config.OPENSTATES_JURISDICTION,
        "bill_id": bill_id,
        "page": 1,
        "per_page": 1
    }
    try:
        resp = requests.get(Config.OPENSTATES_BASE_URL, headers=headers, params=params, timeout=Config.API_TIMEOUT)
        if resp.status_code == 401:
            return {"error": "Invalid Open States API key."}
        if resp.status_code != 200:
            return {"error": f"API error {resp.status_code}"}
        data = resp.json()
        if not data.get("results"):
            params["bill_id"] = bill_number.strip().upper()
            resp = requests.get(Config.OPENSTATES_BASE_URL, headers=headers, params=params, timeout=Config.API_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
        if not data.get("results"):
            return {"error": f"Bill '{bill_number}' not found in Ohio."}
        bill = data["results"][0]
        versions = []
        for v in bill.get("versions", []):
            version_info = {"date": v.get("date"), "title": v.get("note", "Unknown"), "url": None}
            links = v.get("links", [])
            for link in links:
                url = link.get("url")
                if url:
                    version_info["url"] = url
                    break
            versions.append(version_info)
        text_url = versions[0]["url"] if versions else None
        return {
            "identifier": bill.get("identifier"),
            "title": bill.get("title"),
            "session": bill.get("session"),
            "updated_at": bill.get("updated_at"),
            "text_url": text_url,
            "abstract": bill.get("abstract") or bill.get("title"),
            "status": bill.get("status"),
            "classification": bill.get("classification", []),
            "subjects": bill.get("subjects", []),
            "versions": versions,
            "sponsors": bill.get("sponsors", []),
            "chamber": bill.get("chamber", "")
        }
    except Exception as e:
        return {"error": f"Network error: {str(e)}"}

@st.cache_data(ttl=3600, show_spinner="Extracting full text...")
def extract_bill_text(url: str) -> str:
    if not url:
        return "[No text URL available]"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            if not PDF_SUPPORT:
                return "[PDF extraction requires pdfplumber]"
            with pdfplumber.open(BytesIO(resp.content)) as pdf:
                pages = pdf.pages[:Config.MAX_PDF_PAGES]
                text = "\n".join(p.extract_text() or "" for p in pages)
                return text[:Config.MAX_EXTRACT_CHARS]
        elif 'text/html' in content_type or url.endswith(('.htm', '.html')):
            if not HTML_SUPPORT:
                return "[HTML extraction requires beautifulsoup4]"
            soup = BeautifulSoup(resp.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text[:Config.MAX_EXTRACT_CHARS]
        else:
            return f"[Unsupported content type: {content_type}]"
    except Exception as e:
        return f"[Extraction failed: {str(e)}]"

# -------------------- COMPARISON ENGINE --------------------
class ComplianceAnalyzer:
    @staticmethod
    def semantic_similarity(text1: str, text2: str) -> Dict:
        if not text1 or not text2:
            return {"error": "Missing text"}
        sents1 = tokenize_sentences(text1)[:100]
        sents2 = tokenize_sentences(text2)[:100]
        if not sents1 or not sents2:
            return {"error": "No sentences to compare"}
        model = load_semantic_model()
        if model is not None and Config.ENABLE_SEMANTIC:
            try:
                emb1 = model.encode(sents1, convert_to_tensor=True)
                emb2 = model.encode(sents2, convert_to_tensor=True)
                cos_scores = util.cos_sim(emb1, emb2)
                similarities = []
                for i, sent in enumerate(sents1):
                    best_idx = int(cos_scores[i].argmax())
                    best_score = float(cos_scores[i][best_idx])
                    similarities.append({
                        "policy_sentence": sent,
                        "bill_sentence": sents2[best_idx],
                        "similarity": best_score,
                        "risk": risk_level_from_score(best_score)
                    })
                overall = float(cos_scores.mean())
                return {
                    "method": "sentence-transformers",
                    "overall_similarity": overall,
                    "compliance_risk": risk_level_from_score(overall),
                    "sentence_analysis": similarities[:25]
                }
            except Exception as e:
                st.warning(f"Semantic model failed, falling back to TF‑IDF: {e}")
        # TF-IDF fallback
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            all_sents = sents1 + sents2
            vectorizer = TfidfVectorizer().fit_transform(all_sents)
            vectors = vectorizer.toarray()
            vec1 = vectors[:len(sents1)]
            vec2 = vectors[len(sents1):]
            similarities = []
            for i, v1 in enumerate(vec1):
                if vec2.shape[0] == 0:
                    continue
                sims = cosine_similarity([v1], vec2)[0]
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                normalized_score = min(1.0, best_score * 2.0)
                similarities.append({
                    "policy_sentence": sents1[i],
                    "bill_sentence": sents2[best_idx],
                    "similarity": best_score,
                    "risk": risk_level_from_score(normalized_score)
                })
            overall = np.mean([s["similarity"] for s in similarities]) if similarities else 0.0
            normalized_overall = min(1.0, overall * 2.0)
            return {
                "method": "TF-IDF",
                "overall_similarity": overall,
                "compliance_risk": risk_level_from_score(normalized_overall),
                "sentence_analysis": similarities[:25]
            }
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}

    @staticmethod
    def classic_diff(text1: str, text2: str) -> List[Dict]:
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        differ = difflib.SequenceMatcher(None, lines1, lines2)
        changes = []
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == 'replace':
                changes.append({"type": "replace", "old": lines1[i1:i2], "new": lines2[j1:j2]})
            elif tag == 'delete':
                changes.append({"type": "delete", "old": lines1[i1:i2]})
            elif tag == 'insert':
                changes.append({"type": "insert", "new": lines2[j1:j2]})
        return changes

# -------------------- MONITORED BILLS AUTO‑CHECK --------------------
def check_all_monitored_bills(supabase: Client, api_key: str, company_id: str) -> List[Dict]:
    try:
        resp = supabase.table("monitored_bills") \
            .select("*") \
            .eq("company_id", company_id) \
            .execute()
        monitored = resp.data
    except Exception as e:
        st.error(f"Failed to load monitored bills: {e}")
        return []
    changed = []
    for bill in monitored:
        fresh = get_bill_by_number(bill["bill_id"], api_key)
        if "error" in fresh:
            continue
        if not fresh.get("text_url"):
            continue
        current_text = extract_bill_text(fresh["text_url"])
        current_hash = compute_text_hash(current_text)
        if current_hash != bill.get("last_text_hash"):
            try:
                supabase.table("monitored_bills") \
                    .update({
                        "last_text_hash": current_hash,
                        "last_checked": datetime.utcnow().isoformat(),
                        "bill_title": fresh.get("title", "")
                    }) \
                    .eq("id", bill["id"]) \
                    .execute()
            except Exception as e:
                st.error(f"Failed to update bill {bill['bill_id']}: {e}")
                continue
            changed.append({
                "bill_id": bill["bill_id"],
                "bill_title": fresh.get("title", ""),
                "old_hash": bill.get("last_text_hash"),
                "new_hash": current_hash
            })
    return changed

# -------------------- PDF REPORT GENERATOR --------------------
class ComplianceReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'HR Sentinel Compliance Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(240,240,240)
        self.cell(0,8,title,0,1,'L',1)
        self.ln(2)
    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        try:
            self.multi_cell(0,5, text.encode('latin-1', errors='replace').decode('latin-1'))
        except:
            self.multi_cell(0,5, "[Text could not be rendered]")
        self.ln()
    def risk_meter(self, risk_level):
        self.set_font('Arial', 'B', 10)
        color = {"LOW": (0,128,0), "MEDIUM": (255,165,0), "HIGH": (255,0,0)}.get(risk_level, (0,0,0))
        self.set_text_color(*color)
        self.cell(0,6, f"Compliance Risk: {risk_level}", 0,1)
        self.set_text_color(0,0,0)

def generate_pdf_report(bill_data: Dict, handbook_text: str, analysis: Dict) -> bytes:
    if not PDF_REPORT_AVAILABLE:
        return b"PDF generation requires fpdf library."
    pdf = ComplianceReportPDF()
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Compliance Gap Analysis',0,1,'C')
    pdf.ln(5)
    pdf.chapter_title('Bill Information')
    bill_info = f"Bill: {bill_data.get('identifier','N/A')}\n"
    bill_info += f"Title: {bill_data.get('title','N/A')}\n"
    bill_info += f"Session: {bill_data.get('session','N/A')}\n"
    bill_info += f"Status: {bill_data.get('status','N/A')}\n"
    bill_info += f"Last Updated: {bill_data.get('updated_at','N/A')}"
    pdf.chapter_body(bill_info)
    pdf.chapter_title('Risk Assessment')
    if "overall_similarity" in analysis:
        sim = analysis["overall_similarity"]
        risk = analysis.get("compliance_risk","UNKNOWN")
        pdf.risk_meter(risk)
        pdf.chapter_body(f"Overall Semantic Similarity: {sim:.1%}")
    else:
        pdf.chapter_body("Semantic analysis not available.")
    if analysis.get("sentence_analysis"):
        risky = [s for s in analysis["sentence_analysis"] if s.get("risk")=="HIGH"]
        if risky:
            pdf.chapter_title('High‑Risk Policy Gaps')
            for i,r in enumerate(risky[:10]):
                pdf.set_font('Arial','B',9)
                pdf.cell(0,5,f"{i+1}. Policy Sentence:",0,1)
                pdf.set_font('Arial','',9)
                pdf.multi_cell(0,4,r['policy_sentence'][:300])
                pdf.set_font('Arial','I',9)
                pdf.cell(0,4,f"Similarity: {r['similarity']:.1%}",0,1)
                pdf.ln(2)
    pdf.set_y(-20)
    pdf.set_font('Arial','I',8)
    pdf.cell(0,10,f'Generated by HR Sentinel on {datetime.now().strftime("%Y-%m-%d %H:%M")}',0,0,'C')
    try:
        return pdf.output(dest='S').encode('latin-1', errors='replace')
    except:
        return b"PDF generation failed."

# -------------------- UI MODULES (ALL FILTERED BY company_id) --------------------
def render_dashboard(supabase: Client, company_id: str):
    st.title("🛡️ Compliance Risk Overview")
    try:
        status_data = supabase.table("compliance_status_view") \
            .select("*") \
            .eq("company_id", company_id) \
            .execute().data
    except Exception as e:
        st.error(f"Could not load compliance data: {e}")
        status_data = []
    expired = sum(1 for r in status_data if r.get("calculated_status") == "Expired")
    due = sum(1 for r in status_data if r.get("calculated_status") == "Due Soon")
    compliant = sum(1 for r in status_data if r.get("calculated_status") == "Compliant")
    total = expired + due + compliant
    risk_score = 100 if total == 0 else max(0, 100 - ((expired * 5) + (due * 2)))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Score", f"{risk_score}/100")
    col2.metric("Expired Items", expired)
    col3.metric("Due Soon", due)
    col4.metric("Compliant", compliant)
    st.divider()
    st.subheader("Active Compliance Records")
    st.dataframe(status_data, use_container_width=True)

def render_employees(supabase: Client, company_id: str):
    st.title("Workforce Management")
    with st.expander("➕ Add New Employee", expanded=False):
        with st.form("add_employee"):
            first = st.text_input("First Name")
            last = st.text_input("Last Name")
            hire = st.date_input("Hire Date", value=date.today())
            status = st.selectbox("Employment Status", ["Active", "LOA", "Terminated"])
            flsa = st.selectbox("FLSA Classification", ["Exempt", "Non-Exempt"])
            work_auth = st.date_input("Work Authorization Expiration (optional)", value=None)
            submitted = st.form_submit_button("Create Employee")
            if submitted and first and last:
                try:
                    supabase.table("employees").insert({
                        "company_id": company_id,
                        "first_name": first,
                        "last_name": last,
                        "hire_date": hire.isoformat(),
                        "employment_status": status,
                        "flsa_classification": flsa,
                        "work_auth_expiration": work_auth.isoformat() if work_auth else None
                    }).execute()
                    log_action(supabase, "employee_created", {"name": f"{first} {last}"})
                    st.success(f"Employee {first} {last} added.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add employee: {e}")
    try:
        employees = supabase.table("employees") \
            .select("*") \
            .eq("company_id", company_id) \
            .execute().data
        st.dataframe(employees, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load employees: {e}")

def render_requirements(supabase: Client, company_id: str):
    st.title("Compliance Requirements Engine")
    try:
        categories = supabase.table("compliance_categories") \
            .select("*") \
            .eq("active", True) \
            .execute().data
        category_map = {c["name"]: c["id"] for c in categories}
    except Exception as e:
        st.error(f"Could not load categories: {e}")
        category_map = {}
    with st.expander("➕ Create New Requirement", expanded=False):
        with st.form("add_requirement"):
            title = st.text_input("Requirement Title")
            category = st.selectbox("Category", list(category_map.keys()))
            renewal = st.number_input("Renewal Period (days)", min_value=0, value=365)
            mandatory = st.checkbox("Mandatory", value=True)
            submitted = st.form_submit_button("Create Requirement")
            if submitted and title and category:
                try:
                    supabase.table("compliance_requirements").insert({
                        "company_id": company_id,
                        "title": title,
                        "category_id": category_map[category],
                        "renewal_period_days": renewal,
                        "mandatory": mandatory
                    }).execute()
                    log_action(supabase, "requirement_created", {"title": title})
                    st.success(f"Requirement '{title}' created.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create requirement: {e}")
    try:
        requirements = supabase.table("compliance_requirements") \
            .select("*") \
            .eq("company_id", company_id) \
            .execute().data
        st.dataframe(requirements, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load requirements: {e}")

def render_compliance_tracking(supabase: Client, company_id: str):
    st.title("Employee Compliance Assignments")
    try:
        employees = supabase.table("employees") \
            .select("id, first_name, last_name") \
            .eq("company_id", company_id) \
            .eq("employment_status", "Active") \
            .execute().data
        emp_map = {f"{e['first_name']} {e['last_name']}": e["id"] for e in employees}
    except Exception as e:
        st.error(f"Could not load employees: {e}")
        emp_map = {}
    try:
        requirements = supabase.table("compliance_requirements") \
            .select("id, title") \
            .eq("company_id", company_id) \
            .execute().data
        req_map = {r["title"]: r["id"] for r in requirements}
    except Exception as e:
        st.error(f"Could not load requirements: {e}")
        req_map = {}
    with st.expander("➕ Assign Requirement to Employee", expanded=False):
        with st.form("assign_requirement"):
            emp = st.selectbox("Employee", list(emp_map.keys()) if emp_map else [])
            req = st.selectbox("Requirement", list(req_map.keys()) if req_map else [])
            completion = st.date_input("Completion Date", value=date.today())
            expiration = st.date_input("Expiration Date", value=date.today() + timedelta(days=365))
            submitted = st.form_submit_button("Assign")
            if submitted and emp and req:
                try:
                    supabase.table("employee_compliance_records").insert({
                        "employee_id": emp_map[emp],
                        "requirement_id": req_map[req],
                        "status": "Compliant",
                        "completion_date": completion.isoformat(),
                        "expiration_date": expiration.isoformat()
                    }).execute()
                    log_action(supabase, "compliance_assigned", {"employee": emp, "requirement": req})
                    st.success("Compliance record created.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to assign: {e}")
    try:
        records = supabase.table("compliance_status_view") \
            .select("*") \
            .eq("company_id", company_id) \
            .execute().data
        st.dataframe(records, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load compliance records: {e}")

def render_legislative_intelligence(supabase: Client, company_id: str, api_key: str):
    st.title("⚖️ Legislative Intelligence")
    
    # Load handbook content for this company
    handbook_content = ""
    try:
        handbook_resp = supabase.table("handbooks") \
            .select("content") \
            .eq("company_id", company_id) \
            .limit(1) \
            .execute()
        if handbook_resp.data:
            handbook_content = handbook_resp.data[0]["content"]
    except Exception as e:
        st.warning(f"Could not load handbook: {e}")
    
    # Sidebar: Monitored Bills
    with st.sidebar:
        st.markdown("### 🔔 Monitored Bills")
        try:
            monitored = supabase.table("monitored_bills") \
                .select("*") \
                .eq("company_id", company_id) \
                .execute().data
            if monitored:
                for m in monitored:
                    col1, col2 = st.columns([4,1])
                    with col1:
                        st.markdown(f"**{m['bill_id']}**")
                        st.caption(m.get('bill_title', '')[:40] + "...")
                    with col2:
                        if st.button("✕", key=f"remove_{m['id']}"):
                            supabase.table("monitored_bills").delete().eq("id", m["id"]).execute()
                            st.rerun()
                if st.button("🔄 Check All for Updates", use_container_width=True):
                    with st.spinner("Checking monitored bills..."):
                        changed = check_all_monitored_bills(supabase, api_key, company_id)
                        if changed:
                            for b in changed:
                                st.error(f"🚨 **{b['bill_id']}** – Updated!")
                        else:
                            st.success("All monitored bills are current.")
            else:
                st.info("No bills monitored.")
        except Exception as e:
            st.error(f"Error loading monitored bills: {e}")
    
    # Main tabs
    tab_search, tab_handbook, tab_analysis = st.tabs(["🔎 Search Bills", "📘 Handbook", "📊 Gap Analysis"])
    
    with tab_search:
        st.subheader("Find Ohio Legislation")
        search_method = st.radio("Search by", ["Bill Number", "Keyword"], horizontal=True)
        if search_method == "Bill Number":
            bill_input = st.text_input("Enter bill number (e.g., HB33, SB 1)", value="HB33")
            if st.button("📥 Fetch Bill", type="primary"):
                with st.spinner("Fetching..."):
                    bill = get_bill_by_number(bill_input, api_key)
                    if "error" in bill:
                        st.error(bill["error"])
                    else:
                        st.session_state.current_bill = bill
                        st.session_state.bill_text = None
                        st.success(f"Loaded {bill['identifier']}")
        else:
            col1, col2 = st.columns([3,1])
            with col1:
                keyword = st.text_input("Keyword (e.g., 'minimum wage', 'healthcare')")
            with col2:
                year = st.selectbox("Year", [0] + list(range(2026,2020,-1)), format_func=lambda x: "All" if x==0 else str(x))
            if keyword:
                with st.spinner("Searching..."):
                    results = search_bills(keyword, api_key, year)
                    if results:
                        bill_titles = [f"{r['identifier']}: {r['title'][:80]}..." for r in results]
                        selected_idx = st.selectbox("Select a bill", range(len(bill_titles)), format_func=lambda i: bill_titles[i])
                        selected = results[selected_idx]
                        if st.button("📥 Load Selected Bill", type="primary"):
                            bill = get_bill_by_number(selected["identifier"], api_key)
                            if "error" in bill:
                                st.error(bill["error"])
                            else:
                                st.session_state.current_bill = bill
                                st.session_state.bill_text = None
                                st.success(f"Loaded {bill['identifier']}")
                    else:
                        st.warning("No substantive bills found.")
        
        if st.session_state.get("current_bill"):
            bill = st.session_state.current_bill
            st.divider()
            st.success(f"**{bill['identifier']}** – {bill['title']}")
            cols = st.columns(4)
            cols[0].metric("Session", bill.get('session','N/A'))
            cols[1].metric("Status", str(bill.get('status','N/A')).capitalize())
            cols[2].metric("Updated", bill.get('updated_at','')[:10] if bill.get('updated_at') else 'N/A')
            cols[3].metric("Chamber", bill.get('chamber','N/A').capitalize())
            with st.expander("📄 Bill Abstract & Versions"):
                st.markdown(f"**Abstract:** {bill.get('abstract','No abstract.')}")
                if bill.get('versions'):
                    st.markdown("**Versions:**")
                    for v in bill['versions'][:3]:
                        if v.get('url'):
                            st.markdown(f"- [{v.get('title','Version')}]({v['url']})")
            if bill.get('text_url'):
                col1, col2 = st.columns([1,1])
                with col1:
                    if st.button("📄 Extract Full Text", use_container_width=True):
                        with st.spinner("Extracting..."):
                            st.session_state.bill_text = extract_bill_text(bill['text_url'])
                with col2:
                    if st.button("🔔 Monitor This Bill", use_container_width=True):
                        if st.session_state.get("bill_text"):
                            text_hash = compute_text_hash(st.session_state.bill_text)
                            try:
                                supabase.table("monitored_bills").insert({
                                    "company_id": company_id,
                                    "bill_id": bill['identifier'],
                                    "bill_title": bill['title'],
                                    "last_text_hash": text_hash,
                                    "last_checked": datetime.utcnow().isoformat()
                                }).execute()
                                log_action(supabase, "monitor_bill", {"bill_id": bill['identifier']})
                                st.success("Now monitoring this bill.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to monitor: {e}")
                        else:
                            st.warning("Extract bill text first.")
            if st.session_state.get("bill_text"):
                with st.expander("📜 Full Bill Text (preview)"):
                    st.text_area("Bill content", st.session_state.bill_text[:5000], height=200, disabled=True)
                    if len(st.session_state.bill_text) > 5000:
                        st.caption(f"*Showing first 5,000 of {len(st.session_state.bill_text):,} characters*")
    
    with tab_handbook:
        st.subheader("Employee Handbook")
        handbook_text = st.text_area(
            "Handbook content",
            value=st.session_state.get("handbook_content", handbook_content),
            height=400,
            placeholder="Paste your employee handbook or policy text here...",
            key="handbook_text_area"
        )
        
        # Version note - defined BEFORE save button
        version_note = st.text_input("Version note (optional)", placeholder="e.g., Updated PTO policy")
        
        col_save, _ = st.columns([1,3])
        with col_save:
            if st.button("💾 Save Handbook", type="primary", use_container_width=True):
                if handbook_text.strip():
                    try:
                        # Upsert handbook
                        existing = supabase.table("handbooks") \
                            .select("id") \
                            .eq("company_id", company_id) \
                            .execute()
                        
                        if existing.data:
                            supabase.table("handbooks") \
                                .update({
                                    "content": handbook_text,
                                    "updated_at": datetime.utcnow().isoformat()
                                }) \
                                .eq("company_id", company_id) \
                                .execute()
                        else:
                            supabase.table("handbooks").insert({
                                "company_id": company_id,
                                "content": handbook_text
                            }).execute()
                        
                        # Save version with note
                        supabase.table("handbook_versions").insert({
                            "company_id": company_id,
                            "content": handbook_text,
                            "version_note": version_note
                        }).execute()
                        
                        log_action(supabase, "handbook_saved", {"note": version_note})
                        st.success("Handbook saved.")
                        st.session_state.handbook_content = handbook_text
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
        
        st.divider()
        st.subheader("📚 Version History")
        try:
            versions = supabase.table("handbook_versions") \
                .select("*") \
                .eq("company_id", company_id) \
                .order("created_at", desc=True) \
                .limit(10) \
                .execute().data
            for v in versions:
                with st.expander(f"Version from {v['created_at'][:16]} – {v.get('version_note', 'No note')}"):
                    st.text(v['content'][:1000] + ("..." if len(v['content']) > 1000 else ""))
                    if st.button("Restore this version", key=f"restore_{v['id']}"):
                        try:
                            supabase.table("handbooks") \
                                .update({
                                    "content": v['content'],
                                    "updated_at": datetime.utcnow().isoformat()
                                }) \
                                .eq("company_id", company_id) \
                                .execute()
                            st.success("Handbook restored.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Restore failed: {e}")
        except Exception as e:
            st.warning(f"Could not load versions: {e}")
    
    with tab_analysis:
        st.subheader("Compliance Gap Analysis")
        if st.session_state.get("bill_text") and st.session_state.get("handbook_content"):
            if st.button("🔬 Run Full Analysis", type="primary"):
                with st.spinner("🧠 Analyzing semantic alignment..."):
                    result = ComplianceAnalyzer.semantic_similarity(
                        st.session_state.handbook_content,
                        st.session_state.bill_text
                    )
                    st.session_state.analysis_result = result
            if st.session_state.get("analysis_result"):
                res = st.session_state.analysis_result
                if "error" in res:
                    st.error(res["error"])
                else:
                    sim = res.get("overall_similarity", 0)
                    risk = res.get("compliance_risk", "UNKNOWN")
                    risk_color = {"LOW":"green","MEDIUM":"orange","HIGH":"red"}.get(risk,"gray")
                    col1, col2, col3 = st.columns([1,1,2])
                    col1.metric("Similarity", f"{sim:.1%}")
                    col2.markdown(f"**Risk**  \n:<span style='color:{risk_color};font-size:1.8rem;font-weight:bold'>{risk}</span>", unsafe_allow_html=True)
                    col3.info(f"**Method:** {res.get('method','N/A')}")
                    if res.get("sentence_analysis"):
                        with st.expander("⚠️ High‑Risk Sentences", expanded=True):
                            risky = [s for s in res["sentence_analysis"] if s["risk"]=="HIGH"][:7]
                            if risky:
                                for r in risky:
                                    st.markdown(f"**Policy:** {r['policy_sentence']}")
                                    st.markdown(f"→ **Bill:** {r['bill_sentence'][:200]}...")
                                    st.markdown(f"*Similarity: {r['similarity']:.1%}*")
                                    st.divider()
                            else:
                                st.success("No high‑risk sentences detected.")
                    with st.expander("📝 Line‑by‑Line Difference"):
                        diff = ComplianceAnalyzer.classic_diff(
                            st.session_state.handbook_content,
                            st.session_state.bill_text
                        )
                        for change in diff[:10]:
                            st.markdown(f"**Change Type:** {change['type']}")
                            if "old" in change:
                                st.code("\n".join(change["old"]), language="text")
                            if "new" in change:
                                st.code("\n".join(change["new"]), language="text")
                            st.divider()
                    if st.session_state.get("current_bill"):
                        pdf_bytes = generate_pdf_report(
                            st.session_state.current_bill,
                            st.session_state.handbook_content,
                            res
                        )
                        st.download_button(
                            "📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
        else:
            if not st.session_state.get("bill_text"):
                st.info("👆 Extract a bill's full text first.")
            if not st.session_state.get("handbook_content"):
                st.info("👆 Save your employee handbook.")

def render_policy_governance(supabase: Client, company_id: str):
    st.title("📋 Policy Governance")
    tab_redliner, tab_timeline = st.tabs(["📝 Policy Redliner", "📅 Timeline Projector"])
    with tab_redliner:
        st.subheader("Statutory vs. Internal Policy Comparator")
        col1, col2 = st.columns(2)
        with col1:
            policy_text = st.text_area("Current policy text", height=250, key="redliner_policy")
        with col2:
            statute_text = st.text_area("New statutory text", height=250, key="redliner_statute")
        if st.button("Run Gap Analysis", key="run_redliner"):
            if policy_text and statute_text:
                sents1 = tokenize_sentences(policy_text)
                sents2 = tokenize_sentences(statute_text)
                differ = difflib.SequenceMatcher(None, sents1, sents2)
                for tag, i1, i2, j1, j2 in differ.get_opcodes():
                    if tag == 'equal':
                        for sent in sents1[i1:i2]:
                            with st.expander(f"✅ {sent[:60]}...", expanded=False):
                                st.caption(sent)
                    elif tag == 'replace':
                        st.markdown("**:orange[🔄 MODIFIED]**")
                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.markdown("**Policy (old)**")
                            for sent in sents1[i1:i2]:
                                st.error(sent)
                        with col_right:
                            st.markdown("**Statute (new)**")
                            for sent in sents2[j1:j2]:
                                st.success(sent)
                    elif tag == 'delete':
                        st.markdown("**:red[❌ REMOVED]**")
                        for sent in sents1[i1:i2]:
                            st.error(sent)
                    elif tag == 'insert':
                        st.markdown("**:green[➕ ADDED]**")
                        for sent in sents2[j1:j2]:
                            st.success(sent)
            else:
                st.warning("Both fields required.")
    with tab_timeline:
        st.subheader("Compliance Work‑Back Schedule")
        effective_date = st.date_input("Statutory Effective Date", value=date.today() + timedelta(days=90))
        if st.button("Generate Timeline", key="gen_timeline"):
            milestones = [
                ("Audit of Current Policies", 90),
                ("Drafting of Policy Updates", 60),
                ("Legal Counsel Review", 45),
                ("Executive Sign-off", 30),
                ("Manager Training", 15),
                ("Employee Notification", 7),
                ("Go-Live / Effective Date", 0)
            ]
            data = []
            for task, days in milestones:
                due = effective_date - timedelta(days=days)
                data.append({
                    "Milestone": task,
                    "Due Date": due.strftime('%Y-%m-%d'),
                    "Lead Days": days,
                    "Phase": "Preparation" if days > 0 else "Execution"
                })
            df = pd.DataFrame(data)
            st.table(df)
            csv = df.to_csv(index=False)
            st.download_button("📥 Export Timeline (CSV)", csv, "compliance_timeline.csv")

def render_tuition_module(supabase: Client, company_id: str):
    st.title("💰 Tuition Reimbursement Auditor")
    st.caption(f"IRS §127 limit: ${Config.IRS_SEC_127_LIMIT:,.2f} | Min tenure: {Config.MIN_TENURE_YEARS} year")
    with st.expander("📥 Download CSV Template"):
        template_df = pd.DataFrame(columns=['EmployeeID','TenureYears','RequestAmount','DegreeProgram'])
        st.download_button("Download Template", data=template_df.to_csv(index=False),
                           file_name="tuition_audit_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload Request CSV", type=['csv'])
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
            (df['TenureYears'] < Config.MIN_TENURE_YEARS),
            (df['RequestAmount'] > Config.IRS_SEC_127_LIMIT)
        ]
        status_choices = ['Ineligible', 'Eligible (Taxable)']
        basis_choices = [f"Tenure < {Config.MIN_TENURE_YEARS}yr", f"Exceeds IRS ${Config.IRS_SEC_127_LIMIT:,.0f}"]
        df['Decision'] = np.select(conditions, status_choices, default='Eligible (Tax‑Free)')
        df['Basis'] = np.select(conditions, basis_choices, default='Meets criteria')
        df['Taxable_Amount'] = np.where(df['Decision'] == 'Eligible (Taxable)',
                                         df['RequestAmount'] - Config.IRS_SEC_127_LIMIT, 0.0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Processed", len(df))
        col2.metric("Tax‑Free", len(df[df['Decision'] == 'Eligible (Tax‑Free)']))
        col3.metric("Taxable", len(df[df['Decision'] == 'Eligible (Taxable)']))
        col4.metric("Total Exposure", f"${df['Taxable_Amount'].sum():,.2f}")
        st.dataframe(df[['EmployeeID','RequestAmount','Decision','Taxable_Amount','Basis']],
                     use_container_width=True, hide_index=True)
        st.download_button("📄 Download Audit Report (CSV)",
                           data=df.to_csv(index=False),
                           file_name=f"tuition_audit_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        layout=Config.LAYOUT,
        initial_sidebar_state="expanded",
        page_icon="🛡️"
    )
    st.markdown("""
    <style>
        .stApp { background-color: #f8fafc; }
        .stButton>button { border-radius: 8px; font-weight: 500; }
        .stMetric { background-color: white; padding: 1rem; border-radius: 0.75rem; box-shadow: 0 2px 5px rgba(0,0,0,0.03); }
        div[data-testid="stExpander"] { background-color: white; border-radius: 0.5rem; border: 1px solid #e2e8f0; }
        .stAlert { border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

    supabase = get_supabase_client()
    if not supabase:
        st.error("Supabase connection failed. Check your secrets.")
        st.stop()

    company_id = get_or_create_default_company(supabase)
    if not company_id:
        st.error("Company setup failed.")
        st.stop()

    api_key = st.secrets.get("OPENSTATES_API_KEY") if hasattr(st, 'secrets') else None

    st.sidebar.markdown("## 🛡️ HR Sentinel")
    st.sidebar.caption("Workforce Compliance Infrastructure")

    # ----- MODULES DICTIONARY – all functions receive dependencies via lambdas -----
    modules = {
        "Dashboard": lambda: render_dashboard(supabase, company_id),
        "Workforce": lambda: render_employees(supabase, company_id),
        "Requirements": lambda: render_requirements(supabase, company_id),
        "Compliance Tracking": lambda: render_compliance_tracking(supabase, company_id),
        "Legislative Intelligence": lambda: render_legislative_intelligence(supabase, company_id, api_key),
        "Policy Governance": lambda: render_policy_governance(supabase, company_id),
        "Tuition Auditor": lambda: render_tuition_module(supabase, company_id),
    }

    page = st.sidebar.radio("Modules", list(modules.keys()))

    st.sidebar.divider()
    with st.sidebar.expander("🔧 System Status", expanded=False):
        st.write(f"**Supabase:** {'✅' if supabase else '❌'}")
        st.write(f"**Open States API:** {'✅' if api_key else '❌'}")
        st.write(f"**PDF extraction:** {'✅' if PDF_SUPPORT else '❌'}")
        st.write(f"**Semantic AI:** {'✅' if SEMANTIC_AVAILABLE else '⚠️ TF‑IDF'}")
        st.write(f"**PDF Reports:** {'✅' if PDF_REPORT_AVAILABLE else '❌'}")

    # Initialize session state for bill data if not present
    if "current_bill" not in st.session_state:
        st.session_state.current_bill = None
    if "bill_text" not in st.session_state:
        st.session_state.bill_text = None
    if "handbook_content" not in st.session_state:
        st.session_state.handbook_content = ""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    # Execute the selected module
    modules[page]()

if __name__ == "__main__":
    main()
