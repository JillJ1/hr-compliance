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
        self.set_fill_color(240, 240, 240)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(2)
    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        try:
            self.multi_cell(0, 5, text.encode('latin-1', errors='replace').decode('latin-1'))
        except:
            self.multi_cell(0, 5, "[Text could not be rendered]")
        self.ln()
    def risk_meter(self, risk_level):
        self.set_font('Arial', 'B', 10)
        color = {"LOW": (0, 128, 0), "MEDIUM": (255, 165, 0), "HIGH": (255, 0, 0)}.get(risk_level, (0, 0, 0))
        self.set_text_color(*color)
        self.cell(0, 6, f"Compliance Risk: {risk_level}", 0, 1)
        self.set_text_color(0, 0, 0)

def generate_pdf_report(bill_data: Dict, handbook_text: str, analysis: Dict) -> bytes:
    if not PDF_REPORT_AVAILABLE:
        return b"PDF generation requires fpdf library."
    pdf = ComplianceReportPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Compliance Gap Analysis', 0, 1, 'C')
    pdf.ln(5)
    pdf.chapter_title('Bill Information')
    bill_info = f"Bill: {bill_data.get('identifier', 'N/A')}\n"
    bill_info += f"Title: {bill_data.get('title', 'N/A')}\n"
    bill_info += f"Session: {bill_data.get('session', 'N/A')}\n"
    bill_info += f"Status: {bill_data.get('status', 'N/A')}\n"
    bill_info += f"Last Updated: {bill_data.get('updated_at', 'N/A')}"
    pdf.chapter_body(bill_info)
    pdf.chapter_title('Risk Assessment')
    if "overall_similarity" in analysis:
        sim = analysis["overall_similarity"]
        risk = analysis.get("compliance_risk", "UNKNOWN")
        pdf.risk_meter(risk)
        pdf.chapter_body(f"Overall Semantic Similarity: {sim:.1%}")
    else:
        pdf.chapter_body("Semantic analysis not available.")
    if analysis.get("sentence_analysis"):
        risky = [s for s in analysis["sentence_analysis"] if s.get("risk") == "HIGH"]
        if risky:
            pdf.chapter_title('High‑Risk Policy Gaps')
            for i, r in enumerate(risky[:10]):
                pdf.set_font('Arial', 'B', 9)
                pdf.cell(0, 5, f"{i+1}. Policy Sentence:", 0, 1)
                pdf.set_font('Arial', '', 9)
                pdf.multi_cell(0, 4, r['policy_sentence'][:300])
                pdf.set_font('Arial', 'I', 9)
                pdf.cell(0, 4, f"Similarity: {r['similarity']:.1%}", 0, 1)
                pdf.ln(2)
    pdf.set_y(-20)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, f'Generated by HR Sentinel on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
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

# -------------------- COMPLIANCE INTELLIGENCE - PLACEHOLDER VERSION --------------------
def render_compliance_intelligence(supabase: Client, company_id: str, api_key: str = None):
    """🚀 Compliance Intelligence with PLACEHOLDER data - APIs to be connected later"""
    st.title("⚖️ Compliance Intelligence")
    
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
            st.session_state.handbook_content = handbook_content
    except Exception as e:
        st.warning(f"Could not load handbook: {e}")
    
    # Main tabs - All placeholders ready for real APIs
    tab_osha, tab_settlements, tab_regs, tab_dol, tab_handbook = st.tabs([
        "🚨 OSHA Citations", 
        "⚖️ Settlements",
        "📋 Federal Regs",
        "📝 DOL Guidance",
        "📘 Handbook"
    ])
    
    with tab_osha:
        st.subheader("OSHA Citation Lookup")
        st.caption("🔍 PLACEHOLDER - OSHA API connection pending")
        
        col1, col2 = st.columns(2)
        with col1:
            naics = st.text_input("NAICS Code (optional)", 
                                 placeholder="e.g., 541611, 238160",
                                 help="Find your NAICS code at naics.com")
        with col2:
            state = st.selectbox("State", 
                               ["", "OH", "CA", "TX", "NY", "FL", "IL", "PA"],
                               index=0)
        
        if st.button("🔍 Search OSHA Citations", type="primary"):
            with st.spinner("Fetching OSHA enforcement data..."):
                # TODO: Replace with actual OSHA API
                # API Endpoint: https://www.osha.gov/enforcement/data/inspections
                # Documentation: https://www.osha.gov/developers
                
                st.info("📌 **OSHA API Integration Coming Soon**")
                st.markdown("""
                **Sample OSHA Citation Data (For Demonstration):**
                
                🚨 **Fall Protection - General Requirements**  
                - 📊 156 citations in your industry  
                - 💰 Average penalty: $4,321  
                - 📋 Standard: 1926.501  
                ---
                
                🚨 **Hazard Communication**  
                - 📊 98 citations in your industry  
                - 💰 Average penalty: $3,890  
                - 📋 Standard: 1910.1200  
                ---
                
                🚨 **Respiratory Protection**  
                - 📊 67 citations in your industry  
                - 💰 Average penalty: $2,456  
                - 📋 Standard: 1910.134  
                ---
                """)
        
        log_action(supabase, "osha_search_placeholder", {
            "naics": naics,
            "state": state,
            "timestamp": datetime.now().isoformat()
        })
    
    with tab_settlements:
        st.subheader("Recent FLSA Settlements")
        st.caption("⚖️ PLACEHOLDER - CourtListener API connection pending")
        
        days = st.slider("Look back period (days)", 7, 90, 30)
        
        if st.button("🔄 Refresh Settlements"):
            with st.spinner("Fetching court cases..."):
                # TODO: Replace with actual CourtListener API
                # API Endpoint: https://www.courtlistener.com/api/rest/v3/
                # Get free key: https://www.courtlistener.com/api/register/
                
                st.info("📌 **CourtListener API Integration Coming Soon**")
                st.markdown("""
                **Sample FLSA Settlement Data (For Demonstration):**
                
                💰 **Martinez v. Amazon Logistics - $8,400,000**  
                - **Issue:** Off-the-clock work  
                - **Class Size:** 4,200 employees  
                - **Filed:** 2024-02-15  
                
                💰 **Johnson v. Walmart Inc. - $3,200,000**  
                - **Issue:** Missed meal breaks  
                - **Class Size:** 1,850 employees  
                - **Filed:** 2024-02-10  
                
                💰 **Williams v. Target Corp - $2,100,000**  
                - **Issue:** Overtime misclassification  
                - **Class Size:** 950 employees  
                - **Filed:** 2024-02-05  
                
                💰 **Brown v. Starbucks - $1,800,000**  
                - **Issue:** Remote work time tracking  
                - **Class Size:** 3,100 employees  
                - **Filed:** 2024-01-28  
                """)
    
    with tab_regs:
        st.subheader("📋 Federal Register - New Regulations")
        st.caption("🇺🇸 PLACEHOLDER - Federal Register API connection pending")
        
        days_reg = st.selectbox("Time period", [7, 30, 60, 90], index=1, 
                               format_func=lambda x: f"Last {x} days")
        
        with st.spinner("Checking federal regulations..."):
            # TODO: Replace with actual Federal Register API
            # API Endpoint: https://www.federalregister.gov/api/v1/
            # Documentation: https://www.federalregister.gov/developers/api/v1
            
            st.info("📌 **Federal Register API Integration Coming Soon**")
            st.markdown("""
            **Sample Federal Register Data (For Demonstration):**
            
            📌 **Improving Protections for Workers in Temporary Agricultural Employment**  
            - **Agency:** Wage and Hour Division  
            - **Publication Date:** 2024-02-16  
            - **Effective Date:** 2024-03-18  
            - **Document #:** 2024-12345  
            
            📌 **Equal Employment Opportunity for Individuals with Disabilities**  
            - **Agency:** EEOC  
            - **Publication Date:** 2024-02-10  
            - **Effective Date:** 2024-04-01  
            - **Document #:** 2024-67890  
            
            📌 **Updating the Davis-Bacon and Related Acts Regulations**  
            - **Agency:** Wage and Hour Division  
            - **Publication Date:** 2024-02-05  
            - **Effective Date:** 2024-05-15  
            - **Document #:** 2024-54321  
            """)
    
    with tab_dol:
        st.subheader("📝 DOL Opinion Letters")
        st.caption("Official Wage and Hour Division guidance - PLACEHOLDER")
        
        query = st.text_input("Search DOL guidance", 
                             placeholder="e.g., remote work, overtime, meal breaks",
                             key="dol_search")
        
        if query and len(query) > 2:
            with st.spinner("Searching DOL opinion letters..."):
                # TODO: Replace with actual DOL API
                # API Endpoint: https://api.dol.gov/V1/Wage_Hour
                # Get free key: https://developer.dol.gov/signup
                
                st.info("📌 **DOL API Integration Coming Soon**")
                st.markdown("""
                **Sample DOL Opinion Letters (For Demonstration):**
                
                📄 **Opinion Letter: FLSA2024-12** - March 15, 2024
                
                **Question:** Must employees be compensated for mandatory virtual training outside work hours?
                
                **Answer:** Yes. If the training is required and directly related to the employee's job, 
                it constitutes "hours worked" under the FLSA and must be compensated.
                
                ---
                
                📄 **Opinion Letter: FLSA2024-08** - February 22, 2024
                
                **Question:** Are remote employees entitled to overtime for responding to emails after hours?
                
                **Answer:** Yes. Time spent working outside of scheduled hours, including responding to 
                emails or calls, must be counted as hours worked for overtime purposes.
                
                ---
                
                📄 **Opinion Letter: FLSA2024-03** - January 10, 2024
                
                **Question:** Does the FLSA require compensation for time spent in security screenings?
                
                **Answer:** Under the Portal-to-Portal Act, time spent in security screenings is compensable 
                if it is an integral and indispensable part of the employee's principal activities.
                """)
    
    with tab_handbook:
        st.subheader("📘 Employee Handbook")
        st.caption("Version control with audit trail")
        
        handbook_text = st.text_area(
            "Handbook content",
            value=st.session_state.get("handbook_content", handbook_content),
            height=400,
            placeholder="Paste your employee handbook or policy text here...",
            key="handbook_text_area_final"
        )
        
        # ✅ Version note - DEFINED BEFORE save button
        version_note = st.text_input("Version note (optional)", 
                                    placeholder="e.g., Updated remote work policy, Added AI usage policy")
        
        col_save, _ = st.columns([1, 3])
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
                        
                        # ✅ Save version with note - version_note is defined
                        supabase.table("handbook_versions").insert({
                            "company_id": company_id,
                            "content": handbook_text,
                            "version_note": version_note if version_note else "No version note"
                        }).execute()
                        
                        log_action(supabase, "handbook_saved", {"note": version_note})
                        st.success("✅ Handbook saved successfully")
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
            
            if versions:
                for v in versions:
                    created_at = v.get('created_at', '')
                    if len(created_at) > 10:
                        created_at = created_at[:10]
                    
                    with st.expander(f"📅 {created_at} – {v.get('version_note', 'No note')}"):
                        st.text(v['content'][:1000] + ("..." if len(v['content']) > 1000 else ""))
                        if st.button("↩️ Restore this version", key=f"restore_{v['id']}"):
                            try:
                                supabase.table("handbooks") \
                                    .update({
                                        "content": v['content'],
                                        "updated_at": datetime.utcnow().isoformat()
                                    }) \
                                    .eq("company_id", company_id) \
                                    .execute()
                                st.success("✅ Handbook restored")
                                st.session_state.handbook_content = v['content']
                                st.rerun()
                            except Exception as e:
                                st.error(f"Restore failed: {e}")
            else:
                st.info("No version history yet. Save your first handbook version.")
        except Exception as e:
            st.warning(f"Could not load versions: {e}")

def render_policy_governance(supabase: Client, company_id: str):
    st.title("📋 Policy Governance")
    tab_redliner, tab_timeline = st.tabs(["📝 Policy Redliner", "📅 Timeline Projector"])
    
    with tab_redliner:
        st.subheader("Statutory vs. Internal Policy Comparator")
        col1, col2 = st.columns(2)
        with col1:
            policy_text = st.text_area("Current policy text", height=250, 
                                      placeholder="Paste your current policy...",
                                      key="redliner_policy")
        with col2:
            statute_text = st.text_area("New statutory text", height=250, 
                                       placeholder="Paste the new law/regulation...",
                                       key="redliner_statute")
        
        if st.button("Run Gap Analysis", key="run_redliner", type="primary"):
            if policy_text and statute_text:
                sents1 = tokenize_sentences(policy_text)
                sents2 = tokenize_sentences(statute_text)
                differ = difflib.SequenceMatcher(None, sents1, sents2)
                
                st.subheader("📊 Gap Analysis Results")
                
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
                            st.error(f"~~{sent}~~")
                    elif tag == 'insert':
                        st.markdown("**:green[➕ ADDED]**")
                        for sent in sents2[j1:j2]:
                            st.success(sent)
            else:
                st.warning("Both policy text and statutory text are required.")
    
    with tab_timeline:
        st.subheader("Compliance Work‑Back Schedule")
        st.caption("Plan your compliance implementation timeline")
        
        effective_date = st.date_input("Statutory Effective Date", 
                                      value=date.today() + timedelta(days=90),
                                      help="When does the new law take effect?")
        
        if st.button("Generate Timeline", key="gen_timeline", type="primary"):
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
                status = "⚠️ Due Soon" if 0 < days <= 15 else "📅 Planned"
                if days == 0:
                    status = "🎯 Effective Date"
                elif due < date.today():
                    status = "❌ Overdue"
                    
                data.append({
                    "Milestone": task,
                    "Due Date": due.strftime('%Y-%m-%d'),
                    "Days Before Effective": days,
                    "Status": status
                })
            
            df = pd.DataFrame(data)
            st.table(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Export Timeline (CSV)",
                data=csv,
                file_name=f"compliance_timeline_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

def render_tuition_module(supabase: Client, company_id: str):
    st.title("💰 Tuition Reimbursement Auditor")
    st.caption(f"IRS §127 limit: ${Config.IRS_SEC_127_LIMIT:,.2f} | Min tenure: {Config.MIN_TENURE_YEARS} year")
    
    with st.expander("📥 Download CSV Template"):
        template_df = pd.DataFrame(columns=['EmployeeID', 'TenureYears', 'RequestAmount', 'DegreeProgram'])
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            "Download Template",
            data=csv_template,
            file_name="tuition_audit_template.csv",
            mime="text/csv"
        )
    
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
        
        st.dataframe(
            df[['EmployeeID', 'RequestAmount', 'Decision', 'Taxable_Amount', 'Basis']],
            use_container_width=True,
            hide_index=True
        )
        
        st.download_button(
            "📄 Download Audit Report (CSV)",
            data=df.to_csv(index=False),
            file_name=f"tuition_audit_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

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
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0px 0px; padding: 8px 16px; background-color: white; }
        .stTabs [aria-selected="true"] { background-color: #e6f7ff; border-bottom: 2px solid #1890ff; }
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

    st.sidebar.markdown("## 🛡️ HR Sentinel")
    st.sidebar.caption("Workforce Compliance Infrastructure")

    # ----- MODULES DICTIONARY – Open States REMOVED -----
    modules = {
        "Dashboard": lambda: render_dashboard(supabase, company_id),
        "Workforce": lambda: render_employees(supabase, company_id),
        "Requirements": lambda: render_requirements(supabase, company_id),
        "Compliance Tracking": lambda: render_compliance_tracking(supabase, company_id),
        "Compliance Intelligence": lambda: render_compliance_intelligence(supabase, company_id, None),
        "Policy Governance": lambda: render_policy_governance(supabase, company_id),
        "Tuition Auditor": lambda: render_tuition_module(supabase, company_id),
    }

    page = st.sidebar.radio("Modules", list(modules.keys()))

    st.sidebar.divider()
    with st.sidebar.expander("🔧 System Status", expanded=False):
        st.write(f"**Supabase:** {'✅' if supabase else '❌'}")
        st.write(f"**PDF extraction:** {'✅' if PDF_SUPPORT else '❌'}")
        st.write(f"**Semantic AI:** {'✅' if SEMANTIC_AVAILABLE else '⚠️ TF‑IDF'}")
        st.write(f"**PDF Reports:** {'✅' if PDF_REPORT_AVAILABLE else '❌'}")
        st.write(f"**Compliance APIs:** ⏳ Coming Soon")
        st.caption("OSHA, CourtListener, Federal Register, DOL")

    # Initialize session state
    if "handbook_content" not in st.session_state:
        st.session_state.handbook_content = ""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    # Execute the selected module
    modules[page]()

if __name__ == "__main__":
    main()
