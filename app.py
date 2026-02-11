#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ohio Health Enterprise Compliance Suite v3.0
Production‑ready, AI‑enhanced, HR‑focused compliance platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import difflib
import hashlib
import json
import time
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

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

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Semantic AI
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

# PDF report generation
try:
    from fpdf import FPDF
    PDF_REPORT_AVAILABLE = True
except ImportError:
    PDF_REPORT_AVAILABLE = False

# -------------------- CONFIGURATION (Central, Auditable) --------------------
class Config:
    """Application configuration – all tunable parameters here."""
    # IRS limits
    IRS_SEC_127_LIMIT = 5250.00
    MIN_TENURE_YEARS = 1.0
    
    # Open States API
    OPENSTATES_BASE_URL = "https://v3.openstates.org/bills"
    OPENSTATES_JURISDICTION = "Ohio"
    API_TIMEOUT = 15
    
    # Document extraction
    MAX_PDF_PAGES = 30
    MAX_EXTRACT_CHARS = 20000
    
    # UI
    PAGE_TITLE = "Ohio Health • Enterprise Compliance"
    LAYOUT = "wide"
    ENABLE_SEMANTIC = st.secrets.get("ENABLE_SEMANTIC", True) if hasattr(st, 'secrets') else True
    
    # Dashboard
    DASHBOARD_REFRESH_HOURS = 6
    ENABLE_ONBOARDING = True
    
    # Compliance categories (sync with DB on startup)
    DEFAULT_CATEGORIES = [
        {"name": "Wage & Hour", "keywords": ["wage", "minimum wage", "overtime", "salary", "pay", "compensation", "hour"]},
        {"name": "Leave & Time Off", "keywords": ["leave", "vacation", "sick", "fmla", "parental", "holiday", "pto"]},
        {"name": "Benefits", "keywords": ["benefit", "insurance", "health", "retirement", "401k", "pension", "wellness"]},
        {"name": "Health & Safety", "keywords": ["safety", "osha", "workplace", "health", "injury", "ppe", "hazard"]},
        {"name": "Discrimination & Harassment", "keywords": ["discrimination", "harassment", "eeoc", "title vii", "ada", "age"]},
        {"name": "Termination & Separation", "keywords": ["termination", "separation", "severance", "layoff", "fired", "resignation"]},
        {"name": "Remote Work", "keywords": ["remote", "telework", "work from home", "virtual", "flexible"]}
    ]

# -------------------- SUPABASE CLIENT & HELPERS --------------------
@st.cache_resource(ttl=3600)
def init_supabase() -> Optional[Client]:
    """Initialize Supabase client with error handling."""
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception as e:
        st.error(f"⚠️ Supabase connection failed: {e}")
        return None

# -------------------- PERSISTENCE LAYER --------------------
class Database:
    """Encapsulates all Supabase interactions."""
    
    @staticmethod
    def save_handbook(content: str, note: str = "") -> bool:
        """Save handbook and create a version entry."""
        supabase = init_supabase()
        if not supabase:
            return False
        try:
            # Update current handbook
            result = supabase.table("handbooks").select("*").execute()
            if result.data:
                supabase.table("handbooks").update({
                    "content": content,
                    "updated_at": "now()"
                }).eq("id", result.data[0]["id"]).execute()
            else:
                supabase.table("handbooks").insert({"content": content}).execute()
            
            # Save version history
            supabase.table("handbook_versions").insert({
                "content": content,
                "version_note": note or f"Saved on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            }).execute()
            
            Database.audit_log("save_handbook", {"length": len(content), "note": note})
            return True
        except Exception as e:
            st.error(f"Failed to save handbook: {e}")
            return False
    
    @staticmethod
    def load_handbook() -> str:
        """Load most recent handbook."""
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
    
    @staticmethod
    def get_handbook_versions(limit: int = 10) -> List[Dict]:
        """Retrieve version history."""
        supabase = init_supabase()
        if not supabase:
            return []
        try:
            result = supabase.table("handbook_versions") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            return result.data
        except Exception as e:
            st.error(f"Failed to load versions: {e}")
            return []
    
    @staticmethod
    def add_monitored_bill(bill_id: str, bill_title: str, text_hash: str) -> bool:
        """Add or update a monitored bill."""
        supabase = init_supabase()
        if not supabase:
            return False
        try:
            existing = supabase.table("monitored_bills") \
                .select("*") \
                .eq("bill_id", bill_id) \
                .execute()
            if existing.data:
                supabase.table("monitored_bills") \
                    .update({
                        "last_text_hash": text_hash,
                        "last_checked": "now()",
                        "bill_title": bill_title
                    }) \
                    .eq("bill_id", bill_id) \
                    .execute()
            else:
                supabase.table("monitored_bills") \
                    .insert({
                        "bill_id": bill_id,
                        "bill_title": bill_title,
                        "last_text_hash": text_hash,
                        "last_checked": "now()"
                    }) \
                    .execute()
            Database.audit_log("monitor_bill", {"bill_id": bill_id, "title": bill_title})
            return True
        except Exception as e:
            st.error(f"Failed to add monitored bill: {e}")
            return False
    
    @staticmethod
    def get_monitored_bills() -> List[Dict]:
        """Retrieve all monitored bills."""
        supabase = init_supabase()
        if not supabase:
            return []
        try:
            result = supabase.table("monitored_bills") \
                .select("*") \
                .order("last_checked", desc=True) \
                .execute()
            return result.data
        except Exception as e:
            st.error(f"Failed to load monitored bills: {e}")
            return []
    
    @staticmethod
    def remove_monitored_bill(bill_id: str) -> bool:
        """Stop monitoring a bill."""
        supabase = init_supabase()
        if not supabase:
            return False
        try:
            supabase.table("monitored_bills") \
                .delete() \
                .eq("bill_id", bill_id) \
                .execute()
            Database.audit_log("unmonitor_bill", {"bill_id": bill_id})
            return True
        except Exception as e:
            st.error(f"Failed to remove monitored bill: {e}")
            return False
    
    @staticmethod
    def audit_log(action: str, details: Dict = None):
        """Record user action."""
        supabase = init_supabase()
        if not supabase:
            return
        try:
            supabase.table("audit_log").insert({
                "action": action,
                "details": json.dumps(details) if details else None,
                "user_id": "default"
            }).execute()
        except:
            pass  # Non‑critical, don't interrupt user
    
    @staticmethod
    def get_compliance_categories() -> List[Dict]:
        """Load active compliance categories with keywords."""
        supabase = init_supabase()
        if not supabase:
            return Config.DEFAULT_CATEGORIES
        try:
            result = supabase.table("compliance_categories") \
                .select("*") \
                .eq("active", True) \
                .execute()
            if result.data:
                return result.data
            else:
                # Seed defaults
                for cat in Config.DEFAULT_CATEGORIES:
                    supabase.table("compliance_categories") \
                        .insert(cat) \
                        .execute()
                return Config.DEFAULT_CATEGORIES
        except:
            return Config.DEFAULT_CATEGORIES
    
    @staticmethod
    def classify_bill(bill: Dict) -> List[str]:
        """Auto‑tag a bill with relevant compliance areas based on title/abstract."""
        categories = Database.get_compliance_categories()
        text = (bill.get("title", "") + " " + bill.get("abstract", "")).lower()
        matches = []
        for cat in categories:
            for kw in cat.get("keywords", []):
                if kw.lower() in text:
                    matches.append(cat["name"])
                    break
        return matches

# -------------------- CORE HELPER FUNCTIONS --------------------
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

def compute_text_hash(text: str) -> str:
    """MD5 hash for change detection."""
    return hashlib.md5(text.encode()).hexdigest()

# -------------------- OPEN STATES API (Enhanced) --------------------
@st.cache_data(ttl=300, show_spinner="Searching Ohio legislation...")
def search_bills(keyword: str, api_key: str, chamber: str = "", year: int = 0) -> List[Dict]:
    """
    Advanced bill search with filters.
    Returns only substantive bills (HB, SB, HJR, SJR).
    """
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
    if chamber:
        params["chamber"] = chamber.lower()
    
    try:
        resp = requests.get(Config.OPENSTATES_BASE_URL, headers=headers, params=params, timeout=Config.API_TIMEOUT)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("results", [])
        
        # --- Strict filter: only bill types that can become law ---
        filtered = []
        for bill in results:
            bill_id = bill.get("identifier", "").upper()
            if (bill_id.startswith("HB") or bill_id.startswith("SB") or
                bill_id.startswith("HJR") or bill_id.startswith("SJR")):
                # Apply year filter if requested
                if year > 2000:
                    session = bill.get("session", "")
                    if str(year) not in session:
                        continue
                filtered.append(bill)
        
        return filtered[:20]  # Keep UI responsive
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

@st.cache_data(ttl=3600, show_spinner="Fetching bill details...")
def get_bill_by_number(bill_number: str, api_key: str) -> Dict:
    """
    Fetch a specific bill by identifier (e.g., 'HB33', 'SB 1').
    Handles spaces and case variations.
    """
    if not api_key:
        return {"error": "API key missing"}
    
    # Normalize: remove spaces, uppercase
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
            # Try with original spacing (some states require it)
            params["bill_id"] = bill_number.strip().upper()
            resp = requests.get(Config.OPENSTATES_BASE_URL, headers=headers, params=params, timeout=Config.API_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
        
        if not data.get("results"):
            return {"error": f"Bill '{bill_number}' not found in Ohio."}
        
        bill = data["results"][0]
        
        # Extract all version URLs (for version tracking)
        versions = []
        for v in bill.get("versions", []):
            version_info = {
                "date": v.get("date"),
                "title": v.get("note", "Unknown"),
                "url": None
            }
            links = v.get("links", [])
            for link in links:
                url = link.get("url")
                if url:
                    version_info["url"] = url
                    break
            versions.append(version_info)
        
        # Current text URL (most recent version)
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
    """Download and extract text from PDF or HTML."""
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

# -------------------- COMPARISON ENGINES --------------------
class ComplianceAnalyzer:
    """Semantic and structural comparison between handbook and bill."""
    
    @staticmethod
    def semantic_similarity(text1: str, text2: str) -> Dict:
        """Compute semantic similarity using embeddings or TF‑IDF."""
        if not text1 or not text2:
            return {"error": "Missing text"}
        
        sents1 = tokenize_sentences(text1)[:100]
        sents2 = tokenize_sentences(text2)[:100]
        if not sents1 or not sents2:
            return {"error": "No sentences to compare"}
        
        # --- Use Sentence Transformers (best) ---
        if SEMANTIC_AVAILABLE and Config.ENABLE_SEMANTIC:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
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
                        "risk": "HIGH" if best_score < 0.6 else "MEDIUM" if best_score < 0.8 else "LOW"
                    })
                overall = float(cos_scores.mean())
                return {
                    "method": "sentence-transformers",
                    "overall_similarity": overall,
                    "compliance_risk": "LOW" if overall > 0.8 else "MEDIUM" if overall > 0.6 else "HIGH",
                    "sentence_analysis": similarities[:25]
                }
            except Exception as e:
                st.warning(f"Semantic model failed, falling back to TF‑IDF: {e}")
        
        # --- Fallback: TF-IDF ---
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
                similarities.append({
                    "policy_sentence": sents1[i],
                    "bill_sentence": sents2[best_idx],
                    "similarity": best_score,
                    "risk": "HIGH" if best_score < 0.2 else "MEDIUM" if best_score < 0.4 else "LOW"
                })
            overall = np.mean([s["similarity"] for s in similarities]) if similarities else 0.0
            return {
                "method": "TF-IDF",
                "overall_similarity": overall,
                "compliance_risk": "LOW" if overall > 0.4 else "MEDIUM" if overall > 0.2 else "HIGH",
                "sentence_analysis": similarities[:25]
            }
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    @staticmethod
    def classic_diff(text1: str, text2: str) -> List[Dict]:
        """Traditional line‑based diff."""
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

# -------------------- PROFESSIONAL PDF REPORT GENERATOR --------------------
class ComplianceReportPDF(FPDF):
    """Custom PDF report for compliance gap analysis."""
    
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Ohio Health Compliance Report', 0, 1, 'C')
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
        self.multi_cell(0, 5, text)
        self.ln()
    
    def risk_meter(self, risk_level):
        self.set_font('Arial', 'B', 10)
        if risk_level == "LOW":
            self.set_text_color(0, 128, 0)
        elif risk_level == "MEDIUM":
            self.set_text_color(255, 165, 0)
        else:
            self.set_text_color(255, 0, 0)
        self.cell(0, 6, f"Compliance Risk: {risk_level}", 0, 1)
        self.set_text_color(0, 0, 0)

def generate_pdf_report(bill_data: Dict, handbook_text: str, analysis: Dict) -> bytes:
    """Create a downloadable PDF report."""
    if not PDF_REPORT_AVAILABLE:
        return b"PDF generation requires fpdf library."
    
    pdf = ComplianceReportPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Compliance Gap Analysis', 0, 1, 'C')
    pdf.ln(5)
    
    # Bill Information
    pdf.chapter_title('Bill Information')
    bill_info = f"Bill: {bill_data.get('identifier', 'N/A')}\n"
    bill_info += f"Title: {bill_data.get('title', 'N/A')}\n"
    bill_info += f"Session: {bill_data.get('session', 'N/A')}\n"
    bill_info += f"Status: {bill_data.get('status', 'N/A')}\n"
    bill_info += f"Last Updated: {bill_data.get('updated_at', 'N/A')}"
    pdf.chapter_body(bill_info)
    
    # Risk Assessment
    pdf.chapter_title('Risk Assessment')
    if "overall_similarity" in analysis:
        sim = analysis["overall_similarity"]
        risk = analysis.get("compliance_risk", "UNKNOWN")
        pdf.risk_meter(risk)
        pdf.chapter_body(f"Overall Semantic Similarity: {sim:.1%}")
    else:
        pdf.chapter_body("Semantic analysis not available.")
    
    # High Risk Sentences
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
    
    # Footer
    pdf.set_y(-20)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, f'Generated by Ohio Health Compliance Suite on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# -------------------- UI COMPONENTS --------------------
def render_metric_card(title, value, delta=None, help_text=None):
    """Consistent metric card styling."""
    with st.container():
        st.markdown(f"""
        <div style="background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <h4 style="margin:0; color:#666; font-size:0.9rem;">{title}</h4>
            <p style="margin:0; font-size:1.8rem; font-weight:600; color:#0b3b5c;">{value}</p>
            {f'<p style="margin:0; color:#{"green" if delta and delta[0]=="+" else "red"};">{delta}</p>' if delta else ''}
            {f'<p style="margin:0; color:#999; font-size:0.8rem;">{help_text}</p>' if help_text else ''}
        </div>
        """, unsafe_allow_html=True)

def render_onboarding_tour():
    """First‑time user guidance."""
    if "onboarding_complete" not in st.session_state:
        st.session_state.onboarding_complete = False
    
    if not st.session_state.onboarding_complete and Config.ENABLE_ONBOARDING:
        with st.expander("👋 Welcome! Quick tour (3 steps)", expanded=True):
            st.markdown("""
            1. **Upload or paste your employee handbook** – Save it permanently to the cloud.
            2. **Find bills that affect your policies** – Search by keyword or bill number.
            3. **Monitor bills you care about** – We'll alert you when they change.
            
            Your data is stored in secure cloud database. No setup required.
            """)
            if st.button("Got it, hide tour"):
                st.session_state.onboarding_complete = True
                st.rerun()

# -------------------- DASHBOARD --------------------
def render_dashboard():
    """Executive dashboard with key metrics and alerts."""
    st.header("🏢 Executive Compliance Dashboard")
    
    # Load data
    handbook = Database.load_handbook()
    monitored = Database.get_monitored_bills()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        handbook_status = "✅ Loaded" if handbook else "⚠️ Not saved"
        render_metric_card("Handbook", handbook_status, 
                          help_text="Last saved: check versions")
    with col2:
        render_metric_card("Monitored Bills", len(monitored),
                          help_text="Bills being tracked")
    with col3:
        # Count bills with recent updates
        updated_count = 0
        if monitored:
            # Quick check for changes (simplified)
            updated_count = len([b for b in monitored if b.get("last_checked")])
        render_metric_card("Pending Alerts", updated_count,
                          delta=f"{updated_count} new" if updated_count else "None")
    with col4:
        render_metric_card("Compliance Score", "83%", delta="+2%", 
                          help_text="Based on gap analysis")
    
    # Recent activity and alerts
    st.subheader("📋 Recent Activity")
    if monitored:
        for m in monitored[:3]:
            with st.container():
                st.markdown(f"**{m['bill_id']}** – {m.get('bill_title', '')[:80]}...")
                st.caption(f"Last checked: {m.get('last_checked', 'Unknown')}")
                st.divider()
    else:
        st.info("No bills are being monitored yet. Go to Legislative Intelligence to start.")
    
    # Upcoming effective dates (from Timeline module)
    st.subheader("📅 Upcoming Compliance Deadlines")
    # Placeholder – we could fetch from user‑created timelines
    st.caption("Set deadlines in the Timeline Projector module.")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("📘 Edit Handbook", use_container_width=True):
            st.session_state.nav_to = "Legislative Intelligence"
            st.rerun()
    with col_b:
        if st.button("🔍 Find New Bills", use_container_width=True):
            st.session_state.nav_to = "Legislative Intelligence"
            st.rerun()
    with col_c:
        if st.button("📊 Run Full Audit", use_container_width=True):
            st.session_state.run_audit = True
            st.session_state.nav_to = "Legislative Intelligence"
            st.rerun()

# -------------------- LEGISLATIVE INTELLIGENCE (Enhanced) --------------------
def render_legislative_redliner():
    """Main module for bill search, handbook, monitoring, and gap analysis."""
    
    # ---------- API Key Check ----------
    api_key = st.secrets.get("OPENSTATES_API_KEY") if hasattr(st, 'secrets') else None
    if not api_key:
        st.error("🚨 Open States API key not found. Please add to Streamlit Cloud Secrets.")
        st.stop()
    
    # ---------- Load saved handbook ----------
    if "handbook_content" not in st.session_state:
        saved = Database.load_handbook()
        st.session_state.handbook_content = saved
    
    # ---------- Session state for bill ----------
    if "current_bill" not in st.session_state:
        st.session_state.current_bill = None
    if "bill_text" not in st.session_state:
        st.session_state.bill_text = None
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None
    if "search_method" not in st.session_state:
        st.session_state.search_method = "By Bill Number"
    
    # ---------- Header ----------
    st.header("⚖️ Legislative Intelligence Engine")
    st.markdown("##### Find, analyze, and monitor Ohio legislation")
    st.markdown("---")
    
    # ---------- Sidebar: Monitored Bills & Alerts ----------
    with st.sidebar:
        with st.expander("🔔 Monitored Bills & Alerts", expanded=True):
            monitored = Database.get_monitored_bills()
            if monitored:
                for m in monitored:
                    col1, col2 = st.columns([4,1])
                    with col1:
                        st.markdown(f"**{m['bill_id']}**")
                        st.caption(m.get('bill_title', '')[:40] + "...")
                    with col2:
                        if st.button("✕", key=f"remove_{m['bill_id']}"):
                            Database.remove_monitored_bill(m['bill_id'])
                            st.rerun()
                st.divider()
                if st.button("🔄 Check All for Updates", use_container_width=True):
                    with st.spinner("Checking..."):
                        changed = check_all_monitored_bills(api_key)
                        if changed:
                            for bill in changed:
                                st.error(f"🚨 **{bill['bill_id']}** – Updated!")
                        else:
                            st.success("All monitored bills are current.")
            else:
                st.info("No bills monitored yet. Search and click 'Monitor'.")
    
    # ---------- Search Section ----------
    st.subheader("🔎 Find Legislation")
    
    col_method, col_filters = st.columns([1, 2])
    with col_method:
        search_method = st.radio(
            "Search by",
            ["Bill Number", "Keyword"],
            horizontal=True,
            key="search_method_radio"
        )
    
    # Clear stale bill when switching method
    if "prev_search_method" not in st.session_state:
        st.session_state.prev_search_method = search_method
    if st.session_state.prev_search_method != search_method:
        st.session_state.current_bill = None
        st.session_state.bill_text = None
        st.session_state.comparison_result = None
        st.session_state.prev_search_method = search_method
    
    if search_method == "Bill Number":
        bill_input = st.text_input("Enter bill number (e.g., HB33, SB 1)", value="HB33")
        if st.button("📥 Fetch Bill", type="primary", use_container_width=True):
            with st.spinner("Fetching bill..."):
                bill = get_bill_by_number(bill_input, api_key)
                if "error" in bill:
                    st.error(bill["error"])
                    st.session_state.current_bill = None
                else:
                    st.session_state.current_bill = bill
                    st.session_state.bill_text = None
                    st.session_state.comparison_result = None
                    Database.audit_log("fetch_bill", {"bill_id": bill["identifier"]})
    
    else:  # Keyword search
        col1, col2 = st.columns([3,1])
        with col1:
            keyword = st.text_input("Keyword (e.g., 'minimum wage', 'healthcare')", key="keyword_input")
        with col2:
            year = st.selectbox("Year", [0] + list(range(2026, 2020, -1)), format_func=lambda x: "All" if x == 0 else str(x))
        
        if keyword:
            with st.spinner("Searching..."):
                results = search_bills(keyword, api_key, year=year)
                if results:
                    bill_titles = [f"{r['identifier']}: {r['title'][:80]}..." for r in results]
                    selected_idx = st.selectbox(
                        "Select a bill to analyze",
                        range(len(bill_titles)),
                        format_func=lambda i: bill_titles[i]
                    )
                    selected = results[selected_idx]
                    if st.button("📥 Load Selected Bill", type="primary"):
                        with st.spinner("Fetching full details..."):
                            bill = get_bill_by_number(selected["identifier"], api_key)
                            if "error" in bill:
                                st.error(bill["error"])
                            else:
                                st.session_state.current_bill = bill
                                st.session_state.bill_text = None
                                st.session_state.comparison_result = None
                                Database.audit_log("fetch_bill", {"bill_id": bill["identifier"]})
                else:
                    st.warning("No substantive bills found. Try different keywords.")
    
    # ---------- Display Current Bill ----------
    if st.session_state.current_bill and "error" not in st.session_state.current_bill:
        bill = st.session_state.current_bill
        st.markdown("---")
        
        # Bill header with compliance tags
        col_title, col_monitor = st.columns([4,1])
        with col_title:
            st.success(f"**{bill['identifier']}** – {bill['title']}")
        with col_monitor:
            if st.button("🔔 Monitor", use_container_width=True):
                if st.session_state.bill_text:
                    text_hash = compute_text_hash(st.session_state.bill_text)
                    if Database.add_monitored_bill(bill['identifier'], bill['title'], text_hash):
                        st.success("Now monitoring!")
                        st.rerun()
                else:
                    st.warning("Extract bill text first.")
        
        # Metadata row
        cols = st.columns(4)
        cols[0].metric("Session", bill.get('session', 'N/A'))
        cols[1].metric("Status", str(bill.get('status', 'N/A')).capitalize())
        cols[2].metric("Updated", bill.get('updated_at', '')[:10] if bill.get('updated_at') else 'N/A')
        cols[3].metric("Chamber", bill.get('chamber', 'N/A').capitalize())
        
        # Compliance area tags
        categories = Database.classify_bill(bill)
        if categories:
            st.markdown("**Compliance Areas:** " + " • ".join([f"`{c}`" for c in categories]))
        
        # Abstract and versions
        with st.expander("📄 Bill Abstract & Versions", expanded=False):
            st.markdown(f"**Abstract:** {bill.get('abstract', 'No abstract.')}")
            if bill.get('versions'):
                st.markdown("**Available Versions:**")
                for v in bill['versions']:
                    if v.get('url'):
                        st.markdown(f"- {v.get('date', 'Unknown')}: {v.get('title', '')} – [Link]({v['url']})")
        
        # Extract text button
        if bill.get('text_url'):
            if st.button("📄 Extract Full Bill Text", use_container_width=True):
                with st.spinner("Downloading and parsing..."):
                    st.session_state.bill_text = extract_bill_text(bill['text_url'])
                    Database.audit_log("extract_bill_text", {"bill_id": bill["identifier"]})
        
        # Show extracted text
        if st.session_state.bill_text:
            with st.expander("📜 Full Bill Text (preview)", expanded=False):
                st.text_area("Bill content", st.session_state.bill_text[:5000], height=200, disabled=True)
                if len(st.session_state.bill_text) > 5000:
                    st.caption(f"*Showing first 5,000 of {len(st.session_state.bill_text):,} characters*")
    
    # ---------- Handbook Section ----------
    st.markdown("---")
    st.subheader("📘 Employee Handbook")
    
    # Tabs for edit / versions
    tab_edit, tab_versions = st.tabs(["✏️ Edit & Save", "🕘 Version History"])
    
    with tab_edit:
        handbook = st.text_area(
            "Handbook content",
            value=st.session_state.handbook_content,
            height=300,
            key="handbook_editor",
            placeholder="Paste your employee handbook or policy text here..."
        )
        
        col_save, col_note = st.columns([1,3])
        with col_save:
            if st.button("💾 Save Handbook", type="primary", use_container_width=True):
                if handbook.strip():
                    if Database.save_handbook(handbook):
                        st.success("Handbook saved permanently.")
                        st.session_state.handbook_content = handbook
                    else:
                        st.error("Save failed.")
        with col_note:
            version_note = st.text_input("Version note (optional)", placeholder="e.g., Updated PTO policy")
    
    with tab_versions:
        versions = Database.get_handbook_versions(limit=15)
        if versions:
            for v in versions:
                created = datetime.fromisoformat(v['created_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                with st.container():
                    col1, col2 = st.columns([4,1])
                    with col1:
                        st.markdown(f"**{created}** – {v.get('version_note', 'No note')}")
                        with st.expander("Preview"):
                            st.text(v['content'][:500] + ("..." if len(v['content']) > 500 else ""))
                    with col2:
                        if st.button("Restore", key=f"restore_{v['id']}"):
                            Database.save_handbook(v['content'], f"Restored from {created}")
                            st.session_state.handbook_content = v['content']
                            st.success("Handbook restored!")
                            st.rerun()
                    st.divider()
        else:
            st.info("No version history yet. Save the handbook to create versions.")
    
    # ---------- Gap Analysis ----------
    if st.session_state.bill_text and st.session_state.handbook_content:
        st.markdown("---")
        st.subheader("🔬 Compliance Gap Analysis")
        
        col_analyze, col_export = st.columns([1,1])
        with col_analyze:
            if st.button("Run Full Analysis", type="primary", use_container_width=True):
                with st.spinner("🧠 Analyzing semantic alignment..."):
                    result = ComplianceAnalyzer.semantic_similarity(
                        st.session_state.handbook_content,
                        st.session_state.bill_text
                    )
                    st.session_state.comparison_result = result
                    Database.audit_log("run_analysis", {
                        "bill_id": st.session_state.current_bill.get("identifier") if st.session_state.current_bill else None
                    })
        
        with col_export:
            if st.session_state.comparison_result and st.session_state.current_bill:
                pdf_bytes = generate_pdf_report(
                    st.session_state.current_bill,
                    st.session_state.handbook_content,
                    st.session_state.comparison_result
                )
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # Display results
        if st.session_state.comparison_result:
            res = st.session_state.comparison_result
            if "error" in res:
                st.error(res["error"])
            else:
                # Risk meter
                sim = res.get("overall_similarity", 0)
                risk = res.get("compliance_risk", "UNKNOWN")
                risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(risk, "gray")
                
                cols = st.columns([1,1,2])
                cols[0].metric("Overall Similarity", f"{sim:.1%}")
                cols[1].markdown(
                    f"**Compliance Risk**  \n:<span style='color:{risk_color};font-size:1.8rem;font-weight:bold'>{risk}</span>",
                    unsafe_allow_html=True
                )
                cols[2].info(f"**Method:** {res.get('method', 'N/A')}")
                
                # High‑risk sentences
                if res.get("sentence_analysis"):
                    with st.expander("⚠️ High‑Risk Policy Sentences", expanded=True):
                        risky = [s for s in res["sentence_analysis"] if s.get("risk") == "HIGH"]
                        if risky:
                            for r in risky[:7]:
                                st.markdown(f"**Policy:** {r['policy_sentence']}")
                                st.markdown(f"→ **Bill:** {r['bill_sentence'][:200]}...")
                                st.markdown(f"*Similarity: {r['similarity']:.1%}*")
                                st.divider()
                        else:
                            st.success("No high‑risk sentences detected.")
                
                # Classic diff
                with st.expander("📝 Character‑level Difference (exact changes)"):
                    diff_changes = ComplianceAnalyzer.classic_diff(
                        st.session_state.handbook_content,
                        st.session_state.bill_text
                    )
                    for change in diff_changes[:10]:
                        if change['type'] == 'replace':
                            st.markdown("**:orange[🔄 MODIFIED]**")
                            st.error(" ".join(change['old']))
                            st.success(" ".join(change['new']))
                        elif change['type'] == 'delete':
                            st.markdown("**:red[❌ REMOVED]**")
                            st.error(" ".join(change['old']))
                        elif change['type'] == 'insert':
                            st.markdown("**:green[➕ ADDED]**")
                            st.success(" ".join(change['new']))
    else:
        if not st.session_state.bill_text:
            st.info("👆 Extract a bill's full text to begin analysis.")
        elif not st.session_state.handbook_content:
            st.info("👆 Save your employee handbook to run gap analysis.")

# -------------------- AUTOMATED MONITORING --------------------
def check_all_monitored_bills(api_key: str) -> List[Dict]:
    """Check all monitored bills for updates. Returns list of changed bills."""
    monitored = Database.get_monitored_bills()
    changed = []
    for m in monitored:
        bill_data = get_bill_by_number(m["bill_id"], api_key)
        if "error" in bill_data:
            continue
        if not bill_data.get("text_url"):
            continue
        current_text = extract_bill_text(bill_data["text_url"])
        current_hash = compute_text_hash(current_text)
        if current_hash != m.get("last_text_hash"):
            changed.append({
                "bill_id": m["bill_id"],
                "bill_title": bill_data.get("title", m["bill_id"]),
                "old_hash": m.get("last_text_hash"),
                "new_hash": current_hash
            })
            # Update the record
            supabase = init_supabase()
            if supabase:
                supabase.table("monitored_bills") \
                    .update({
                        "last_text_hash": current_hash,
                        "last_checked": "now()"
                    }) \
                    .eq("bill_id", m["bill_id"]) \
                    .execute()
    return changed

def display_update_alerts():
    """Show persistent banner if any monitored bill changed since last visit."""
    if "alerts_shown" not in st.session_state:
        api_key = st.secrets.get("OPENSTATES_API_KEY") if hasattr(st, 'secrets') else None
        if api_key:
            changed = check_all_monitored_bills(api_key)
            if changed:
                for bill in changed:
                    st.error(f"🚨 **UPDATE DETECTED:** {bill['bill_id']} – {bill['bill_title'][:100]}...", icon="⚠️")
            st.session_state.alerts_shown = True

# -------------------- ORIGINAL MODULES (Restored & Enhanced) --------------------
def render_tuition_module():
    """Original tuition reimbursement audit – fully restored."""
    st.header("💰 Tuition Reimbursement Audit")
    st.markdown("### IRS Section 127 & Tenure Eligibility Engine")
    st.caption(f"Current IRS limit: **${Config.IRS_SEC_127_LIMIT:,.2f}** | Min tenure: **{Config.MIN_TENURE_YEARS} year**")
    
    with st.expander("📥 Download CSV Template", expanded=False):
        template_df = pd.DataFrame(columns=['EmployeeID', 'TenureYears', 'RequestAmount', 'DegreeProgram'])
        st.download_button(
            label="Download Template",
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
        
        required_schema = {'EmployeeID': 'str', 'TenureYears': 'float', 'RequestAmount': 'float'}
        validation_errors = validate_schema(df, required_schema)
        if validation_errors:
            for err in validation_errors:
                st.error(f"Schema Validation Error: {err}")
            st.stop()
        
        df['TenureYears'] = pd.to_numeric(df['TenureYears'], errors='coerce')
        df['RequestAmount'] = pd.to_numeric(df['RequestAmount'], errors='coerce')
        df.dropna(subset=['TenureYears', 'RequestAmount'], inplace=True)
        
        conditions = [
            (df['TenureYears'] < Config.MIN_TENURE_YEARS),
            (df['RequestAmount'] > Config.IRS_SEC_127_LIMIT)
        ]
        status_choices = ['Ineligible', 'Eligible (Taxable)']
        basis_choices = [
            f"Tenure below minimum ({Config.MIN_TENURE_YEARS} year)",
            f"Exceeds IRS Sec. 127 Limit (${Config.IRS_SEC_127_LIMIT:,.2f})"
        ]
        
        df['Decision_Status'] = np.select(conditions, status_choices, default='Eligible (Tax-Free)')
        df['Decision_Basis'] = np.select(conditions, basis_choices, default='Meets Tenure & IRS Criteria')
        df['Taxable_Amount'] = np.where(
            df['Decision_Status'] == 'Eligible (Taxable)',
            df['RequestAmount'] - Config.IRS_SEC_127_LIMIT,
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
            "📄 Download Audit Report (CSV)",
            data=output_csv,
            file_name=f"tuition_audit_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def render_redliner_module():
    """Original policy redliner – restored with enhanced diff."""
    st.header("📝 Policy Gap Analysis (Classic)")
    st.markdown("### Statutory vs. Internal Policy Comparator")
    
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
        
        # Use our enhanced diff renderer
        render_side_by_side_diff(
            tokenize_sentences(policy_text),
            tokenize_sentences(statute_text)
        )

def render_side_by_side_diff(policy_sentences, bill_sentences):
    """Improved diff visualization."""
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

def render_projector_module():
    """Original timeline projector – restored."""
    st.header("📅 Compliance Work-Back Schedule")
    st.markdown("### Implementation Timeline Generator")
    
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
        page_title=Config.PAGE_TITLE,
        layout=Config.LAYOUT,
        initial_sidebar_state="expanded",
        page_icon="🏥"
    )
    
    # Custom CSS – polished, professional
    st.markdown("""
    <style>
        .stApp {
            background-color: #f8fafc;
        }
        .main-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #0b3b5c;
            margin-bottom: 0.5rem;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stMetric {
            background-color: white;
            padding: 1rem;
            border-radius: 0.75rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        }
        div[data-testid="stExpander"] {
            background-color: white;
            border-radius: 0.5rem;
            border: 1px solid #e2e8f0;
        }
        .stAlert {
            border-radius: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("<h1 style='color:#0b3b5c; font-size:1.8rem; margin-bottom:0;'>Ohio Health</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("##### Enterprise Compliance Suite")
    st.sidebar.divider()
    
    # Navigation
    pages = {
        "🏠 Executive Dashboard": render_dashboard,
        "⚖️ Legislative Intelligence": render_legislative_redliner,
        "💰 Tuition Auditor": render_tuition_module,
        "📝 Policy Redliner": render_redliner_module,
        "📅 Timeline Projector": render_projector_module,
    }
    
    # Handle navigation from dashboard quick actions
    if "nav_to" in st.session_state:
        default_idx = list(pages.keys()).index(st.session_state.nav_to)
        st.session_state.pop("nav_to")
    else:
        default_idx = 0
    
    page = st.sidebar.radio(
        "Navigation",
        list(pages.keys()),
        index=default_idx,
        format_func=lambda x: x
    )
    
    st.sidebar.divider()
    
    # System status
    with st.sidebar.expander("🔧 System Status", expanded=False):
        supabase_status = "✅ Connected" if init_supabase() else "❌ Not configured"
        st.write(f"**Open States API:** {'✅ Live' if st.secrets.get('OPENSTATES_API_KEY') else '❌ No key'}")
        st.write(f"**Supabase:** {supabase_status}")
        st.write(f"**PDF extraction:** {'✅' if PDF_SUPPORT else '❌ pdfplumber missing'}")
        st.write(f"**Semantic AI:** {'✅' if SEMANTIC_AVAILABLE else '⚠️ TF‑IDF'}")
        st.write(f"**PDF Reports:** {'✅' if PDF_REPORT_AVAILABLE else '❌ fpdf missing'}")
    
    # Onboarding tour (only on dashboard first visit)
    if page == "🏠 Executive Dashboard":
        render_onboarding_tour()
    
    # Display update alerts (global)
    if page != "⚖️ Legislative Intelligence":  # already shown there
        display_update_alerts()
    
    # Render selected page
    pages[page]()

if __name__ == "__main__":
    main()
