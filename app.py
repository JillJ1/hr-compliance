import streamlit as st
import pandas as pd
import numpy as np
import re
import difflib
from datetime import datetime, timedelta
from io import BytesIO

# --- CONFIGURATION & CONSTANTS ---
class ComplianceConfig:
    """Central configuration for compliance thresholds and statutory limits."""
    # IRS Section 127: Educational Assistance Programs
    IRS_SEC_127_LIMIT = 5250.00
    
    # Internal Policy Constants (To be adjusted based on specific Hospital policy)
    MIN_TENURE_YEARS = 1.0
    
    # UI Configuration
    PAGE_TITLE = "Ohio Health System | Compliance Decision Support"
    LAYOUT = "wide"

# --- HELPER FUNCTIONS ---

def validate_schema(df: pd.DataFrame, required_columns: dict) -> list:
    """
    Validates that the dataframe contains required columns and correct types.
    Returns a list of error messages.
    """
    errors = []
    
    # Check for missing columns
    missing_cols = [col for col in required_columns.keys() if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Basic type validation (can be expanded)
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
    """
    Splits text into sentences for granular comparison.
    Uses regex to handle common sentence delimiters while preserving them.
    """
    if not text:
        return []
    # Split by period, question mark, or exclamation point followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# --- MODULES ---

def render_tuition_module():
    st.header("Tuition Reimbursement Audit")
    st.markdown("### IRS Section 127 & Tenure Eligibility Engine")
    st.markdown(
        """
        **Purpose:** Batch process scholarship requests to determine eligibility and tax liability.
        **Statutory Reference:** 26 U.S. Code § 127 - Educational assistance programs.
        """
    )

    # 1. Template Download
    with st.expander("Download Batch Template", expanded=False):
        st.caption("Use this schema for upload. Do not include sensitive PII in non-secure environments.")
        template_df = pd.DataFrame(columns=['EmployeeID', 'TenureYears', 'RequestAmount', 'DegreeProgram'])
        st.download_button(
            label="Download Empty CSV Schema",
            data=template_df.to_csv(index=False),
            file_name="tuition_audit_template.csv",
            mime="text/csv"
        )

    # 2. Data Ingestion
    uploaded_file = st.file_uploader("Upload Request CSV", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            return

        # 3. Schema Validation
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

        # 4. Processing (Vectorized)
        # Ensure types are numeric for calculation
        df['TenureYears'] = pd.to_numeric(df['TenureYears'], errors='coerce')
        df['RequestAmount'] = pd.to_numeric(df['RequestAmount'], errors='coerce')
        
        # Drop rows that failed coercion
        if df.isna().any().any():
            st.warning("Rows with invalid numeric data were excluded from analysis.")
            df.dropna(subset=['TenureYears', 'RequestAmount'], inplace=True)

        # Logic Definitions
        conditions = [
            (df['TenureYears'] < ComplianceConfig.MIN_TENURE_YEARS),
            (df['RequestAmount'] > ComplianceConfig.IRS_SEC_127_LIMIT)
        ]
        
        choices_status = ['Ineligible', 'Eligible (Taxable)']
        choices_basis = [
            f"Tenure below minimum ({ComplianceConfig.MIN_TENURE_YEARS} year)",
            f"Exceeds IRS Sec. 127 Limit (${ComplianceConfig.IRS_SEC_127_LIMIT:,.2f})"
        ]
        
        # Apply Logic
        df['Decision_Status'] = np.select(conditions, choices_status, default='Eligible (Tax-Free)')
        df['Decision_Basis'] = np.select(conditions, choices_basis, default='Meets Tenure & IRS Criteria')
        
        # Calculate Taxable Exposure
        df['Taxable_Amount'] = np.where(
            df['Decision_Status'] == 'Eligible (Taxable)',
            df['RequestAmount'] - ComplianceConfig.IRS_SEC_127_LIMIT,
            0.0
        )

        # 5. Output Presentation
        st.divider()
        st.subheader("Audit Results")
        
        # Summary Metrics
        total_exposure = df['Taxable_Amount'].sum()
        ineligible_count = len(df[df['Decision_Status'] == 'Ineligible'])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records Processed", len(df))
        m2.metric("Total Taxable Exposure", f"${total_exposure:,.2f}")
        m3.metric("Ineligible Requests", ineligible_count)

        # Detailed Dataframe
        st.dataframe(
            df[['EmployeeID', 'RequestAmount', 'Decision_Status', 'Taxable_Amount', 'Decision_Basis']],
            use_container_width=True,
            hide_index=True
        )
        
        # Download Result
        output_csv = df.to_csv(index=False)
        st.download_button(
            "Download Audit Report",
            data=output_csv,
            file_name=f"audit_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def render_redliner_module():
    st.header("Policy Gap Analysis")
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
        policy_text = st.text_area("Paste current handbook text here:", height=300)
    with col2:
        st.markdown("#### New Statutory Text")
        statute_text = st.text_area("Paste new legislative text here:", height=300)

    if st.button("Run Gap Analysis"):
        if not policy_text or not statute_text:
            st.warning("Both text fields are required for analysis.")
            return

        st.divider()
        st.subheader("Analysis Output")

        # Tokenize
        policy_sentences = tokenize_sentences(policy_text)
        statute_sentences = tokenize_sentences(statute_text)

        # Diff
        differ = difflib.Differ()
        diff = list(differ.compare(policy_sentences, statute_sentences))

        # Render
        for line in diff:
            code = line[:2]
            sentence = line[2:]
            
            if code == "- ":
                st.markdown(f":red[**REMOVED/MODIFIED:**] {sentence}")
                st.caption("Present in Policy, absent in Statute.")
            elif code == "+ ":
                st.markdown(f":green[**ADDITION:**] {sentence}")
                st.caption("Present in Statute, absent in Policy.")
            elif code == "  ":
                with st.expander(f"Unchanged: {sentence[:50]}...", expanded=False):
                    st.write(sentence)

def render_projector_module():
    st.header("Compliance Work-Back Schedule")
    st.markdown("### Implementation Timeline Generator")
    st.markdown("**Purpose:** Generate operational milestones backwards from a statutory effective date.")

    effective_date = st.date_input("Statutory Effective Date")
    
    if st.button("Generate Timeline"):
        # Define Standard Milestones (Days Before Effective Date)
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

# --- MAIN APP SHELL ---

def main():
    st.set_page_config(
        page_title=ComplianceConfig.PAGE_TITLE,
        layout=ComplianceConfig.LAYOUT
    )
    
    # Sidebar Navigation
    st.sidebar.title("Compliance Systems")
    st.sidebar.markdown("Ohio Health System | Internal Tool")
    
    module = st.sidebar.radio(
        "Select Module:",
        ["Tuition Reimbursement", "Policy Redliner", "Timeline Projector"]
    )
    
    st.sidebar.divider()
    st.sidebar.info(f"Configuration loaded.\nIRS Limit: ${ComplianceConfig.IRS_SEC_127_LIMIT:,.2f}")

    # Routing
    if module == "Tuition Reimbursement":
        render_tuition_module()
    elif module == "Policy Redliner":
        render_redliner_module()
    elif module == "Timeline Projector":
        render_projector_module()

if __name__ == "__main__":
    main()
