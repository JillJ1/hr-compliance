# Ohio Health System: Compliance Decision Support System (CDSS)

**Status:** Internal Pilot / Portfolio Concept  
**Stack:** Python, Streamlit, Pandas (Vectorized Logic)

## Executive Summary
This application is a deterministic compliance engine designed to automate high-volume HR audits. It replaces manual spreadsheet calculations with auditable, rule-based logic.

## Modules
1.  **Tuition Reimbursement Audit:** * Validates scholarship requests against **IRS Section 127** limits ($5,250).
    * Enforces internal tenure requirements.
    * Calculates exact taxable exposure for payroll.
2.  **Policy Gap Analysis:**
    * Uses Natural Language Processing (NLP) tokenization to compare internal policy text against new Ohio statutes.
3.  **Implementation Projector:**
    * Generates operational work-back schedules based on statutory effective dates.

## Technical Architecture
* **Zero-Retention:** Stateless architecture ensures no PII is stored after processing.
* **Vectorized Processing:** Uses Pandas vector operations for hospital-scale performance (10k+ rows).
* **Configuration:** Statutory limits are decoupled from logic for easy annual updates.
