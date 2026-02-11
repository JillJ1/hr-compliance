Compliance Decision Support System (CDSS) - Technical Specification

Version: 1.0.0

Status: Pilot / Internal Review

Target User: HR Operations & Compliance Officers

1. Executive Summary

The Compliance Decision Support System (CDSS) is a modular Streamlit application designed to automate high-volume compliance audits and gap analyses. It focuses on deterministic logic to reduce human error in payroll taxation and policy updates. The system operates on a "human-in-the-loop" architecture—it provides data and recommendations, but final execution authority remains with the user.

2. Module Specifications

2.1 Tuition Reimbursement Audit Engine

Purpose: To process bulk scholarship requests and determine eligibility based on internal tenure policies and federal tax liabilities under IRS Section 127.

Inputs:

Format: CSV File

Schema:

EmployeeID (String): Unique identifier.

TenureYears (Float): Employee's length of service.

RequestAmount (Float): Total reimbursement requested in USD.

Processing Logic (Vectorized):

Validation: Enforces schema and numeric types. Invalid rows are excluded from calculation to prevent system failure.

Tenure Check: If TenureYears < MIN_TENURE_YEARS (Config: 1.0), status is Ineligible.

Tax Threshold: If RequestAmount > IRS_SEC_127_LIMIT (Config: $5,250.00), calculate Taxable_Amount as RequestAmount - IRS_SEC_127_LIMIT. Status is Eligible (Taxable).

Default: All other cases are Eligible (Tax-Free).

Outputs:

Processed CSV containing Decision_Status, Decision_Basis, and Taxable_Amount.

Executive summary metrics for total financial exposure.

Risk Considerations:

The system assumes TenureYears is calculated correctly by the source system (e.g., ADP/Workday).

Tax limits are hardcoded in the configuration class and must be updated annually.

2.2 Policy Gap Analysis (Redliner)

Purpose: To identify semantic changes between current internal policies and new legislative text (statutes/regulations).

Inputs:

Source Text: Current Internal Handbook Policy (Text Area).

Target Text: New Statutory Text (Text Area).

Processing Logic:

Tokenization: Splits input blocks into sentences using regex delimiters (., ?, !).

Comparison: Uses difflib.Differ to compare the two lists of sentences.

Classification:

REMOVED/MODIFIED: Sentence exists in Source but not Target.

ADDITION: Sentence exists in Target but not Source.

UNCHANGED: Sentence exists in both (collapsed by default).

Outputs:

Visual "Diff" showing additions in Green and removals in Red.

Focuses on sentence-level context rather than character-level typos.

2.3 Work-Back Scheduler

Purpose: To standardize implementation timelines for new compliance requirements.

Inputs:

Effective Date (Date Picker).

Processing Logic:

Calculates due dates by subtracting standard lead times (e.g., Legal Review = Effective Date - 45 Days) from the target date.

Outputs:

Table of milestones categorized by phase (Preparation vs. Execution).

3. Technical Constraints & Security

Data Persistence: The application is stateless. No data is stored on the server after the session ends.

Privacy: Users are advised via UI tooltips not to upload sensitive PII (SSN, DOB) into the pilot environment.

Dependencies: pandas, streamlit, numpy.
