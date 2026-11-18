import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LoanCheck India — AI Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS — Premium Modern Design & Floating Chatbot Styling
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Tokens ── */
:root {
  --white:      #ffffff;
  --bg:         #f8fafc;
  --surface:    #ffffff;
  --border:     #e2e8f0;
  --blue:       #2563eb;
  --blue-dark:  #0f172a;
  --blue-navy:  #1e3a8a;
  --blue-light: #eff6ff;
  --green:      #16a34a;
  --green-bg:   #f0fdf4;
  --red:        #dc2626;
  --red-bg:     #fef2f2;
  --amber:      #d97706;
  --amber-bg:   #fffbeb;
  --text:       #0f172a;
  --text-2:     #475569;
  --text-3:     #94a3b8;
  --shadow-sm:  0 1px 3px rgba(0,0,0,.06);
  --shadow-md:  0 4px 14px rgba(0,0,0,.08);
  --shadow-lg:  0 10px 30px rgba(0,0,0,.12);
}

/* ── Reset & Container ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container {
  background: var(--bg) !important;
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
}

/* Hide Streamlit Header & Watermarks */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
header[data-testid="stHeader"] {
  display: none !important;
  height: 0 !important;
  visibility: hidden !important;
}

.block-container {
  padding-top: 1.2rem !important;
  padding-left: 1.8rem !important;
  padding-right: 1.8rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 1440px !important;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
  border-right: none !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] span:not([data-testid*="metric"]),
[data-testid="stSidebar"] div:not([data-testid*="metric"]) { 
  color: #f1f5f9 !important; 
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12) !important; }
[data-testid="stSidebar"] label {
  color: #93c5fd !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: .02em !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
  background: rgba(255,255,255,.07) !important;
  border: 1px solid rgba(255,255,255,.16) !important;
  border-radius: 10px !important;
  color: white !important;
}
[data-testid="stSidebar"] input {
  background: rgba(255,255,255,.07) !important;
  border: 1px solid rgba(255,255,255,.16) !important;
  border-radius: 10px !important;
  color: white !important;
}

/* Primary Predict Button in Sidebar */
[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  padding: 14px !important;
  box-shadow: 0 4px 16px rgba(37,99,235,.4) !important;
  transition: all .25s ease !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
  background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
  box-shadow: 0 6px 22px rgba(37,99,235,.6) !important;
  transform: translateY(-2px) !important;
}

/* Secondary Action Buttons */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
  background: rgba(255,255,255,.08) !important;
  border: 1px solid rgba(255,255,255,.2) !important;
  color: #bfdbfe !important;
  border-radius: 8px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
}

/* ── Main Content Dark Text Enforcement ── */
[data-testid="stMain"],
[data-testid="stMain"] p,
[data-testid="stMain"] span:not([style*="color: white"]):not([style*="color:white"]),
[data-testid="stMain"] label,
[data-testid="stMain"] h1:not(.hero-title),
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6 {
  color: #0f172a !important;
}

/* ── Hero Header Banner Pure White Text ── */
.hero-header-banner,
.hero-header-banner *,
.hero-header-banner h1,
.hero-title,
h1.hero-title,
[data-testid="stMain"] h1.hero-title,
[data-testid="stMain"] .hero-title,
[data-testid="stMain"] .hero-header-banner * {
  color: #ffffff !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"],
[data-testid="metric-container"] {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 14px !important;
  padding: 16px 20px !important;
  box-shadow: 0 2px 6px rgba(0,0,0,.04) !important;
  transition: transform .2s, box-shadow .2s !important;
}
[data-testid="stMetric"]:hover,
[data-testid="metric-container"]:hover {
  box-shadow: 0 6px 16px rgba(0,0,0,.08) !important;
  transform: translateY(-2px);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] * {
  color: #64748b !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: .06em !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] div,
div[data-testid="metric-container"] [data-testid="stMetricValue"] * {
  color: #0f172a !important;
  font-size: 22px !important;
  font-weight: 800 !important;
}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
  font-size: 12px !important;
  font-weight: 700 !important;
}

/* ── Progress Bar ── */
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, #2563eb, #10b981) !important;
  border-radius: 99px !important;
}

/* ── Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px !important;
  background-color: transparent !important;
  border-bottom: 2px solid #e2e8f0 !important;
}
.stTabs [data-baseweb="tab"] {
  height: 44px !important;
  white-space: pre !important;
  border-radius: 10px 10px 0 0 !important;
  background: #f1f5f9 !important;
  border: 1px solid #e2e8f0 !important;
  border-bottom: none !important;
  padding: 0 18px !important;
}
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {
  color: #475569 !important;
  font-weight: 600 !important;
  font-size: 14px !important;
}
.stTabs [aria-selected="true"] {
  background: #ffffff !important;
  border-top: 3px solid #1d4ed8 !important;
  border-left: 1px solid #cbd5e1 !important;
  border-right: 1px solid #cbd5e1 !important;
  border-bottom: 2px solid #ffffff !important;
}
.stTabs [aria-selected="true"] *,
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {
  color: #1d4ed8 !important;
  font-weight: 800 !important;
}

/* ── Floating FAB Button (Bottom-Right) ── */
div[class*="st-key-floating_fab_box"] {
  position: fixed !important;
  bottom: 24px !important;
  right: 24px !important;
  z-index: 999999 !important;
}
div[class*="st-key-floating_fab_box"] button {
  background: linear-gradient(135deg, #1d4ed8 0%, #4f46e5 100%) !important;
  color: white !important;
  border-radius: 99px !important;
  padding: 10px 22px !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  border: 2px solid rgba(255, 255, 255, 0.4) !important;
  box-shadow: 0 8px 24px rgba(29, 78, 216, 0.45) !important;
  transition: all 0.25s ease !important;
}
div[class*="st-key-floating_fab_box"] button:hover {
  transform: translateY(-2px) scale(1.03) !important;
  box-shadow: 0 12px 30px rgba(29, 78, 216, 0.6) !important;
}

/* ── Floating Chat Popup Card ── */
div[class*="st-key-floating_chat_popup"] {
  position: fixed !important;
  bottom: 82px !important;
  right: 24px !important;
  width: 400px !important;
  max-width: calc(100vw - 32px) !important;
  height: 540px !important;
  background: #ffffff !important;
  border-radius: 18px !important;
  box-shadow: 0 16px 45px rgba(15, 23, 42, 0.3) !important;
  border: 1px solid #cbd5e1 !important;
  z-index: 999998 !important;
  display: flex !important;
  flex-direction: column !important;
  padding: 14px !important;
  overflow: hidden !important;
  animation: slideUp 0.25s ease-out !important;
}
@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px) scale(0.96); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

div[class*="st-key-floating_chat_popup"] [data-testid="stChatMessage"] {
  padding: 8px 12px !important;
  border-radius: 12px !important;
  margin-bottom: 6px !important;
}
div[class*="st-key-floating_chat_popup"] [data-testid="stChatMessage"] p {
  font-size: 13px !important;
  line-height: 1.4 !important;
  margin-bottom: 0 !important;
}
div[class*="st-key-floating_chat_popup"] button {
  font-size: 11px !important;
  padding: 4px 8px !important;
  border-radius: 8px !important;
}

/* Hide Streamlit elements in sidebar header */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

USD_TO_INR = 83

# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model')
    if not os.path.exists(model_dir):
        model_dir = os.path.join(base_dir, 'LoanApp', 'model')
    model     = joblib.load(os.path.join(model_dir, 'loan_model.pkl'))
    scaler    = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    features  = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    threshold = joblib.load(os.path.join(model_dir, 'threshold.pkl'))
    return model, scaler, features, threshold

model, scaler, features, threshold = load_model()

# ============================================================
# HELPER FUNCTIONS & ENCODINGS
# ============================================================
def credit_grade(score):
    if score < 580:   return 0
    elif score < 640: return 1
    elif score < 700: return 2
    elif score < 750: return 3
    else:             return 4

def credit_label(score):
    if score < 580:   return "Poor"
    elif score < 640: return "Fair"
    elif score < 700: return "Good"
    elif score < 750: return "Very Good"
    else:             return "Excellent"

def credit_range(score):
    if score < 580:   return "< 580"
    elif score < 640: return "580–639"
    elif score < 700: return "640–699"
    elif score < 750: return "700–749"
    else:             return "750+"

def credit_color(score):
    if score < 580:   return "#dc2626"
    elif score < 640: return "#ea580c"
    elif score < 700: return "#d97706"
    elif score < 750: return "#16a34a"
    else:             return "#0d9488"

home_map   = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 0}
intent_map = {
    'Personal': 0, 'Education': 1, 'Medical': 2,
    'Business / Venture': 3, 'Home Improvement': 4, 'Debt Consolidation': 5
}
edu_map = {
    'High School': 0, 'Associate': 1, 'Bachelor': 2,
    'Master': 3, 'Doctorate': 4
}

def calculate_emi(principal, annual_rate_pct, tenure_months):
    if tenure_months <= 0 or principal <= 0: return 0.0
    r = (annual_rate_pct / 12) / 100
    if r == 0: return principal / tenure_months
    factor = (1 + r) ** tenure_months
    return principal * r * factor / (factor - 1)

def build_features(person_age, gender, education, income_usd,
                   emp_exp, home_ownership, loan_usd,
                   loan_intent, int_rate, credit_score,
                   cred_hist, had_default):
    monthly_usd = income_usd / 12
    lpi = loan_usd / income_usd if income_usd > 0 else 0
    gender_enc   = 1 if gender == 'Male' else 0
    edu_enc      = edu_map.get(education, 2)
    home_enc     = home_map.get(home_ownership, 0)
    intent_enc   = intent_map.get(loan_intent, 0)
    default_enc  = 1 if had_default == "Yes (Active Default)" else 0
    c_grade      = credit_grade(credit_score)
    cred_band    = min(3, max(0, cred_hist // 2))

    return {
        'person_age':                 person_age,
        'gender_enc':                 gender_enc,
        'edu_enc':                    edu_enc,
        'person_emp_exp':             emp_exp,
        'home_enc':                   home_enc,
        'low_emp_flag':               int(emp_exp < 1),
        'credit_grade':               c_grade,
        'fine_credit_score':          credit_score,
        'had_default':                default_enc,
        'cb_person_cred_hist_length': cred_hist,
        'cred_hist_band':             cred_band,
        'short_hist_flag':            int(cred_hist < 2),
        'monthly_income':             monthly_usd,
        'loan_percent_income':        lpi,
        'high_debt_flag':             int(lpi > 0.35),
        'low_income_flag':            int(monthly_usd < 2000),
        'loan_amnt':                  loan_usd,
        'loan_int_rate':              int_rate,
        'intent_enc':                 intent_enc,
        'income_x_grade':             monthly_usd * c_grade,
        'debt_x_intent':              lpi * intent_enc,
        'age_x_emp':                  person_age * emp_exp,
        'income_debt_ratio':          monthly_usd / (loan_usd + 1),
        'credit_x_hist':              credit_score * cred_hist,
        'rate_burden':                int_rate * lpi,
        'emp_income_combo':           emp_exp * monthly_usd,
    }

def apply_hard_rules(income_inr, loan_inr, monthly_inr,
                     had_default, credit_score, emp_exp, lpi, foir):
    reasons = []
    if had_default == "Yes (Active Default)":
        reasons.append("Active loan default on file — automatic rejection across all Indian lenders")
    elif had_default == "Settled":
        weak_factors = 0
        if credit_score < 650:  weak_factors += 1
        if foir > 50:           weak_factors += 1
        if monthly_inr < 25000: weak_factors += 1
        if lpi > 0.50:          weak_factors += 1
        if emp_exp < 1:         weak_factors += 1
        if weak_factors >= 3:
            reasons.append(f"Settled default + {weak_factors} high-risk flags — combined risk exceeds bank threshold")
    if credit_score < 550:
        reasons.append(f"CIBIL score {credit_score} is below 550 minimum baseline")
    if monthly_inr < 15000:
        reasons.append(f"Monthly income ₹{monthly_inr:,.0f} below ₹15,000 baseline cutoff")
    if loan_inr > income_inr * 10:
        reasons.append(f"Requested loan (₹{loan_inr:,.0f}) is {loan_inr/income_inr:.1f}× annual income (Max cap: 10×)")
    if foir > 55:
        reasons.append(f"FOIR {foir:.1f}% exceeds 55% debt capacity cap")
    return len(reasons) > 0, reasons

# ============================================================
# MULTI-BANK EVALUATION MATRIX (Nationalised & Private)
# ============================================================
BANKS_DATABASE = [
    {
        "name": "State Bank of India (SBI)",
        "type": "Nationalised Public Bank",
        "min_salary": 15000,
        "min_cibil": 700,
        "max_foir": 50,
        "rate_range": "11.15% - 15.30%",
        "processing_fee": "1.0% (Min ₹1,000)",
        "icon": "🏛️"
    },
    {
        "name": "Bank of Baroda (BOB)",
        "type": "Nationalised Public Bank",
        "min_salary": 15000,
        "min_cibil": 680,
        "max_foir": 50,
        "rate_range": "10.90% - 14.75%",
        "processing_fee": "1.0% + GST",
        "icon": "🏛️"
    },
    {
        "name": "Indian Overseas Bank (IOB)",
        "type": "Nationalised Public Bank",
        "min_salary": 15000,
        "min_cibil": 675,
        "max_foir": 50,
        "rate_range": "11.00% - 15.00%",
        "processing_fee": "0.75% + GST",
        "icon": "🏛️"
    },
    {
        "name": "Punjab National Bank (PNB)",
        "type": "Nationalised Public Bank",
        "min_salary": 15000,
        "min_cibil": 680,
        "max_foir": 50,
        "rate_range": "10.95% - 14.85%",
        "processing_fee": "1.0%",
        "icon": "🏛️"
    },
    {
        "name": "HDFC Bank",
        "type": "Private Bank",
        "min_salary": 25000,
        "min_cibil": 720,
        "max_foir": 55,
        "rate_range": "10.50% - 21.00%",
        "processing_fee": "Up to ₹4,999",
        "icon": "🏦"
    },
    {
        "name": "ICICI Bank",
        "type": "Private Bank",
        "min_salary": 25000,
        "min_cibil": 700,
        "max_foir": 55,
        "rate_range": "10.65% - 16.00%",
        "processing_fee": "Up to 2.25%",
        "icon": "🏦"
    },
    {
        "name": "Axis Bank",
        "type": "Private Bank",
        "min_salary": 15000,
        "min_cibil": 700,
        "max_foir": 50,
        "rate_range": "10.75% - 22.00%",
        "processing_fee": "1.0% to 2.0%",
        "icon": "🏦"
    },
    {
        "name": "Bajaj Finance",
        "type": "NBFC Financial Services",
        "min_salary": 20000,
        "min_cibil": 650,
        "max_foir": 60,
        "rate_range": "13.00% - 26.00%",
        "processing_fee": "Up to 3.99%",
        "icon": "💼"
    }
]

def evaluate_multi_bank(monthly_inr, credit_score, foir_pct, had_default, emp_exp):
    results = []
    for b in BANKS_DATABASE:
        status = "Eligible"
        reasons = []
        
        if had_default == "Yes (Active Default)":
            status = "Ineligible"
            reasons.append("Active Default")
        else:
            if monthly_inr < b["min_salary"]:
                status = "Ineligible"
                reasons.append(f"Income < ₹{b['min_salary']:,.0f}")
            if credit_score < b["min_cibil"]:
                status = "Ineligible" if credit_score < b["min_cibil"] - 30 else "Borderline"
                reasons.append(f"CIBIL < {b['min_cibil']}")
            if foir_pct > b["max_foir"]:
                status = "Ineligible" if foir_pct > b["max_foir"] + 5 else "Borderline"
                reasons.append(f"FOIR > {b['max_foir']}%")
            if emp_exp < 1:
                if status == "Eligible": status = "Borderline"
                reasons.append("Exp < 1 yr")

        results.append({
            "bank": b["name"],
            "type": b["type"],
            "status": status,
            "reasons": reasons,
            "rate": b["rate_range"],
            "fee": b["processing_fee"],
            "icon": b["icon"]
        })
    return results

# ============================================================
# CONTEXT-AWARE CHATBOT KB & ENGINE
# ============================================================
def get_context_chat_response(msg, context_data=None):
    msg_clean = msg.lower().strip()
    
    # 1. Personal Context Specific Queries
    if any(k in msg_clean for k in ["my status", "my chances", "my loan", "why rejected", "will i", "chances", "eligible", "eligibility", "my profile"]):
        if context_data and context_data.get("calculated"):
            c = context_data
            status_text = "LIKELY APPROVED" if c["approved"] else "LIKELY REJECTED"
            response = f"**📊 Live Analysis for Your Profile:**\n\n"
            response += f"• **Overall Status:** {status_text} ({c['display_prob']*100:.1f}% Probability)\n"
            response += f"• **CIBIL Score:** {c['credit_score']} ({credit_label(c['credit_score'])})\n"
            response += f"• **FOIR Ratio:** {c['foir_pct']:.1f}% (Ideal: ≤ 40%)\n"
            response += f"• **Monthly Income:** ₹{c['monthly_inr']:,.0f}\n\n"
            
            if not c["approved"]:
                response += "**Quick fixes to get approved:**\n"
                if c["foir_pct"] > 40:
                    response += "• Lower your monthly EMI by extending loan tenure or reducing loan amount.\n"
                if c["credit_score"] < 700:
                    response += "• Clear credit card dues to bring CIBIL above 700.\n"
                if c["had_default"] != "No":
                    response += "• Obtain a No Dues Certificate (NDC) for your default record.\n"
                response += "• Consider applying with a co-applicant to pool incomes.\n"
            else:
                response += "🎉 Your profile is strong! You meet eligibility benchmarks for SBI, BOB, IOB, PNB, and HDFC."
            return response
        return "Please fill in the applicant details in the sidebar to analyze your live profile."

    # 2. Nationalised Banks Info
    if any(k in msg_clean for k in ["bob", "bank of baroda", "iob", "indian overseas bank", "pnb", "punjab national bank", "sbi", "state bank", "nationalised", "government bank", "public bank"]):
        return """**🏛️ Public & Nationalised Banks (SBI, BOB, IOB, PNB):**

• **Min Monthly Income:** ₹15,000/month  
• **Target CIBIL Score:** 675–700+  
• **Max FOIR Limit:** 50%  
• **Interest Rate Range:** 10.90% to 15.30%  
• **Advantages:** Lower interest rates, transparent terms, no hidden fees, flexible tenure up to 72–84 months."""

    # 3. Private Banks Info
    if any(k in msg_clean for k in ["hdfc", "icici", "axis", "private bank"]):
        return """**🏦 Private Banks (HDFC, ICICI, Axis):**

• **Min Monthly Income:** ₹25,000 (HDFC/ICICI), ₹15,000 (Axis)  
• **Target CIBIL:** 700–720+  
• **Max FOIR:** 50%–55%  
• **Interest Rates:** 10.50% to 22.00%  
• **Advantage:** Quick digital approvals and fast disbursal within 24–48 hours."""

    # 4. Credit Score Query
    if any(k in msg_clean for k in ["credit score", "cibil", "credit rating"]):
        return """**📊 CIBIL Credit Score Bands:**

• **750+ (Excellent):** Instant approval, lowest interest rates (~10.5%–11.5%).  
• **700–749 (Good):** Standard approval at nationalised & private banks.  
• **640–699 (Fair):** Possible approval with higher interest (~14%–18%).  
• **< 640 (Poor):** High risk of rejection.

*Tip: Keep credit card balance below 30% of your credit limit to boost CIBIL.*"""

    # 5. FOIR Query
    if any(k in msg_clean for k in ["foir", "fixed obligation", "debt ratio", "debt-to-income"]):
        return """**💰 FOIR (Fixed Obligation to Income Ratio):**

`FOIR = (Total Existing & Proposed EMIs ÷ Net Monthly Income) × 100`

• **≤ 40%:** Ideal zone (Highest approval odds) ✅  
• **40% – 50%:** Borderline / Case-by-case ⚠️  
• **> 50%:** High risk / Likely rejection at SBI, PNB, BOB ❌  

*Tip: Increasing tenure lowers EMI and immediately improves your FOIR!*"""

    # 6. EMI Query
    if any(k in msg_clean for k in ["emi", "monthly payment", "installment"]):
        return """**📅 EMI Calculation:**

`EMI = P × r × (1+r)ⁿ ÷ ((1+r)ⁿ − 1)`

Where **P** = Principal, **r** = Monthly rate, **n** = Tenure in months.  
*Tip: Keep your total EMIs under 40% of your monthly salary for best approval odds.*"""

    # 7. Default Query
    if any(k in msg_clean for k in ["default", "missed payment", "settled", "cibil defaulter", "overdue", "npa"]):
        return """**⚠️ Past Default & Settlement Impact:**

• **Active Default:** Automatic rejection across 100% of Indian banks.  
• **Settled Account:** High risk; can be approved if CIBIL is >700, income is high, and FOIR is <40%.  
• **Recovery Steps:** Clear outstanding dues → Obtain No Dues Certificate (NDC) → Request bank to update CIBIL status to 'Closed'."""

    # 8. Improvement Tips
    if any(k in msg_clean for k in ["improve", "tips", "increase chances", "boost", "how to get approved"]):
        return """**🚀 Top Ways to Improve Approval Odds:**

1. ✅ Clear existing credit card balances to lower FOIR below 40%.
2. ✅ Extend loan tenure (e.g. from 24 to 48 months) to reduce monthly EMI.
3. ✅ Check nationalised banks (SBI, BOB) first — they have more flexible salary thresholds.
4. ✅ Add a earning co-applicant with stable income.
5. ✅ Maintain minimum 1–2 years stability in current employment."""

    # 9. Required Documents
    if any(k in msg_clean for k in ["document", "documents", "docs", "papers", "id proof", "salary slip"]):
        return """**📄 Standard Document Checklist:**

**Salaried Applicants:**
• Aadhaar Card & PAN Card  
• Last 3 months' salary slips  
• Last 6 months' bank account statement  
• Form 16 / Latest ITR  

**Self-Employed Applicants:**
• PAN Card, Aadhaar Card & Business Proof  
• Past 2 years' ITR with P&L computation  
• 12 months' primary bank statement"""

    # 10. Greetings
    if any(k in msg_clean for k in ["hi", "hello", "hey", "namaste", "help"]):
        return """👋 **Hello! I'm your AI Loan Assistant.**

I can help you with:
• **Your Profile Chances** (*"What are my chances?"*)
• **Bank Criteria** (*"What are SBI rules?"* or *"HDFC requirements?"*)
• **CIBIL Score tips** (*"How to improve CIBIL?"*)
• **FOIR & EMI explanations** (*"What is FOIR?"*)
• **Required Documents** (*"What documents do I need?"*)

Ask me any question below!"""

    # 11. Default Fallback
    return """I am your **AI Loan Assistant**! I can answer questions about:
- **Your Personal Profile Eligibility** (*"Why rejected?"* or *"My chances"*)
- **Bank Rules** (*"SBI requirements"* or *"Bank of Baroda"*)
- **CIBIL Score & FOIR calculations**
- **Document checklist & tips to get approved**

Try asking: *"What are SBI rules?"* or *"What is FOIR?"* or *"My status"*."""

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi! 👋 I'm your AI Loan Assistant. Fill out the applicant form or ask me any question about CIBIL, FOIR, or bank rules!"}
    ]
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# ============================================================
# TOP HEADER BAR
# ============================================================
st.markdown("""
<div class="hero-header-banner" style="
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
  border-radius: 18px;
  padding: 24px 32px;
  margin-bottom: 24px;
  box-shadow: 0 8px 30px rgba(15,23,42,.18);
  display: flex;
  align-items: center;
  justify-content: space-between;
  border: 1px solid rgba(255,255,255,0.12);
">
  <div>
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:32px">🏦</span>
      <h1 class="hero-title" style="color:#ffffff !important;margin:0;font-size:1.9rem;font-weight:800;
                 letter-spacing:-0.02em;font-family:'Plus Jakarta Sans',sans-serif;text-shadow:0 2px 8px rgba(0,0,0,0.3);">
        LoanCheck India — AI Loan Approval & Multi-Bank Engine
      </h1>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — INPUT FORM
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 4px 6px 4px;">
      <div style="color:white;font-size:19px;font-weight:800;letter-spacing:-.01em;">📋 Applicant Profile</div>
      <div style="color:#94a3b8;font-size:12px;">Fill parameters to run prediction & multi-bank check</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#38bdf8;font-size:11px;font-weight:700;letter-spacing:.06em;margin:10px 0 2px 0;">PERSONAL & CREDENTIALS</p>', unsafe_allow_html=True)
    person_age     = st.slider("Age", 18, 70, 30)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    education      = st.selectbox("Education Level", ["Bachelor", "Master", "High School", "Associate", "Doctorate"])
    emp_exp        = st.slider("Work Experience (years)", 0, 40, 3)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"],
                                   format_func=lambda x: {"RENT":"🏠 Renting","MORTGAGE":"🏦 Mortgage","OWN":"✅ Own Home","OTHER":"🔄 Other"}[x])
    cred_hist      = st.slider("Credit History Length (years)", 0, 30, 5)
    had_default    = st.selectbox("Previous Loan Default Record", ["No", "Yes (Active Default)", "Settled"])

    st.markdown('<p style="color:#38bdf8;font-size:11px;font-weight:700;letter-spacing:.06em;margin:14px 0 2px 0;">FINANCIAL & LOAN DETAILS (₹)</p>', unsafe_allow_html=True)
    income_inr    = st.number_input("Annual Income (₹)", min_value=100000, max_value=50000000, value=600000, step=50000, format="%d")
    loan_inr      = st.number_input("Loan Amount (₹)", min_value=50000, max_value=10000000, value=500000, step=50000, format="%d")
    int_rate      = st.slider("Interest Rate (% p.a.)", 5.0, 25.0, 11.5, step=0.1)
    tenure_months = st.slider("Tenure (months)", 6, 84, 36, step=6)
    loan_intent   = st.selectbox("Loan Purpose", ["Personal", "Education", "Medical", "Business / Venture", "Home Improvement", "Debt Consolidation"])
    credit_score  = st.slider("Credit Score (CIBIL)", 300, 850, 700)

    st.markdown("---")
    predict_btn = st.button("🔍 Run Instant Loan Prediction", use_container_width=True, type="primary")

# ============================================================
# COMPUTE LIVE FINANCIAL METRICS
# ============================================================
income_usd  = income_inr / USD_TO_INR
loan_usd    = loan_inr   / USD_TO_INR
monthly_inr = income_inr / 12
emi_inr     = calculate_emi(loan_inr, int_rate, tenure_months)
foir_pct    = (emi_inr / monthly_inr * 100) if monthly_inr > 0 else 0.0
feat_dict   = build_features(person_age, gender, education, income_usd, emp_exp,
                             home_ownership, loan_usd, loan_intent, int_rate,
                             credit_score, cred_hist, had_default)
lpi = feat_dict['loan_percent_income']

# ============================================================
# LIVE TOP DASHBOARD CARDS
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Monthly Income", f"₹{monthly_inr:,.0f}")
c2.metric("🏷️ Loan Amount", f"₹{loan_inr:,.0f}")
c3.metric("📅 Monthly EMI", f"₹{emi_inr:,.0f}")

foir_tag = "✅ Ideal" if foir_pct <= 40 else ("⚠️ Marginal" if foir_pct <= 50 else "❌ High Risk")
c4.metric("📊 FOIR", f"{foir_pct:.1f}%", delta=foir_tag, delta_color="normal" if foir_pct <= 40 else "inverse")

lpi_tag = "✅ Safe" if lpi <= 0.35 else "⚠️ High"
c5.metric("📈 Loan vs Income", f"{lpi*100:.1f}%", delta=lpi_tag, delta_color="normal" if lpi <= 0.35 else "inverse")

# Prepare evaluation context
hard_reject, hard_reasons = apply_hard_rules(income_inr, loan_inr, monthly_inr, had_default, credit_score, emp_exp, lpi, foir_pct)
input_df     = pd.DataFrame([feat_dict])[features]
input_scaled = scaler.transform(input_df)
ml_prob      = model.predict_proba(input_scaled)[0][1]

# Hybrid score logic
if had_default == "Yes (Active Default)": default_pts = 0
elif had_default == "Settled": default_pts = 50
else: default_pts = 100

credit_pts  = 100 if credit_score >= 750 else (80 if credit_score >= 700 else (55 if credit_score >= 640 else 20))
income_pts  = 100 if monthly_inr >= 75000 else (75 if monthly_inr >= 25000 else 30)
lti_pts     = 100 if lpi <= 0.25 else (75 if lpi <= 0.35 else 20)
emp_pts     = 100 if emp_exp >= 3 else (70 if emp_exp >= 1 else 30)
foir_pts    = 100 if foir_pct <= 35 else (75 if foir_pct <= 50 else 10)

hybrid = (default_pts*0.25 + credit_pts*0.20 + income_pts*0.15 + lti_pts*0.15 + emp_pts*0.10 + foir_pts*0.15)
final  = (ml_prob*100 * 0.50) + (hybrid * 0.50)

if hard_reject:
    display_prob = min(final, 30.0) / 100
    approved = False
else:
    display_prob = final / 100
    approved = final >= 55

context_payload = {
    "calculated": True,
    "approved": approved,
    "display_prob": display_prob,
    "credit_score": credit_score,
    "foir_pct": foir_pct,
    "monthly_inr": monthly_inr,
    "had_default": had_default
}

# ============================================================
# MAIN TABS SYSTEM
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prediction & Risk Scorecard",
    "🏛️ Multi-Bank Eligibility Matrix",
    "💡 Counterfactual Loan Optimizer",
    "📊 Amortization & Financial Health"
])

# ------------------------------------------------------------
# TAB 1: PREDICTION & RISK SCORECARD
# ------------------------------------------------------------
with tab1:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    
    # BANNER RESULT
    if approved:
        b_bg, b_border, b_icon, b_title, b_sub = "linear-gradient(135deg, #065f46, #047857)", "#10b981", "✅", "LIKELY APPROVED", "Application matches risk benchmarks for standard approval."
    else:
        b_bg, b_border, b_icon, b_title, b_sub = "linear-gradient(135deg, #7f1d1d, #991b1b)", "#f87171", "❌", "LIKELY REJECTED", "Application poses elevated credit risk. Check recommendations in tab 3."

    st.markdown(f"""
    <div style="background:{b_bg};border:1px solid {b_border}55;border-radius:16px;padding:24px 28px;margin-bottom:20px;box-shadow:0 6px 20px rgba(0,0,0,.15);">
      <div style="display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:24px;">
        <div style="font-size:48px">{b_icon}</div>
        <div>
          <div style="color:white;font-size:24px;font-weight:800;">{b_title}</div>
          <div style="color:rgba(255,255,255,.8);font-size:13.5px;margin-top:4px;">{b_sub}</div>
        </div>
        <div style="text-align:center;background:rgba(255,255,255,.12);padding:14px 24px;border-radius:12px;border:1px solid rgba(255,255,255,.2);">
          <div style="color:rgba(255,255,255,.7);font-size:10px;font-weight:700;letter-spacing:.08em;">APPROVAL PROBABILITY</div>
          <div style="color:white;font-size:36px;font-weight:800;">{display_prob*100:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(min(display_prob, 1.0)))

    if hard_reject:
        st.markdown("#### ❌ Critical Rejection Factors")
        for r in hard_reasons:
            st.error(f"• {r}")

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("#### ⚖️ Bank Weightage Score")
        factors = [
            ("Default History", default_pts, 0.25),
            ("Credit Score", credit_pts, 0.20),
            ("Income Level", income_pts, 0.15),
            ("Loan-to-Income", lti_pts, 0.15),
            ("FOIR Ratio", foir_pts, 0.15),
            ("Employment Exp", emp_pts, 0.10)
        ]
        for fname, pts, wt in factors:
            b_col = "#16a34a" if pts >= 75 else ("#d97706" if pts >= 40 else "#dc2626")
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px;margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:600;margin-bottom:6px;">
                <span>{fname}</span>
                <span style="color:{b_col};">{pts}/100 <span style="color:#94a3b8;font-size:11px;">({int(wt*100)}% weight)</span></span>
              </div>
              <div style="background:#f1f5f9;height:7px;border-radius:99px;overflow:hidden;">
                <div style="background:{b_col};width:{pts}%;height:100%;border-radius:99px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### 📋 Detailed Risk Factor Scorecard")
        scorecard = [
            ("Prior Default", had_default, "❌ High Risk" if had_default != "No" else "✅ Clean", "#dc2626" if had_default != "No" else "#16a34a"),
            ("CIBIL Score", f"{credit_score} ({credit_label(credit_score)})", "✅ Strong" if credit_score>=700 else "⚠️ Moderate", credit_color(credit_score)),
            ("Monthly Income", f"₹{monthly_inr:,.0f}", "✅ Eligible" if monthly_inr>=25000 else "⚠️ Low", "#16a34a" if monthly_inr>=25000 else "#d97706"),
            ("FOIR Burden", f"{foir_pct:.1f}%", "✅ Safe" if foir_pct<=40 else "❌ High", "#16a34a" if foir_pct<=40 else "#dc2626"),
            ("Work Experience", f"{emp_exp} years", "✅ Stable" if emp_exp>=3 else "⚠️ Short", "#16a34a" if emp_exp>=3 else "#d97706")
        ]
        for name, val, status, color in scorecard:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;background:white;border:1px solid #e2e8f0;border-radius:10px;padding:10px 16px;margin-bottom:8px;">
              <div>
                <div style="font-size:13px;font-weight:700;color:#0f172a;">{name}</div>
                <div style="font-size:12px;color:#64748b;">{val}</div>
              </div>
              <div style="font-size:13px;font-weight:700;color:{color};">{status}</div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 2: MULTI-BANK ELIGIBILITY MATRIX
# ------------------------------------------------------------
with tab2:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("#### 🏛️ Multi-Bank Eligibility Comparison (Public Nationalised & Private Lenders)")
    st.caption("Real-world criteria evaluation across major Indian Banks & NBFCs")

    bank_results = evaluate_multi_bank(monthly_inr, credit_score, foir_pct, had_default, emp_exp)

    cols = st.columns(2)
    for idx, b in enumerate(bank_results):
        target_col = cols[idx % 2]
        
        if b["status"] == "Eligible":
            badge_bg, badge_col, badge_text = "#f0fdf4", "#16a34a", "✅ Highly Eligible"
            border_col = "#bbf7d0"
        elif b["status"] == "Borderline":
            badge_bg, badge_col, badge_text = "#fffbeb", "#d97706", "⚠️ Borderline / Review"
            border_col = "#fde68a"
        else:
            badge_bg, badge_col, badge_text = "#fef2f2", "#dc2626", "❌ Ineligible"
            border_col = "#fecaca"

        with target_col:
            st.markdown(f"""
            <div style="background:white;border:1px solid {border_col};border-radius:14px;padding:16px 20px;margin-bottom:14px;box-shadow:0 2px 6px rgba(0,0,0,.03);">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                <div>
                  <span style="font-size:18px;margin-right:6px;">{b['icon']}</span>
                  <span style="font-size:15px;font-weight:800;color:#0f172a;">{b['bank']}</span>
                  <div style="font-size:11px;color:#64748b;margin-left:26px;">{b['type']}</div>
                </div>
                <div style="background:{badge_bg};color:{badge_col};padding:4px 10px;border-radius:99px;font-size:11px;font-weight:700;">
                  {badge_text}
                </div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;background:#f8fafc;padding:8px 12px;border-radius:8px;margin-top:8px;">
                <div>Est. Interest Rate: <b>{b['rate']}</b></div>
                <div>Processing Fee: <b>{b['fee']}</b></div>
              </div>
              {'<div style="font-size:11px;color:#dc2626;margin-top:6px;"><b>Reason:</b> ' + ', '.join(b['reasons']) + '</div>' if b['reasons'] else ''}
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 3: COUNTERFACTUAL LOAN OPTIMIZER ("WHAT-IF")
# ------------------------------------------------------------
with tab3:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("#### 💡 Interactive \"What-If\" Counterfactual Optimizer")
    st.caption("Automatically calculates parameter adjustments needed to achieve loan approval")

    if approved and not hard_reject:
        st.success("🎉 Your current profile is already in the **APPROVAL** zone! You don't need emergency counterfactual changes.")

    st.markdown("##### 🛠️ Recommended Action Scenarios:")
    
    opt_col1, opt_col2 = st.columns(2, gap="large")

    with opt_col1:
        # Scenario 1: Reduce Loan Amount
        target_max_loan = monthly_inr * 0.40 * tenure_months / 1.15
        if target_max_loan < loan_inr:
            st.markdown(f"""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:16px;margin-bottom:12px;">
              <div style="font-size:14px;font-weight:700;color:#1d4ed8;">Option 1: Reduce Loan Amount</div>
              <div style="font-size:12.5px;color:#1e3a8a;margin-top:4px;">
                Reduce loan from <b>₹{loan_inr:,.0f}</b> to <b>₹{max(50000, target_max_loan):,.0f}</b> to drop your FOIR below 40%.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Scenario 2: Increase Tenure
        if tenure_months < 60:
            new_tenure = min(84, tenure_months + 24)
            new_emi = calculate_emi(loan_inr, int_rate, new_tenure)
            new_foir = (new_emi / monthly_inr) * 100
            st.markdown(f"""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:16px;margin-bottom:12px;">
              <div style="font-size:14px;font-weight:700;color:#16a34a;">Option 2: Extend Tenure</div>
              <div style="font-size:12.5px;color:#15803d;margin-top:4px;">
                Increase tenure from <b>{tenure_months}</b> to <b>{new_tenure} months</b>. EMI drops to ₹{new_emi:,.0f}/mo (FOIR: {new_foir:.1f}%).
              </div>
            </div>
            """, unsafe_allow_html=True)

    with opt_col2:
        # Scenario 3: CIBIL Score Improvement
        if credit_score < 720:
            st.markdown(f"""
            <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:12px;padding:16px;margin-bottom:12px;">
              <div style="font-size:14px;font-weight:700;color:#d97706;">Option 3: Improve CIBIL Score</div>
              <div style="font-size:12.5px;color:#92400e;margin-top:4px;">
                Boosting score from <b>{credit_score}</b> to <b>720+</b> opens instant approval across BOB, PNB, SBI, and HDFC.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Scenario 4: Add Co-Applicant Income
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:12px;padding:16px;margin-bottom:12px;">
          <div style="font-size:14px;font-weight:700;color:#334155;">Option 4: Add Co-Applicant</div>
          <div style="font-size:12.5px;color:#475569;margin-top:4px;">
            Adding a spouse or parent with ₹25,000/mo income increases joint household capability and satisfies debt ratios.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 4: AMORTIZATION & FINANCIAL HEALTH
# ------------------------------------------------------------
with tab4:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("#### 📊 Loan Repayment Breakdown & Amortization")

    total_repayment = emi_inr * tenure_months
    total_interest  = total_repayment - loan_inr

    am1, am2 = st.columns([1, 1])

    with am1:
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:14px;padding:20px;box-shadow:0 2px 6px rgba(0,0,0,.04);">
          <h5 style="color:#0f172a;margin-bottom:12px;">Cost Summary</h5>
          <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9;">
            <span>Principal Amount:</span><b>₹{loan_inr:,.0f}</b>
          </div>
          <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9;">
            <span>Total Interest Payable:</span><b style="color:#dc2626">₹{total_interest:,.0f}</b>
          </div>
          <div style="display:flex;justify-content:space-between;padding:8px 0;">
            <span>Total Repayment Amount:</span><b style="color:#2563eb">₹{total_repayment:,.0f}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with am2:
        # Pie chart visualization
        fig, ax = plt.subplots(figsize=(4, 2.8))
        ax.pie([loan_inr, max(0, total_interest)], labels=['Principal', 'Interest'],
               colors=['#2563eb', '#f87171'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

# ============================================================
# FLOATING CHATBOT WIDGET (BOTTOM-RIGHT FIXED POPUP)
# ============================================================

with st.sidebar:
    st.markdown("---")
    st.markdown('<p style="color:#38bdf8;font-size:12px;font-weight:700;letter-spacing:.06em;">💬 AI LOAN CHATBOT</p>', unsafe_allow_html=True)
    sidebar_label = "✖ Close Chat Popup" if st.session_state.chat_open else "💬 Open AI Loan Assistant"
    if st.button(sidebar_label, use_container_width=True, key="sidebar_chat_btn"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

# Floating FAB button fixed in bottom right corner
with st.container(key="floating_fab_box"):
    fab_label = "✖ Close Assistant" if st.session_state.chat_open else "💬 Ask AI Assistant"
    if st.button(fab_label, key="fab_chat_toggle"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

# Floating Chat Popup Card — ALL chat elements are strictly INSIDE this popup!
if st.session_state.chat_open:
    with st.container(key="floating_chat_popup"):
        # Header row
        p_c1, p_c2 = st.columns([5, 1])
        with p_c1:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="font-size:22px">🤖</span>
              <div>
                <div style="font-weight:800;font-size:14px;color:#0f172a;line-height:1.2;">AI Loan Assistant</div>
                <div style="font-size:11px;color:#16a34a;font-weight:600;">● Live Profile Active</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with p_c2:
            if st.button("✖", key="popup_close_btn", help="Close"):
                st.session_state.chat_open = False
                st.rerun()

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

        # Quick question chips
        q1, q2, q3 = st.columns(3)
        if q1.button("FOIR?", key="q_foir", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "What is FOIR?"})
            res = get_context_chat_response("What is FOIR?", context_payload)
            st.session_state.chat_history.append({"role": "assistant", "content": res})
            st.rerun()
        if q2.button("SBI Rules", key="q_sbi", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "What are SBI requirements?"})
            res = get_context_chat_response("What are SBI requirements?", context_payload)
            st.session_state.chat_history.append({"role": "assistant", "content": res})
            st.rerun()
        if q3.button("My Chances", key="q_chances", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "What are my chances?"})
            res = get_context_chat_response("What are my chances?", context_payload)
            st.session_state.chat_history.append({"role": "assistant", "content": res})
            st.rerun()

        # Messages container inside the popup
        chat_box = st.container(height=280)
        with chat_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
                    st.markdown(msg["content"])

        # Input form inside the popup
        with st.form("popup_chat_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns([5, 1])
            user_msg = f_col1.text_input("Ask question", placeholder="Ask about FOIR, CIBIL, banks...", label_visibility="collapsed")
            f_submit = f_col2.form_submit_button("➤")
            if f_submit and user_msg:
                st.session_state.chat_history.append({"role": "user", "content": user_msg})
                res = get_context_chat_response(user_msg, context_payload)
                st.session_state.chat_history.append({"role": "assistant", "content": res})
                st.rerun()

        # Bottom actions
        b_c1, b_c2 = st.columns([1, 1])
        if b_c1.button("🗑️ Clear History", key="clear_chat_btn", use_container_width=True):
            st.session_state.chat_history = [{"role": "assistant", "content": "Chat history reset! How can I assist you with your loan application?"}]
            st.rerun()
        if b_c2.button("📄 Documents?", key="q_docs", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "What documents are required?"})
            res = get_context_chat_response("What documents are required?", context_payload)
            st.session_state.chat_history.append({"role": "assistant", "content": res})
            st.rerun()