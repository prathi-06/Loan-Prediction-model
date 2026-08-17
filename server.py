"""
LoanCheck India — Flask Backend
Serves the ML-powered loan prediction API and the frontend static files.
"""

from flask import Flask, jsonify, request, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder="static", static_url_path="/static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load Model Artifacts ──────────────────────────────────────
model     = joblib.load(os.path.join(BASE_DIR, "model", "loan_model.pkl"))
scaler    = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
features  = joblib.load(os.path.join(BASE_DIR, "model", "feature_names.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "model", "threshold.pkl"))

# ── Encodings ────────────────────────────────────────────────
HOME_MAP = {"RENT": 0, "MORTGAGE": 1, "OWN": 2, "OTHER": 0}
INTENT_MAP = {
    "Personal": 0, "Education": 1, "Medical": 2,
    "Business": 3, "HomeImprovement": 4, "DebtConsolidation": 5
}
EDU_MAP = {
    "HighSchool": 0, "Associate": 1, "Bachelor": 2,
    "Master": 3, "Doctorate": 4
}

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

def credit_color(score):
    if score < 580:   return "#ef4444"
    elif score < 640: return "#f97316"
    elif score < 700: return "#eab308"
    elif score < 750: return "#22c55e"
    else:             return "#0ea5e9"

def calculate_emi(principal, annual_rate_pct, tenure_months):
    if tenure_months <= 0 or principal <= 0:
        return 0.0
    r = (annual_rate_pct / 12) / 100
    if r == 0:
        return principal / tenure_months
    factor = (1 + r) ** tenure_months
    return principal * r * factor / (factor - 1)

def apply_hard_rules(income_inr, loan_inr, monthly_inr,
                     had_default, credit_score, emp_exp, lpi, foir):
    reasons = []
    if had_default == "ActiveDefault":
        reasons.append("Active loan default on record — rejected by all Indian lenders.")
    elif had_default == "Settled":
        weak = sum([
            credit_score < 650,
            foir > 50,
            monthly_inr < 25000,
            lpi > 0.50,
            emp_exp < 1
        ])
        if weak >= 3:
            reasons.append(f"Settled default + {weak} other high-risk factors — risk too high.")
    if credit_score < 500:
        reasons.append(f"CIBIL {credit_score} is critically below the 500 floor.")
    if monthly_inr < 15000:
        reasons.append(f"Monthly income ₹{monthly_inr:,.0f} is below ₹15,000 SBI baseline.")
    if loan_inr > income_inr * 10:
        reasons.append(f"Loan is {loan_inr/income_inr:.1f}× annual income — exceeds 10× hard cap.")
    if foir > 65:
        reasons.append(f"FOIR {foir:.1f}% exceeds 65% extreme cap.")
    return len(reasons) > 0, reasons

# ── Banks Database ────────────────────────────────────────────
BANKS = [
    {"name": "State Bank of India", "short": "SBI", "type": "Nationalised", "icon": "🏛️",
     "min_salary": 15000, "min_cibil": 700, "max_foir": 50, "rate": "11.15% – 15.30%", "fee": "1.0% (min ₹1,000)"},
    {"name": "Bank of Baroda", "short": "BOB", "type": "Nationalised", "icon": "🏛️",
     "min_salary": 15000, "min_cibil": 680, "max_foir": 50, "rate": "10.90% – 14.75%", "fee": "1.0% + GST"},
    {"name": "Indian Overseas Bank", "short": "IOB", "type": "Nationalised", "icon": "🏛️",
     "min_salary": 15000, "min_cibil": 675, "max_foir": 50, "rate": "11.00% – 15.00%", "fee": "0.75% + GST"},
    {"name": "Punjab National Bank", "short": "PNB", "type": "Nationalised", "icon": "🏛️",
     "min_salary": 15000, "min_cibil": 680, "max_foir": 50, "rate": "10.95% – 14.85%", "fee": "1.0%"},
    {"name": "HDFC Bank", "short": "HDFC", "type": "Private", "icon": "🏦",
     "min_salary": 25000, "min_cibil": 720, "max_foir": 55, "rate": "10.50% – 21.00%", "fee": "Up to ₹4,999"},
    {"name": "ICICI Bank", "short": "ICICI", "type": "Private", "icon": "🏦",
     "min_salary": 25000, "min_cibil": 700, "max_foir": 55, "rate": "10.65% – 16.00%", "fee": "Up to 2.25%"},
    {"name": "Axis Bank", "short": "AXIS", "type": "Private", "icon": "🏦",
     "min_salary": 15000, "min_cibil": 700, "max_foir": 50, "rate": "10.75% – 22.00%", "fee": "1.0% – 2.0%"},
    {"name": "Bajaj Finance", "short": "BAJAJ", "type": "NBFC", "icon": "💼",
     "min_salary": 20000, "min_cibil": 650, "max_foir": 60, "rate": "13.00% – 26.00%", "fee": "Up to 3.99%"},
]

def evaluate_banks(monthly_inr, credit_score, foir_pct, had_default, emp_exp):
    results = []
    for b in BANKS:
        status, reasons = "eligible", []
        if had_default == "ActiveDefault":
            status = "ineligible"; reasons.append("Active Default")
        else:
            if monthly_inr < b["min_salary"]:
                status = "ineligible"; reasons.append(f"Income < ₹{b['min_salary']:,.0f}")
            if credit_score < b["min_cibil"] - 30:
                status = "ineligible"; reasons.append(f"CIBIL < {b['min_cibil'] - 30}")
            elif credit_score < b["min_cibil"]:
                if status == "eligible": status = "borderline"
                reasons.append(f"CIBIL near {b['min_cibil']} threshold")
            if foir_pct > b["max_foir"] + 5:
                status = "ineligible"; reasons.append(f"FOIR > {b['max_foir']+5}%")
            elif foir_pct > b["max_foir"]:
                if status == "eligible": status = "borderline"
                reasons.append(f"FOIR above {b['max_foir']}%")
            if emp_exp < 1 and status == "eligible":
                status = "borderline"; reasons.append("Exp < 1 yr")
        results.append({**b, "status": status, "reasons": reasons})
    return results

# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR + "/static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    d = request.json

    # Parse inputs
    income_inr           = float(d["income_inr"])
    has_coapplicant      = d.get("has_coapplicant", "No")
    co_applicant_income  = float(d.get("co_applicant_income") or 0) if has_coapplicant == "Yes" else 0.0
    co_applicant_default = d.get("co_applicant_default", "No") if has_coapplicant == "Yes" else "No"
    loan_inr             = float(d["loan_inr"])
    int_rate             = float(d["int_rate"])
    tenure_months        = int(d["tenure_months"])
    credit_score         = int(d["credit_score"])
    cred_hist            = int(d["cred_hist"])
    emp_exp              = float(d["emp_exp"])
    person_age           = int(d["person_age"])
    gender               = d["gender"]
    education            = d["education"]
    home_ownership       = d["home_ownership"]
    loan_intent          = d["loan_intent"]
    had_default          = d["had_default"]

    # Joint Default Evaluation
    if has_coapplicant == "Yes" and (co_applicant_default == "ActiveDefault" or had_default == "ActiveDefault"):
        joint_default = "ActiveDefault"
    elif has_coapplicant == "Yes" and (co_applicant_default == "Settled" or had_default == "Settled"):
        joint_default = "Settled"
    else:
        joint_default = had_default

    # Joint income calculation (feeds both ML model & banking rules)
    total_income_inr = income_inr + co_applicant_income
    monthly_inr  = total_income_inr / 12
    income_usd   = total_income_inr / 83
    loan_usd     = loan_inr   / 83
    monthly_usd  = income_usd / 12
    lpi          = loan_usd / income_usd if income_usd > 0 else 0
    emi_inr      = calculate_emi(loan_inr, int_rate, tenure_months)
    foir_pct     = (emi_inr / monthly_inr * 100) if monthly_inr > 0 else 0

    # Feature build — Joint income & joint default directly update XGBoost ML features
    feat = {
        "person_age":                 person_age,
        "gender_enc":                 1 if gender == "Male" else 0,
        "edu_enc":                    EDU_MAP.get(education, 2),
        "person_emp_exp":             emp_exp,
        "home_enc":                   HOME_MAP.get(home_ownership, 0),
        "low_emp_flag":               int(emp_exp < 1),
        "credit_grade":               credit_grade(credit_score),
        "fine_credit_score":          credit_score,               # Continuous CIBIL score
        "had_default":                1 if joint_default == "ActiveDefault" else 0,
        "cb_person_cred_hist_length": cred_hist,
        "cred_hist_band":             min(3, max(0, cred_hist // 2)),  # 0-3 band
        "short_hist_flag":            int(cred_hist < 2),
        "monthly_income":             monthly_usd,             # ← Joint income (ML prediction)
        "loan_percent_income":        lpi,                     # ← Joint LTI (ML prediction)
        "high_debt_flag":             int(lpi > 0.35),          # ← Joint debt flag
        "low_income_flag":            int(monthly_usd < 2000),  # ← Joint income flag
        "loan_amnt":                  loan_usd,
        "loan_int_rate":              int_rate,
        "intent_enc":                 INTENT_MAP.get(loan_intent, 0),
        # ── Interaction features (break had_default monopoly) ──────────────
        "income_x_grade":             monthly_usd * credit_grade(credit_score),
        "debt_x_intent":              lpi * INTENT_MAP.get(loan_intent, 0),
        "age_x_emp":                  person_age * emp_exp,
        "income_debt_ratio":          monthly_usd / (loan_usd + 1),
        "credit_x_hist":              credit_score * cred_hist,
        "rate_burden":                int_rate * lpi,
        "emp_income_combo":           emp_exp * monthly_usd,
    }

    import pandas as pd
    inp = pd.DataFrame([feat])[features]
    inp_scaled = scaler.transform(inp)
    ml_prob = float(model.predict_proba(inp_scaled)[0][1])

    # Hard rules
    hard_reject, hard_reasons = apply_hard_rules(
        total_income_inr, loan_inr, monthly_inr,
        joint_default, credit_score, emp_exp, lpi, foir_pct
    )

    # ── BALANCED Hybrid Score (CIBIL not dominant) ────────────
    if joint_default == "ActiveDefault":
        default_pts = 0
    elif joint_default == "Settled":
        default_pts = 45
    else:
        default_pts = 100

    # Credit score — softened band (does not dominate)
    if credit_score >= 750:   credit_pts = 100
    elif credit_score >= 700: credit_pts = 82
    elif credit_score >= 660: credit_pts = 65  # ← softened (was 55)
    elif credit_score >= 620: credit_pts = 48  # ← softened (was 30)
    elif credit_score >= 580: credit_pts = 30
    else:                     credit_pts = 10

    # Income — weighted more heavily now
    if monthly_inr >= 100000:  income_pts = 100
    elif monthly_inr >= 50000: income_pts = 88
    elif monthly_inr >= 25000: income_pts = 72
    elif monthly_inr >= 15000: income_pts = 50
    else:                      income_pts = 20

    # FOIR — significant weight
    if foir_pct <= 30:   foir_pts = 100
    elif foir_pct <= 40: foir_pts = 85
    elif foir_pct <= 50: foir_pts = 60
    elif foir_pct <= 60: foir_pts = 30
    else:                foir_pts = 5

    # Employment
    if emp_exp >= 5:   emp_pts = 100
    elif emp_exp >= 3: emp_pts = 85
    elif emp_exp >= 1: emp_pts = 65
    else:              emp_pts = 25

    # Loan-to-Income
    if lpi <= 0.20:   lti_pts = 100
    elif lpi <= 0.30: lti_pts = 82
    elif lpi <= 0.40: lti_pts = 60
    elif lpi <= 0.55: lti_pts = 35
    else:             lti_pts = 10

    # Credit history bonus (not penalty if low)
    if cred_hist >= 7:   hist_pts = 100
    elif cred_hist >= 4: hist_pts = 80
    elif cred_hist >= 2: hist_pts = 60
    else:                hist_pts = 40

    # ── Balanced weights: no single factor > 20% ──────────────
    hybrid = (
        default_pts * 0.20 +
        credit_pts  * 0.15 +
        income_pts  * 0.20 +
        foir_pts    * 0.15 +
        emp_pts     * 0.15 +
        lti_pts     * 0.10 +
        hist_pts    * 0.05
    )

    # Factor-of-safety: ML covers model signals, hybrid covers bank logic
    final = (ml_prob * 100 * 0.45) + (hybrid * 0.55)

    if hard_reject:
        display_prob = min(final, 28.0)
        approved = False
    else:
        display_prob = final
        approved = final >= 52

    # Strengths & risks
    strengths, risks = [], []
    if has_coapplicant == "Yes" and co_applicant_income > 0:
        strengths.append(f"Joint application with co-applicant income (₹{co_applicant_income:,.0f}/yr) — directly improves ML prediction & joint FOIR")
    if has_coapplicant == "Yes" and co_applicant_default == "ActiveDefault":
        risks.append("Co-applicant has an ACTIVE DEFAULT on record — causes automatic joint rejection across all lenders")
    elif has_coapplicant == "Yes" and co_applicant_default == "Settled":
        risks.append("Co-applicant has a SETTLED/WRITTEN-OFF loan on record — high risk flag for joint application")

    if had_default == "No" and (has_coapplicant == "No" or co_applicant_default == "No"):
        strengths.append("Clean repayment history — strong positive signal")
    elif had_default == "Settled":
        risks.append("Primary applicant has a settled loan on record — high risk review")
    elif had_default == "ActiveDefault":
        risks.append("Primary applicant has an active default — automatic rejection across all lenders")

    if credit_score >= 750: strengths.append(f"Excellent CIBIL {credit_score} — best interest rates available")
    elif credit_score >= 700: strengths.append(f"Good CIBIL {credit_score} — standard bank approval")
    elif credit_score < 640: risks.append(f"CIBIL {credit_score} is below preferred 640+ threshold")

    if monthly_inr >= 50000: strengths.append(f"Strong joint income ₹{monthly_inr:,.0f}/mo — exceeds all bank minimums")
    elif monthly_inr >= 25000: strengths.append(f"Adequate joint income ₹{monthly_inr:,.0f}/mo — meets HDFC/ICICI standard")
    elif monthly_inr < 15000: risks.append(f"Joint income ₹{monthly_inr:,.0f}/mo below SBI's ₹15,000 floor")

    if foir_pct <= 40: strengths.append(f"FOIR {foir_pct:.1f}% — within ideal ≤40% zone")
    elif foir_pct > 55: risks.append(f"FOIR {foir_pct:.1f}% — exceeds acceptable 55% ceiling")
    else: risks.append(f"FOIR {foir_pct:.1f}% — in the 40–55% borderline zone")

    if emp_exp >= 3: strengths.append(f"{emp_exp:.0f} years employment — stable income confirmed")
    elif emp_exp < 1: risks.append("Zero work experience — high risk for larger loan amounts")

    if lpi <= 0.30: strengths.append(f"Loan-to-income {lpi*100:.1f}% — well within safe range")
    elif lpi > 0.50: risks.append(f"Loan-to-income {lpi*100:.1f}% — above the preferred 50% ceiling")

    # Multi-bank evaluation
    bank_results = evaluate_banks(monthly_inr, credit_score, foir_pct, had_default, emp_exp)

    # What-if suggestions
    whatif = []
    if foir_pct > 40:
        new_tenure = min(84, tenure_months + 12)
        new_emi = calculate_emi(loan_inr, int_rate, new_tenure)
        new_foir = (new_emi / monthly_inr * 100) if monthly_inr > 0 else 0
        whatif.append({
            "title": f"Extend Tenure to {new_tenure} Months",
            "desc": f"New EMI: ₹{new_emi:,.0f}/mo → FOIR drops to {new_foir:.1f}%",
            "type": "green"
        })
    if loan_inr > monthly_inr * 30:
        reduced = monthly_inr * 30
        whatif.append({
            "title": f"Reduce Loan to ₹{reduced:,.0f}",
            "desc": f"Keeps loan within 2.5× annual income safety range",
            "type": "blue"
        })
    if credit_score < 720:
        whatif.append({
            "title": "Improve CIBIL to 720+",
            "desc": "Pay outstanding dues & keep credit card usage below 30% for 3 months",
            "type": "amber"
        })
    if has_coapplicant == "No":
        whatif.append({
            "title": "Add a Co-Applicant",
            "desc": "Select 'Yes' under Co-Applicant status and enter income to boost joint ML prediction & FOIR",
            "type": "purple"
        })

    # Amortization schedule (yearly summary starting from Month 0)
    amort = [{
        "month": 0,
        "balance": round(loan_inr, 0),
        "total_paid": 0,
        "total_interest": 0
    }]
    balance = loan_inr
    monthly_rate = (int_rate / 12) / 100
    total_paid = total_interest = 0
    for month in range(1, tenure_months + 1):
        interest = balance * monthly_rate
        principal_payment = emi_inr - interest
        balance = max(0, balance - principal_payment)
        total_paid += emi_inr
        total_interest += interest
        if month % 12 == 0 or month == tenure_months:
            amort.append({
                "month": month,
                "balance": round(balance, 0),
                "total_paid": round(total_paid, 0),
                "total_interest": round(total_interest, 0)
            })

    return jsonify({
        "approved": approved,
        "display_prob": round(display_prob, 1),
        "ml_prob": round(ml_prob * 100, 1),
        "hybrid": round(hybrid, 1),
        "hard_reject": hard_reject,
        "hard_reasons": hard_reasons,
        "factors": {
            "default_pts": default_pts,
            "credit_pts": credit_pts,
            "income_pts": income_pts,
            "foir_pts": foir_pts,
            "emp_pts": emp_pts,
            "lti_pts": lti_pts,
            "hist_pts": hist_pts,
        },
        "metrics": {
            "monthly_inr": round(monthly_inr, 0),
            "emi_inr": round(emi_inr, 0),
            "foir_pct": round(foir_pct, 1),
            "lpi_pct": round(lpi * 100, 1),
            "total_repayment": round(emi_inr * tenure_months, 0),
            "total_interest": round(emi_inr * tenure_months - loan_inr, 0),
            "credit_label": credit_label(credit_score),
            "credit_color": credit_color(credit_score),
        },
        "strengths": strengths,
        "risks": risks,
        "bank_results": bank_results,
        "whatif": whatif,
        "amort": amort,
    })


@app.route("/chat", methods=["POST"])
def chat():
    d = request.json
    msg = d.get("message", "").lower().strip()
    ctx = d.get("context", {})

    # Context-aware answers
    if any(k in msg for k in ["my chances", "my loan", "why rejected", "will i", "my status", "my profile"]):
        if ctx.get("has_result"):
            prob = ctx.get("prob", 0)
            approved = ctx.get("approved", False)
            foir = ctx.get("foir", 0)
            cibil = ctx.get("cibil", 0)
            income = ctx.get("income", 0)
            status = "✅ LIKELY APPROVED" if approved else "❌ LIKELY REJECTED"
            reply = f"**Your Current Profile Analysis:**\n\n📊 **Status:** {status} ({prob:.1f}% probability)\n💳 **CIBIL:** {cibil} ({credit_label(cibil)})\n📈 **FOIR:** {foir:.1f}%\n💰 **Monthly Income:** ₹{income:,.0f}\n\n"
            if not approved:
                reply += "**Quick fixes to get approved:**\n"
                if foir > 40: reply += "• Extend loan tenure to lower your monthly EMI and FOIR\n"
                if cibil < 700: reply += "• Pay off credit card dues to push CIBIL above 700\n"
                reply += "• Consider adding a co-applicant"
            else:
                reply += "Your profile is strong! Compare bank offers on the Bank Matrix tab."
            return jsonify({"reply": reply})
        return jsonify({"reply": "Please fill in the form and click **Predict** first, so I can analyse your personal profile!"})

    if any(k in msg for k in ["bob", "baroda", "iob", "overseas", "pnb", "punjab national", "nationalised", "government bank", "public bank", "sbi"]):
        return jsonify({"reply": "**🏛️ Nationalised/Public Bank Requirements (SBI, BOB, IOB, PNB):**\n\n• **Min Monthly Income:** ₹15,000\n• **Min CIBIL:** 675–700\n• **Max FOIR:** 50%\n• **Interest Rates:** 10.90% – 15.30%\n• **Advantage:** Lower rates, no hidden charges, flexible tenure up to 84 months\n• **SBI** is the most lenient with rural/semi-urban applicants"})

    if any(k in msg for k in ["hdfc", "icici", "axis", "private bank"]):
        return jsonify({"reply": "**🏦 Private Bank Requirements (HDFC, ICICI, Axis):**\n\n• **Min Monthly Income:** ₹25,000 (HDFC/ICICI), ₹15,000 (Axis)\n• **Min CIBIL:** 700–720\n• **Max FOIR:** 50–55%\n• **Interest Rates:** 10.50% – 22.00%\n• **Advantage:** Instant digital approval, quick disbursal in 24–48 hours"})

    if any(k in msg for k in ["cibil", "credit score", "credit rating"]):
        return jsonify({"reply": "**📊 CIBIL Score Impact on Loans:**\n\n| Score | Rating | Typical Rate |\n|-------|--------|--------------|\n| 750+ | Excellent | 10.5%–11.5% |\n| 700–749 | Good | 11.5%–14% |\n| 660–699 | Fair | 14%–18% |\n| 580–659 | Poor | High risk |\n| <580 | Very Poor | Likely rejected |\n\n**Note:** CIBIL is *one of many* factors. A high income & low FOIR can partly compensate for a lower score."})

    if any(k in msg for k in ["foir", "debt ratio", "fixed obligation"]):
        return jsonify({"reply": "**💰 FOIR (Fixed Obligation to Income Ratio):**\n\n`FOIR = All EMIs ÷ Monthly Income × 100`\n\n• **≤ 40%** → Ideal zone ✅\n• **40–50%** → Borderline, case-by-case ⚠️\n• **50–55%** → Risky, higher rate ⚠️\n• **> 55%** → Usually rejected ❌\n\n*Tip: Extending tenure reduces EMI and improves FOIR*"})

    if any(k in msg for k in ["emi", "monthly payment", "installment"]):
        return jsonify({"reply": "**📅 EMI Formula:**\n\n`EMI = P × r × (1+r)ⁿ ÷ ((1+r)ⁿ − 1)`\n\nWhere: P = Principal, r = Monthly Rate, n = Tenure in months\n\n**Example:** ₹5L @ 12% for 36 months → **₹16,607/mo**\n\nKeep EMI below 40% of monthly income for best approval odds."})

    if any(k in msg for k in ["default", "defaulted", "overdue", "npa"]):
        return jsonify({"reply": "**⚠️ Loan Default Impact:**\n\n• **Active Default (unpaid):** Auto-rejected at all Indian banks & NBFCs\n• **Settled:** Treated as high-risk; approval with strong income/CIBIL possible\n• Stays on CIBIL report for **7 years**\n\n**Recovery Path:**\n1. Clear all dues → Get NDC (No Dues Certificate)\n2. Request bank to update CIBIL to 'Closed'\n3. Wait 2–3 years → Reapply with improved profile"})

    if any(k in msg for k in ["improve", "tips", "how to get", "increase chances", "boost"]):
        return jsonify({"reply": "**🚀 Top 6 Ways to Improve Loan Approval:**\n\n1. ✅ Clear all existing EMI dues on time\n2. ✅ Keep CIBIL credit card usage under 30%\n3. ✅ Reduce FOIR below 40% (reduce debt or extend tenure)\n4. ✅ Stay employed at current firm for 3+ years\n5. ✅ Add a co-applicant with stable income\n6. ✅ Apply to nationalised banks (SBI, BOB) first — more flexible criteria"})

    if any(k in msg for k in ["documents", "docs", "papers", "required"]):
        return jsonify({"reply": "**📄 Documents Required:**\n\n**Salaried Applicants:**\n• Aadhaar Card + PAN Card\n• Last 3 salary slips\n• 6-month bank statement\n• Form 16 / IT Returns\n• Employment letter\n\n**Self-Employed:**\n• ITR for past 2 years\n• GST Registration & Returns\n• 12-month business bank statement\n• CA-certified P&L statement"})

    if any(k in msg for k in ["hi", "hello", "hey", "namaste", "hii", "help"]):
        return jsonify({"reply": "👋 **Hello! I'm your AI Loan Assistant.**\n\nI can help you with:\n• 🏛️ Nationalised bank requirements (SBI, BOB, IOB, PNB)\n• 💳 CIBIL score & credit tips\n• 📊 FOIR & EMI calculations\n• ⚠️ Default recovery steps\n• 🚀 Approval improvement tips\n• 📄 Document checklist\n\nOr click **Predict** and I'll analyse **your specific profile** live!"})

    return jsonify({"reply": "I can help with **CIBIL scores, FOIR, EMI, bank requirements (SBI/BOB/IOB/PNB/HDFC/ICICI), defaults, documents, or your personal profile**.\n\nTry: *\"What are SBI requirements?\"* or *\"How to improve CIBIL?\"* or *\"What is FOIR?\"*"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
