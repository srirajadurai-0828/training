BANK_NAME = "Horizon Bank"
BANK_TAGLINE = "Banking Built Around You"
BANK_WEBSITE = "www.horizonbank.com"
BANK_HELPLINE = "1800-XXX-XXXX (Toll-free, 24x7)"
BANK_EMAIL = "support@horizonbank.com"
BANK_APP_NAME = "Horizon Mobile"

BANK_HOURS = "Mon–Sat, 9:00 AM – 5:00 PM"

SAVINGS_ACCOUNT = {
    "min_balance": "₹1,000 (Metro) / ₹500 (Rural)",
    "interest_rate": "3.5% p.a.",
    "non_maint_charge": "₹100/quarter if balance falls below minimum",
    "features": [
        "Debit card",
        "Net banking",
        "UPI",
        "SMS alerts",
        "Passbook"
    ],
}

SALARY_ACCOUNT = {
    "min_balance": "Zero balance",
    "interest_rate": "3.5% p.a.",
    "features": [
        "Free debit card",
        "Unlimited ATM transactions",
        "Overdraft up to 2x salary"
    ],
}

CURRENT_ACCOUNT = {
    "min_balance": "₹10,000",
    "features": [
        "Unlimited transactions",
        "Overdraft facility",
        "Multi-city cheque book"
    ],
}

FIXED_DEPOSIT = {
    "min_amount": "₹1,000",
    "tenure_range": "7 days to 10 years",
    "interest_rate": "6.5%–7.2% p.a. (varies by tenure)",
    "senior_citizen_bonus": "0.5% extra",
}

RECURRING_DEPOSIT = {
    "min_monthly": "₹100",
    "tenure_range": "6 months to 10 years",
    "interest_rate": "6.0%–7.0% p.a.",
}

PERSONAL_LOAN = {
    "amount_range": "₹50,000 – ₹25,00,000",
    "interest_rate": "10.5%–18% p.a. (based on credit score)",
    "tenure": "12–60 months",
    "processing_fee": "1%–2% of loan amount",
    "disbursal_time": "48 hours post approval",
}

HOME_LOAN = {
    "amount_range": "Up to ₹5 Crore",
    "interest_rate": "8.5%–9.5% p.a.",
    "tenure": "Up to 30 years",
    "processing_fee": "0.5% of loan amount",
}

VEHICLE_LOAN = {
    "interest_rate": "9.0%–11.5% p.a.",
    "ltv": "Up to 90% of vehicle cost",
    "tenure": "12–84 months",
}

EDUCATION_LOAN = {
    "amount_range": "Up to ₹40 Lakh (abroad) / ₹20 Lakh (India)",
    "interest_rate": "9.5%–12% p.a.",
    "moratorium": "Course duration + 12 months",
}

CREDIT_CARD = {
    "variants": [
        "Horizon Classic",
        "Horizon Gold",
        "Horizon Platinum"
    ],
    "reward_rate": "1 point per ₹100 spent",
    "annual_fee": "₹500–₹2,000 (waived on ₹1L+ annual spend)",
    "credit_limit": "Based on income and credit score",
}

CHARGES = {
    "neft": "Free via app; ₹2–₹25 at branch",
    "rtgs": "Free via app; ₹25–₹50 at branch",
    "imps": "Free up to ₹1,000; ₹5 above that",
    "atm_own": "Free (unlimited)",
    "atm_other": "5 free/month; ₹20 per transaction after",
    "dd_issuance": "₹50–₹150",
    "cheque_bounce": "₹500 per instance",
    "stop_payment": "₹100 per cheque",
    "locker_small": "₹1,500/year",
    "locker_medium": "₹3,000/year",
}

ESCALATION = {
    "nodal_officer_email": "nodal@horizonbank.com",
    "banking_ombudsman": "https://cms.rbi.org.in",
    "grievance_portal": "www.horizonbank.com/grievance",
}


def get_bank_context() -> str:

    return f"""
BANK INFORMATION (use this to answer questions about the bank):

- Name: {BANK_NAME}
- Tagline: {BANK_TAGLINE}
- Website: {BANK_WEBSITE}
- Helpline: {BANK_HELPLINE}
- Email: {BANK_EMAIL}
- Mobile App: {BANK_APP_NAME}
- Branch Hours: {BANK_HOURS}

KEY PRODUCTS:

- Savings Account:
  Min balance {SAVINGS_ACCOUNT['min_balance']}
  Interest {SAVINGS_ACCOUNT['interest_rate']}

- Salary Account:
  {SALARY_ACCOUNT['min_balance']} balance
  Interest {SALARY_ACCOUNT['interest_rate']}

- Fixed Deposit:
  {FIXED_DEPOSIT['tenure_range']}
  Interest {FIXED_DEPOSIT['interest_rate']}

- Recurring Deposit:
  from {RECURRING_DEPOSIT['min_monthly']}/month
  {RECURRING_DEPOSIT['interest_rate']}

- Personal Loan:
  {PERSONAL_LOAN['amount_range']}
  Rate {PERSONAL_LOAN['interest_rate']}
  Disbursal {PERSONAL_LOAN['disbursal_time']}

- Home Loan:
  up to ₹5 Crore
  Rate {HOME_LOAN['interest_rate']}
  Tenure {HOME_LOAN['tenure']}

- Credit Cards:
  {', '.join(CREDIT_CARD['variants'])}

KEY CHARGES:

- NEFT/RTGS/IMPS:
  {CHARGES['neft']} /
  {CHARGES['rtgs']} /
  {CHARGES['imps']}

- ATM (own network):
  {CHARGES['atm_own']}

- ATM (other banks):
  {CHARGES['atm_other']}

ESCALATION:

- Grievance portal:
  {ESCALATION['grievance_portal']}

- Nodal officer:
  {ESCALATION['nodal_officer_email']}

- RBI Ombudsman:
  {ESCALATION['banking_ombudsman']}
""".strip()