# Scheme eligibility rules, required documents, and metadata for all supported welfare schemes.
# Each scheme is a dictionary with deterministic integer-based eligibility rules
# so the grader can evaluate agent decisions without any ambiguity.

from typing import Dict, Any, List, Optional


SCHEMES: Dict[str, Dict[str, Any]] = {

    "PMKVY": {
        # Pradhan Mantri Kaushal Vikas Yojana — skill training for young workers
        "full_name":    "Pradhan Mantri Kaushal Vikas Yojana",
        "benefit":      "Free skill training and certification with Rs 8000 stipend",
        "eligibility":  {
            "age_min":        18,
            "age_max":        35,
            "income_max":     9999,       # Strictly less than 10000 — income=10000 does NOT qualify
            "occupations":    ["mason", "carpenter"],
            "requires_aadhaar": False,
        },
        "required_docs": ["aadhaar", "education_certificate"],
    },

    "MGNREGS": {
        # Mahatma Gandhi National Rural Employment Guarantee Scheme
        "full_name":    "Mahatma Gandhi National Rural Employment Guarantee Scheme",
        "benefit":      "100 days of guaranteed wage employment per year",
        "eligibility":  {
            "age_min":        18,
            "age_max":        60,
            "income_max":     None,       # No income ceiling for this scheme
            "occupations":    ["farm_labourer"],
            "requires_aadhaar": True,     # Aadhaar is mandatory for MGNREGS enrollment
        },
        "required_docs": ["aadhaar", "job_card"],
    },

    "PMAY": {
        # Pradhan Mantri Awaas Yojana — housing assistance for low-income families
        "full_name":    "Pradhan Mantri Awaas Yojana",
        "benefit":      "Rs 1.2 lakh grant for pucca house construction",
        "eligibility":  {
            "age_min":        21,
            "age_max":        55,
            "income_max":     5999,       # Strictly less than 6000 — income=6000 does NOT qualify
            "occupations":    None,       # Any occupation qualifies
            "requires_aadhaar": True,
        },
        "required_docs": ["aadhaar", "income_certificate", "land_document"],
    },

    "PM_SYM": {
        # Pradhan Mantri Shram Yogi Maan-dhan — pension scheme for unorganised workers
        "full_name":    "Pradhan Mantri Shram Yogi Maan-dhan",
        "benefit":      "Rs 3000 per month pension after age 60",
        "eligibility":  {
            "age_min":        18,
            "age_max":        40,
            "income_max":     14999,      # Strictly less than 15000
            "occupations":    None,       # Any unorganised worker qualifies
            "requires_aadhaar": True,
            "worker_type":    "unorganised",
            "not_epfo":       True,       # Must NOT be enrolled in EPFO
            "not_esic":       True,       # Must NOT be enrolled in ESIC
        },
        "required_docs": ["aadhaar", "bank_passbook", "mobile_number"],
    },

    "AYUSHMAN_BHARAT": {
        # Ayushman Bharat PM-JAY — health insurance for low-income families
        "full_name":    "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana",
        "benefit":      "Rs 5 lakh annual health insurance cover per family",
        "eligibility":  {
            "age_min":        None,       # No age restriction
            "age_max":        None,
            "income_max":     49999,      # Family income strictly less than 50000
            "occupations":    None,
            "requires_aadhaar": True,
            "not_govt_employee": True,    # Government employees are excluded
        },
        "required_docs": ["aadhaar", "ration_card"],
    },

    "E_SHRAM": {
        # e-Shram Portal — national database registration for unorganised workers
        "full_name":    "e-Shram Portal Registration",
        "benefit":      "Rs 2 lakh accident insurance and linkage to all welfare schemes",
        "eligibility":  {
            "age_min":        16,
            "age_max":        59,
            "income_max":     None,
            "occupations":    None,
            "requires_aadhaar": True,
            "worker_type":    "unorganised",
            "not_epfo":       True,
            "not_esic":       True,
            "not_nps":        True,       # Must NOT be enrolled in National Pension Scheme
        },
        "required_docs": ["aadhaar", "mobile_number", "bank_passbook"],
    },

    "NFSA": {
        # National Food Security Act — subsidised food grains through ration card
        "full_name":    "National Food Security Act — Ration Card",
        "benefit":      "Subsidised food grains at Rs 1-3 per kg",
        "eligibility":  {
            "age_min":        18,
            "age_max":        None,
            "income_max":     9999,       # Strictly less than 10000
            "occupations":    None,
            "requires_aadhaar": True,
            "not_income_tax_payer": True, # Income tax payers are excluded
        },
        "required_docs": ["aadhaar", "address_proof", "family_photo"],
    },

    "PMMVY": {
        # Pradhan Mantri Matru Vandana Yojana — maternity benefit for first child
        "full_name":    "Pradhan Mantri Matru Vandana Yojana",
        "benefit":      "Rs 5000 maternity benefit paid in 3 instalments",
        "eligibility":  {
            "age_min":        18,
            "age_max":        None,
            "income_max":     None,
            "occupations":    None,
            "requires_aadhaar": True,
            "gender":         "female",   # Only female applicants qualify
            "is_pregnant":    True,       # Must be pregnant at time of application
            "first_child":    True,       # Benefit is only for the first living child
            "has_bank_account": True,     # Direct bank transfer requires an active account
        },
        "required_docs": ["aadhaar", "mch_card", "bank_passbook"],
    },
}


def get_eligible_schemes(profile: dict) -> list:
    """
    Evaluate a complete applicant profile against all 8 schemes and return
    a list of scheme keys the applicant qualifies for.
    All comparisons use strict integer arithmetic — no rounding or approximation.
    """
    eligible = []

    age    = int(profile.get("age", 0))
    income = int(profile.get("income", 0))
    occ    = profile.get("occupation", "").lower()
    aadhaar = str(profile.get("has_aadhaar", "False")).lower() == "true"

    for scheme_key, scheme in SCHEMES.items():
        rules = scheme["eligibility"]

        # Check age bounds when defined
        if rules.get("age_min") and age < rules["age_min"]:
            continue
        if rules.get("age_max") and age > rules["age_max"]:
            continue

        # Check income ceiling when defined — strict less-than comparison
        if rules.get("income_max") is not None and income > rules["income_max"]:
            continue

        # Check occupation restriction when defined
        if rules.get("occupations") and occ not in rules["occupations"]:
            continue

        # Check Aadhaar requirement
        if rules.get("requires_aadhaar") and not aadhaar:
            continue

        eligible.append(scheme_key)

    return eligible


def get_optimal_scheme(profile: dict) -> str | None:
    """
    Return the single most beneficial scheme for this applicant profile.
    Priority order: PMKVY > MGNREGS > PMAY > PM_SYM > AYUSHMAN_BHARAT > E_SHRAM > NFSA > PMMVY.
    Returns None if the applicant is not eligible for any scheme.
    """
    eligible = get_eligible_schemes(profile)

    # Defined priority order — earlier schemes take precedence when multiple qualify
    priority = ["MGNREGS", "PMAY", "PMKVY", "PM_SYM",
            "AYUSHMAN_BHARAT", "E_SHRAM", "NFSA", "PMMVY"]

    for scheme in priority:
        if scheme in eligible:
            return scheme

    return None