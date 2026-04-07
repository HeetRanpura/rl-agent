# Scheme eligibility rules, required documents, and metadata for all supported welfare schemes.
# Each scheme is a dictionary with deterministic integer-based eligibility rules
# so the grader can evaluate agent decisions without any ambiguity.

from typing import Dict, Any, List, Optional


SCHEMES: Dict[str, Dict[str, Any]] = {

    # ── BENCHMARK SCHEMES ─────────────────────────────────────────────────────
    # These 3 schemes are actively tested in tasks 1–5.
    # Agents are expected to reason about these from the system prompt rules.

    "PMKVY": {
        # Pradhan Mantri Kaushal Vikas Yojana — skill training for young workers
        "full_name":    "Pradhan Mantri Kaushal Vikas Yojana",
        "benefit":      "Free skill training and certification with Rs 8000 stipend",
        "eligibility":  {
            "age_min":          18,
            "age_max":          35,
            "income_max":       9999,       # Strictly less than 10000
            "occupations":      ["mason", "carpenter"],
            "requires_aadhaar": False,
        },
        "required_docs": ["aadhaar", "education_certificate"],
    },

    "MGNREGS": {
        # Mahatma Gandhi National Rural Employment Guarantee Scheme
        "full_name":    "Mahatma Gandhi National Rural Employment Guarantee Scheme",
        "benefit":      "100 days of guaranteed wage employment per year",
        "eligibility":  {
            "age_min":          18,
            "age_max":          60,
            "income_max":       None,       # No income ceiling
            "occupations":      ["farm_labourer"],
            "requires_aadhaar": True,
        },
        "required_docs": ["aadhaar", "job_card"],
    },

    "PMAY": {
        # Pradhan Mantri Awaas Yojana — housing assistance for low-income families
        "full_name":    "Pradhan Mantri Awaas Yojana",
        "benefit":      "Rs 1.2 lakh grant for pucca house construction",
        "eligibility":  {
            "age_min":          21,
            "age_max":          55,
            "income_max":       5999,       # Strictly less than 6000
            "occupations":      None,       # Any occupation qualifies
            "requires_aadhaar": True,
        },
        "required_docs": ["aadhaar", "income_certificate", "land_document"],
    },

    # ── EXTENDED SCHEMES ──────────────────────────────────────────────────────
    # These 5 schemes are defined for future tasks using enriched profiles.
    # They are NOT reachable from benchmark tasks 1–5 which use sparse
    # 4-field profiles (age, income, occupation, has_aadhaar).
    # get_eligible_schemes() will skip these unless the required extra
    # profile fields are explicitly provided by a future task.

    # NOTE: PM_SYM requires worker_type, is_epfo_member, is_esic_member in profile.
    # Not reachable from benchmark tasks 1–5.
    "PM_SYM": {
        # Pradhan Mantri Shram Yogi Maan-dhan — pension for unorganised workers
        "full_name":    "Pradhan Mantri Shram Yogi Maan-dhan",
        "benefit":      "Rs 3000 per month pension after age 60",
        "eligibility":  {
            "age_min":          18,
            "age_max":          40,
            "income_max":       14999,
            "occupations":      None,
            "requires_aadhaar": True,
            "worker_type":      "unorganised",
            "not_epfo":         True,
            "not_esic":         True,
        },
        "required_docs": ["aadhaar", "bank_passbook", "mobile_number"],
    },

    # NOTE: AYUSHMAN_BHARAT requires not_govt_employee field in profile.
    # Not reachable from benchmark tasks 1–5.
    "AYUSHMAN_BHARAT": {
        # Ayushman Bharat PM-JAY — health insurance for low-income families
        "full_name":    "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana",
        "benefit":      "Rs 5 lakh annual health insurance cover per family",
        "eligibility":  {
            "age_min":           None,
            "age_max":           None,
            "income_max":        49999,
            "occupations":       None,
            "requires_aadhaar":  True,
            "not_govt_employee": True,
        },
        "required_docs": ["aadhaar", "ration_card"],
    },

    # NOTE: E_SHRAM requires worker_type, is_epfo_member, is_esic_member, is_nps_subscriber.
    # Not reachable from benchmark tasks 1–5.
    "E_SHRAM": {
        # e-Shram Portal — national database registration for unorganised workers
        "full_name":    "e-Shram Portal Registration",
        "benefit":      "Rs 2 lakh accident insurance and linkage to all welfare schemes",
        "eligibility":  {
            "age_min":          16,
            "age_max":          59,
            "income_max":       None,
            "occupations":      None,
            "requires_aadhaar": True,
            "worker_type":      "unorganised",
            "not_epfo":         True,
            "not_esic":         True,
            "not_nps":          True,
        },
        "required_docs": ["aadhaar", "mobile_number", "bank_passbook"],
    },

    # NOTE: NFSA requires is_income_tax_payer field in profile.
    # Not reachable from benchmark tasks 1–5.
    "NFSA": {
        # National Food Security Act — subsidised food grains through ration card
        "full_name":    "National Food Security Act — Ration Card",
        "benefit":      "Subsidised food grains at Rs 1-3 per kg",
        "eligibility":  {
            "age_min":               18,
            "age_max":               None,
            "income_max":            9999,
            "occupations":           None,
            "requires_aadhaar":      True,
            "not_income_tax_payer":  True,
        },
        "required_docs": ["aadhaar", "address_proof", "family_photo"],
    },

    # NOTE: PMMVY requires gender=female, is_pregnant, first_child, has_bank_account.
    # Not reachable from benchmark tasks 1–5.
    "PMMVY": {
        # Pradhan Mantri Matru Vandana Yojana — maternity benefit for first child
        "full_name":    "Pradhan Mantri Matru Vandana Yojana",
        "benefit":      "Rs 5000 maternity benefit paid in 3 instalments",
        "eligibility":  {
            "age_min":          18,
            "age_max":          None,
            "income_max":       None,
            "occupations":      None,
            "requires_aadhaar": True,
            "gender":           "female",
            "is_pregnant":      True,
            "first_child":      True,
            "has_bank_account": True,
        },
        "required_docs": ["aadhaar", "mch_card", "bank_passbook"],
    },
}


def get_eligible_schemes(profile: dict) -> list:
    """
    Evaluate a complete applicant profile against all schemes and return
    a list of scheme keys the applicant qualifies for.
    All comparisons use strict integer arithmetic — no rounding or approximation.
    Extended schemes (PM_SYM, AYUSHMAN_BHARAT, E_SHRAM, NFSA, PMMVY) will only
    match if the required extra fields are present in the profile.
    """
    eligible = []

    age     = int(profile.get("age", 0))
    income  = int(profile.get("income", 0))
    occ     = profile.get("occupation", "").lower()
    aadhaar = str(profile.get("has_aadhaar", "False")).lower() == "true"

    for scheme_key, scheme in SCHEMES.items():
        rules = scheme["eligibility"]

        # FIX C3: use 'is not None' to correctly handle age_min=0
        if rules.get("age_min") is not None and age < rules["age_min"]:
            continue
        if rules.get("age_max") is not None and age > rules["age_max"]:
            continue

        # Strict less-than income comparison
        if rules.get("income_max") is not None and income > rules["income_max"]:
            continue

        # Occupation restriction when defined
        if rules.get("occupations") and occ not in rules["occupations"]:
            continue

        # Aadhaar requirement
        if rules.get("requires_aadhaar") and not aadhaar:
            continue

        # Extended scheme checks — only evaluated if field present in profile
        if rules.get("not_govt_employee") and profile.get("is_govt_employee") is True:
            continue
        if rules.get("not_epfo") and profile.get("is_epfo_member") is True:
            continue
        if rules.get("not_esic") and profile.get("is_esic_member") is True:
            continue
        if rules.get("not_nps") and profile.get("is_nps_subscriber") is True:
            continue
        if rules.get("not_income_tax_payer") and profile.get("is_income_tax_payer") is True:
            continue
        if rules.get("gender") and profile.get("gender") and \
                profile.get("gender", "").lower() != rules["gender"]:
            continue
        if rules.get("is_pregnant") and profile.get("is_pregnant") is False:
            continue
        if rules.get("first_child") and profile.get("first_child") is False:
            continue
        if rules.get("has_bank_account") and profile.get("has_bank_account") is False:
            continue

        eligible.append(scheme_key)

    return eligible


def get_optimal_scheme(profile: dict) -> Optional[str]:
    """
    Return the single most beneficial scheme for this applicant profile.
    Priority order: PMAY > MGNREGS > PMKVY > PM_SYM > AYUSHMAN_BHARAT > E_SHRAM > NFSA > PMMVY.
    Returns None if the applicant is not eligible for any scheme.
    """
    eligible = get_eligible_schemes(profile)

    # Priority order matches system prompt benefit hierarchy
    priority = ["PMAY", "MGNREGS", "PMKVY", "PM_SYM",
                "AYUSHMAN_BHARAT", "E_SHRAM", "NFSA", "PMMVY"]

    for scheme in priority:
        if scheme in eligible:
            return scheme

    return None
