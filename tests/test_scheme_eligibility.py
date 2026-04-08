"""
Unit tests for scheme eligibility logic and grader score computation.

All functions under test are pure and deterministic — no mocking needed.
Run from project root: pytest tests/
"""

import pytest
from server.schemes import get_eligible_schemes, get_optimal_scheme
from server.scheme_env_environment import _compute_grader_score


# =========================================================
# PMKVY BOUNDARY TESTS
# =========================================================

def test_pmkvy_qualifies_age_lower_bound():
    profile = {"age": 18, "income": 9999, "occupation": "mason", "has_aadhaar": "True"}
    assert "PMKVY" in get_eligible_schemes(profile)


def test_pmkvy_qualifies_age_upper_bound():
    profile = {"age": 35, "income": 9999, "occupation": "mason", "has_aadhaar": "True"}
    assert "PMKVY" in get_eligible_schemes(profile)


def test_pmkvy_disqualifies_age_exceeded():
    profile = {"age": 36, "income": 9999, "occupation": "mason", "has_aadhaar": "True"}
    assert "PMKVY" not in get_eligible_schemes(profile)


def test_pmkvy_disqualifies_income_exceeded():
    profile = {"age": 18, "income": 10000, "occupation": "mason", "has_aadhaar": "True"}
    assert "PMKVY" not in get_eligible_schemes(profile)


def test_pmkvy_disqualifies_wrong_occupation():
    profile = {"age": 18, "income": 9999, "occupation": "farm_labourer", "has_aadhaar": "True"}
    assert "PMKVY" not in get_eligible_schemes(profile)


# =========================================================
# MGNREGS BOUNDARY TESTS
# =========================================================

def test_mgnregs_qualifies_age_lower_bound():
    profile = {"age": 18, "occupation": "farm_labourer", "has_aadhaar": "True", "income": 0}
    assert "MGNREGS" in get_eligible_schemes(profile)


def test_mgnregs_qualifies_age_upper_bound():
    profile = {"age": 60, "occupation": "farm_labourer", "has_aadhaar": "True", "income": 0}
    assert "MGNREGS" in get_eligible_schemes(profile)


def test_mgnregs_disqualifies_age_exceeded():
    profile = {"age": 61, "occupation": "farm_labourer", "has_aadhaar": "True", "income": 0}
    assert "MGNREGS" not in get_eligible_schemes(profile)


def test_mgnregs_disqualifies_no_aadhaar():
    profile = {"age": 30, "occupation": "farm_labourer", "has_aadhaar": "False", "income": 0}
    assert "MGNREGS" not in get_eligible_schemes(profile)


# =========================================================
# PMAY BOUNDARY TESTS
# =========================================================

def test_pmay_qualifies_age_lower_bound():
    profile = {"age": 21, "income": 5999, "has_aadhaar": "True", "occupation": "mason"}
    assert "PMAY" in get_eligible_schemes(profile)


def test_pmay_disqualifies_income_at_threshold():
    profile = {"age": 21, "income": 6000, "has_aadhaar": "True", "occupation": "mason"}
    assert "PMAY" not in get_eligible_schemes(profile)


def test_pmay_qualifies_age_upper_bound():
    profile = {"age": 55, "income": 5999, "has_aadhaar": "True", "occupation": "mason"}
    assert "PMAY" in get_eligible_schemes(profile)


def test_pmay_disqualifies_age_exceeded():
    profile = {"age": 56, "income": 5999, "has_aadhaar": "True", "occupation": "mason"}
    assert "PMAY" not in get_eligible_schemes(profile)


# =========================================================
# get_optimal_scheme() PRIORITY TESTS
# =========================================================

def test_optimal_prefers_pmay_over_pmkvy():
    # Age 21-35, income < 6000, mason — eligible for both PMAY and PMKVY
    profile = {"age": 25, "income": 5000, "occupation": "mason", "has_aadhaar": "True"}
    assert get_optimal_scheme(profile) == "PMAY"


def test_optimal_mgnregs_only():
    profile = {"age": 40, "income": 50000, "occupation": "farm_labourer", "has_aadhaar": "True"}
    assert get_optimal_scheme(profile) == "MGNREGS"


def test_optimal_none_when_no_scheme():
    # High income, wrong occupation, no aadhaar
    profile = {"age": 40, "income": 99999, "occupation": "lawyer", "has_aadhaar": "False"}
    assert get_optimal_scheme(profile) is None


# =========================================================
# _compute_grader_score TESTS
# =========================================================

def test_grader_score_perfect():
    score = _compute_grader_score(
        task=1, base_score=1.0, step_count=3,
        noise_queries=0, redundant_queries=0,
    )
    assert score == 1.0


def test_grader_score_noise_penalty():
    # base=1.0, noise_queries=5 → 1.0 - 0.40 = 0.60, clamped to [0.30, 1.0]
    score = _compute_grader_score(
        task=1, base_score=1.0, step_count=3,
        noise_queries=5, redundant_queries=0,
    )
    assert score == pytest.approx(0.60, abs=1e-3)


def test_grader_score_zero_base():
    score = _compute_grader_score(
        task=1, base_score=0.0, step_count=3,
        noise_queries=0, redundant_queries=0,
    )
    assert score == 0.0


def test_grader_score_floor_at_030():
    # Massive penalties should floor at 0.30, not go negative
    score = _compute_grader_score(
        task=1, base_score=1.0, step_count=3,
        noise_queries=50, redundant_queries=50,
    )
    assert score == pytest.approx(0.30, abs=1e-3)
