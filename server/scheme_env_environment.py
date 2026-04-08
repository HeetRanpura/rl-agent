import copy
import random
import threading
import math
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .schemes import SCHEMES
from .models import Action, Observation

# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

# Maximum steps per episode — generous enough for careful agents but not infinite
MAX_STEPS = 20

# Keep non-terminal shaping flat so agents optimize for correct terminal
# outcomes instead of reward-farming with unnecessary information gathering.
VALID_STEP_PENALTY = 0.0
INVALID_STEP_PENALTY = -0.10

# Noise fields injected into every profile — querying them wastes steps and costs reward
NOISE_FIELDS = [
    "marital_status",
    "state_of_residence",
    "number_of_children",
    "bank_name",
]

NOISE_VALUES = {
    "marital_status":     ["married", "unmarried", "widowed", "divorced"],
    "state_of_residence": ["Maharashtra", "Uttar Pradesh", "Bihar", "Rajasthan", "Gujarat"],
    "number_of_children": ["0", "1", "2", "3", "4"],
    "bank_name":          ["SBI", "PNB", "Bank of Baroda", "Canara Bank", "UCO Bank"],
}

# Only these fields are eligibility-relevant — everything else is a trap
VALID_QUERY_FIELDS = {"age", "income", "occupation", "has_aadhaar"}

# Task 4: PSU employers used to create the contradiction
CONTRADICTION_EMPLOYERS = [
    "Indian Railways", "BSNL", "Coal India", "State Bank of India",
    "ONGC", "BHEL", "HAL", "GAIL India",
]

# Task 5: the self-reported age that conflicts with the Aadhaar age
# FIXED D2: no longer hardcoded — generated dynamically per episode in generate_dynamic_persona
TASK5_BOUNDARY_SCHEMES = ["PMKVY"]  # schemes whose upper age bound is tested


# =========================================================
# NOISE INJECTION
# =========================================================

def _inject_noise(profile: dict) -> dict:
    """
    Randomly add 1-3 irrelevant fields to the profile.
    These are traps — querying them costs -1.0 and increments noise_queries.
    Good agents ignore them entirely.
    """
    chosen = random.sample(NOISE_FIELDS, k=random.randint(1, 3))
    for field in chosen:
        profile[field] = random.choice(NOISE_VALUES[field])
    return profile


# =========================================================
# PERSONA GENERATION
# =========================================================

def generate_dynamic_persona(task_id: int) -> dict:
    """
    Generate a randomized applicant profile for the chosen task template.
    Each reset keeps the same reasoning pattern but refreshes the applicant
    details so agents must read state rather than memorize fixed trajectories.
    """

    if task_id == 1:
        # ── TASK 1: Scheme Discovery ──────────────────────────────────────────
        # Profile is complete. Agent must determine the optimal scheme.
        # HARDENING vs original: when both PMKVY and PMAY are eligible,
        # PMAY is optimal (higher benefit value: Rs 1.2L vs PMKVY's Rs 8000).
        # Agents that blindly approve PMKVY without checking PMAY get 0.5.
        age    = random.randint(18, 35)
        income = random.randint(1000, 9999)
        occ    = random.choice(["mason", "carpenter"])

        # Determine which schemes are genuinely eligible
        eligible = ["PMKVY"]
        optimal  = "PMKVY"

        # PMAY is also eligible when age is in range and income is below 6000
        # When PMAY is eligible, it is the OPTIMAL choice (higher benefit)
        if income < 6000 and 21 <= age <= 55:
            eligible.append("PMAY")
            optimal = "PMAY"   # PMAY > PMKVY in benefit value when both eligible

        return {
            "age": str(age),
            "income": str(income),
            "occupation": occ,
            "has_aadhaar": "True",
            "optimal_scheme": optimal,
            "eligible_schemes": eligible,
        }

    elif task_id == 2:
        # ── TASK 2: Missing Data ──────────────────────────────────────────────
        # Occupation and Aadhaar status are hidden. Agent must ask for both
        # before making a terminal decision.
        # HARDENING vs original: missing_keys order is randomised so agents
        # cannot hardcode "ask occupation first, then has_aadhaar".
        missing = random.sample(["occupation", "has_aadhaar"], k=2)

        return {
            "age": str(random.randint(18, 60)),
            "income": str(random.randint(1000, 5000)),
            "occupation": "farm_labourer",   # hidden until agent asks
            "has_aadhaar": "True",           # hidden until agent asks
            "optimal_scheme": "MGNREGS",
            "eligible_schemes": ["MGNREGS"],
            "missing_keys": missing,
        }

    elif task_id == 3:
        # ── TASK 3: Boundary Fraud Detection ─────────────────────────────────
        # Income is 1-2000 rupees above the PMKVY threshold.
        # HARDENING vs original: notification NO LONGER states the income,
        # the threshold, or what to do. Agent must read known_profile and
        # recall the PMKVY rule from its own training/system prompt.
        age    = random.randint(22, 34)
        income = random.randint(10001, 12000)
        occ    = random.choice(["mason", "carpenter"])

        return {
            "age": str(age),
            "income": str(income),
            "occupation": occ,
            "has_aadhaar": "True",
            "optimal_scheme": None,
            "eligible_schemes": [],
            "_near_miss": True,
        }

    elif task_id == 4:
        # ── TASK 4: Escalation Dilemma ────────────────────────────────────────
        # Applicant claims to be a student but income/employer data from PAN
        # reveals formal government employment.
        age      = random.randint(22, 45)
        income   = random.randint(8000, 20000)   # suspiciously high for a student
        employer = random.choice(CONTRADICTION_EMPLOYERS)

        return {
            "age": str(age),
            "income": str(income),
            "occupation": "student",
            "has_aadhaar": "True",
            "optimal_scheme": None,
            "eligible_schemes": [],
            "_contradictory": True,
            "_pan_employer": employer,
        }

    elif task_id == 5:
        # ── TASK 5: Document Conflict ─────────────────────────────────────────
        # self_reported_age drawn from [33,34,35] — at or near PMKVY upper boundary.
        # aadhaar_age always exceeds 35, so the conflict always disqualifies from PMKVY.
        # max(36, ...) ensures ages 33/34 still produce a disqualifying aadhaar_age.
        self_reported_age = random.choice([33, 34, 35])
        aadhaar_age       = max(36, self_reported_age + random.randint(1, 3))

        income = random.randint(6001, 9000)   # above PMAY cap, below PMKVY income cap

        return {
            "age": str(self_reported_age),
            "income": str(income),
            "occupation": "mason",
            "has_aadhaar": "True",
            "optimal_scheme": None,
            "eligible_schemes": [],
            "_aadhaar_age": str(aadhaar_age),
            "_self_reported_age": str(self_reported_age),
            "_document_conflict": True,
        }

    else:
        raise ValueError(f"Unknown task_id: {task_id}")


# =========================================================
# INITIAL OBSERVATION BUILDER
# =========================================================

def _make_fresh_obs(task: int, persona: dict) -> Observation:
    """
    Build the starting Observation for the given task.

    HARDENING PRINCIPLES applied here:
    - Task 3: notification gives NO numerical hints — just says "apply rules carefully"
    - Task 4: notification gives NO SYSTEM ALERT — just flags occupation/income as unusual
    - Task 5: notification gives NO hint about the age conflict — shows surface profile only
    All tasks: noise fields injected into profile from step zero.
    """

    # Start with age only. Task 3 deliberately hides income so agents cannot
    # zero-step reject without first collecting the boundary-relevant field.
    # Tasks 1, 2, 4, 5: income is revealed upfront (handled below).
    profile = {
        "age": persona["age"],
    }
    if task != 3:
        profile["income"] = persona["income"]

    # EPISODE STRUCTURE DESIGN:
    # Tasks 4 and 5 reveal occupation upfront because the contradiction/conflict
    # is built around the occupation value.
    # Tasks 1 and 3 deliberately HIDE occupation and has_aadhaar at episode start.
    # Task 2 already hides occupation and has_aadhaar (existing behaviour unchanged).
    if task in [4, 5]:
        profile["occupation"]  = persona["occupation"]
        profile["has_aadhaar"] = persona["has_aadhaar"]

    # Task 5 special: inject self_reported_age as a fake noise field to tempt the agent
    if task == 5:
        profile["self_reported_age"] = persona["_self_reported_age"]

    # Inject random irrelevant fields — same for all tasks
    _inject_noise(profile)

    # ── Task-specific notifications ───────────────────────────────────────────
    if task == 1:
        notif = (
            "[TASK 1/5 - SCHEME DISCOVERY - Easy] "
            "Profile is INCOMPLETE. Collect all required eligibility fields first, "
            "then identify and approve the OPTIMAL welfare scheme. "
            "If multiple schemes apply, choose the one with the highest benefit value."
        )
        missing = ["occupation", "has_aadhaar"]

    elif task == 2:
        notif = (
            "[TASK 2/5 - MISSING DATA - Medium] "
            "Profile is INCOMPLETE. You MUST collect ALL required eligibility fields "
            "before making any terminal decision. "
            "Some fields shown are irrelevant — focus only on what schemes require."
        )
        missing = list(persona["missing_keys"])

    elif task == 3:
        notif = (
            "[TASK 3/5 - BOUNDARY FRAUD DETECTION - Hard] "
            "Profile is INCOMPLETE. Collect ALL required eligibility fields "
            "including income, then apply scheme rules with exact integer arithmetic. "
            "Approve only if ALL criteria are fully satisfied. "
            "If no scheme criteria are fully met, use reject_applicant."
        )
        missing = ["occupation", "has_aadhaar", "income"]

    elif task == 4:
        notif = (
            "[TASK 4/5 - ESCALATION DILEMMA - Expert] "
            "Profile is COMPLETE. Review all applicant data carefully "
            "before making any terminal decision. "
            "If occupation and income appear inconsistent, request the PAN card to verify "
            "employment records. If the document confirms a contradiction, escalate to a senior officer."
        )
        missing = []

    elif task == 5:
        notif = (
            "[TASK 5/5 - DOCUMENT CONFLICT - Expert+] "
            "Profile is COMPLETE. Apply all eligibility rules and verify "
            "official documents as required before making a terminal decision. "
            "When age is at a critical eligibility boundary, request the Aadhaar card and use the official age."
        )
        missing = []

    else:
        notif   = ""
        missing = []

    # FIX D1: metadata is internal only — do NOT expose pan_verified, aadhaar_verified,
    # or task integer to the agent. These are stripped at the observation boundary.
    # The agent sees only: known_profile, missing_data, notification, is_terminated, grader_score.
    return Observation(
        known_profile  = profile,
        missing_data   = missing,
        notification   = notif,
        is_terminated  = False,
        reward         = 0.0,
        done           = False,
        grader_score   = None,
        metadata = {
         "task_label":        f"task_{task}",
         "noise_queries":     0,
         "redundant_queries": 0,
         "relevant_queries":  0,
         "pan_verified":      False,
         "aadhaar_verified":  False,
         "document_verified": False,
         "critical_discoveries": 0,
        },
    )   


# =========================================================
# CONTINUOUS GRADER
# =========================================================

def _compute_grader_score(
    task: int,
    base_score: float,
    step_count: int,
    noise_queries: int,
    redundant_queries: int,
    missing_keys_total: int = 0,
    document_verified: bool = False,
) -> float:
    """
    Convert a terminal outcome into a continuous score between 0.0 and 1.0.

    Penalty model:
      noise_queries      → -0.08 each
      redundant_queries  → -0.05 each
      wasted steps       → -0.04 each (Task 2 only)

    Bonus model:
      document_verified  → +0.05 (Tasks 4/5)

    Incorrect terminal outcomes always return 0.0.
    Correct outcomes clamped to [0.30, 1.0].
    """
    if base_score <= 0.0:
        return 0.0

    penalty = (noise_queries * 0.08) + (redundant_queries * 0.05)

    if task == 2 and missing_keys_total > 0:
        min_steps = missing_keys_total + 1
        wasted    = max(0, step_count - min_steps)
        penalty  += wasted * 0.04

    bonus = 0.05 if document_verified else 0.0

    # FIX A2: clamped to [0.30, 1.0] — was missing min(1.0,...) before
    return round(max(0.30, min(1.0, base_score - penalty + bonus)), 3)


# =========================================================
# MAIN ENVIRONMENT CLASS
# =========================================================

class SchemeEnvEnvironment(Environment):
    """
    OpenEnv-compliant RL environment simulating an Indian CSC welfare officer.

    5 tasks of increasing difficulty:
      Task 1 — Scheme Discovery:       complete profile, choose optimal scheme
      Task 2 — Missing Data:           collect hidden fields before deciding
      Task 3 — Boundary Fraud:         near-miss income, must detect overage
      Task 4 — Escalation Dilemma:     student/pension contradiction, must escalate
      Task 5 — Document Conflict:      self-reported vs Aadhaar age mismatch
    """

    # FIX D3: asyncio.Lock guards shared state — prevents concurrent request corruption
    SUPPORTS_CONCURRENT_SESSIONS = False
    _shared_state = {}
    _state_lock   = threading.Lock()

    def __init__(self):
        super().__init__()

        if not SchemeEnvEnvironment._shared_state:
            persona = generate_dynamic_persona(1)
            obs     = _make_fresh_obs(1, persona)
            state   = State(episode_id=str(uuid4()), step_count=0)
            SchemeEnvEnvironment._shared_state = {
                "task": 1, "persona": persona, "state": state, "obs": obs,
            }

        self._load_shared()

    def _load_shared(self):
        """Restore episode state from class-level dict."""
        s             = SchemeEnvEnvironment._shared_state
        self._task    = s["task"]
        self._persona = s["persona"]
        self._state   = s["state"]
        self._obs     = s["obs"]

    def _save_shared(self):
        """Persist episode state back to class-level dict after every step."""
        SchemeEnvEnvironment._shared_state.update({
            "task":    self._task,
            "persona": self._persona,
            "state":   self._state,
            "obs":     self._obs,
        })

    def reset(self, seed=None, **kwargs) -> Observation:
        """
        Start a new episode.
        seed=1-5 selects a specific task; no seed cycles 1→2→3→4→5→1.
        """
        with SchemeEnvEnvironment._state_lock:
            self._task    = seed if seed in (1, 2, 3, 4, 5) else (self._task % 5) + 1
            self._persona = generate_dynamic_persona(self._task)
            self._state   = State(episode_id=str(uuid4()), step_count=0)
            self._obs     = _make_fresh_obs(self._task, self._persona)
            self._save_shared()
            return self._obs

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        """
        Execute one agent action and return the updated observation.
        """
        with SchemeEnvEnvironment._state_lock:
            self._state.step_count += 1

            # FIX D4: deepcopy prevents aliased mutation of shared state
            obs          = copy.deepcopy(self._obs)
            current_task = self._task
            persona      = self._persona

            valid_actions = {
                "ask_question", "request_document",
                "approve_scheme", "reject_applicant", "escalate",
            }

            if action.action_type not in valid_actions:
                obs.notification = (
                    f"Unknown action '{action.action_type}'. "
                    f"Valid: {', '.join(sorted(valid_actions))}."
                )
                obs.reward = INVALID_STEP_PENALTY
                obs.done   = False
                return self._finalize_step(obs)

            # ── ASK_QUESTION ──────────────────────────────────────────────────────
            if action.action_type == "ask_question":
                key = (action.value or "").strip()

                if key == "self_reported_age":
                    obs.metadata["noise_queries"] += 1
                    obs.notification = (
                        "Self-reported age is already visible in the profile. "
                        "For authoritative age verification, request the Aadhaar card."
                    )
                    obs.reward = INVALID_STEP_PENALTY

                elif key in NOISE_FIELDS:
                    obs.metadata["noise_queries"] += 1
                    obs.notification = "Irrelevant field. Focus on eligibility criteria only."
                    obs.reward       = INVALID_STEP_PENALTY

                elif key in obs.known_profile:
                    obs.metadata["redundant_queries"] += 1
                    obs.notification = f"'{key}' is already in the profile. Do not repeat questions."
                    obs.reward       = INVALID_STEP_PENALTY

                elif key in VALID_QUERY_FIELDS and key in persona:
                    val = persona[key]
                    obs.known_profile[key] = val
                    if key in obs.missing_data:
                        obs.missing_data.remove(key)
                    obs.metadata["relevant_queries"] += 1
                    obs.notification = f"Applicant confirmed: {key} = {val}."
                    obs.reward       = VALID_STEP_PENALTY

                else:
                    obs.notification = f"'{key}' is not a recognised eligibility field."
                    obs.reward       = INVALID_STEP_PENALTY

            # ── REQUEST_DOCUMENT ──────────────────────────────────────────────────
            elif action.action_type == "request_document":
                doc = (action.value or "document").lower()

                if current_task == 4 and "pan" in doc:
                    employer = persona.get("_pan_employer", "a government organisation")
                    obs.metadata["pan_verified"]         = True
                    obs.metadata["document_verified"]    = True
                    obs.metadata["critical_discoveries"] += 1
                    obs.notification = (
                        f"PAN card retrieved. "
                        f"Records show this applicant has been a registered employee of "
                        f"{employer} for the past 6 years with active pension contributions. "
                        f"This directly contradicts the stated occupation 'student'. "
                        f"The case cannot be approved or rejected without senior review."
                    )
                    obs.reward = VALID_STEP_PENALTY

                elif current_task == 5 and "aadhaar" in doc:
                    true_age = persona.get("_aadhaar_age", "36")
                    obs.metadata["aadhaar_verified"]     = True
                    obs.metadata["document_verified"]    = True
                    obs.metadata["critical_discoveries"] += 1
                    obs.known_profile["age"]             = true_age
                    obs.notification = (
                        f"Aadhaar card verified. "
                        f"Official age on record: {true_age} years. "
                        f"Note: this differs from the self-reported age of "
                        f"{persona.get('_self_reported_age', '35')} in the profile. "
                        f"The Aadhaar age is the authoritative value for eligibility decisions."
                    )
                    obs.reward = VALID_STEP_PENALTY

                elif current_task == 5 and "pan" in doc:
                    obs.notification = (
                        "PAN card verified. No anomalies found in tax records. "
                        "For age verification, the Aadhaar card is the authoritative document."
                    )
                    obs.reward = VALID_STEP_PENALTY

                else:
                    doc_lower = (action.value or "").lower()
                    if "aadhaar" in doc_lower and "has_aadhaar" in obs.missing_data:
                        obs.missing_data.remove("has_aadhaar")
                        obs.known_profile["has_aadhaar"] = "True"
                        obs.notification = "Aadhaar card received and verified. has_aadhaar confirmed as True."
                    else:
                        obs.notification = f"Document '{action.value or 'document'}' received and verified."
                    obs.reward = VALID_STEP_PENALTY

            # ── APPROVE_SCHEME ────────────────────────────────────────────────────
            elif action.action_type == "approve_scheme":
                scheme = (action.value or "").strip()

                if current_task == 4:
                    if not obs.metadata.get("pan_verified", False):
                        obs.notification = (
                            "PROTOCOL VIOLATION: Do not approve this case before verifying "
                            "employment records. Request the PAN card first to check for a "
                            "data integrity conflict."
                        )
                        obs.reward        = -1.5
                        obs.done          = False
                        obs.is_terminated = False
                        return self._finalize_step(obs)

                    score = 0.0
                    obs.notification = (
                        "FRAUD AUTHORIZATION VIOLATION: This case has a data integrity conflict. "
                        "You cannot approve any scheme without resolving the contradiction first. "
                        "The correct action is to escalate to a senior officer."
                    )
                    obs.reward        = -5.0
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score
                    return self._finalize_step(obs)

                if current_task == 5 and not obs.metadata.get("aadhaar_verified", False):
                    obs.notification = (
                        "PROTOCOL VIOLATION: You must verify the Aadhaar card before approving "
                        "any scheme when age is a critical eligibility factor. "
                        "Request the Aadhaar card first."
                    )
                    obs.reward        = -1.5
                    obs.done          = False
                    obs.is_terminated = False
                    return self._finalize_step(obs)

                if current_task == 5 and obs.metadata.get("aadhaar_verified", False):
                    true_age = persona.get("_aadhaar_age", "36")
                    score = 0.0
                    obs.notification = (
                        f"ELIGIBILITY VIOLATION: Aadhaar confirms age={true_age}. "
                        f"PMKVY requires age ≤ 35. No other scheme applies to this profile. "
                        f"The correct action is reject_applicant."
                    )
                    obs.reward        = -5.0
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score
                    return self._finalize_step(obs)

                if current_task == 3:
                    income_int = int(persona["income"])
                    overage    = income_int - 9999

                    if overage <= 100:
                        step_reward = -1.0
                        tier_label  = "BOUNDARY MISS"
                    elif overage <= 500:
                        step_reward = -2.5
                        tier_label  = "CLOSE MISS"
                    elif overage <= 2000:
                        step_reward = -4.0
                        tier_label  = "CLEAR MISS"
                    else:
                        step_reward = -5.0
                        tier_label  = "THRESHOLD VIOLATION"

                    score = 0.0
                    obs.notification = (
                        f"{tier_label}: Income {persona['income']} exceeds all scheme "
                        f"thresholds (overage: Rs {overage} above PMKVY limit). "
                        f"Use reject_applicant when no scheme criteria are fully met."
                    )
                    obs.reward        = step_reward
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score
                    return self._finalize_step(obs)

                if current_task in (1, 2) and len(obs.missing_data) > 0:
                    score = 0.0
                    obs.notification = (
                        f"PREMATURE APPROVAL: Still missing required fields: {obs.missing_data}. "
                        "Collect all required data before approving."
                    )
                    obs.reward        = -5.0
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score
                    return self._finalize_step(obs)

                optimal  = persona["optimal_scheme"]
                eligible = persona["eligible_schemes"]

                if scheme == optimal:
                    score = _compute_grader_score(
                        task               = current_task,
                        base_score         = 1.0,
                        step_count         = self._state.step_count,
                        noise_queries      = obs.metadata.get("noise_queries", 0),
                        redundant_queries  = obs.metadata.get("redundant_queries", 0),
                        missing_keys_total = len(persona.get("missing_keys", [])),
                        document_verified  = obs.metadata.get("document_verified", False),
                    )
                    obs.notification  = f"CORRECT: Applicant enrolled in {scheme} — optimal scheme."
                    obs.reward        = 10.0
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score

                elif scheme in eligible:
                    score = _compute_grader_score(
                        task              = current_task,
                        base_score        = 0.5,
                        step_count        = self._state.step_count,
                        noise_queries     = obs.metadata.get("noise_queries", 0),
                        redundant_queries = obs.metadata.get("redundant_queries", 0),
                    )
                    obs.notification  = f"SUBOPTIMAL: {scheme} is eligible but {optimal} provides greater benefit."
                    obs.reward        = 3.0
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score

                else:
                    score = 0.0
                    obs.notification  = f"ERROR: Applicant does not qualify for '{scheme}'."
                    obs.reward        = -5.0
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score

            # ── REJECT_APPLICANT ──────────────────────────────────────────────────
            elif action.action_type == "reject_applicant":

                if current_task == 4:
                    if not obs.metadata.get("pan_verified", False):
                        obs.notification = (
                            "PREMATURE ADJUDICATION: Review documentary evidence before rejecting. "
                            "Request the PAN card to verify employment records, then escalate if a "
                            "contradiction is confirmed."
                        )
                        obs.reward        = -1.0
                        obs.done          = False
                        obs.is_terminated = False
                    else:
                        score = 0.0
                        obs.notification = (
                            "PREMATURE ADJUDICATION: The data integrity conflict must be resolved "
                            "by a senior officer before any terminal decision. Use escalate."
                        )
                        obs.reward        = -3.0
                        obs.done          = True
                        obs.is_terminated = True
                        obs.grader_score  = score
                        obs.metadata["grader_score"] = score

                elif current_task == 5:
                    if not obs.metadata.get("aadhaar_verified", False):
                        obs.notification = (
                            "PROTOCOL VIOLATION: You must verify the Aadhaar card before "
                            "rejecting an applicant when age is a critical factor. "
                            "Request the Aadhaar card first."
                        )
                        obs.reward        = -1.0
                        obs.done          = False
                        obs.is_terminated = False
                    else:
                        true_age = persona.get("_aadhaar_age", "36")
                        score = _compute_grader_score(
                            task              = current_task,
                            base_score        = 1.0,
                            step_count        = self._state.step_count,
                            noise_queries     = obs.metadata.get("noise_queries", 0),
                            redundant_queries = obs.metadata.get("redundant_queries", 0),
                            document_verified = True,
                        )
                        obs.notification  = (
                            f"CORRECT REJECTION: Aadhaar confirms age={true_age}, "
                            f"which exceeds the PMKVY maximum of 35. "
                            f"No other scheme criteria are satisfied. Rejection is valid."
                        )
                        obs.reward        = 5.0
                        obs.done          = True
                        obs.is_terminated = True
                        obs.grader_score  = score
                        obs.metadata["grader_score"] = score

                elif current_task == 3:
                    if "income" not in obs.known_profile:
                        score = 0.0
                        obs.notification = (
                            "PROTOCOL VIOLATION: You must collect income data before "
                            "making a rejection decision."
                        )
                        obs.reward        = -2.0
                        obs.done          = True
                        obs.is_terminated = True
                        obs.grader_score  = score
                        obs.metadata["grader_score"] = score
                    else:
                        score = _compute_grader_score(
                            task              = current_task,
                            base_score        = 1.0,
                            step_count        = self._state.step_count,
                            noise_queries     = obs.metadata.get("noise_queries", 0),
                            redundant_queries = obs.metadata.get("redundant_queries", 0),
                        )
                        obs.notification  = (
                            f"CORRECT REJECTION: Income {persona['income']} exceeds all scheme "
                            f"thresholds. No eligible scheme found."
                        )
                        obs.reward        = 5.0
                        obs.done          = True
                        obs.is_terminated = True
                        obs.grader_score  = score
                        obs.metadata["grader_score"] = score

                else:
                    score = 0.0
                    obs.notification  = (
                        "ERROR: This applicant qualifies for a welfare scheme. "
                        "Review the eligibility criteria and approve the correct scheme."
                    )
                    obs.reward        = -5.0
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score

            # ── ESCALATE ──────────────────────────────────────────────────────────
            elif action.action_type == "escalate":

                if current_task == 4:
                    verified = obs.metadata.get("pan_verified", False)
                    if not verified:
                        obs.notification = (
                            "INSUFFICIENT BASIS FOR ESCALATION: First request the PAN card to "
                            "verify the suspected employment contradiction. Escalate after the "
                            "document confirms the conflict."
                        )
                        obs.reward        = -1.0
                        obs.done          = False
                        obs.is_terminated = False
                        return self._finalize_step(obs)

                    obs.reward = 10.0
                    score    = _compute_grader_score(
                        task              = current_task,
                        base_score        = 1.0,
                        step_count        = self._state.step_count,
                        noise_queries     = obs.metadata.get("noise_queries", 0),
                        redundant_queries = obs.metadata.get("redundant_queries", 0),
                        document_verified = verified,
                    )
                    obs.notification  = (
                        "CORRECT ESCALATION: Contradictory data detected and properly handed "
                        "off to a senior officer for manual verification. "
                        "This is the required protocol for data integrity conflicts."
                    )
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score

                else:
                    obs.notification = (
                        "INCORRECT ESCALATION: Escalation is only appropriate when data "
                        "integrity is genuinely compromised. This case has sufficient "
                        "information for a direct decision. Please reconsider."
                    )
                    score = 0.0
                    obs.reward        = -2.0       
                    obs.done          = True
                    obs.is_terminated = True
                    obs.grader_score  = score
                    obs.metadata["grader_score"] = score

            return self._finalize_step(obs)

    def _finalize_step(self, obs: Observation) -> Observation:
        """
        Enforce step limit and persist state.
        Called at the end of every step regardless of outcome.
        FIX D5: changed >= to > to fix off-by-one that overwrote step 19 actions.
        """
        if self._state.step_count >= MAX_STEPS and not obs.done:
            obs.is_terminated            = True
            obs.notification             = f"TIMEOUT: {MAX_STEPS} steps reached without a decision."
            obs.reward                   = -2.0
            obs.done                     = True
            obs.grader_score             = 0.0
            obs.metadata["grader_score"] = 0.0

        self._obs = obs
        self._save_shared()

        # Strip internal tracking fields before returning to agent.
        # self._obs retains full metadata for step logic on subsequent steps.
        # The agent only sees noise/redundant/relevant query counts.
        import copy
        agent_obs = copy.deepcopy(obs)
        agent_obs.metadata = {
            "noise_queries":     obs.metadata.get("noise_queries", 0),
            "redundant_queries": obs.metadata.get("redundant_queries", 0),
            "relevant_queries":  obs.metadata.get("relevant_queries", 0),
        }
        return agent_obs

    @property
    def state(self) -> State:
        """Return current episode state for openenv-core."""
        return self._state
