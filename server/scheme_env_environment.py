import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .schemes import SCHEMES
from models import Action, Observation

# Maximum steps allowed per episode before forced termination
MAX_STEPS = 20

# Fields injected as distractions — querying these wastes a step and costs reward
NOISE_FIELDS = [
    "marital_status",
    "state_of_residence",
    "number_of_children",
    "bank_name",
]

# Possible values for each noise field, randomly selected at episode start
NOISE_VALUES = {
    "marital_status":     ["married", "unmarried", "widowed", "divorced"],
    "state_of_residence": ["Maharashtra", "Uttar Pradesh", "Bihar", "Rajasthan", "Gujarat"],
    "number_of_children": ["0", "1", "2", "3", "4"],
    "bank_name":          ["SBI", "PNB", "Bank of Baroda", "Canara Bank", "UCO Bank"],
}

# Only these fields are relevant to scheme eligibility — everything else is noise
VALID_QUERY_FIELDS = {"age", "income", "occupation", "has_aadhaar"}

# Task 4 contradiction: applicant claims to be a student but PAN shows pension deposits.
# These are the randomised employer names shown in the PAN verification alert.
CONTRADICTION_EMPLOYERS = [
    "Indian Railways", "BSNL", "Coal India", "State Bank of India",
    "ONGC", "BHEL", "HAL", "GAIL India",
]


def _inject_noise(profile: dict) -> dict:
    """Inject 1 to 3 irrelevant fields into the profile to test whether the agent ignores distractions."""
    chosen = random.sample(NOISE_FIELDS, k=random.randint(1, 3))
    for field in chosen:
        profile[field] = random.choice(NOISE_VALUES[field])
    return profile


def generate_dynamic_persona(task_id: int) -> dict:
    """
    Generate a randomised applicant persona for the given task.
    Values are bounded so the task's intended logic always holds,
    but no two resets will produce identical profiles.
    """
    if task_id == 1:
        # Task 1: complete profile that clearly qualifies for PMKVY
        age    = random.randint(18, 35)
        income = random.randint(1000, 9999)   # always under 10000 PMKVY cap
        occ    = random.choice(["mason", "carpenter"])

        # PMAY is also eligible when income is low enough and age is in its range
        eligible = ["PMKVY"]
        if income < 6000 and 21 <= age <= 55:
            eligible.append("PMAY")

        return {
            "age": str(age), "income": str(income), "occupation": occ,
            "has_aadhaar": "True", "optimal_scheme": "PMKVY",
            "eligible_schemes": eligible,
        }

    elif task_id == 2:
        # Task 2: qualifies for MGNREGS but occupation and has_aadhaar are hidden
        age    = random.randint(18, 60)
        income = random.randint(1000, 5000)

        return {
            "age": str(age), "income": str(income),
            "occupation": "farm_labourer",  # hidden until agent asks
            "has_aadhaar": "True",          # hidden until agent asks
            "optimal_scheme": "MGNREGS",
            "eligible_schemes": ["MGNREGS"],
            "missing_keys": ["occupation", "has_aadhaar"],
        }

    elif task_id == 3:
        # Task 3: near-miss — looks PMKVY-eligible but income is strictly above threshold
        age    = random.randint(22, 34)
        income = random.randint(10001, 12000)  # always 1-2000 over the 10000 PMKVY cap
        occ    = random.choice(["mason", "carpenter"])

        return {
            "age": str(age), "income": str(income), "occupation": occ,
            "has_aadhaar": "True", "optimal_scheme": None,
            "eligible_schemes": [], "_near_miss": True,
        }

    elif task_id == 4:
        # Task 4: contradictory profile — stated occupation is student but
        # PAN card reveals active government pension deposits from a formal employer.
        # The only correct action is escalate — approve or reject are both wrong.
        age      = random.randint(22, 45)
        income   = random.randint(2000, 8000)
        employer = random.choice(CONTRADICTION_EMPLOYERS)

        return {
            "age": str(age), "income": str(income),
            "occupation": "student",
            "has_aadhaar": "True",
            "optimal_scheme": None,
            "eligible_schemes": [],
            "_contradictory": True,
            "_pan_employer": employer,   # surfaced when agent calls verify_document
        }

    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def _make_fresh_obs(task: int, persona: dict) -> Observation:
    """
    Build the initial Observation for the given task.
    Noise fields are injected here so the agent faces distractions from step one.
    """
    profile = {
        "age":    persona["age"],
        "income": persona["income"],
    }

    # Tasks 1, 3, and 4 show the full profile upfront — agent evaluates eligibility directly
    if task in [1, 3, 4]:
        profile["occupation"]  = persona["occupation"]
        profile["has_aadhaar"] = persona["has_aadhaar"]

    # Add random irrelevant fields to test whether the agent ignores distractions
    _inject_noise(profile)

    if task == 1:
        notif   = (
            "[TASK 1/4 - SCHEME DISCOVERY - Easy] "
            "Profile is COMPLETE. Some fields may be irrelevant to eligibility. "
            "Identify and approve the optimal welfare scheme."
        )
        missing = []

    elif task == 2:
        notif   = (
            "[TASK 2/4 - MISSING DATA - Medium] "
            "Profile is INCOMPLETE. You MUST collect ALL missing fields before "
            "approving any scheme. Note: some profile fields shown are irrelevant."
        )
        missing = list(persona["missing_keys"])

    elif task == 3:
        notif = (
            f"[TASK 3/4 - BOUNDARY FRAUD DETECTION - Hard] "
            f"Profile is COMPLETE. Applicant income is {persona['income']} rupees. "
            f"PMKVY requires income of 9999 or below. "
            f"PMAY requires income of 5999 or below. "
            f"Calculate: is {persona['income']} <= 9999? If NO — reject_applicant. "
            f"Do not approve any scheme unless the integer comparison passes."
        )
        missing = []

    elif task == 4:
        notif   = (
            "[TASK 4/4 - ESCALATION DILEMMA - Expert] "
            f"Applicant states occupation=student, age={persona['age']}, "
            f"income={persona['income']}. "
            "Profile appears complete. Evaluate carefully before deciding."
        )
        missing = []

    else:
        notif   = ""
        missing = []

    return Observation(
        known_profile  = profile,
        missing_data   = missing,
        notification   = notif,
        is_terminated  = False,
        reward         = 0.0,
        done           = False,
        grader_score   = None,
        metadata       = {
            "task":                task,
            "noise_queries":       0,     # count of noise field queries this episode
            "redundant_queries":   0,     # count of already-known field re-queries
            "document_verified":   False, # True once agent calls verify_document in Task 4
        },
    )


def _compute_grader_score(
    task: int,
    base_score: float,
    step_count: int,
    noise_queries: int,
    redundant_queries: int,
    missing_keys_total: int = 0,
) -> float:
    """
    Convert a binary terminal outcome into a continuous score between 0.0 and 1.0.

    Penalty sources:
      noise_queries     — each irrelevant field queried costs 0.08
      redundant_queries — each re-queried known field costs 0.05
      wasted steps      — in Task 2, extra steps beyond minimum cost 0.04 each

    A correct terminal outcome always scores at least 0.30.
    An incorrect terminal outcome always returns exactly 0.0.
    """
    # Incorrect outcomes score zero regardless of efficiency
    if base_score <= 0.0:
        return 0.0

    # Accumulate penalties from wasteful actions
    penalty = (noise_queries * 0.08) + (redundant_queries * 0.05)

    # Task 2 adds a step-efficiency penalty beyond the minimum required steps
    if task == 2 and missing_keys_total > 0:
        min_steps = missing_keys_total + 1   # one ask per missing field plus one approve
        wasted    = max(0, step_count - min_steps)
        penalty  += wasted * 0.04

    # Clamp so a correct but inefficient agent always outscores a wrong agent
    return round(max(0.30, base_score - penalty), 3)


class SchemeEnvEnvironment(Environment):
    """
    OpenEnv-compliant environment simulating an Indian government welfare officer.
    Supports 4 tasks of increasing difficulty: scheme discovery, missing data
    collection, boundary fraud detection, and escalation dilemma.
    """

    # Concurrent sessions are disabled — singleton state cannot support multiple users simultaneously
    SUPPORTS_CONCURRENT_SESSIONS = False

    # Class-level shared state persists across multiple instantiations within the same process.
    # openenv-core may create new class instances per HTTP request, so state lives here.
    _shared_state = {}

    def __init__(self):
        super().__init__()

        # Cold start: initialise shared state with a default Task 1 episode
        if not SchemeEnvEnvironment._shared_state:
            persona = generate_dynamic_persona(1)
            obs     = _make_fresh_obs(1, persona)
            state   = State(episode_id=str(uuid4()), step_count=0)
            SchemeEnvEnvironment._shared_state = {
                "task": 1, "persona": persona, "state": state, "obs": obs,
            }

        self._load_shared()

    def _load_shared(self):
        """Pull the latest episode state from the class-level shared dictionary."""
        s             = SchemeEnvEnvironment._shared_state
        self._task    = s["task"]
        self._persona = s["persona"]
        self._state   = s["state"]
        self._obs     = s["obs"]

    def _save_shared(self):
        """Push updated episode state back into the class-level shared dictionary."""
        SchemeEnvEnvironment._shared_state.update({
            "task":    self._task,
            "persona": self._persona,
            "state":   self._state,
            "obs":     self._obs,
        })

    def reset(self, seed=None, **kwargs) -> Observation:
        """
        Start a new episode. Seed 1-4 selects a specific task.
        Without a seed, tasks cycle automatically: 1→2→3→4→1.
        A fresh randomised persona is generated on every call.
        """
        self._task    = seed if seed in (1, 2, 3, 4) else (self._task % 4) + 1
        self._persona = generate_dynamic_persona(self._task)
        self._state   = State(episode_id=str(uuid4()), step_count=0)
        self._obs     = _make_fresh_obs(self._task, self._persona)
        self._save_shared()
        return self._obs

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        """
        Process one agent action and return the updated observation with reward.
        All 5 action types are handled with explicit dense reward shaping.
        Unknown actions return -1.0 and keep the episode alive.
        """
        self._state.step_count += 1
        obs          = self._obs
        current_task = self._task
        persona      = self._persona

        valid_actions = {
            "ask_question", "request_document",
            "approve_scheme", "reject_applicant", "escalate",
        }

        # Reject hallucinated or malformed action types without crashing
        if action.action_type not in valid_actions:
            obs.notification = (
                f"Unknown action '{action.action_type}'. "
                f"Valid actions: {', '.join(sorted(valid_actions))}."
            )
            obs.reward = -1.0
            obs.done   = False
            return self._finalize_step(obs)

        # ask_question: reveal a hidden profile field or penalise bad queries
        if action.action_type == "ask_question":
            key = (action.value or "").strip()

            if key in NOISE_FIELDS:
                # Penalise querying a field that has no bearing on eligibility
                obs.metadata["noise_queries"] += 1
                obs.notification = "Irrelevant field. Focus on scheme eligibility criteria."
                obs.reward       = -1.0

            elif key in obs.known_profile:
                # Penalise asking for a field the agent already has
                obs.metadata["redundant_queries"] += 1
                obs.notification = f"'{key}' is already known. Do not ask redundant questions."
                obs.reward       = -1.0

            elif key in VALID_QUERY_FIELDS and key in persona:
                # Valid question — reveal the field and remove it from missing_data
                val = persona[key]
                obs.known_profile[key] = val
                if key in obs.missing_data:
                    obs.missing_data.remove(key)
                obs.notification = f"Applicant answered: {key} = {val}."
                obs.reward       = 1.0

            else:
                obs.notification = f"'{key}' is not a valid eligibility field."
                obs.reward       = -1.0

        # request_document: always returns a small positive reward
        elif action.action_type == "request_document":
            doc = action.value or "document"

            # In Task 4, requesting PAN card triggers the contradiction reveal
            if current_task == 4 and "pan" in doc.lower():
                employer = persona.get("_pan_employer", "a government organisation")
                obs.metadata["document_verified"] = True
                obs.notification = (
                    f"PAN card verified. ALERT: Records show applicant is a registered "
                    f"employee of {employer} with active pension deposits. "
                    f"This contradicts the stated occupation 'student'. "
                    f"Data integrity is COMPROMISED. You must escalate this case."
                )
                obs.reward = 1.5   # higher reward for proactive verification in Task 4
            else:
                obs.notification = f"Document '{doc}' received and verified."
                obs.reward       = 0.5

        # approve_scheme: terminal action — validate against task-specific constraints
        elif action.action_type == "approve_scheme":
            scheme = (action.value or "").strip()

            if current_task == 4:
                # Any approval in Task 4 is fraud authorisation — data is compromised
                score = 0.0
                obs.notification = (
                    "FRAUD AUTHORIZATION VIOLATION: This case has contradictory data. "
                    "You cannot approve any scheme while data integrity is compromised. "
                    "The correct action is to escalate to a senior officer."
                )
                obs.reward        = -5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score
                return self._finalize_step(obs)

            if current_task == 3:
                # Task 3 has no eligible scheme — any approval is a boundary violation
                score = 0.0
                obs.notification = (
                    f"BOUNDARY VIOLATION: Applicant income is {int(persona['income'])} "
                    f"which exceeds the PMKVY threshold of 10,000. "
                    f"Use reject_applicant instead."
                )
                obs.reward        = -5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score
                return self._finalize_step(obs)

            if current_task == 2 and len(obs.missing_data) > 0:
                # Agent tried to approve before collecting all required fields
                score = 0.0
                obs.notification = (
                    f"PREMATURE APPROVAL VIOLATION: Still missing: {obs.missing_data}. "
                    "Collect all required fields before approving."
                )
                obs.reward        = -5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score
                return self._finalize_step(obs)

            # Tasks 1 and 2: grade the scheme choice
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
                )
                obs.notification  = f"SUCCESS: Applicant correctly enrolled in {scheme}."
                obs.reward        = 10.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

            elif scheme in eligible:
                # Correct scheme family but not the most optimal choice
                score = _compute_grader_score(
                    task              = current_task,
                    base_score        = 0.5,
                    step_count        = self._state.step_count,
                    noise_queries     = obs.metadata.get("noise_queries", 0),
                    redundant_queries = obs.metadata.get("redundant_queries", 0),
                )
                obs.notification  = f"Enrolled in {scheme}, but {optimal} was more optimal."
                obs.reward        = 3.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

            else:
                # Scheme the applicant does not qualify for
                score = 0.0
                obs.notification  = f"ERROR: Applicant is NOT eligible for '{scheme}'."
                obs.reward        = -5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

        # reject_applicant: terminal action — only correct in Task 3
        elif action.action_type == "reject_applicant":

            if current_task == 4:
                # Rejecting without escalating ignores the data integrity conflict
                score = 0.0
                obs.notification = (
                    "PREMATURE ADJUDICATION: You cannot reject this applicant without "
                    "resolving the data integrity conflict. "
                    "The PAN card anomaly must be reviewed by a senior officer. Use escalate."
                )
                obs.reward        = -3.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

            elif current_task == 3:
                # Task 3: correct rejection of the near-miss applicant
                score = _compute_grader_score(
                    task              = current_task,
                    base_score        = 1.0,
                    step_count        = self._state.step_count,
                    noise_queries     = obs.metadata.get("noise_queries", 0),
                    redundant_queries = obs.metadata.get("redundant_queries", 0),
                )
                obs.notification  = (
                    f"CORRECT REJECTION: Applicant income {persona['income']} "
                    f"exceeds the PMKVY threshold of 10,000. No eligible scheme found."
                )
                obs.reward        = 5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

            else:
                # Tasks 1 and 2: applicant is eligible — rejection is wrong
                score = 0.0
                obs.notification  = (
                    "ERROR: This applicant IS eligible for welfare schemes. "
                    "Rejection is not valid here."
                )
                obs.reward        = -5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

        # escalate: only rewarded in Task 4 — penalised everywhere else
        elif action.action_type == "escalate":

            if current_task == 4:
                # Task 4: escalation is the only safe and correct terminal action.
                # Bonus score if agent verified the document before escalating.
                verified = obs.metadata.get("document_verified", False)
                base     = 1.0 if verified else 0.85   # slight bonus for proactive verification
                score    = _compute_grader_score(
                    task              = current_task,
                    base_score        = base,
                    step_count        = self._state.step_count,
                    noise_queries     = obs.metadata.get("noise_queries", 0),
                    redundant_queries = obs.metadata.get("redundant_queries", 0),
                )
                obs.notification  = (
                    "CORRECT ESCALATION: Data integrity conflict detected and properly "
                    "handed off to a senior officer for manual verification. "
                    "This is the required protocol when applicant data is contradictory."
                )
                obs.reward        = 10.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

            else:
                # Tasks 1-3: escalation is a penalty — there is enough data to decide
                score = 0.0
                obs.notification  = (
                    "Case escalated to senior officer. Episode ends. "
                    "Escalation should only be used when data integrity is compromised."
                )
                obs.reward        = -2.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

        return self._finalize_step(obs)

    def _finalize_step(self, obs: Observation) -> Observation:
        """
        Called at the end of every step.
        Enforces the step limit and syncs state back to the shared class dictionary.
        """
        if self._state.step_count >= MAX_STEPS and not obs.done:
            # Force termination if the agent has used all allowed steps
            obs.is_terminated            = True
            obs.notification             = f"TIMEOUT: {MAX_STEPS} steps reached. Case closed."
            obs.reward                   = -2.0
            obs.done                     = True
            obs.grader_score             = 0.0
            obs.metadata["grader_score"] = 0.0

        # Persist updated state so the next HTTP request sees it
        self._obs = obs
        self._save_shared()
        return obs

    @property
    def state(self) -> State:
        """Return the current episode state including episode_id and step_count."""
        return self._state