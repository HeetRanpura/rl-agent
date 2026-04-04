import copy
import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import Action, Observation

# Maximum number of steps allowed per episode before forced termination
MAX_STEPS = 20

# Fields that are injected as distractions — querying these wastes a step and costs reward
NOISE_FIELDS = [
    "marital_status",
    "state_of_residence",
    "number_of_children",
    "bank_name",
]

# Possible values for each noise field — randomly selected at episode start
NOISE_VALUES = {
    "marital_status":     ["married", "unmarried", "widowed", "divorced"],
    "state_of_residence": ["Maharashtra", "Uttar Pradesh", "Bihar", "Rajasthan", "Gujarat"],
    "number_of_children": ["0", "1", "2", "3", "4"],
    "bank_name":          ["SBI", "PNB", "Bank of Baroda", "Canara Bank", "UCO Bank"],
}

# Only these fields are actually relevant to scheme eligibility decisions
VALID_QUERY_FIELDS = {"age", "income", "occupation", "has_aadhaar"}


def _inject_noise(profile: dict) -> dict:
    """Add 1 to 3 random irrelevant fields into the applicant profile to test agent focus."""
    chosen = random.sample(NOISE_FIELDS, k=random.randint(1, 3))
    for field in chosen:
        profile[field] = random.choice(NOISE_VALUES[field])
    return profile


def generate_dynamic_persona(task_id: int) -> dict:
    """
    Generate a randomised applicant profile for the given task.
    Values are randomised within bounds that guarantee the task's intended logic holds.
    No two resets produce identical profiles, preventing agents from memorising answers.
    """
    if task_id == 1:
        # Task 1: Profile qualifies for PMKVY — age, occupation, and income all within range
        age    = random.randint(18, 35)
        income = random.randint(1000, 9999)
        occ    = random.choice(["mason", "carpenter"])

        # PMAY is also eligible when income is low enough and age is in the PMAY range
        eligible = ["PMKVY"]
        if income < 6000 and 21 <= age <= 55:
            eligible.append("PMAY")

        return {
            "age": str(age), "income": str(income), "occupation": occ,
            "has_aadhaar": "True", "optimal_scheme": "PMKVY",
            "eligible_schemes": eligible,
        }

    elif task_id == 2:
        # Task 2: Profile qualifies for MGNREGS but occupation and has_aadhaar are hidden
        age    = random.randint(18, 60)
        income = random.randint(1000, 5000)

        return {
            "age": str(age), "income": str(income),
            "occupation": "farm_labourer", "has_aadhaar": "True",
            "optimal_scheme": "MGNREGS", "eligible_schemes": ["MGNREGS"],
            # These keys are hidden from the initial observation — agent must ask for them
            "missing_keys": ["occupation", "has_aadhaar"],
        }

    elif task_id == 3:
        # Task 3: Near-miss profile — looks PMKVY-eligible but income is strictly above threshold
        age    = random.randint(22, 34)
        income = random.randint(10001, 12000)   # Always above the 10000 PMKVY cap
        occ    = random.choice(["mason", "carpenter"])

        return {
            "age": str(age), "income": str(income), "occupation": occ,
            "has_aadhaar": "True", "optimal_scheme": None,
            "eligible_schemes": [], "_near_miss": True,
        }

    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def _make_fresh_obs(task: int, persona: dict) -> Observation:
    """
    Build the initial Observation for the given task using the generated persona.
    Noise fields are injected here so the agent sees distractions from the start.
    """
    # Start with the two fields always visible regardless of task
    profile = {
        "age":    persona["age"],
        "income": persona["income"],
    }

    # Task 1 and 3 show the full profile upfront — agent just needs to evaluate eligibility
    if task in [1, 3]:
        profile["occupation"]  = persona["occupation"]
        profile["has_aadhaar"] = persona["has_aadhaar"]

    # Inject random irrelevant fields to test whether the agent ignores distractions
    _inject_noise(profile)

    if task == 1:
        notif   = (
            "[TASK 1/3 - SCHEME DISCOVERY - Easy] Profile is COMPLETE. "
            "Some fields may be irrelevant to eligibility. "
            "Identify and approve the optimal welfare scheme."
        )
        missing = []

    elif task == 2:
        notif   = (
            "[TASK 2/3 - MISSING DATA - Medium] Profile is INCOMPLETE. "
            "You MUST collect ALL missing fields before approving any scheme. "
            "Note: some profile fields shown are irrelevant to eligibility."
        )
        missing = list(persona["missing_keys"])

    elif task == 3:
        notif   = (
            "[TASK 3/3 - BOUNDARY FRAUD DETECTION - Hard] Profile is COMPLETE. "
            "Review ALL scheme income thresholds with strict integer precision. "
            "Approve only if ALL criteria are satisfied. Otherwise use reject_applicant."
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
            "task":              task,
            "noise_queries":     0,   # incremented each time agent asks a noise field
            "redundant_queries": 0,   # incremented each time agent asks a known field
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
      - noise_queries:     each irrelevant field queried costs 0.08
      - redundant_queries: each already-known field re-queried costs 0.05
      - wasted steps:      in Task 2, steps beyond the minimum needed cost 0.04 each

    A correct outcome can never score below 0.30 — even a slow correct agent beats a wrong one.
    An incorrect terminal outcome always returns 0.0 regardless of efficiency.
    """
    # Wrong terminal outcomes are always 0.0 — no partial credit for wrong decisions
    if base_score <= 0.0:
        return 0.0

    # Calculate total penalty from noise and redundant queries
    penalty = (noise_queries * 0.08) + (redundant_queries * 0.05)

    # In Task 2, penalise extra steps beyond the minimum required to collect missing fields
    if task == 2 and missing_keys_total > 0:
        min_steps = missing_keys_total + 1   # one ask per missing field, plus one approve
        wasted    = max(0, step_count - min_steps)
        penalty  += wasted * 0.04

    # Apply penalty to base score, but floor at 0.30 so correct agents always outscore wrong ones
    return round(max(0.30, base_score - penalty), 3)


class SchemeEnvEnvironment(Environment):
    """
    OpenEnv-compatible environment simulating an Indian government welfare officer.
    The agent must interview applicants, identify the correct scheme, and enroll them.
    """

    # Disable concurrent sessions — singleton state cannot support multiple simultaneous users
    SUPPORTS_CONCURRENT_SESSIONS = False

    # Class-level shared state persists across multiple instantiations within the same process.
    # This is necessary because openenv-core may create new class instances per HTTP request.
    _shared_state = {}

    def __init__(self):
        super().__init__()

        # On cold start, initialise shared state with a default Task 1 episode
        if not SchemeEnvEnvironment._shared_state:
            persona = generate_dynamic_persona(1)
            obs     = _make_fresh_obs(1, persona)
            state   = State(episode_id=str(uuid4()), step_count=0)
            SchemeEnvEnvironment._shared_state = {
                "task": 1, "persona": persona, "state": state, "obs": obs,
            }

        # Load current shared state into instance variables for this request
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
        Start a new episode. If seed is 1, 2, or 3, select that specific task.
        Otherwise cycle through tasks automatically: 1 -> 2 -> 3 -> 1.
        Generates a fresh randomised persona on every call to prevent memorisation.
        """
        self._task    = seed if seed in (1, 2, 3) else (self._task % 3) + 1
        self._persona = generate_dynamic_persona(self._task)
        self._state   = State(episode_id=str(uuid4()), step_count=0)
        self._obs     = _make_fresh_obs(self._task, self._persona)
        self._save_shared()
        return self._obs

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        """
        Process one agent action and return the updated observation with reward signal.
        All 5 action types are handled with explicit reward shaping.
        Unknown or malformed actions return -1.0 and keep the episode alive.
        """
        self._state.step_count += 1
        obs          = self._obs
        current_task = self._task
        persona      = self._persona

        valid_actions = {
            "ask_question", "request_document",
            "approve_scheme", "reject_applicant", "escalate",
        }

        # Reject hallucinated or misspelled action types without crashing the server
        if action.action_type not in valid_actions:
            obs.notification = (
                f"Unknown action '{action.action_type}'. "
                f"Valid actions: {', '.join(sorted(valid_actions))}."
            )
            obs.reward = -1.0
            obs.done   = False
            return self._finalize_step(obs)

        if action.action_type == "ask_question":
            key = (action.value or "").strip()

            if key in NOISE_FIELDS:
                # Penalise querying a field that has no bearing on scheme eligibility
                obs.metadata["noise_queries"] += 1
                obs.notification = "Irrelevant field. Focus on scheme eligibility criteria."
                obs.reward       = -1.0

            elif key in obs.known_profile:
                # Penalise re-asking a field the agent already has in its profile
                obs.metadata["redundant_queries"] += 1
                obs.notification = f"'{key}' is already known. Do not ask redundant questions."
                obs.reward       = -1.0

            elif key in VALID_QUERY_FIELDS and key in persona:
                # Valid question — reveal the field value and remove it from missing_data
                val = persona[key]
                obs.known_profile[key] = val
                if key in obs.missing_data:
                    obs.missing_data.remove(key)
                obs.notification = f"Applicant answered: {key} = {val}."
                obs.reward       = 1.0

            else:
                # Field name is not recognised as a valid eligibility field
                obs.notification = f"'{key}' is not a valid eligibility field."
                obs.reward       = -1.0

        elif action.action_type == "request_document":
            # Document requests always succeed with a small positive reward
            obs.notification = f"Document '{action.value or 'document'}' received and verified."
            obs.reward       = 0.5

        elif action.action_type == "approve_scheme":
            scheme = (action.value or "").strip()

            if current_task == 3:
                # Task 3 has no eligible scheme — any approval is a boundary violation
                obs.notification = (
                    f"BOUNDARY VIOLATION: Applicant income is {int(persona['income'])} "
                    f"which exceeds the threshold. Use reject_applicant instead."
                )
                obs.reward        = -5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = 0.0
                obs.metadata["grader_score"] = 0.0
                return self._finalize_step(obs)

            if current_task == 2 and len(obs.missing_data) > 0:
                # Agent tried to approve before collecting all required data
                obs.notification = (
                    f"PREMATURE APPROVAL VIOLATION: Still missing: {obs.missing_data}. "
                    "Collect all required fields before approving."
                )
                obs.reward        = -5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = 0.0
                obs.metadata["grader_score"] = 0.0
                return self._finalize_step(obs)

            optimal  = persona["optimal_scheme"]
            eligible = persona["eligible_schemes"]

            if scheme == optimal:
                # Agent selected the best possible scheme for this applicant
                score = _compute_grader_score(
                    task               = current_task,
                    base_score         = 1.0,
                    step_count         = self._state.step_count,
                    noise_queries      = obs.metadata.get("noise_queries", 0),
                    redundant_queries  = obs.metadata.get("redundant_queries", 0),
                    missing_keys_total = len(persona.get("missing_keys", [])),
                )
                obs.notification = f"SUCCESS: Applicant correctly enrolled in {scheme}."
                obs.reward        = 10.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

            elif scheme in eligible:
                # Agent picked an eligible but suboptimal scheme
                score = _compute_grader_score(
                    task              = current_task,
                    base_score        = 0.5,
                    step_count        = self._state.step_count,
                    noise_queries     = obs.metadata.get("noise_queries", 0),
                    redundant_queries = obs.metadata.get("redundant_queries", 0),
                )
                obs.notification = f"Enrolled in {scheme}, but {optimal} was more optimal."
                obs.reward        = 3.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

            else:
                # Agent picked a scheme the applicant is not eligible for
                obs.notification = f"ERROR: Applicant is NOT eligible for '{scheme}'."
                obs.reward        = -5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = 0.0
                obs.metadata["grader_score"] = 0.0

        elif action.action_type == "reject_applicant":
            if current_task == 3:
                # Task 3 is the only task where rejection is the correct terminal action
                score = _compute_grader_score(
                    task              = current_task,
                    base_score        = 1.0,
                    step_count        = self._state.step_count,
                    noise_queries     = obs.metadata.get("noise_queries", 0),
                    redundant_queries = obs.metadata.get("redundant_queries", 0),
                )
                obs.notification = (
                    f"CORRECT REJECTION: Applicant income {persona['income']} "
                    f"exceeds the PMKVY threshold. No eligible scheme found."
                )
                obs.reward        = 5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = score
                obs.metadata["grader_score"] = score

            else:
                # Rejecting an eligible applicant is always wrong
                obs.notification = (
                    "ERROR: This applicant IS eligible for welfare schemes. "
                    "Rejection is not valid here."
                )
                obs.reward        = -5.0
                obs.done          = True
                obs.is_terminated = True
                obs.grader_score  = 0.0
                obs.metadata["grader_score"] = 0.0

        elif action.action_type == "escalate":
            # Escalation is never the right answer in Tasks 1-3 — always a penalty
            obs.notification = (
                "Case escalated to senior officer. Episode ends. "
                "Escalation should only be used when data integrity is compromised."
            )
            obs.reward        = -2.0
            obs.done          = True
            obs.is_terminated = True
            obs.grader_score  = 0.0
            obs.metadata["grader_score"] = 0.0

        return self._finalize_step(obs)

    def _finalize_step(self, obs: Observation) -> Observation:
        """
        Called at the end of every step. Handles timeout enforcement and
        syncs the updated observation back to the shared class-level state.
        """
        if self._state.step_count >= MAX_STEPS and not obs.done:
            # Force episode termination if agent has used all allowed steps
            obs.is_terminated            = True
            obs.notification             = f"TIMEOUT: {MAX_STEPS} steps reached. Case closed."
            obs.reward                   = -2.0
            obs.done                     = True
            obs.grader_score             = 0.0
            obs.metadata["grader_score"] = 0.0

        # Persist updated observation to shared state so next HTTP request sees it
        self._obs = obs
        self._save_shared()
        return obs

    @property
    def state(self) -> State:
        """Return the current episode state including episode_id and step_count."""
        return self._state