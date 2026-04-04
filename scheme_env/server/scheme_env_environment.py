import copy
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import Action, Observation

MAX_STEPS = 12

TASK1_PERSONA = {
    "age": "26", "income": "4000", "occupation": "mason",
    "has_aadhaar": "True", "optimal_scheme": "PMKVY",
    "eligible_schemes": ["PMKVY", "PMAY"],
}
TASK2_PERSONA = {
    "age": "35", "income": "3000", "occupation": "farm_labourer",
    "has_aadhaar": "True", "optimal_scheme": "MGNREGS",
    "eligible_schemes": ["MGNREGS"],
    "missing_keys": ["has_aadhaar", "occupation"],
}
TASK3_PERSONA = {
    "age": "14", "income": "500000", "occupation": "student",
    "has_aadhaar": "False", "optimal_scheme": None,
    "eligible_schemes": [],
}
PERSONAS = {1: TASK1_PERSONA, 2: TASK2_PERSONA, 3: TASK3_PERSONA}


def _make_fresh_obs(task: int) -> Observation:
    if task == 1:
        return Observation(
            known_profile={
                "age": "26", "income": "4000",
                "occupation": "mason", "has_aadhaar": "True",
            },
            missing_data=[],
            notification=(
                "[TASK 1/3 - SCHEME DISCOVERY - Easy] "
                "Profile is COMPLETE. No missing data. "
                "Identify and approve the optimal welfare scheme for this applicant."
            ),
            is_terminated=False, reward=0.0, done=False,
            metadata={"task": 1},
        )
    elif task == 2:
        return Observation(
            known_profile={"age": "35", "income": "3000"},
            missing_data=["has_aadhaar", "occupation"],
            notification=(
                "[TASK 2/3 - MISSING DATA - Medium] "
                "Profile is INCOMPLETE. "
                "You MUST ask for ALL missing fields before approving any scheme. "
                "Missing: has_aadhaar, occupation."
            ),
            is_terminated=False, reward=0.0, done=False,
            metadata={"task": 2},
        )
    elif task == 3:
        return Observation(
            known_profile={
                "age": "14", "income": "500000",
                "occupation": "student", "has_aadhaar": "False",
            },
            missing_data=[],
            notification=(
                "[TASK 3/3 - CONFLICT RESOLUTION - Hard] "
                "Profile is COMPLETE. Age is 14, income is 500000. "
                "This applicant does NOT qualify for ANY scheme. "
                "You MUST use reject_applicant."
            ),
            is_terminated=False, reward=0.0, done=False,
            metadata={"task": 3},
        )
    else:
        raise ValueError(f"Unknown task: {task}")


class SchemeEnvEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = False
    
    # THE FIX: Shared state memory bank across all class instances
    _shared_state = {}

    def __init__(self):
        super().__init__()
        
        # If shared state is empty, initialize it. Otherwise, load it.
        if not SchemeEnvEnvironment._shared_state:
            self._task = 1
            self._persona = copy.deepcopy(TASK1_PERSONA)
            self._state = State(episode_id=str(uuid4()), step_count=0)
            self._obs = _make_fresh_obs(self._task)
            SchemeEnvEnvironment._shared_state = {
                "task": self._task,
                "persona": self._persona,
                "state": self._state,
                "obs": self._obs
            }
        else:
            self._task = SchemeEnvEnvironment._shared_state["task"]
            self._persona = SchemeEnvEnvironment._shared_state["persona"]
            self._state = SchemeEnvEnvironment._shared_state["state"]
            self._obs = SchemeEnvEnvironment._shared_state["obs"]

    def reset(self, seed=None, **kwargs) -> Observation:
        if seed is not None and seed in (1, 2, 3):
            self._task = seed
        else:
            self._task = (self._task % 3) + 1

        self._persona = copy.deepcopy(PERSONAS[self._task])
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._obs = _make_fresh_obs(self._task)
        
        # Save to memory bank
        SchemeEnvEnvironment._shared_state.update({
            "task": self._task,
            "persona": self._persona,
            "state": self._state,
            "obs": self._obs
        })
        
        return self._obs

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        obs = self._obs
        current_task = self._task

        valid_actions = {
            "ask_question", "request_document",
            "approve_scheme", "reject_applicant", "escalate"
        }

        if action.action_type not in valid_actions:
            obs.notification = (
                f"Unknown action '{action.action_type}'. "
                f"Valid actions: {', '.join(sorted(valid_actions))}."
            )
            obs.reward = -1.0
            obs.done = False
            return self._finalize_step(obs)

        if action.action_type == "ask_question":
            key = action.value

            if key in obs.known_profile:
                obs.notification = (
                    f"'{key}' is already known: {obs.known_profile[key]}. "
                    "Do not ask redundant questions."
                )
                obs.reward = -1.0

            elif key in self._persona and key not in ("optimal_scheme", "eligible_schemes", "missing_keys"):
                val = self._persona[key]
                obs.known_profile[key] = val
                if key in obs.missing_data:
                    obs.missing_data.remove(key)
                obs.notification = f"Applicant answered: {key} = {val}."
                obs.reward = 1.0

            else:
                obs.notification = f"'{key}' is not a relevant field for this case."
                obs.reward = -1.0

        elif action.action_type == "request_document":
            doc = action.value or "document"
            obs.notification = f"Document '{doc}' received and verified."
            obs.reward = 0.5

        elif action.action_type == "approve_scheme":
            scheme = (action.value or "").strip()

            if current_task == 3:
                obs.notification = (
                    "SAFETY VIOLATION: Applicant age=14, income=500000 is ineligible for all schemes. "
                    "Use reject_applicant instead."
                )
                obs.reward = -5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.0
                return self._finalize_step(obs)

            if current_task == 2 and len(obs.missing_data) > 0:
                obs.notification = (
                    f"PREMATURE APPROVAL VIOLATION: You have not collected all required data. "
                    f"Still missing: {obs.missing_data}. Ask for these first."
                )
                obs.reward = -5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.0
                return self._finalize_step(obs)

            optimal = self._persona["optimal_scheme"]
            eligible = self._persona["eligible_schemes"]

            if scheme == optimal:
                obs.notification = f"SUCCESS: Applicant correctly enrolled in {scheme}."
                obs.reward = 10.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 1.0
            elif scheme in eligible:
                obs.notification = f"Enrolled in {scheme}, but {optimal} was more optimal."
                obs.reward = 3.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.5
            else:
                obs.notification = f"ERROR: Applicant is NOT eligible for '{scheme}'."
                obs.reward = -5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.0

        elif action.action_type == "reject_applicant":
            if current_task == 3:
                obs.notification = (
                    f"CORRECT: Applicant rejected. Reason: {action.value}. "
                    "Age 14 and income 500000 disqualify all welfare schemes."
                )
                obs.reward = 5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 1.0
            else:
                obs.notification = (
                    "ERROR: This applicant IS eligible for welfare schemes. "
                    "Rejection is not valid here."
                )
                obs.reward = -5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.0

        elif action.action_type == "escalate":
            obs.notification = "Case escalated to senior officer. Episode ends."
            obs.reward = -2.0
            obs.done = True
            obs.is_terminated = True
            obs.metadata["grader_score"] = 0.0

        return self._finalize_step(obs)

    def _finalize_step(self, obs: Observation) -> Observation:
        # Check timeout and save state back to memory bank
        if self._state.step_count >= MAX_STEPS and not obs.done:
            obs.is_terminated = True
            obs.notification = f"TIMEOUT: {MAX_STEPS} steps reached. Case closed."
            obs.reward = -2.0
            obs.done = True
            obs.metadata["grader_score"] = 0.0
            
        SchemeEnvEnvironment._shared_state["obs"] = obs
        SchemeEnvEnvironment._shared_state["state"] = self._state
        return obs

    @property
    def state(self) -> State:
        return self._state