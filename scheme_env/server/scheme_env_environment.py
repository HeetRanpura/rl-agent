import copy
import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import Action, Observation

MAX_STEPS = 12

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

VALID_QUERY_FIELDS = {"age", "income", "occupation", "has_aadhaar"}

def _inject_noise(profile: dict) -> dict:
    chosen = random.sample(NOISE_FIELDS, k=random.randint(1, 3))
    for field in chosen:
        profile[field] = random.choice(NOISE_VALUES[field])
    return profile

def generate_dynamic_persona(task_id: int) -> dict:
    if task_id == 1:
        age    = random.randint(18, 35)
        income = random.randint(1000, 9999)
        occ    = random.choice(["mason", "carpenter"])

        eligible = ["PMKVY"]
        if income < 6000 and 21 <= age <= 55:
            eligible.append("PMAY")

        return {
            "age": str(age), "income": str(income), "occupation": occ,
            "has_aadhaar": "True", "optimal_scheme": "PMKVY", "eligible_schemes": eligible,
        }
    elif task_id == 2:
        age    = random.randint(18, 60)
        income = random.randint(1000, 5000)

        return {
            "age": str(age), "income": str(income), "occupation": "farm_labourer",
            "has_aadhaar": "True", "optimal_scheme": "MGNREGS", "eligible_schemes": ["MGNREGS"],
            "missing_keys": ["occupation", "has_aadhaar"],
        }
    elif task_id == 3:
        age    = random.randint(22, 34)         
        income = random.randint(10001, 12000)   
        occ    = random.choice(["mason", "carpenter"])

        return {
            "age": str(age), "income": str(income), "occupation": occ,
            "has_aadhaar": "True", "optimal_scheme": None, "eligible_schemes": [],
            "_near_miss": True,   
        }
    elif task_id == 4:
        age    = random.randint(22, 45)
        income = random.randint(2000, 8000)

        return {
            "age": str(age), "income": str(income), "occupation": "student",
            "has_aadhaar": "True", "optimal_scheme": None, "eligible_schemes": [],
            "_contradictory": True,   
        }
    else:
        raise ValueError(f"Unknown task_id: {task_id}")

def _make_fresh_obs(task: int, persona: dict) -> Observation:
    profile = {
        "age": persona["age"],
        "income": persona["income"],
    }
    
    if task in [1, 3, 4]:
        profile["occupation"] = persona["occupation"]
        profile["has_aadhaar"] = persona["has_aadhaar"]
        
    _inject_noise(profile)  

    if task == 1:
        notif = "[TASK 1/4 - SCHEME DISCOVERY - Easy] Profile is COMPLETE. Some fields may be irrelevant to eligibility. Identify and approve the optimal welfare scheme."
        missing = []
    elif task == 2:
        notif = f"[TASK 2/4 - MISSING DATA - Medium] Profile is INCOMPLETE. You MUST collect ALL missing fields before approving any scheme. Note: some profile fields shown are irrelevant."
        missing = list(persona["missing_keys"])
    elif task == 3:
        notif = "[TASK 3/4 - BOUNDARY FRAUD DETECTION - Hard] Profile is COMPLETE. Review ALL scheme income thresholds with strict integer precision. Approve only if ALL criteria are satisfied. Otherwise use reject_applicant."
        missing = []
    elif task == 4:
        notif = "[TASK 4/4 - ESCALATION DILEMMA - Expert] Profile is COMPLETE. SYSTEM ALERT: Applicant PAN card shows active government pension deposits inconsistent with stated occupation 'student'. Data integrity is COMPROMISED."
        missing = []

    return Observation(
        known_profile=profile, missing_data=missing, notification=notif,
        is_terminated=False, reward=0.0, done=False, metadata={"task": task},
    )

class SchemeEnvEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False
    _shared_state = {}

    def __init__(self):
        super().__init__()
        if not SchemeEnvEnvironment._shared_state:
            persona = generate_dynamic_persona(1)
            obs     = _make_fresh_obs(1, persona)
            state   = State(episode_id=str(uuid4()), step_count=0)
            SchemeEnvEnvironment._shared_state = {"task": 1, "persona": persona, "state": state, "obs": obs}
        self._load_shared()

    def _load_shared(self):
        s = SchemeEnvEnvironment._shared_state
        self._task, self._persona, self._state, self._obs = s["task"], s["persona"], s["state"], s["obs"]

    def _save_shared(self):
        SchemeEnvEnvironment._shared_state.update({"task": self._task, "persona": self._persona, "state": self._state, "obs": self._obs})

    def reset(self, seed=None, **kwargs) -> Observation:
        self._task = seed if seed in (1, 2, 3, 4) else (self._task % 4) + 1
        self._persona = generate_dynamic_persona(self._task)
        self._state   = State(episode_id=str(uuid4()), step_count=0)
        self._obs     = _make_fresh_obs(self._task, self._persona)
        self._save_shared()
        return self._obs

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        obs, current_task, persona = self._obs, self._task, self._persona

        valid_actions = {"ask_question", "request_document", "approve_scheme", "reject_applicant", "escalate"}

        if action.action_type not in valid_actions:
            obs.notification = f"Unknown action. Valid actions: {', '.join(sorted(valid_actions))}."
            obs.reward, obs.done = -1.0, False
            return self._finalize_step(obs)

        if action.action_type == "ask_question":
            key = (action.value or "").strip()
            if key in NOISE_FIELDS:
                obs.notification, obs.reward = "Irrelevant field. Focus on scheme eligibility criteria.", -1.0
            elif key in obs.known_profile:
                obs.notification, obs.reward = f"'{key}' is already known. Do not ask redundant questions.", -1.0
            elif key in VALID_QUERY_FIELDS and key in persona:
                val = persona[key]
                obs.known_profile[key] = val
                if key in obs.missing_data:
                    obs.missing_data.remove(key)
                obs.notification, obs.reward = f"Applicant answered: {key} = {val}.", 1.0
            else:
                obs.notification, obs.reward = f"'{key}' is not a valid field.", -1.0

        elif action.action_type == "request_document":
            obs.notification, obs.reward = f"Document '{action.value or 'document'}' received and verified.", 0.5

        elif action.action_type == "approve_scheme":
            scheme = (action.value or "").strip()
            
            if current_task == 4:
                obs.notification = "FRAUD AUTHORIZATION VIOLATION: This case has contradictory data. The correct action is to escalate to a senior officer."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = -5.0, True, True, 0.0
                return self._finalize_step(obs)
                
            if current_task == 3:
                obs.notification = f"BOUNDARY VIOLATION: Applicant income is {int(persona['income'])} — threshold breached. Use reject_applicant."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = -5.0, True, True, 0.0
                return self._finalize_step(obs)

            if current_task == 2 and len(obs.missing_data) > 0:
                obs.notification = f"PREMATURE APPROVAL VIOLATION: You have not collected all required data. Still missing: {obs.missing_data}."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = -5.0, True, True, 0.0
                return self._finalize_step(obs)

            optimal, eligible = persona["optimal_scheme"], persona["eligible_schemes"]
            if scheme == optimal:
                obs.notification = f"SUCCESS: Applicant correctly enrolled in {scheme}."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = 10.0, True, True, 1.0
            elif scheme in eligible:
                obs.notification = f"Enrolled in {scheme}, but {optimal} was the more optimal scheme."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = 3.0, True, True, 0.5
            else:
                obs.notification = f"ERROR: Applicant is NOT eligible for '{scheme}'."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = -5.0, True, True, 0.0

        elif action.action_type == "reject_applicant":
            if current_task == 4:
                obs.notification = "PREMATURE ADJUDICATION: You cannot reject this applicant without resolving the data integrity conflict. Use escalate."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = -3.0, True, True, 0.0
            elif current_task == 3:
                obs.notification = f"CORRECT REJECTION: Applicant income {persona['income']} exceeds the PMKVY threshold. No eligible scheme found."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = 5.0, True, True, 1.0
            else:
                obs.notification = "ERROR: This applicant IS eligible for welfare schemes. Rejection is not valid here."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = -5.0, True, True, 0.0

        elif action.action_type == "escalate":
            if current_task == 4:
                obs.notification = "CORRECT ESCALATION: Data integrity conflict detected and properly handed off to a senior officer."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = 10.0, True, True, 1.0
            else:
                obs.notification = "Case escalated to senior officer. Episode ends. Escalation should only be used when data integrity is compromised."
                obs.reward, obs.done, obs.is_terminated, obs.metadata["grader_score"] = -2.0, True, True, 0.0

        return self._finalize_step(obs)

    def _finalize_step(self, obs: Observation) -> Observation:
        if self._state.step_count >= MAX_STEPS and not obs.done:
            obs.is_terminated = True
            obs.notification = f"TIMEOUT: {MAX_STEPS} steps reached. Case closed."
            obs.reward, obs.done, obs.metadata["grader_score"] = -2.0, True, 0.0
        self._obs = obs
        self._save_shared()
        return obs

    @property
    def state(self) -> State:
        return self._state