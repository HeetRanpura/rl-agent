import os
import sys
import json
import time
import urllib.request
import re

sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI

# All credentials read from environment variables — never hardcoded
API_BASE_URL   = os.getenv("API_BASE_URL",   "https://api.openai.com/v1")
MODEL_NAME     = os.getenv("MODEL_NAME",     "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENV_URL        = os.getenv("ENV_URL",        "http://localhost:7860")

client    = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
BENCHMARK = "scheme_env"
MAX_STEPS = 20

TASK_NAMES = {
    1: "scheme_discovery",
    2: "missing_data",
    3: "boundary_fraud",
    4: "escalation_dilemma",
}


def _post(path: str, body: dict) -> dict:
    """Send a JSON POST request to the environment server and return the parsed response."""
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(
        ENV_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def env_reset(task: int) -> dict:
    """Reset the environment to a specific task and return the initial observation."""
    return _post("/reset", {"seed": task})


def env_step(action_type: str, value: str) -> dict:
    """Send one action to the environment and return the resulting observation."""
    return _post("/step", {"action": {"action_type": action_type, "value": value}})


def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] structured log line required by the hackathon spec."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    """Emit a [STEP] structured log line required by the hackathon spec."""
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    """Emit the [END] structured log line required by the hackathon spec."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = """You are a CSC (Common Service Centre) operator in rural India.
Your job is to help villagers enroll in the correct government welfare scheme.

=== WORKFLOW — follow this phase order strictly ===
PHASE 1 GATHER:   If missing_data is not empty, ask for ONE field at a time using ask_question.
PHASE 2 IDENTIFY: Once missing_data is empty, determine the best scheme using eligibility rules.
PHASE 3 DECIDE:   Use approve_scheme for eligible applicants, reject_applicant for ineligible ones.
PHASE 4 SAFETY:   If you suspect data fraud or contradiction, use verify_document on PAN card first.
                  If the PAN card reveals a contradiction — you MUST use escalate immediately.

=== AVAILABLE ACTIONS — respond ONLY with valid JSON ===
{"action_type": "ask_question",    "value": "<field_name>"}
  Valid field_names ONLY: age, income, occupation, has_aadhaar

{"action_type": "request_document","value": "<doc_name>"}
  Use "pan_card" when you suspect data integrity issues.

{"action_type": "approve_scheme",  "value": "<scheme_name>"}
  Valid schemes: PMKVY, MGNREGS, PMAY

{"action_type": "reject_applicant","value": "<reason>"}

{"action_type": "escalate",        "value": ""}

=== SCHEME ELIGIBILITY — ALL conditions must be simultaneously true ===
PMKVY   : age 18-35, occupation mason OR carpenter, income STRICTLY < 10000
MGNREGS : age 18-60, occupation farm_labourer, has_aadhaar = True
PMAY    : age 21-55, any occupation, income STRICTLY < 6000, has_aadhaar = True
REJECT if no scheme criteria are fully satisfied.

=== CRITICAL RULES ===
NOISE FIELDS: known_profile may contain marital_status, bank_name, state_of_residence,
number_of_children. These are TRAPS. NEVER use ask_question for these fields (-1.0 reward).

INCOME PRECISION: Thresholds are strict integers.
income=10000 does NOT qualify for PMKVY (requires STRICTLY < 10000).
income=6000 does NOT qualify for PMAY (requires STRICTLY < 6000).

MISSING DATA: If missing_data is not empty, ask before deciding.
Approving with missing_data not empty costs -5.0 and ends the episode.

ESCALATION DILEMMA: If the applicant claims to be a student but PAN card shows
government pension deposits — this is a data integrity conflict.
First call request_document with value "pan_card" to confirm the contradiction.
Then immediately use escalate. Approving or rejecting is wrong (-5.0 or -3.0).

Respond ONLY with a JSON object. No explanation. No markdown. Just JSON."""


def get_agent_action(observation: dict, history: list):
    """
    Call the LLM with the current observation and conversation history.
    Extracts JSON from the response even if the model adds surrounding text.
    Falls back to escalate if JSON parsing fails completely.
    """
    obs_text = (
        f"known_profile: {observation.get('known_profile', {})}\n"
        f"missing_data: {observation.get('missing_data', [])}\n"
        f"notification: {observation.get('notification', '')}\n"
        f"is_terminated: {observation.get('is_terminated', False)}\n"
        f"What is your next action? Respond with JSON only."
    )

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-8:]
        + [{"role": "user", "content": obs_text}]
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            max_tokens=120, temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"action_type": "escalate", "value": ""}, f"API_ERROR: {e}"

    # Extract JSON even if the model wraps it in markdown or prose
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return {"action_type": "escalate", "value": ""}, raw


def run_episode(task: int) -> float:
    """Run one complete episode for the given task and return the grader score."""
    task_name = TASK_NAMES[task]

    # Emit [START] log required by hackathon spec
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(task)
    except Exception as e:
        print(f"[ERROR] env_reset failed for task {task}: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    obs          = result.get("observation", result)
    grader_score = 0.0
    rewards      = []
    history      = []
    step         = 0

    print(f"\n{'='*60}", flush=True)
    print(f"  TASK {task}/4 — {task_name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Profile : {obs.get('known_profile', {})}", flush=True)
    print(f"  Missing : {obs.get('missing_data', [])}", flush=True)
    print(f"  Notif   : {str(obs.get('notification', ''))[:120]}", flush=True)

    while step < MAX_STEPS:
        step += 1

        if obs.get("is_terminated", False):
            grader_score = obs.get("grader_score") or obs.get("metadata", {}).get("grader_score", 0.0)
            break

        action, raw_response = get_agent_action(obs, history)
        action_type = action.get("action_type", "escalate")
        value       = action.get("value", "") or ""

        history.append({"role": "assistant", "content": raw_response})

        try:
            step_result = env_step(action_type, value)
        except Exception as e:
            log_step(step=step, action=f"{action_type}({value!r})",
                     reward=0.0, done=False, error=str(e))
            continue

        obs          = step_result.get("observation", step_result)
        reward       = step_result.get("reward", 0.0)
        done         = step_result.get("done", False)
        notification = str(obs.get("notification", ""))

        rewards.append(reward)

        action_str = f"{action_type}({value!r})"

        # Emit [STEP] log required by hackathon spec
        log_step(step=step, action=action_str, reward=reward, done=done, error=None)

        print(f"  Step {step:02d}: {action_str} -> reward={reward}, done={done}", flush=True)
        print(f"           {notification[:100]}", flush=True)

        history.append({
            "role":    "user",
            "content": f"reward={reward}, notification={notification}",
        })

        if done:
            grader_score = obs.get("grader_score") or obs.get("metadata", {}).get("grader_score", None)
            if grader_score is None:
                if reward >= 10.0:  grader_score = 1.0
                elif reward >= 5.0: grader_score = 1.0
                elif reward >= 3.0: grader_score = 0.5
                else:               grader_score = 0.0
            break

        time.sleep(0.3)

    grader_score = float(grader_score or 0.0)
    success      = grader_score >= 1.0

    # Emit [END] log required by hackathon spec
    log_end(success=success, steps=step, score=grader_score, rewards=rewards)
    print(f"\n  GRADER SCORE: {grader_score:.3f} / 1.0", flush=True)
    return grader_score


def main():
    print(f"\n{'='*60}", flush=True)
    print(f"  SCHEME ENV — INFERENCE EVALUATION", flush=True)
    print(f"  Model : {MODEL_NAME}", flush=True)
    print(f"  Env   : {ENV_URL}", flush=True)
    print(f"{'='*60}", flush=True)

    scores = {}
    for task in [1, 2, 3, 4]:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            print(f"\n  [ERROR] Task {task} failed: {e}", flush=True)
            scores[task] = 0.0
        time.sleep(1)

    avg = sum(scores.values()) / len(scores)

    # Final score block — exact format required by hackathon spec
    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL GRADER SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Task 1 (Scheme Discovery)    : {scores[1]:.1f} / 1.0", flush=True)
    print(f"  Task 2 (Missing Data)        : {scores[2]:.1f} / 1.0", flush=True)
    print(f"  Task 3 (Boundary Fraud)      : {scores[3]:.1f} / 1.0", flush=True)
    print(f"  Task 4 (Escalation Dilemma)  : {scores[4]:.1f} / 1.0", flush=True)
    print(f"  Average                      : {avg:.2f} / 1.0", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()