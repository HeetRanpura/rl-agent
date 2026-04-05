import os
import sys
import json
import time
import urllib.request
import re

sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI

# All credentials from environment — never hardcoded
API_BASE_URL   = os.getenv("API_BASE_URL",   "https://router.huggingface.co/v1")
MODEL_NAME     = os.getenv("MODEL_NAME",     "Qwen/Qwen2.5-7B-Instruct")
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
    5: "document_conflict",
}


def _post(path: str, body: dict) -> dict:
    """POST JSON to the environment server and return parsed response."""
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(
        ENV_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def env_reset(task: int) -> dict:
    return _post("/reset", {"seed": task})


def env_step(action_type: str, value: str) -> dict:
    return _post("/step", {"action": {"action_type": action_type, "value": value}})


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# =========================================================
# SYSTEM PROMPT — OPTION A HARDENED VERSION
#
# KEY DIFFERENCES from original:
# 1. No escalation recipe — agent must infer when to escalate
# 2. No income/threshold hints in task descriptions
# 3. Explicit PMAY > PMKVY priority when both eligible
# 4. Document verification workflow described generically
# 5. No "SYSTEM ALERT" pattern matching guidance
# =========================================================

SYSTEM_PROMPT = """You are a CSC (Common Service Centre) operator evaluating welfare scheme applications in rural India.
Your decisions directly affect whether vulnerable citizens receive government support.

=== AVAILABLE ACTIONS — respond ONLY with valid JSON ===
{"action_type": "ask_question",    "value": "<field_name>"}
  Use ONLY for: age, income, occupation, has_aadhaar

{"action_type": "request_document","value": "<document_name>"}
  Use to verify official records when self-reported data seems unreliable.
  Key documents: "aadhaar_card" (identity/age), "pan_card" (employment/income)

{"action_type": "approve_scheme",  "value": "<scheme_name>"}
  Valid schemes: PMKVY, MGNREGS, PMAY

{"action_type": "reject_applicant","value": "<reason>"}

{"action_type": "escalate",        "value": ""}
  Use ONLY when data integrity is genuinely compromised and a human supervisor
  must review before any decision can be made.

=== SCHEME ELIGIBILITY — ALL conditions must be simultaneously true ===
PMKVY (skill training, Rs 8,000 stipend):
  age 18 to 35 inclusive
  occupation: mason OR carpenter
  income: 9999 or below (income of 10000 FAILS)
  Aadhaar: not required

MGNREGS (100 days employment guarantee):
  age 18 to 60 inclusive
  occupation: farm_labourer ONLY
  has_aadhaar: must be True

PMAY (housing grant Rs 1.2 lakh):
  age 21 to 55 inclusive
  occupation: any
  income: 5999 or below (income of 6000 FAILS)
  has_aadhaar: must be True

=== PRIORITY RULE ===
When multiple schemes apply to the same applicant:
  PMAY > PMKVY (PMAY provides Rs 1.2 lakh vs PMKVY's Rs 8,000 — always choose the higher benefit)
  Always approve the scheme with the greatest financial benefit.

=== DECISION WORKFLOW ===
Step 1 — CHECK MISSING DATA: If missing_data is not empty, ask for one field at a time.
Step 2 — VERIFY IF NEEDED: If any profile field seems inconsistent or self-reported data 
          may conflict with official records, request the relevant document BEFORE deciding.
Step 3 — APPLY RULES: Check ALL eligibility conditions with exact integer arithmetic.
          Convert income strings to integers before comparing to thresholds.
Step 4 — DECIDE: approve_scheme if eligible, reject_applicant if not.
          escalate ONLY if you discovered a genuine data integrity conflict that 
          cannot be resolved without a senior officer.

=== CRITICAL RULES ===
NOISE TRAP: known_profile may contain marital_status, bank_name, state_of_residence,
number_of_children, self_reported_age. These are IRRELEVANT. Never ask about them.
Querying noise fields costs -1.0 and reduces your final score.

INTEGER PRECISION: income thresholds are absolute integers.
  PMKVY:  income must be ≤ 9999. income=10000 DISQUALIFIES.
  PMAY:   income must be ≤ 5999. income=6000  DISQUALIFIES.

AGE VERIFICATION: When age is a critical eligibility factor and you have reason
to doubt the self-reported age, request the aadhaar_card to get the official age.
The Aadhaar age overrides any self-reported age in the profile.

DATA INTEGRITY: If official documents reveal information that directly contradicts
the applicant's stated profile — occupation, employment status, age — you cannot
make a terminal decision without senior review. Escalate.

Respond ONLY with a JSON object. No explanation. No markdown. Just JSON."""


def get_agent_action(observation: dict, history: list):
    """
    Query the LLM with current observation and conversation history.
    Extracts JSON from the response even if the model adds surrounding text.
    Falls back to escalate on complete JSON parse failure.
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
        + history[-10:]
        + [{"role": "user", "content": obs_text}]
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            max_tokens=150, temperature=0.0,
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
    """Run one complete episode and return the grader score."""
    task_name = TASK_NAMES[task]
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
    print(f"  TASK {task}/5 — {task_name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Profile : {obs.get('known_profile', {})}", flush=True)
    print(f"  Missing : {obs.get('missing_data', [])}", flush=True)
    print(f"  Notif   : {str(obs.get('notification', ''))[:140]}", flush=True)

    while step < MAX_STEPS:
        step += 1

        if obs.get("is_terminated", False):
            grader_score = (
                obs.get("grader_score")
                or obs.get("metadata", {}).get("grader_score", 0.0)
            )
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

        log_step(step=step, action=action_str, reward=reward, done=done, error=None)
        print(f"  Step {step:02d}: {action_str} -> reward={reward}, done={done}", flush=True)
        print(f"           {notification[:120]}", flush=True)

        history.append({
            "role":    "user",
            "content": f"reward={reward}, notification={notification}",
        })

        if done:
            grader_score = obs.get("grader_score") or obs.get("metadata", {}).get("grader_score", None)
            if grader_score is None:
                if reward >= 10.0: grader_score = 1.0
                elif reward >= 5.0: grader_score = 1.0
                elif reward >= 3.0: grader_score = 0.5
                else:               grader_score = 0.0
            break

        time.sleep(0.3)

    grader_score = float(grader_score or 0.0)
    success      = grader_score >= 1.0

    log_end(success=success, steps=step, score=grader_score, rewards=rewards)
    print(f"\n  GRADER SCORE: {grader_score:.3f} / 1.0", flush=True)
    return grader_score


def main():
    print(f"\n{'='*60}", flush=True)
    print(f"  SCHEME ENV — OPTION A EVALUATION", flush=True)
    print(f"  Model : {MODEL_NAME}", flush=True)
    print(f"  Env   : {ENV_URL}", flush=True)
    print(f"{'='*60}", flush=True)

    scores = {}
    for task in [1, 2, 3, 4, 5]:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            print(f"\n  [ERROR] Task {task} failed: {e}", flush=True)
            scores[task] = 0.0
        time.sleep(1)

    avg = sum(scores.values()) / len(scores)

    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL GRADER SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Task 1 (Scheme Discovery)    : {scores[1]:.3f} / 1.0", flush=True)
    print(f"  Task 2 (Missing Data)        : {scores[2]:.3f} / 1.0", flush=True)
    print(f"  Task 3 (Boundary Fraud)      : {scores[3]:.3f} / 1.0", flush=True)
    print(f"  Task 4 (Escalation Dilemma)  : {scores[4]:.3f} / 1.0", flush=True)
    print(f"  Task 5 (Document Conflict)   : {scores[5]:.3f} / 1.0", flush=True)
    print(f"  Average                      : {avg:.3f} / 1.0", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()