import os
import sys
import json
import time
import urllib.request
import re

sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI

# ── Environment variables (all required by hackathon spec) ──────────────────
API_BASE_URL   = os.getenv("API_BASE_URL",   "https://api.openai.com/v1")
MODEL_NAME     = os.getenv("MODEL_NAME",     "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENV_URL        = os.getenv("ENV_URL",        "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

BENCHMARK  = "scheme_env"
MAX_STEPS  = 20

TASK_NAMES = {
    1: "scheme_discovery",
    2: "missing_data",
    3: "conflict_resolution",
}

# ── HTTP helpers ────────────────────────────────────────────────────────────
def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(
        ENV_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def env_reset(task: int) -> dict:
    return _post("/reset", {"seed": task})

def env_step(action_type: str, value: str) -> dict:
    return _post("/step", {"action": {"action_type": action_type, "value": value}})

# ── Structured log functions (MUST match spec exactly) ──────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── System prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a CSC (Common Service Centre) operator in rural India.
Your job is to help villagers enroll in the correct government welfare scheme.

=== WORKFLOW (follow this phase order strictly) ===
PHASE 1 — GATHER: If missing_data is not empty, ask for ONE field at a time using ask_question.
PHASE 2 — IDENTIFY: Once missing_data is empty, determine the best scheme using eligibility rules.
PHASE 3 — DECIDE: approve_scheme with the optimal scheme, or reject_applicant if ineligible.
SPECIAL: If notification contains SYSTEM ALERT about compromised/contradictory data → escalate immediately.

=== AVAILABLE ACTIONS (respond ONLY with valid JSON) ===
{"action_type": "ask_question",    "value": "<field_name>"}
  Valid field_names ONLY: age, income, occupation, has_aadhaar
{"action_type": "request_document","value": "<doc_name>"}
{"action_type": "approve_scheme",  "value": "<scheme_name>"}
  Valid schemes: PMKVY, MGNREGS, PMAY
{"action_type": "reject_applicant","value": "<reason>"}
{"action_type": "escalate",        "value": ""}

=== SCHEME ELIGIBILITY (ALL conditions must be simultaneously true) ===
PMKVY   : age 18-35, occupation mason OR carpenter, income STRICTLY < 10000
MGNREGS : age 18-60, occupation farm_labourer, has_aadhaar = True
PMAY    : age 21-55, any occupation, income STRICTLY < 6000, has_aadhaar = True

REJECT if: no scheme criteria fully satisfied.

=== CRITICAL RULES ===
- NOISE: known_profile may contain marital_status, bank_name, state_of_residence,
  number_of_children. These are TRAPS. NEVER use ask_question for these fields.
  Doing so gives -1.0 reward.
- INCOME PRECISION: income=10000 does NOT qualify for PMKVY (requires STRICTLY < 10000).
  income=6000 does NOT qualify for PMAY (requires STRICTLY < 6000).
- REDUNDANCY: Never ask for a field already in known_profile. Costs -1.0 reward.
- MISSING DATA: If missing_data is not empty, you MUST ask before deciding.
  Approving while missing_data is not empty costs -5.0 reward and ends episode.
- ESCALATION: Only for SYSTEM ALERT contradictions. Using escalate on a normal
  case costs -2.0 reward.

Respond ONLY with a JSON object. No explanation. No markdown. Just JSON."""

# ── Agent action ────────────────────────────────────────────────────────────
def get_agent_action(observation: dict, history: list):
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
            model=MODEL_NAME,
            messages=messages,
            max_tokens=100,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"action_type": "escalate", "value": ""}, f"API_ERROR: {e}"

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return {"action_type": "escalate", "value": ""}, raw

# ── Episode runner ──────────────────────────────────────────────────────────
def run_episode(task: int) -> float:
    task_name = TASK_NAMES[task]

    # ── [START] log — required by spec ──
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

    # Console header (informational only)
    print(f"\n{'='*60}", flush=True)
    print(f"  TASK {task}/3 — {task_name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Profile : {obs.get('known_profile', {})}", flush=True)
    print(f"  Missing : {obs.get('missing_data', [])}", flush=True)
    print(f"  Notif   : {str(obs.get('notification', ''))[:120]}", flush=True)

    while step < MAX_STEPS:
        step += 1

        if obs.get("is_terminated", False):
            grader_score = obs.get("metadata", {}).get("grader_score", 0.0)
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
            print(f"  [ERROR] env_step failed: {e}", flush=True)
            continue

        obs          = step_result.get("observation", step_result)
        reward       = step_result.get("reward", 0.0)
        done         = step_result.get("done", False)
        notification = str(obs.get("notification", ""))

        rewards.append(reward)

        action_str = f"{action_type}({value!r})"

        # ── [STEP] log — required by spec ──
        log_step(step=step, action=action_str, reward=reward,
                 done=done, error=None)

        # Console detail (informational)
        print(f"  Step {step:02d}: {action_str} -> reward={reward}, done={done}",
              flush=True)
        print(f"           {notification[:100]}", flush=True)

        history.append({
            "role":    "user",
            "content": f"reward={reward}, notification={notification}",
        })

        if done:
            grader_score = obs.get("metadata", {}).get("grader_score", None)
            if grader_score is None:
                if reward >= 10.0:   grader_score = 1.0
                elif reward >= 5.0:  grader_score = 1.0
                elif reward >= 3.0:  grader_score = 0.5
                else:                grader_score = 0.0
            break

        time.sleep(0.3)

    grader_score = float(grader_score or 0.0)
    success      = grader_score >= 1.0

    # ── [END] log — required by spec ──
    log_end(success=success, steps=step, score=grader_score, rewards=rewards)

    print(f"\n  GRADER SCORE: {grader_score:.3f} / 1.0", flush=True)
    return grader_score

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}", flush=True)
    print(f"  SCHEME ENV — INFERENCE EVALUATION", flush=True)
    print(f"  Model : {MODEL_NAME}", flush=True)
    print(f"  Env   : {ENV_URL}", flush=True)
    print(f"{'='*60}", flush=True)

    scores = {}
    for task in [1, 2, 3]:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            print(f"\n  [ERROR] Task {task} failed: {e}", flush=True)
            scores[task] = 0.0
        time.sleep(1)

    avg = sum(scores.values()) / len(scores)

    # ── Final scores — exact format required by spec ──
    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL GRADER SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Task 1 (Scheme Discovery)    : {scores[1]:.1f} / 1.0", flush=True)
    print(f"  Task 2 (Missing Data)        : {scores[2]:.1f} / 1.0", flush=True)
    print(f"  Task 3 (Conflict Resolution) : {scores[3]:.1f} / 1.0", flush=True)
    print(f"  Average                      : {avg:.2f} / 1.0", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()