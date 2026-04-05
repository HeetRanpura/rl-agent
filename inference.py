import os
import sys
import json
import time
import urllib.request
import re

sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI

# =========================================================
# ENVIRONMENT CONFIGURATION
# All credentials read from environment — never hardcoded.
# =========================================================
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
# SYSTEM PROMPT
#
# This is what a real CSC operator knows on day one of the job.
# It contains:
#   - Scheme eligibility rules (job training, not hints)
#   - Available actions and their JSON format
#   - Response format requirement with <think> block
#
# It does NOT contain:
#   - Which fields are traps (agent learns via -1.0 penalty)
#   - Which field to ask next (agent decides from missing_data)
#   - Task-specific guidance (agent reasons from observation)
#   - Any DECISION recommendation (agent must conclude independently)
#   - Any hint about what profile patterns mean
#
# This design ensures the reward signal is the only teacher.
# The agent improves by experiencing consequences, not by reading
# instructions that solve the task for it.
# =========================================================

SYSTEM_PROMPT = """You are a CSC (Common Service Centre) operator evaluating welfare scheme applications in rural India.
Your decisions directly affect whether vulnerable citizens receive government support.
You must reason carefully and act only on verified information.

=== MANDATORY RESPONSE FORMAT ===
You MUST respond in this exact format for every action:
<think>
[Your step-by-step reasoning. Check what data you have, what is missing,
what the eligibility rules say, and what action is appropriate.]
</think>
{"action_type": "...", "value": "..."}

Never output JSON without the <think> block. Never skip reasoning.

=== AVAILABLE ACTIONS ===

{"action_type": "ask_question", "value": "<field_name>"}
  Request information from the applicant.
  Valid field names: age, income, occupation, has_aadhaar

{"action_type": "request_document", "value": "<document_name>"}
  Request an official document for verification.
  aadhaar_card: verifies identity and official age
  pan_card: verifies employment history and income source

{"action_type": "approve_scheme", "value": "<scheme_name>"}
  Enroll the applicant in a welfare scheme.
  Valid scheme names: PMKVY, MGNREGS, PMAY

{"action_type": "reject_applicant", "value": "<reason>"}
  Reject the application with a clear reason.

{"action_type": "escalate", "value": ""}
  Hand off to a senior officer for manual review.

=== SCHEME ELIGIBILITY RULES ===
All conditions must be simultaneously true. Use strict integer arithmetic.

PMKVY — Pradhan Mantri Kaushal Vikas Yojana (Rs 8,000 skill training stipend):
  Age: 18 to 35 inclusive
  Occupation: mason OR carpenter
  Income: 9,999 or below (income of 10,000 does NOT qualify)
  Aadhaar: not required

MGNREGS — Mahatma Gandhi National Rural Employment Guarantee Scheme (100 days employment):
  Age: 18 to 60 inclusive
  Occupation: farm_labourer ONLY
  Aadhaar: required (has_aadhaar must be True)

PMAY — Pradhan Mantri Awaas Yojana (Rs 1.2 lakh housing grant):
  Age: 21 to 55 inclusive
  Occupation: any
  Income: 5,999 or below (income of 6,000 does NOT qualify)
  Aadhaar: required (has_aadhaar must be True)

When multiple schemes apply, choose the one with the highest financial benefit.
Benefit values: PMAY (Rs 1.2 lakh) > MGNREGS (100 days wages) > PMKVY (Rs 8,000)

=== DECISION PRINCIPLES ===

1. Do not make any terminal decision (approve, reject, escalate) while
   missing_data is not empty. Collect all required information first.

2. Apply eligibility rules with exact integer comparisons.
   income=9999 qualifies for PMKVY. income=10000 does not.

3. If official documents reveal information that contradicts the stated
   profile, the contradiction must be reviewed by a senior officer.

4. If no scheme criteria are met, reject the applicant with a clear reason.

5. Escalation is reserved for genuine data integrity conflicts discovered
   through document verification — not for uncertainty or eligibility failures."""


def get_agent_action(observation: dict, history: list):
    """
    Query the LLM with the current observation.

    The user prompt contains only raw observation data:
    - known_profile: what the agent has gathered so far
    - missing_data: fields still needed
    - notification: environment feedback on last action
    - is_terminated: episode state

    No eligibility hints, no DECISION lines, no field directives.
    The agent must reason from the observation and scheme rules alone.
    This is the correct design for an RL training environment —
    the reward signal teaches, not the prompt.
    """
    profile      = observation.get('known_profile', {})
    missing      = observation.get('missing_data', [])
    notification = observation.get('notification', '')
    terminated   = observation.get('is_terminated', False)

    obs_text = (
        f"Current application state:\n"
        f"known_profile: {profile}\n"
        f"missing_data: {missing}\n"
        f"notification: {notification}\n"
        f"is_terminated: {terminated}\n\n"
        f"Reason through this carefully in <think> tags, then output your next action as JSON."
    )

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-10:]
        + [{"role": "user", "content": obs_text}]
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            # 500 tokens: allows full <think> reasoning (~300) plus JSON (~30)
            max_tokens=500, temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"action_type": "escalate", "value": ""}, f"API_ERROR: {e}"

    # Extract and log <think> reasoning block.
    # Logged separately so reasoning is visible even if JSON extraction
    # modifies the raw string. This enables future reward shaping on
    # reasoning quality, not just terminal action correctness.
    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    thinking    = think_match.group(1).strip() if think_match else ""
    if thinking:
        print(f"  Reasoning: {thinking[:500]}", flush=True)

    # Extract JSON — handles markdown fences, think tags, prose wrapping.
    # The regex finds the first complete JSON object in the response.
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        # Fallback returns ask_question, not escalate.
        # Escalate fallback gives 0.75 on Task 4 by luck, masking JSON
        # formatting failures as if they were correct reasoning decisions.
        return {"action_type": "ask_question", "value": "occupation"}, raw


def run_episode(task: int) -> float:
    """
    Run one complete episode for the given task and return the grader score.

    Episode flow:
    1. Reset environment to get initial observation
    2. Loop: get agent action → step environment → observe reward
    3. Continue until episode terminates or MAX_STEPS reached
    4. Return grader_score from terminal observation
    """
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