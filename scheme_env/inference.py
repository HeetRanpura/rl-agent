import os
import sys
import json
import time
import urllib.request
import re

sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(ENV_URL + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def env_reset(task: int) -> dict: return _post("/reset", {"seed": task})
def env_step(action_type: str, value: str) -> dict: return _post("/step", {"action": {"action_type": action_type, "value": value}})

SYSTEM_PROMPT = """You are an Indian Government Welfare Officer AI agent. Evaluate applicants and enroll them in the correct welfare scheme.

Available actions — respond ONLY with valid JSON, nothing else:
1. {"action_type": "ask_question", "value": "<field_name>"} (Valid fields: age, income, occupation, has_aadhaar)
2. {"action_type": "request_document", "value": "<doc_name>"}
3. {"action_type": "approve_scheme", "value": "<scheme_name>"} (Options: PMKVY, MGNREGS, PMAY)
4. {"action_type": "reject_applicant", "value": "<reason>"}
5. {"action_type": "escalate", "value": ""}

Scheme eligibility rules (ALL conditions must be met):
- PMKVY: age 18-35, occupation mason OR carpenter, income STRICTLY < 10000
- MGNREGS: age 18-60, occupation farm_labourer, has_aadhaar = True
- PMAY: age 21-55, any occupation, income STRICTLY < 6000, has_aadhaar = True

IMPORTANT RULES:
- NOISE FIELDS: The known_profile may contain irrelevant fields (e.g., marital_status, bank_name). NEVER ask about these.
- INCOME THRESHOLDS: Thresholds are strict integers. income=10000 does NOT qualify for PMKVY.
- MISSING DATA: Pick ONE missing field and ask_question.
- ESCALATION: If the notification contains a SYSTEM ALERT about compromised data, you MUST use escalate.

Respond ONLY with JSON. No explanation."""

def get_agent_action(observation: dict, history: list):
    obs_text = f"known_profile: {observation.get('known_profile', {})}\nmissing_data: {observation.get('missing_data', [])}\nnotification: {observation.get('notification', '')}\nWhat is your next action? Respond with JSON only."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-8:] + [{"role": "user", "content": obs_text}]
    response = client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=100, temperature=0.0)
    raw = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match: raw = match.group(0)
    try: return json.loads(raw), raw
    except json.JSONDecodeError: return {"action_type": "escalate", "value": ""}, raw

def run_episode(task: int) -> float:
    result = env_reset(task)
    obs, step, grader_score, history = result.get("observation", result), 0, 0.0, []
    
    print(f"\n{'='*60}\n  TASK {task}/4\n{'='*60}")
    print(f"  Profile : {obs.get('known_profile', {})}\n  Missing : {obs.get('missing_data', [])}")

    while step < 12:
        step += 1
        if obs.get("is_terminated", False):
            grader_score = obs.get("metadata", {}).get("grader_score", 0.0)
            break

        action, raw_response = get_agent_action(obs, history)
        action_type, value = action.get("action_type", "escalate"), action.get("value", "")
        history.append({"role": "assistant", "content": raw_response})

        step_result = env_step(action_type, value)
        obs, reward, done = step_result.get("observation", step_result), step_result.get("reward", 0.0), step_result.get("done", False)
        
        print(f"  Step {step:02d}: {action_type}({value!r}) -> reward={reward}, done={done}")
        history.append({"role": "user", "content": f"reward={reward}, notification={obs.get('notification', '')}"})

        if done:
            grader_score = obs.get("metadata", {}).get("grader_score", None)
            if grader_score is None:
                grader_score = 1.0 if reward in (10.0, 5.0) else (0.5 if reward == 3.0 else 0.0)
            break
        time.sleep(0.3)
    
    print(f"  GRADER SCORE: {grader_score:.1f} / 1.0")
    return grader_score

def main():
    print(f"\n{'='*60}\n  SCHEME ENV EVALUATION\n  Model: {MODEL_NAME}\n{'='*60}")
    scores = {task: run_episode(task) for task in [1, 2, 3, 4]}
    avg = sum(scores.values()) / 4
    print(f"\n{'='*60}\n  FINAL SCORES: Avg {avg:.2f}/1.0\n{'='*60}")

if __name__ == "__main__": main()