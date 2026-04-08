"""
Manual smoke test — verifies each task resets correctly and 
that basic correct actions score as expected.
Run with server already running on localhost:7860.
"""
import json
import urllib.request

ENV_URL = "http://localhost:7860"

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        ENV_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())

def reset(task):
    return post("/reset", {"seed": task})

def step(action_type, value=""):
    return post("/step", {"action": {"action_type": action_type, "value": value}})

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition

print("\n" + "="*60)
print("SMOKE TEST — scheme_env")
print("="*60)

all_pass = True

# ── TASK 1: approve optimal scheme immediately ─────────────────
print("\nTask 1 — Scheme Discovery")
r = reset(1)
obs = r.get("observation", r)
profile = obs.get("known_profile", {})
missing = obs.get("missing_data", [])
print(f"  Profile at reset: {profile}")
print(f"  Missing: {missing}")

# Ask for missing fields first
for field in list(missing):
    r2 = step("ask_question", field)
    obs = r2.get("observation", r2)

# Now approve — pick PMKVY as safe bet for Task 1
r3 = step("approve_scheme", "PMKVY")
obs = r3.get("observation", r3)
reward = r3.get("reward", 0)
done = r3.get("done", False)
score = obs.get("grader_score")
notif = obs.get("notification", "")
all_pass &= check("Task 1 terminates on approve", done, f"done={done}")
all_pass &= check("Task 1 reward positive or suboptimal", reward >= 3.0, f"reward={reward}")
all_pass &= check("Task 1 grader_score set", score is not None, f"score={score}")
print(f"  Notification: {notif[:100]}")

# ── TASK 2: premature approve should fail ──────────────────────
print("\nTask 2 — Missing Data (premature approval)")
r = reset(2)
obs = r.get("observation", r)
missing = obs.get("missing_data", [])
print(f"  Missing at reset: {missing}")
r2 = step("approve_scheme", "MGNREGS")
obs2 = r2.get("observation", r2)
reward2 = r2.get("reward", 0)
done2 = r2.get("done", False)
score2 = obs2.get("grader_score")
all_pass &= check("Task 2 premature approve terminates", done2, f"done={done2}")
all_pass &= check("Task 2 premature approve scores 0.0", score2 == 0.0, f"score={score2}")
all_pass &= check("Task 2 premature approve reward negative", reward2 < 0, f"reward={reward2}")

# ── TASK 2: correct flow ───────────────────────────────────────
print("\nTask 2 — Missing Data (correct flow)")
r = reset(2)
obs = r.get("observation", r)
missing = list(obs.get("missing_data", []))
for field in missing:
    r2 = step("ask_question", field)
    obs = r2.get("observation", r2)
r3 = step("approve_scheme", "MGNREGS")
obs3 = r3.get("observation", r3)
reward3 = r3.get("reward", 0)
done3 = r3.get("done", False)
score3 = obs3.get("grader_score")
all_pass &= check("Task 2 correct flow terminates", done3, f"done={done3}")
all_pass &= check("Task 2 correct flow reward positive", reward3 >= 3.0, f"reward={reward3}")
all_pass &= check("Task 2 correct flow score > 0", score3 and score3 > 0, f"score={score3}")

# ── TASK 3: reject without collecting income should fail ───────
print("\nTask 3 — Boundary Fraud (reject before asking income)")
r = reset(3)
obs = r.get("observation", r)
profile = obs.get("known_profile", {})
missing = obs.get("missing_data", [])
print(f"  Profile at reset: {profile}")
print(f"  Missing: {missing}")
all_pass &= check("Task 3 income hidden at reset", "income" not in profile, 
                  f"income in profile={list(profile.keys())}")
r2 = step("reject_applicant", "INCOME_TOO_HIGH")
obs2 = r2.get("observation", r2)
reward2 = r2.get("reward", 0)
done2 = r2.get("done", False)
score2 = obs2.get("grader_score")
all_pass &= check("Task 3 premature reject scores 0.0", score2 == 0.0, f"score={score2}")
all_pass &= check("Task 3 premature reject reward negative", reward2 < 0, f"reward={reward2}")

# ── TASK 3: correct flow ───────────────────────────────────────
print("\nTask 3 — Boundary Fraud (correct flow)")
r = reset(3)
obs = r.get("observation", r)
missing = list(obs.get("missing_data", []))
for field in missing:
    step("ask_question", field)
r2 = step("reject_applicant", "INCOME_TOO_HIGH")
obs2 = r2.get("observation", r2)
reward2 = r2.get("reward", 0)
done2 = r2.get("done", False)
score2 = obs2.get("grader_score")
all_pass &= check("Task 3 correct reject terminates", done2, f"done={done2}")
all_pass &= check("Task 3 correct reject reward positive", reward2 >= 5.0, f"reward={reward2}")
all_pass &= check("Task 3 correct reject score > 0.30", score2 and score2 >= 0.30, f"score={score2}")

# ── TASK 4: escalate WITHOUT pan card should soft-block ────────
print("\nTask 4 — Escalation (escalate before PAN — should soft-block)")
r = reset(4)
obs = r.get("observation", r)
r2 = step("escalate", "MANUAL_REVIEW_REQUIRED")
obs2 = r2.get("observation", r2)
done2 = r2.get("done", False)
reward2 = r2.get("reward", 0)
all_pass &= check("Task 4 escalate before PAN is non-terminal", not done2, f"done={done2}")
all_pass &= check("Task 4 escalate before PAN reward negative", reward2 < 0, f"reward={reward2}")

# ── TASK 4: correct flow ───────────────────────────────────────
print("\nTask 4 — Escalation (correct flow)")
r = reset(4)
r2 = step("request_document", "pan_card")
obs2 = r2.get("observation", r2)
notif2 = obs2.get("notification", "")
all_pass &= check("Task 4 PAN reveals contradiction", "contradiction" in notif2.lower() 
                  or "pension" in notif2.lower(), f"notif={notif2[:80]}")
r3 = step("escalate", "MANUAL_REVIEW_REQUIRED")
obs3 = r3.get("observation", r3)
reward3 = r3.get("reward", 0)
done3 = r3.get("done", False)
score3 = obs3.get("grader_score")
all_pass &= check("Task 4 correct escalate terminates", done3, f"done={done3}")
all_pass &= check("Task 4 correct escalate reward=10", reward3 == 10.0, f"reward={reward3}")
all_pass &= check("Task 4 correct escalate score > 0.80", score3 and score3 >= 0.80, f"score={score3}")

# ── TASK 5: correct flow ───────────────────────────────────────
print("\nTask 5 — Document Conflict (correct flow)")
r = reset(5)
obs = r.get("observation", r)
profile = obs.get("known_profile", {})
print(f"  Self-reported age in profile: {profile.get('age', 'N/A')}")
r2 = step("request_document", "aadhaar_card")
obs2 = r2.get("observation", r2)
notif2 = obs2.get("notification", "")
all_pass &= check("Task 5 Aadhaar reveals age conflict", 
                  "official age" in notif2.lower() or "differs" in notif2.lower(),
                  f"notif={notif2[:80]}")
r3 = step("reject_applicant", "AGE_EXCEEDED")
obs3 = r3.get("observation", r3)
reward3 = r3.get("reward", 0)
done3 = r3.get("done", False)
score3 = obs3.get("grader_score")
all_pass &= check("Task 5 correct reject terminates", done3, f"done={done3}")
all_pass &= check("Task 5 correct reject reward=5", reward3 == 5.0, f"reward={reward3}")
all_pass &= check("Task 5 correct reject score > 0.80", score3 and score3 >= 0.80, f"score={score3}")

# ── WRONG ESCALATION TERMINAL CHECK ───────────────────────────
print("\nWrong escalation on Task 1 (should now be terminal)")
r = reset(1)
obs = r.get("observation", r)
missing = list(obs.get("missing_data", []))
for field in missing:
    step("ask_question", field)
r2 = step("escalate", "MANUAL_REVIEW_REQUIRED")
obs2 = r2.get("observation", r2)
done2 = r2.get("done", False)
reward2 = r2.get("reward", 0)
all_pass &= check("Wrong escalation Task 1 is terminal", done2, f"done={done2}")
all_pass &= check("Wrong escalation reward=-2.0", reward2 == -2.0, f"reward={reward2}")

# ── SUMMARY ───────────────────────────────────────────────────
print("\n" + "="*60)
if all_pass:
    print("ALL TESTS PASSED — environment logic is correct")
else:
    print("SOME TESTS FAILED — review output above before running inference")
print("="*60 + "\n")