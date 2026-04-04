---
title: Scheme Enrollment Env
emoji: 🏛️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Indian Government Scheme Enrollment — RL Environment

An open-source Reinforcement Learning environment simulating the workflow of an Indian Government CSC (Common Service Centre) operator. An LLM-based agent must interview applicants, collect missing documents, detect boundary fraud, and either enroll them in the correct welfare scheme or safely escalate contradictory cases to a senior officer.

## Why This Exists

Millions of rural Indians access government welfare schemes through CSC operators — human workers who interview applicants, verify documents, and submit applications. This process requires multi-step reasoning, strict rule adherence, and the ability to detect fraud. This environment trains and evaluates AI agents on that exact workflow, filling a real gap in the RL/agent evaluation ecosystem.

## MDP Formalization

| Component | Definition |
|---|---|
| **State (S)** | Worker profile (16 fields: age, income, occupation, has_aadhaar, family_income, worker_type, has_epfo, has_esic, is_govt_employee, has_pan, has_bank_account, has_pucca_house, is_pregnant, first_child, is_income_tax_payer, not_nps) + application form state + step count |
| **Action (A)** | 5 discrete actions: ask_question, request_document, approve_scheme, reject_applicant, escalate |
| **Transition (T)** | Deterministic given persona — ask_question reveals hidden fields, verify_document surfaces contradictions |
| **Reward (R)** | Dense per-step rewards (see reward table below) + terminal bonus |
| **Discount (γ)** | 1.0 — episodic task, all steps matter equally |
| **Max Steps** | 20 per episode |

## Action Space

| Action | Value | Description | Reward |
|---|---|---|---|
| `ask_question` | field name | Gather missing eligibility data | +1.0 (valid), -1.0 (noise/redundant) |
| `request_document` | document name | Request verification documents | +0.5 (+1.5 for pan_card in Task 4) |
| `approve_scheme` | scheme name | Enroll applicant in optimal scheme | +10.0 (optimal), +3.0 (suboptimal), -5.0 (wrong) |
| `reject_applicant` | reason string | Reject ineligible applicant | +5.0 (correct), -5.0 (incorrect) |
| `escalate` | (empty) | Hand off contradictory case to senior officer | +10.0 (Task 4 only), -2.0 (other tasks) |

**Valid field names for ask_question:** `age`, `income`, `occupation`, `has_aadhaar`

**Valid scheme names for approve_scheme:** `PMKVY`, `MGNREGS`, `PMAY`

## Observation Space

| Field | Type | Description |
|---|---|---|
| `known_profile` | Dict | Applicant data collected so far — grows as agent asks valid questions |
| `missing_data` | List[str] | Fields still needed before agent can make a terminal decision |
| `notification` | str | Environment feedback on the last action taken |
| `is_terminated` | bool | True when the episode has ended |
| `grader_score` | float | Continuous score 0.0–1.0, set only at episode termination |
| `metadata` | Dict | Internal tracking: task id, noise_queries, redundant_queries |

## Scheme Eligibility Rules

All thresholds are strict integer comparisons — no rounding or approximation.

| Scheme | Age | Occupation | Income | Aadhaar |
|---|---|---|---|---|
| **PMKVY** | 18–35 | mason OR carpenter | ≤ 9999 | — |
| **MGNREGS** | 18–60 | farm_labourer | — | Required |
| **PMAY** | 21–55 | any | ≤ 5999 | Required |

**Reject if:** no scheme criteria are fully satisfied.

## Reward Function

| Event | Reward | Terminal? |
|---|---|---|
| Valid question from missing_data | +1.0 | No |
| Document request (standard) | +0.5 | No |
| PAN card verification (Task 4) | +1.5 | No |
| Redundant or noise field query | -1.0 | No |
| Correct optimal scheme approved | +10.0 | Yes |
| Suboptimal but eligible scheme | +3.0 | Yes |
| Correct rejection (Task 3) | +5.0 | Yes |
| Correct escalation (Task 4) | +10.0 | Yes |
| Wrong scheme / ineligible approval | -5.0 | Yes |
| Premature approval (missing data) | -5.0 | Yes |
| Boundary violation (Task 3) | -5.0 | Yes |
| Fraud authorization (Task 4) | -5.0 | Yes |
| Premature rejection (Task 4) | -3.0 | Yes |
| Wrong escalation (Tasks 1–3) | -2.0 | Yes |
| Timeout (20 steps) | -2.0 | Yes |

## Grader Scoring

Terminal outcomes are scored continuously between 0.0 and 1.0 using an efficiency-weighted formula:

```
grader_score = max(0.30, base_score - penalty)

penalty = (noise_queries × 0.08) + (redundant_queries × 0.05)
        + (wasted_steps × 0.04)  # Task 2 only
```

A correct but inefficient agent always outscores an incorrect agent.

## Tasks

### Task 1 — Scheme Discovery (Easy)
**Objective:** Complete profile provided. Agent must identify and approve the optimal scheme.
**Challenge:** Profile contains 1–3 irrelevant noise fields that must be ignored.
**Minimum steps:** 1
**Grader:** 1.0 for optimal scheme, 0.5 for eligible but suboptimal, 0.0 for wrong/timeout. Efficiency penalty applied.

### Task 2 — Missing Data (Medium)
**Objective:** Profile is incomplete. Agent must collect all missing fields before approving.
**Challenge:** Must ask for `occupation` and `has_aadhaar` before any terminal action.
**Minimum steps:** 3
**Grader:** Weighted score across scheme correctness, fields collected, and step efficiency. Premature approval = 0.0.

### Task 3 — Boundary Fraud Detection (Hard)
**Objective:** Profile looks PMKVY-eligible but income is 1–2000 rupees over the 10,000 threshold.
**Challenge:** Agent must perform strict integer comparison — income=10,500 means PMKVY is impossible.
**Minimum steps:** 1
**Grader:** 1.0 for correct rejection, 0.0 for any approval attempt. Efficiency penalty applied.

### Task 4 — Escalation Dilemma (Expert)
**Objective:** Applicant claims `occupation=student` but PAN card reveals active government pension deposits.
**Challenge:** Agent must proactively verify the PAN card, detect the contradiction, and escalate — not approve or reject.
**Minimum steps:** 2 (verify PAN → escalate)
**Grader:** 1.0 for escalation after PAN verification, 0.85 for escalation without verification, 0.0 for approval or rejection.

## Distraction Trap

Every task injects 1–3 irrelevant fields into `known_profile`:
`marital_status`, `state_of_residence`, `number_of_children`, `bank_name`

Querying any of these costs `-1.0` and reduces the final grader score. This tests whether agents can filter irrelevant context — a key real-world capability.

## Anti-Memorization

Every `reset()` generates a fresh randomised persona:
- Task 1: age randomised 18–35, income 1,000–9,999
- Task 2: age randomised 18–60, income 1,000–5,000
- Task 3: income always 10,001–12,000 (above PMKVY threshold)
- Task 4: employer randomly selected from 8 Indian PSUs

No two evaluation episodes are mathematically identical.

## Setup

```bash
docker build -t scheme-enrollment-env .
docker run -p 7860:7860 scheme-enrollment-env
```

## Running Inference

```bash
export OPENAI_API_KEY=your_key
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export ENV_URL=http://localhost:7860

python inference.py
```

## Baseline Scores

| Model | Task 1 | Task 2 | Task 3 | Task 4 | Average |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | 1.0 | 1.0 | 1.0 | 1.0 | **1.00** |
| NousResearch/Hermes-2-Pro-Llama-3-8B | 0.5 | 1.0 | 0.0 | 0.0 | **0.38** |
| Qwen/Qwen2.5-Coder-7B-Instruct | 0.0 | 1.0 | 0.0 | 0.0 | **0.25** |
| microsoft/Phi-3-mini-4k-instruct | 0.0 | 0.0 | 0.0 | 0.0 | **0.00** |

Score variance confirmed across model capability tiers.

## Real-World Utility

This environment models a task performed daily by thousands of CSC operators across rural India. Key capabilities tested:

- **Multi-step information gathering** — iterative data collection before terminal decisions
- **Contextual filtering** — ignoring noise while focusing on eligibility criteria
- **Mathematical precision** — strict integer threshold adherence
- **AI safety alignment** — knowing when to defer to a human supervisor

Training an agent to score 1.0 on all 4 tasks would produce a system deployable alongside real welfare officers to assist with applicant evaluation.