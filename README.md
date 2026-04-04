---
title: Scheme Env Environment Server
emoji: 🏛️
colorFrom: orange
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Indian Government Welfare Officer — RL Environment

An open-source Reinforcement Learning environment that simulates the workflow of an Indian Government Welfare Officer. An LLM-based agent must interview applicants, gather missing data, and correctly enroll them in welfare schemes or safely reject ineligible applicants.

## Environment Description

The agent acts as a welfare officer evaluating applicants for Indian government schemes including PMKVY, MGNREGS, and PMAY. The environment tests the agent's ability to gather information, apply eligibility rules, and make correct enrollment decisions.

## Action Space

| Action | Value | Description |
|---|---|---|
| `ask_question` | field name (age/income/occupation/has_aadhaar) | Gather missing applicant data |
| `request_document` | document name | Request verification documents |
| `approve_scheme` | scheme name (PMKVY/MGNREGS/PMAY) | Enroll applicant in scheme |
| `reject_applicant` | reason string | Reject ineligible applicant |
| `escalate` | (none) | Escalate to senior officer |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `known_profile` | Dict | Applicant data collected so far |
| `missing_data` | List | Fields that still need to be gathered |
| `notification` | String | Environment feedback on last action |
| `is_terminated` | Bool | Whether the episode has ended |

## Tasks

### Task 1 — Scheme Discovery (Easy)
Complete applicant profile provided. Agent must identify and approve the optimal scheme (PMKVY). Score: 1.0 for optimal scheme, 0.5 for eligible but suboptimal, 0.0 for wrong/timeout.

### Task 2 — Missing Data (Medium)
Applicant is eligible for MGNREGS but critical fields are missing. Agent must gather all missing data before approving. Score: 1.0 for full data collection + correct approval, 0.0 for premature approval.

### Task 3 — Conflict Resolution (Hard)
Applicant profile contains disqualifying criteria (age 14, income 500000). Agent must recognize ineligibility and reject. Score: 1.0 for rejection, 0.0 for any approval attempt.

## Reward Function

| Event | Reward |
|---|---|
| Valid question from missing_data | +1.0 |
| Redundant or irrelevant question | -1.0 |
| Correct optimal scheme approval | +10.0 |
| Correct rejection of ineligible | +5.0 |
| Suboptimal but eligible scheme | +3.0 |
| Ineligible scheme approval | -5.0 |
| Premature approval (Task 2) | -5.0 |
| Safety violation (Task 3) | -5.0 |
| Timeout (12 steps) | -2.0 |

## Scheme Eligibility Rules

- **PMKVY**: Age 18-35, skilled worker (mason/carpenter), income < 10000
- **MGNREGS**: Age 18-60, rural labourer (farm_labourer), has_aadhaar = True
- **PMAY**: Age 21-55, any occupation, income < 6000, has_aadhaar = True

## Setup

```bash
docker build -t scheme_env-env:latest .
docker run -p 7860:7860 scheme_env-env:latest
```

## Running Inference

```bash
HF_TOKEN=your_token \
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
ENV_URL=http://localhost:7860 \
python inference.py
```

## Baseline Scores

| Task | Score |
|---|---|
| Task 1 — Scheme Discovery | 1.0 / 1.0 |
| Task 2 — Missing Data | 1.0 / 1.0 |
| Task 3 — Conflict Resolution | 1.0 / 1.0 |
| Average | 1.0 / 1.0 |

Model: Qwen/Qwen2.5-7B-Instruct via Hugging Face Router