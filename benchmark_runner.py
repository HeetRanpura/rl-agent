"""
Scheme Env benchmark runner.

Runs the configured model suite sequentially, captures structured inference logs,
and produces bundled analysis artifacts for each benchmark run.
"""

import os
import asyncio
import csv
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in some runtimes
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()

# =========================================================
# CONFIGURATION
# =========================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_TOKEN = os.getenv("OPENAI_API_KEY", "") or os.getenv("HF_TOKEN", "")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

MAX_CONCURRENT  = 1   # singleton environment — must stay 1
TIMEOUT_SECONDS = 600
N_REPEATS       = 3   # episodes per task for score averaging

# ── model suite across capability tiers ──────────────────────────────────────
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-7B-Instruct",       # confirmed working earlier
    "meta-llama/Llama-3.3-70B-Instruct", # confirmed working earlier  
    "Qwen/Qwen3-Coder-30B-A3B-Instruct", # ✅ Task 1 = 1.0
    "Qwen/Qwen2.5-72B-Instruct",         # ✅ doing real reasoning
    "Qwen/QwQ-32B",                       # reasoning model
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # reasoning model
]
 
MODEL_TIERS = {
    "Qwen/Qwen2.5-7B-Instruct":                  "1-small",
    "meta-llama/Llama-3.3-70B-Instruct":         "3-large",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct":         "4-xlarge",
    "Qwen/Qwen2.5-72B-Instruct":                 "3-large",
    "Qwen/QwQ-32B":                              "4-xlarge",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":  "4-xlarge",
}

TASK_NAMES = {
    1: "Scheme Discovery",
    2: "Missing Data",
    3: "Boundary Fraud",
    4: "Escalation Dilemma",
    5: "Document Conflict",
}

TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORTS_DIR = Path("reports")
RUN_DIR     = REPORTS_DIR / f"report_{TIMESTAMP}"
RESULTS_FILE   = RUN_DIR / f"leaderboard_{TIMESTAMP}.csv"
LOG_DIR        = RUN_DIR / f"logs_{TIMESTAMP}"
MANIFEST_FILE  = RUN_DIR / f"run_manifest_{TIMESTAMP}.json"
ANALYSIS_FILE  = RUN_DIR / f"analysis_{TIMESTAMP}.json"
SUMMARY_FILE   = RUN_DIR / f"summary_{TIMESTAMP}.txt"


# =========================================================
# SERVER HEALTH CHECK
# =========================================================

def _wait_for_server(url: str, retries: int = 12, delay: int = 5) -> None:
    for i in range(retries):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=5)
            print(f"[RUNNER] Server ready at {url}", flush=True)
            return
        except Exception:
            print(f"[RUNNER] Waiting for server... ({i+1}/{retries})", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Server at {url} did not become ready after {retries} attempts")


# =========================================================
# SCORE EXTRACTION
# =========================================================

def extract_std_scores(output_text: str) -> dict:
    """Extract per-task std from STD_JSON lines written by inference.py."""
    stds = {"Task 1": 0.0, "Task 2": 0.0, "Task 3": 0.0, "Task 4": 0.0, "Task 5": 0.0}
    for line in output_text.splitlines():
        if line.startswith("STD_JSON "):
            try:
                data = json.loads(line[len("STD_JSON "):])
                stds[f"Task {data['task']}"] = float(data["std"])
            except Exception:
                pass
    return stds


def extract_scores(output_text: str) -> dict:
    scores = {
        "Task 1": 0.0, "Task 2": 0.0, "Task 3": 0.0,
        "Task 4": 0.0, "Task 5": 0.0, "Average": 0.0,
    }

    # Primary: structured SCORE_JSON lines
    for line in output_text.splitlines():
        if line.startswith("SCORE_JSON "):
            try:
                data = json.loads(line[len("SCORE_JSON "):])
                scores[f"Task {data['task']}"] = float(data["score"])
            except Exception:
                pass

    # Fallback: regex on print output
    if all(scores[f"Task {i}"] == 0.0 for i in range(1, 6)):
        task_matches = re.findall(
            r"TASK\s+([1-5])/[0-9]+.*?GRADER SCORE:\s*([0-9.]+)\s*/\s*1\.0",
            output_text, re.DOTALL,
        )
        for task_number, value in task_matches:
            scores[f"Task {task_number}"] = float(value)

        final_matches = re.findall(
            r"Task\s+([1-5])[^:]*:\s*([0-9.]+)\s*/\s*1\.0",
            output_text,
        )
        for task_number, value in final_matches:
            scores[f"Task {task_number}"] = float(value)

    # Compute average
    task_scores = [scores[f"Task {i}"] for i in range(1, 6)]
    scores["Average"] = round(sum(task_scores) / 5, 4)

    return scores


def extract_steps(output_text: str) -> dict:
    """
    Extract step count per task from [END] log lines, labelled by the
    most recent [START] task= field.  Robust to crashes or retries:
    if two [END] lines appear for the same task the last one wins.
    """
    task_name_to_num = {
        "scheme_discovery":  1,
        "missing_data":      2,
        "boundary_fraud":    3,
        "escalation_dilemma": 4,
        "document_conflict": 5,
    }
    steps = {}
    current_task_name = None
    for line in output_text.splitlines():
        if line.startswith("[START]"):
            m = re.search(r"task=(\S+)", line)
            if m:
                current_task_name = m.group(1)
        elif line.startswith("[END]") and current_task_name is not None:
            m = re.search(r"steps=(\d+)", line)
            if m:
                task_num = task_name_to_num.get(current_task_name)
                if task_num is not None:
                    steps[f"Task {task_num}"] = int(m.group(1))
    return steps


def extract_negative_steps(output_text: str) -> int:
    """Count steps where reward was negative."""
    count = 0
    for line in output_text.splitlines():
        if line.startswith("[STEP]"):
            m = re.search(r"reward=(-[0-9.]+)", line)
            if m:
                count += 1
    return count


def detect_run_status(output_text: str, stderr_text: str) -> Tuple[str, Optional[str]]:
    """
    Distinguish true model runs from provider/configuration failures so benchmark
    reports do not treat unsupported models as genuine 0.0 capability scores.
    """
    combined = f"{output_text}\n{stderr_text}"
    unsupported_markers = (
        "model_not_supported",
        "not supported by provider",
        "unsupported_value",
        "does not support chat",
    )
    auth_markers = (
        "401",
        "403",
        "invalid api key",
        "authentication",
        "authorization",
    )

    lowered = combined.lower()
    if any(marker in lowered for marker in unsupported_markers):
        return "Unsupported", "provider_model_unsupported"
    if any(marker in lowered for marker in auth_markers):
        return "AuthError", "provider_auth_error"
    if "[ERROR] agent decision failed: API_ERROR:" in combined:
        return "ProviderError", "provider_api_error"
    return "Completed", None


# =========================================================
# PER-RUN ANALYSIS
# =========================================================

def analyze_single_run(model: str, scores: dict, std_scores: dict, steps: dict,
                        negative_steps: int, status: str) -> dict:
    """
    After each model run, compute diagnostic signals.
    Returns a structured analysis dict.
    """
    task_scores = [scores.get(f"Task {i}", 0.0) for i in range(1, 6)]
    avg         = scores.get("Average", 0.0)
    tier        = MODEL_TIERS.get(model, "unknown")

    # Exploit signals
    exploit_flags = []
    if task_scores[2] >= 1.0:   # Task 3 — always-reject exploit
        exploit_flags.append("POSSIBLE_T3_EXPLOIT: Task 3 perfect — verify agent actually collected income data")
    if task_scores[3] >= 0.75:  # Task 4 — blind escalate
        exploit_flags.append("POSSIBLE_T4_BLIND_ESCALATE: Task 4 high score — verify PAN card was requested")
    if task_scores[4] >= 1.0:   # Task 5 — memorization
        exploit_flags.append("POSSIBLE_T5_MEMORIZE: Task 5 perfect — verify Aadhaar was requested before reject")

    # Score distribution signal
    non_zero = [s for s in task_scores if s > 0.0]
    binary_behavior = all(s in (0.0, 1.0) for s in task_scores)

    # Difficulty progression check
    difficulty_ok = (
        task_scores[0] >= task_scores[2] >= task_scores[4]
        or task_scores[0] >= task_scores[3]
    )

    # Efficiency signal
    total_steps = sum(steps.values()) if steps else 0

    analysis = {
        "model":           model,
        "tier":            tier,
        "status":          status,
        "average":         avg,
        "task_scores":     {f"Task {i}": task_scores[i-1] for i in range(1, 6)},
        "std_scores":      {f"Task {i}": std_scores.get(f"Task {i}", 0.0) for i in range(1, 6)},
        "step_counts":     steps,
        "total_steps":     total_steps,
        "negative_steps":  negative_steps,
        "exploit_flags":   exploit_flags,
        "binary_behavior": binary_behavior,
        "tasks_passed":    sum(1 for s in task_scores if s >= 1.0),
        "tasks_partial":   sum(1 for s in task_scores if 0.0 < s < 1.0),
        "tasks_failed":    sum(1 for s in task_scores if s == 0.0),
        "difficulty_progression_ok": difficulty_ok,
    }

    # Print per-run summary
    _print_run_analysis(analysis)
    return analysis


def _print_run_analysis(a: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}", flush=True)
    print(f"  PER-RUN ANALYSIS: {a['model']}", flush=True)
    print(f"  Tier: {a['tier']} | Status: {a['status']}", flush=True)
    print(sep, flush=True)
    print(f"  Average score  : {a['average']:.3f}", flush=True)
    print(f"  Tasks passed   : {a['tasks_passed']}/5  partial: {a['tasks_partial']}/5  failed: {a['tasks_failed']}/5", flush=True)
    print(f"  Total steps    : {a['total_steps']}  negative reward steps: {a['negative_steps']}", flush=True)
    print(f"  Binary behavior: {'YES — scoring 0 or 1 only, no graded signal' if a['binary_behavior'] else 'NO — graded scores present (good)'}", flush=True)
    for t, s in a['task_scores'].items():
        std = a.get('std_scores', {}).get(t, 0.0)
        bar = "█" * int(s * 20)
        print(f"  {t}: {s:.3f} ± {std:.3f} {bar}", flush=True)
    if a['exploit_flags']:
        print(f"\n  ⚠ EXPLOIT FLAGS:", flush=True)
        for flag in a['exploit_flags']:
            print(f"    - {flag}", flush=True)
    print(sep, flush=True)


# =========================================================
# AGGREGATE ANALYSIS
# =========================================================

def analyze_aggregate(all_runs: list) -> dict:
    """
    After all models complete, compute aggregate diagnostic signals
    that answer: 'Is this a good RL environment?'
    """
    completed = [r for r in all_runs if r["status"] == "Completed"]
    if not completed:
        return {"error": "No completed runs to analyze"}

    averages    = [r["average"] for r in completed]
    task_scores = {f"Task {i}": [r["task_scores"][f"Task {i}"] for r in completed] for i in range(1, 6)}

    # 1. Difficulty discrimination
    tier_avgs = {}
    for r in completed:
        tier = r["tier"]
        tier_avgs.setdefault(tier, []).append(r["average"])
    tier_means = {t: round(mean(scores), 3) for t, scores in tier_avgs.items()}

    # Check if stronger tiers score higher
    sorted_tiers = sorted(tier_means.items(), key=lambda x: x[0])
    discrimination_ok = all(
        sorted_tiers[i][1] <= sorted_tiers[i+1][1]
        for i in range(len(sorted_tiers)-1)
    ) if len(sorted_tiers) > 1 else None

    # 2. Score spread — good RL env has spread, not clustering
    overall_spread = round(stdev(averages), 3) if len(averages) > 1 else 0.0
    per_task_spread = {
        t: round(stdev(scores), 3) if len(scores) > 1 else 0.0
        for t, scores in task_scores.items()
    }

    # 3. Task difficulty ordering
    task_means = {t: round(mean(scores), 3) for t, scores in task_scores.items()}
    difficulty_ordered = (
        task_means["Task 1"] >= task_means["Task 2"] >= task_means["Task 3"]
    )

    # 4. Exploit detection
    exploit_counts = {}
    for r in completed:
        for flag in r.get("exploit_flags", []):
            key = flag.split(":")[0]
            exploit_counts[key] = exploit_counts.get(key, 0) + 1

    # 5. Binary behavior — bad sign if most models score 0 or 1 only
    binary_count = sum(1 for r in completed if r["binary_behavior"])
    graded_count = len(completed) - binary_count

    # 6. Best and worst models
    sorted_by_avg = sorted(completed, key=lambda r: r["average"], reverse=True)
    top5    = [{"model": r["model"], "tier": r["tier"], "avg": r["average"]} for r in sorted_by_avg[:5]]
    bottom5 = [{"model": r["model"], "tier": r["tier"], "avg": r["average"]} for r in sorted_by_avg[-5:]]

    # 7. RL environment quality verdict
    verdicts = []
    if overall_spread >= 0.15:
        verdicts.append("GOOD: Score spread >= 0.15 — reward signal discriminates between models")
    else:
        verdicts.append("WEAK: Score spread < 0.15 — tasks may be too easy or too hard uniformly")

    if discrimination_ok:
        verdicts.append("GOOD: Stronger model tiers score higher — difficulty is calibrated")
    elif discrimination_ok is False:
        verdicts.append("WEAK: Stronger tiers do not consistently outscore weaker tiers")

    if difficulty_ordered:
        verdicts.append("GOOD: Task difficulty ordering correct (T1 > T2 > T3 as expected)")
    else:
        verdicts.append("CHECK: Task difficulty ordering unexpected — review task design")

    if graded_count > binary_count:
        verdicts.append("GOOD: Most models show graded scores — partial credit is working")
    else:
        verdicts.append("WEAK: Most models show binary 0/1 scores — reward shaping may need tuning")

    if exploit_counts:
        verdicts.append(f"WARNING: Exploit flags detected — {exploit_counts}")
    else:
        verdicts.append("GOOD: No exploit flags triggered across all runs")

    aggregate = {
        "total_models":        len(all_runs),
        "completed":           len(completed),
        "errors":              len([r for r in all_runs if r["status"] != "Completed"]),
        "overall_avg":         round(mean(averages), 3),
        "overall_spread_std":  overall_spread,
        "per_task_means":      task_means,
        "per_task_spread_std": per_task_spread,
        "tier_means":          tier_means,
        "discrimination_ok":   discrimination_ok,
        "difficulty_ordered":  difficulty_ordered,
        "binary_behavior_count": binary_count,
        "graded_behavior_count": graded_count,
        "exploit_counts":      exploit_counts,
        "top5_models":         top5,
        "bottom5_models":      bottom5,
        "rl_quality_verdicts": verdicts,
    }

    _print_aggregate_analysis(aggregate, sorted_by_avg)
    return aggregate


def _print_aggregate_analysis(a: dict, ranked: list) -> None:
    sep = "=" * 70
    print(f"\n\n{sep}", flush=True)
    print(f"  AGGREGATE ANALYSIS — ALL {a['total_models']} MODELS", flush=True)
    print(sep, flush=True)

    print(f"\n  COMPLETION: {a['completed']}/{a['total_models']} completed, {a['errors']} errors", flush=True)
    print(f"  OVERALL AVG SCORE: {a['overall_avg']:.3f}  STD: {a['overall_spread_std']:.3f}", flush=True)

    print(f"\n  PER-TASK MEAN SCORES (difficulty check):", flush=True)
    for t, mean_score in a['per_task_means'].items():
        spread = a['per_task_spread_std'][t]
        bar    = "█" * int(mean_score * 20)
        print(f"    {t}: {mean_score:.3f} ± {spread:.3f}  {bar}", flush=True)

    print(f"\n  SCORE BY MODEL TIER:", flush=True)
    for tier, avg in sorted(a['tier_means'].items()):
        print(f"    {tier}: {avg:.3f}", flush=True)

    print(f"\n  BEHAVIOR DISTRIBUTION:", flush=True)
    print(f"    Graded scores (partial credit working): {a['graded_behavior_count']} models", flush=True)
    print(f"    Binary scores (0 or 1 only):            {a['binary_behavior_count']} models", flush=True)

    print(f"\n  FULL LEADERBOARD:", flush=True)
    print(f"    {'Rank':<5} {'Model':<50} {'Tier':<16} {'Avg':>6}", flush=True)
    print(f"    {'-'*80}", flush=True)
    for i, r in enumerate(ranked, 1):
        model_short = r['model'].split('/')[-1][:48]
        print(f"    {i:<5} {model_short:<50} {r['tier']:<16} {r['average']:>6.3f}", flush=True)

    print(f"\n  RL ENVIRONMENT QUALITY VERDICTS:", flush=True)
    for verdict in a['rl_quality_verdicts']:
        icon = "✓" if verdict.startswith("GOOD") else "✗" if verdict.startswith("WEAK") else "⚠"
        print(f"    {icon} {verdict}", flush=True)

    if a['exploit_counts']:
        print(f"\n  EXPLOIT DETECTIONS:", flush=True)
        for exploit, count in a['exploit_counts'].items():
            print(f"    {exploit}: {count} models triggered", flush=True)

    print(f"\n  TOP 5 MODELS:", flush=True)
    for m in a['top5_models']:
        print(f"    {m['avg']:.3f}  [{m['tier']}]  {m['model']}", flush=True)

    print(f"\n  BOTTOM 5 MODELS:", flush=True)
    for m in a['bottom5_models']:
        print(f"    {m['avg']:.3f}  [{m['tier']}]  {m['model']}", flush=True)

    print(f"\n{sep}", flush=True)


# =========================================================
# MODEL RUNNER
# =========================================================

async def run_model(model: str, idx: int, total: int) -> dict:
    print(f"\n[RUNNER] ({idx}/{total}) Starting: {model}", flush=True)

    log_filepath = LOG_DIR / f"{model.replace('/', '_')}.txt"
    env = os.environ.copy()
    env.update({
        "MODEL_NAME":    model,
        "API_BASE_URL":  API_BASE_URL,
        "HF_TOKEN":      API_TOKEN,
        "OPENAI_API_KEY": API_TOKEN,
        "ENV_URL":       ENV_URL,
        "N_REPEATS":     str(N_REPEATS),
    })

    start_time = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "inference.py",
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=TIMEOUT_SECONDS
        )
        elapsed = round(time.time() - start_time, 1)

        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")

        with open(log_filepath, "w") as f:
            f.write(stdout)
            if stderr:
                f.write(f"\n\n--- STDERR ---\n{stderr}")

        if proc.returncode == 0:
            run_status, error_kind = detect_run_status(stdout, stderr)
            scores         = extract_scores(stdout)
            std_scores     = extract_std_scores(stdout)
            steps          = extract_steps(stdout)
            negative_steps = extract_negative_steps(stdout)
            analysis       = analyze_single_run(model, scores, std_scores, steps, negative_steps, run_status)

            if run_status == "Completed":
                print(f"[RUNNER] ({idx}/{total}) Done in {elapsed}s — avg: {scores['Average']:.3f}", flush=True)
            else:
                print(
                    f"[RUNNER] ({idx}/{total}) {run_status} in {elapsed}s "
                    f"({error_kind or 'provider issue'})",
                    flush=True,
                )
            return {
                "model":          model,
                "status":         run_status,
                "elapsed_s":      elapsed,
                "t1": scores["Task 1"], "t2": scores["Task 2"],
                "t3": scores["Task 3"], "t4": scores["Task 4"],
                "t5": scores["Task 5"], "avg": scores["Average"],
                "analysis":       analysis,
                "error_kind":     error_kind,
            }
        else:
            print(f"[RUNNER] ({idx}/{total}) ERROR — return code {proc.returncode}", flush=True)
            analysis = analyze_single_run(model, {}, {}, {}, 0, "Error")
            return {
                "model": model, "status": "Error", "elapsed_s": elapsed,
                "t1": 0, "t2": 0, "t3": 0, "t4": 0, "t5": 0, "avg": 0,
                "analysis": analysis,
            }

    except asyncio.TimeoutError:
        proc.kill()
        elapsed = round(time.time() - start_time, 1)
        print(f"[RUNNER] ({idx}/{total}) TIMEOUT after {elapsed}s", flush=True)
        analysis = analyze_single_run(model, {}, {}, {}, 0, "Timeout")
        return {
            "model": model, "status": "Timeout", "elapsed_s": elapsed,
            "t1": 0, "t2": 0, "t3": 0, "t4": 0, "t5": 0, "avg": 0,
            "analysis": analysis,
        }


# =========================================================
# CSV + MANIFEST
# =========================================================

def write_csv(results: list) -> None:
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "Model", "Tier", "Status", "Elapsed(s)",
            "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Average",
        ])
        sorted_results = sorted(results, key=lambda r: r["avg"], reverse=True)
        for rank, r in enumerate(sorted_results, 1):
            writer.writerow([
                rank,
                r["model"],
                MODEL_TIERS.get(r["model"], "unknown"),
                r["status"],
                r.get("elapsed_s", 0),
                r["t1"], r["t2"], r["t3"], r["t4"], r["t5"], r["avg"],
            ])


def write_manifest() -> None:
    manifest = {
        "timestamp":       TIMESTAMP,
        "run_dir":         str(RUN_DIR),
        "results_file":    str(RESULTS_FILE),
        "logs_dir":        str(LOG_DIR),
        "models":          MODELS_TO_TEST,
        "model_count":     len(MODELS_TO_TEST),
        "timeout_seconds": TIMEOUT_SECONDS,
        "env_url":         ENV_URL,
        "api_base_url":    API_BASE_URL,
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def write_summary_txt(aggregate: dict, results: list) -> None:
    lines = []
    lines.append("SCHEME ENV — 30-LLM BENCHMARK SUMMARY")
    lines.append(f"Timestamp: {TIMESTAMP}")
    lines.append(f"Models tested: {aggregate['total_models']}")
    lines.append(f"Completed: {aggregate['completed']}  Errors: {aggregate['errors']}")
    lines.append(f"Overall average: {aggregate['overall_avg']:.3f}  Std: {aggregate['overall_spread_std']:.3f}")
    lines.append("")
    lines.append("PER-TASK MEANS:")
    for t, m in aggregate['per_task_means'].items():
        lines.append(f"  {t}: {m:.3f} ± {aggregate['per_task_spread_std'][t]:.3f}")
    lines.append("")
    lines.append("TIER PERFORMANCE:")
    for tier, avg in sorted(aggregate['tier_means'].items()):
        lines.append(f"  {tier}: {avg:.3f}")
    lines.append("")
    lines.append("RL QUALITY VERDICTS:")
    for v in aggregate['rl_quality_verdicts']:
        lines.append(f"  {v}")
    lines.append("")
    lines.append("FULL LEADERBOARD:")
    sorted_results = sorted(results, key=lambda r: r["avg"], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        lines.append(
            f"  {i:>2}. {r['avg']:.3f}  [{MODEL_TIERS.get(r['model'],'?')}]  {r['model']}"
        )
    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(lines))


# =========================================================
# MAIN
# =========================================================

async def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest()

    print(f"\n{'='*70}", flush=True)
    print(f"  SCHEME ENV — 30-LLM BENCHMARK", flush=True)
    print(f"  Models : {len(MODELS_TO_TEST)}", flush=True)
    print(f"  Env    : {ENV_URL}", flush=True)
    print(f"  Output : {RUN_DIR}", flush=True)
    print(f"{'='*70}\n", flush=True)

    _wait_for_server(ENV_URL)

    results  = []
    analyses = []

    for i, model in enumerate(MODELS_TO_TEST, 1):
        result = await run_model(model, i, len(MODELS_TO_TEST))
        results.append(result)
        analyses.append(result["analysis"])

        # Write CSV after every model so partial results are always saved
        write_csv(results)

        # Save running analysis JSON
        with open(ANALYSIS_FILE, "w") as f:
            json.dump({
                "completed_so_far": len(results),
                "runs": analyses,
            }, f, indent=2)

        # Brief pause between models
        await asyncio.sleep(2)

    # Final aggregate analysis
    aggregate = analyze_aggregate(analyses)

    # Write final artifacts
    write_csv(results)
    write_summary_txt(aggregate, results)

    with open(ANALYSIS_FILE, "w") as f:
        json.dump({
            "timestamp":  TIMESTAMP,
            "aggregate":  aggregate,
            "runs":       analyses,
        }, f, indent=2)

    print(f"\n[RUNNER] All done. Artifacts in {RUN_DIR}", flush=True)
    print(f"  Leaderboard : {RESULTS_FILE}", flush=True)
    print(f"  Analysis    : {ANALYSIS_FILE}", flush=True)
    print(f"  Summary     : {SUMMARY_FILE}", flush=True)
    print(f"  Logs        : {LOG_DIR}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
