import os, asyncio, csv, json, re, sys
from datetime import datetime
from pathlib import Path

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_TOKEN = os.getenv("OPENAI_API_KEY", "") or os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# CRITICAL FIX: Must be 1 when running against a local Singleton environment
MAX_CONCURRENT = 1 
TIMEOUT_SECONDS = 600

MODELS_TO_TEST = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "NousResearch/Hermes-2-Pro-Llama-3-8B"
]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORTS_DIR = Path("reports")
RUN_DIR = REPORTS_DIR / f"report_{TIMESTAMP}"
RESULTS_FILE = RUN_DIR / f"leaderboard_{TIMESTAMP}.csv"
LOG_DIR = RUN_DIR / f"logs_{TIMESTAMP}"
MANIFEST_FILE = RUN_DIR / f"run_manifest_{TIMESTAMP}.json"

def extract_scores(output_text: str) -> dict:
    scores = {"Task 1": 0.0, "Task 2": 0.0, "Task 3": 0.0, "Task 4": 0.0, "Average": 0.0}
    task_matches = re.findall(
        r"TASK\s+([1-4])/[0-9]+.*?GRADER SCORE:\s*([0-9.]+)\s*/\s*1\.0",
        output_text,
        re.DOTALL,
    )
    for task_number, value in task_matches:
        scores[f"Task {task_number}"] = float(value)

    final_task_matches = re.findall(
        r"Task\s+([1-4]).*?:\s*([0-9.]+)\s*/\s*1\.0",
        output_text,
    )
    for task_number, value in final_task_matches:
        scores[f"Task {task_number}"] = float(value)

    avg_match = re.search(r"Average\s*:\s*([0-9.]+)\s*/\s*1\.0", output_text)
    if avg_match:
        scores["Average"] = float(avg_match.group(1))
    elif task_matches or final_task_matches:
        scores["Average"] = round(
            sum(scores[f"Task {index}"] for index in range(1, 5)) / 4,
            4,
        )
    return scores

async def run_model(model: str, queue: asyncio.Queue, idx: int, total: int):
    print(f"[{idx}/{total}] Testing: {model}...")
    log_filepath = LOG_DIR / f"{model.replace('/', '_')}.txt"
    env = os.environ.copy()
    env.update({"MODEL_NAME": model, "API_BASE_URL": API_BASE_URL, "HF_TOKEN": API_TOKEN, "ENV_URL": ENV_URL})

    try:
        proc = await asyncio.create_subprocess_exec("python", "inference.py", env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SECONDS)
        
        stdout, stderr = stdout_bytes.decode("utf-8"), stderr_bytes.decode("utf-8")
        with open(log_filepath, "w") as f: f.write(stdout + ("\n\n--- STDERR ---\n" + stderr if stderr else ""))

        if proc.returncode == 0:
            s = extract_scores(stdout)
            await queue.put({"model": model, "status": "Completed", "t1": s["Task 1"], "t2": s["Task 2"], "t3": s["Task 3"], "t4": s["Task 4"], "avg": s["Average"]})
        else:
            await queue.put({"model": model, "status": "Error", "t1": 0, "t2": 0, "t3": 0, "t4": 0, "avg": 0})
    except asyncio.TimeoutError:
        proc.kill()
        await queue.put({"model": model, "status": "Timeout", "t1": 0, "t2": 0, "t3": 0, "t4": 0, "avg": 0})

async def csv_writer(queue: asyncio.Queue, total: int):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Status", "Task 1", "Task 2", "Task 3", "Task 4", "Average Score"])
        for _ in range(total):
            r = await queue.get()
            writer.writerow([r["model"], r["status"], r["t1"], r["t2"], r["t3"], r["t4"], r["avg"]])

def write_manifest():
    manifest = {
        "timestamp": TIMESTAMP,
        "run_dir": str(RUN_DIR),
        "results_file": str(RESULTS_FILE),
        "logs_dir": str(LOG_DIR),
        "models": MODELS_TO_TEST,
        "timeout_seconds": TIMEOUT_SECONDS,
        "env_url": ENV_URL,
        "api_base_url": API_BASE_URL,
    }
    with open(MANIFEST_FILE, "w") as handle:
        json.dump(manifest, handle, indent=2)


async def run_report_generation() -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "benchmark_report.py",
        "--run-dir",
        str(RUN_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    stdout = stdout_bytes.decode("utf-8").strip()
    stderr = stderr_bytes.decode("utf-8").strip()

    if proc.returncode == 0:
        print("\nReport generation completed.")
        if stdout:
            print(stdout)
    else:
        print("\nReport generation failed.")
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)

async def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest()
    queue = asyncio.Queue()
    tasks = [run_model(model, queue, i+1, len(MODELS_TO_TEST)) for i, model in enumerate(MODELS_TO_TEST)]
    writer = asyncio.create_task(csv_writer(queue, len(MODELS_TO_TEST)))
    
    # Run sequentially because MAX_CONCURRENT is 1
    for task in tasks: await task 
    await writer
    await run_report_generation()
    print(f"\nDone! Check {RUN_DIR}")

if __name__ == "__main__": asyncio.run(main())
