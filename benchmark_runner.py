import os, asyncio, csv, re, subprocess
from datetime import datetime

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_TOKEN = os.getenv("HF_TOKEN", "") 
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
RESULTS_FILE, LOG_DIR = f"leaderboard_{TIMESTAMP}.csv", f"logs_{TIMESTAMP}"

def extract_scores(output_text: str) -> dict:
    scores = {"Task 1": 0.0, "Task 2": 0.0, "Task 3": 0.0, "Task 4": 0.0, "Average": 0.0}
    for key in scores.keys():
        match = re.search(f"{key}.*?:\\s*([0-9.]+)\\s*/", output_text)
        if match: scores[key] = float(match.group(1))
    return scores

async def run_model(model: str, queue: asyncio.Queue, idx: int, total: int):
    print(f"[{idx}/{total}] Testing: {model}...")
    log_filepath = os.path.join(LOG_DIR, f"{model.replace('/', '_')}.txt")
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

async def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    queue = asyncio.Queue()
    tasks = [run_model(model, queue, i+1, len(MODELS_TO_TEST)) for i, model in enumerate(MODELS_TO_TEST)]
    writer = asyncio.create_task(csv_writer(queue, len(MODELS_TO_TEST)))
    
    # Run sequentially because MAX_CONCURRENT is 1
    for task in tasks: await task 
    await writer
    print(f"\nDone! Check {RESULTS_FILE}")

if __name__ == "__main__": asyncio.run(main())