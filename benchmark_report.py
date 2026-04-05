#!/usr/bin/env python3
"""Generate graph-first benchmark reports from benchmark_runner artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from textwrap import fill
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scheme_env_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

TASK_NUMBERS = [1, 2, 3, 4, 5]
TASK_NAMES = {
    1: "T1: Scheme Discovery",
    2: "T2: Missing Data",
    3: "T3: Boundary Fraud Detection",
    4: "T4: Escalation Dilemma",
    5: "T5: Document Conflict",
}

THEME = {
    "bg": "#f6f1e8",
    "panel": "#fffdf8",
    "ink": "#172033",
    "muted": "#67778a",
    "grid": "#d8ccbc",
    "border": "#dccfbe",
    "teal": "#0f766e",
    "blue": "#38bdf8",
    "amber": "#f59e0b",
    "coral": "#ef4444",
    "soft_green": "#dff5ef",
    "warning_bg": "#fff4dd",
    "warning_border": "#f7d48c",
    "warning_ink": "#7b4f00",
}

STATUS_COLORS = {
    "Completed": THEME["teal"],
    "Partial": THEME["blue"],
    "Timeout": THEME["amber"],
    "Error": THEME["coral"],
    "Unknown": THEME["muted"],
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.facecolor": THEME["panel"],
        "figure.facecolor": THEME["bg"],
        "axes.edgecolor": THEME["border"],
        "axes.labelcolor": THEME["ink"],
        "xtick.color": THEME["ink"],
        "ytick.color": THEME["ink"],
        "text.color": THEME["ink"],
        "axes.titlecolor": THEME["ink"],
    }
)


@dataclass(frozen=True)
class ParsedAction:
    step: int
    action_type: str
    value: str
    reward: float
    done: bool


@dataclass(frozen=True)
class ParsedTask:
    task_number: int
    task_name: str
    grader_score: float = 0.0
    steps_taken: int = 0
    terminated: bool = False
    actions: list[ParsedAction] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedModelRun:
    model_name: str
    safe_model_slug: str
    status: str
    average_score: float
    tasks: list[ParsedTask]
    total_steps: int
    total_negative_reward_steps: int
    error_type: Optional[str]
    error_message: Optional[str]
    log_path: str
    csv_status: Optional[str] = None
    data_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReportWarning:
    code: str
    message: str


@dataclass(frozen=True)
class ReportBundle:
    timestamp: str
    csv_path: Optional[str]
    logs_dir: str
    models: list[ParsedModelRun]
    warnings: list[ReportWarning]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual reports for scheme_env benchmark runs.")
    parser.add_argument("--timestamp", help="Benchmark timestamp like 20260404_124255.")
    parser.add_argument("--run-dir", type=Path, help="Path to a benchmark run bundle like reports/report_<timestamp>.")
    parser.add_argument("--csv", dest="csv_path", type=Path, help="Path to leaderboard CSV.")
    parser.add_argument("--logs-dir", type=Path, help="Path to logs directory.")
    parser.add_argument("--latest", action="store_true", help="Use the latest discovered artifact pair.")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated report artifacts.")
    parser.add_argument("--include-failed", action="store_true", help="Accepted for compatibility; failed models are included by default.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on parse or reconciliation warnings.")
    args = parser.parse_args()

    selected = int(bool(args.timestamp)) + int(bool(args.latest)) + int(bool(args.run_dir)) + int(bool(args.csv_path or args.logs_dir))
    if selected != 1:
        parser.error("Provide exactly one of --timestamp, --latest, --run-dir, or --csv with --logs-dir.")
    if bool(args.csv_path) != bool(args.logs_dir):
        parser.error("--csv and --logs-dir must be provided together.")
    return args


def discover_artifact_pairs(base_dir: Path) -> list[tuple[str, Optional[Path], Path]]:
    pairs: dict[str, tuple[Optional[Path], Path]] = {}

    for run_dir in sorted(base_dir.glob("reports/report_*")):
        if not run_dir.is_dir():
            continue
        timestamp = run_dir.name.replace("report_", "")
        csv_path = run_dir / f"leaderboard_{timestamp}.csv"
        logs_dir = run_dir / f"logs_{timestamp}"
        if logs_dir.is_dir():
            pairs[timestamp] = (csv_path if csv_path.exists() else None, logs_dir)

    manifest_map: dict[str, dict] = {}
    for manifest_path in base_dir.glob("run_manifest_*.json"):
        try:
            payload = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        timestamp = payload.get("timestamp")
        if timestamp:
            manifest_map[timestamp] = payload

    csv_map = {path.stem.replace("leaderboard_", ""): path for path in base_dir.glob("leaderboard_*.csv")}
    logs_map = {path.name.replace("logs_", ""): path for path in base_dir.glob("logs_*") if path.is_dir()}
    for timestamp, payload in manifest_map.items():
        csv_name = payload.get("results_file")
        logs_name = payload.get("logs_dir")
        if csv_name:
            csv_map.setdefault(timestamp, base_dir / csv_name)
        if logs_name:
            logs_map.setdefault(timestamp, base_dir / logs_name)
    timestamps = sorted(set(csv_map) | set(logs_map) | set(manifest_map))
    root_pairs = [(timestamp, csv_map.get(timestamp), logs_map[timestamp]) for timestamp in timestamps if timestamp in logs_map]
    for timestamp, csv_path, logs_dir in root_pairs:
        pairs.setdefault(timestamp, (csv_path, logs_dir))
    return [(timestamp, value[0], value[1]) for timestamp, value in sorted(pairs.items())]


def resolve_inputs(args: argparse.Namespace, base_dir: Path) -> tuple[str, Optional[Path], Path, Path]:
    if args.run_dir:
        run_dir = args.run_dir
        timestamp = run_dir.name.replace("report_", "")
        csv_path = run_dir / f"leaderboard_{timestamp}.csv"
        logs_dir = run_dir / f"logs_{timestamp}"
        output_dir = run_dir
    elif args.timestamp:
        timestamp = args.timestamp
        run_dir = base_dir / "reports" / f"report_{timestamp}"
        csv_path = run_dir / f"leaderboard_{timestamp}.csv"
        logs_dir = run_dir / f"logs_{timestamp}"
        output_dir = run_dir
    elif args.latest:
        pairs = discover_artifact_pairs(base_dir)
        if not pairs:
            raise SystemExit("No benchmark artifacts found.")
        timestamp, csv_path, logs_dir = pairs[-1]
        output_dir = base_dir / "reports" / f"report_{timestamp}"
    else:
        csv_path = args.csv_path
        logs_dir = args.logs_dir
        timestamp = logs_dir.name.replace("logs_", "")
        output_dir = args.output_dir or (base_dir / "reports" / f"report_{timestamp}")

    if not logs_dir.exists():
        raise SystemExit(f"Logs directory not found: {logs_dir}")
    if csv_path is not None and not csv_path.exists():
        csv_path = None

    output_dir = args.output_dir or output_dir
    return timestamp, csv_path, logs_dir, output_dir


def parse_csv_rows(csv_path: Optional[Path]) -> dict[str, dict[str, str]]:
    if csv_path is None:
        return {}
    with csv_path.open() as handle:
        return {row["Model"]: row for row in csv.DictReader(handle)}


def safe_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace(" ", "_")


def classify_error(stderr: str) -> tuple[Optional[str], Optional[str]]:
    stderr = stderr.strip()
    if not stderr:
        return None, None
    lowered = stderr.lower()
    if "model_not_supported" in lowered or "not supported by any provider" in lowered:
        return "model_not_supported", stderr.splitlines()[-1]
    if "depleted your monthly included credits" in lowered or "purchase pre-paid credits" in lowered:
        return "provider_credits_exhausted", stderr.splitlines()[-1]
    if "timeout" in lowered:
        return "timeout", stderr.splitlines()[-1]
    if "openai." in lowered or "api" in lowered:
        return "api_error", stderr.splitlines()[-1]
    if "traceback" in lowered:
        return "runtime_error", stderr.splitlines()[-1]
    return "unknown_error", stderr.splitlines()[-1]


def parse_actions(task_body: str) -> list[ParsedAction]:
    actions: list[ParsedAction] = []
    pattern = re.compile(
        r"Step\s+(\d+):\s+([a-z_]+)\((.*?)\)\s*->\s*reward=([-0-9.]+),\s*done=(True|False)"
    )
    for step, action_type, raw_value, reward, done in pattern.findall(task_body):
        value = raw_value.strip()
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        actions.append(
            ParsedAction(
                step=int(step),
                action_type=action_type,
                value=value,
                reward=float(reward),
                done=(done == "True"),
            )
        )
    return actions


def parse_tasks(log_text: str) -> list[ParsedTask]:
    matches = list(
        re.finditer(
            r"TASK\s+([1-5])/[0-9]+(.*?)(?=\n={20,}\n\s+TASK\s+[1-5]/[0-9]+|\n={20,}\n\s+FINAL|\Z)",
            log_text,
            re.DOTALL,
        )
    )
    parsed: list[ParsedTask] = []
    for match in matches:
        task_number = int(match.group(1))
        task_body = match.group(2)
        actions = parse_actions(task_body)
        score_match = re.search(r"GRADER SCORE:\s*([0-9.]+)\s*/\s*1\.0", task_body)
        score = float(score_match.group(1)) if score_match else 0.0
        terminated = bool(actions and actions[-1].done) or bool(score_match)
        parsed.append(
            ParsedTask(
                task_number=task_number,
                task_name=TASK_NAMES[task_number],
                grader_score=score,
                steps_taken=len(actions),
                terminated=terminated,
                actions=actions,
            )
        )
    parsed.sort(key=lambda task: task.task_number)
    return parsed


def parse_model_log(log_path: Path, csv_row: Optional[dict[str, str]]) -> tuple[ParsedModelRun, list[ReportWarning]]:
    log_text = log_path.read_text()
    warnings: list[ReportWarning] = []

    model_match = re.search(r"Model:\s*(.+)", log_text)
    model_name = model_match.group(1).strip() if model_match else (csv_row["Model"] if csv_row else log_path.stem)
    tasks = parse_tasks(log_text)
    task_scores = {task.task_number: task.grader_score for task in tasks}

    avg_match = re.search(r"FINAL SCORES:\s*Avg\s*([0-9.]+)\s*/?\s*1\.0", log_text)
    if not avg_match:
        avg_match = re.search(r"Average\s*:\s*([0-9.]+)\s*/\s*1\.0", log_text)
    if avg_match:
        average_score = float(avg_match.group(1))
    elif tasks:
        average_score = round(sum(task_scores.values()) / len(tasks), 4)
    else:
        average_score = 0.0

    final_task_matches = re.findall(r"Task\s+([1-4]).*?:\s*([0-9.]+)\s*/\s*1\.0", log_text)
    if final_task_matches:
        task_summary_map = {int(task_number): float(value) for task_number, value in final_task_matches}
        if not tasks:
            tasks = [
                ParsedTask(
                    task_number=task_number,
                    task_name=TASK_NAMES.get(task_number, f"T{task_number}"),
                    grader_score=score,
                    steps_taken=0,
                    terminated=True,
                    actions=[],
                )
                for task_number, score in sorted(task_summary_map.items())
            ]
        else:
            tasks = [
                ParsedTask(
                    task_number=task.task_number,
                    task_name=task.task_name,
                    grader_score=task_summary_map.get(task.task_number, task.grader_score),
                    steps_taken=task.steps_taken,
                    terminated=task.terminated,
                    actions=task.actions,
                )
                for task in tasks
            ]

    stderr = ""
    if "\n--- STDERR ---\n" in log_text:
        stderr = log_text.split("\n--- STDERR ---\n", 1)[1]
    error_type, error_message = classify_error(stderr)

    total_steps = sum(task.steps_taken for task in tasks)
    total_negative_reward_steps = sum(
        1 for task in tasks for action in task.actions if action.reward < 0
    )

    completed_tasks = sum(1 for task in tasks if task.terminated)
    expected_tasks = len(tasks) if tasks else 0
    if expected_tasks and completed_tasks == expected_tasks:
        status = "Completed"
    elif error_type == "timeout" or (csv_row and csv_row.get("Status") == "Timeout"):
        status = "Timeout"
    elif completed_tasks > 0:
        status = "Partial"
    elif error_type:
        status = "Error"
    else:
        status = csv_row.get("Status") if csv_row else "Unknown"

    data_warnings: list[str] = []
    if csv_row:
        csv_avg = float(csv_row["Average Score"])
        if round(csv_avg, 4) != round(average_score, 4):
            warning = (
                f"CSV average {csv_avg:.2f} disagreed with log average {average_score:.2f} for {model_name}; "
                "using log-derived values."
            )
            data_warnings.append(warning)
            warnings.append(ReportWarning(code="csv_log_mismatch", message=warning))

    if not tasks:
        message = f"No task blocks parsed from {log_path.name}."
        data_warnings.append(message)
        warnings.append(ReportWarning(code="parse_fallback", message=message))

    return (
        ParsedModelRun(
            model_name=model_name,
            safe_model_slug=safe_slug(model_name),
            status=status,
            average_score=average_score,
            tasks=tasks,
            total_steps=total_steps,
            total_negative_reward_steps=total_negative_reward_steps,
            error_type=error_type,
            error_message=error_message,
            log_path=str(log_path),
            csv_status=csv_row.get("Status") if csv_row else None,
            data_warnings=data_warnings,
        ),
        warnings,
    )


def active_task_numbers(models: list[ParsedModelRun]) -> list[int]:
    numbers = sorted({task.task_number for model in models for task in model.tasks})
    return numbers or [1, 2, 3, 4, 5]


def reconcile_bundle(timestamp: str, csv_path: Optional[Path], logs_dir: Path) -> ReportBundle:
    csv_rows = parse_csv_rows(csv_path)
    warnings: list[ReportWarning] = []
    models: list[ParsedModelRun] = []

    for log_path in sorted(logs_dir.glob("*.txt")):
        matched_row = None
        for model_name, row in csv_rows.items():
            if safe_slug(model_name) == log_path.stem:
                matched_row = row
                break
        model_run, model_warnings = parse_model_log(log_path, matched_row)
        models.append(model_run)
        warnings.extend(model_warnings)

    log_slugs = {model.safe_model_slug for model in models}
    for model_name in csv_rows:
        if safe_slug(model_name) not in log_slugs:
            warnings.append(
                ReportWarning(
                    code="missing_log",
                    message=f"CSV contains {model_name} but no corresponding log file was found.",
                )
            )

    if csv_path is None:
        warnings.append(ReportWarning(code="missing_csv", message="CSV not found; report was built from logs only."))

    models.sort(key=lambda model: (-model.average_score, model.model_name))
    return ReportBundle(
        timestamp=timestamp,
        csv_path=str(csv_path) if csv_path else None,
        logs_dir=str(logs_dir),
        models=models,
        warnings=warnings,
    )


def _short_label(text: str, width: int) -> str:
    return fill(text, width=width)


def _style_axes(ax: plt.Axes, title: str, subtitle: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=18, fontweight="bold", loc="left", pad=24)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color=THEME["muted"], va="bottom")
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.7, color=THEME["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME["border"])
    ax.spines["bottom"].set_color(THEME["border"])


def _add_badge(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.92,
        0.955,
        text,
        ha="right",
        va="center",
        fontsize=9,
        color=THEME["teal"],
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.4", "facecolor": "#daf3ef", "edgecolor": "#b8e6dd"},
    )


def create_average_score_chart(models: list[ParsedModelRun], output_dir: Path) -> Path:
    fig = plt.figure(figsize=(14.2, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 4.0], wspace=0.0)
    label_ax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=label_ax)
    y_positions = list(range(len(models)))
    colors = [STATUS_COLORS.get(model.status, THEME["muted"]) for model in models]
    bars = ax.barh(y_positions, [model.average_score for model in models], color=colors, edgecolor="none", height=0.64)

    label_ax.set_facecolor(THEME["panel"])
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(-0.5, len(models) - 0.5)
    label_ax.set_xticks([])
    label_ax.set_yticks(y_positions)
    label_ax.set_yticklabels([])
    label_ax.tick_params(axis="y", length=0)
    for spine in ("top", "left", "bottom"):
        label_ax.spines[spine].set_color(THEME["border"])
    label_ax.spines["right"].set_color(THEME["border"])
    label_ax.spines["right"].set_linewidth(1.2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Average benchmark score", fontsize=11.5)
    _style_axes(ax, "Model Leaderboard", "Higher is better. Status, step cost, and failures are annotated per model.", "")
    _add_badge(fig, "Overall ranking")
    ax.spines["left"].set_visible(False)

    for bar, model in zip(bars, models):
        y_center = bar.get_y() + bar.get_height() / 2
        label_ax.text(
            0.03,
            y_center + 0.03,
            _short_label(model.model_name, 20),
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color=THEME["ink"],
        )
        value_x = model.average_score + 0.015 if model.average_score > 0 else 0.02
        ax.text(value_x, y_center, f"{model.average_score:.2f}", va="center", fontsize=11, fontweight="bold")
        descriptor = f"{model.status}  |  {model.total_steps} steps"
        if model.error_type:
            descriptor += f"  |  {model.error_type}"
        label_ax.text(0.03, y_center - 0.19, _short_label(descriptor, 28), va="center", fontsize=9.2, color=THEME["muted"])

    fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.12)
    path = output_dir / "average_scores.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def create_task_heatmap(models: list[ParsedModelRun], output_dir: Path) -> Path:
    task_numbers = active_task_numbers(models)
    score_matrix = []
    step_matrix = []
    for model in models:
        task_map = {task.task_number: task for task in model.tasks}
        score_matrix.append([task_map.get(number, ParsedTask(number, TASK_NAMES[number])).grader_score for number in task_numbers])
        step_matrix.append([task_map.get(number, ParsedTask(number, TASK_NAMES[number])).steps_taken for number in task_numbers])

    cmap = LinearSegmentedColormap.from_list("benchmark_heat", ["#fff2c9", "#79cfc0", "#39a8df", "#1b2868"])
    fig, ax = plt.subplots(figsize=(13.8, 7.6))
    image = ax.imshow(score_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(task_numbers)))
    ax.set_xticklabels([_short_label(TASK_NAMES[number], 16) for number in task_numbers])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_short_label(model.model_name, 26) for model in models])
    _style_axes(ax, "Per-Task Reliability Heatmap", "Cell color shows score. Secondary annotation shows step count per task.", "Model")
    ax.set_xlabel("Benchmark task", fontsize=11.5)
    _add_badge(fig, "Failure localization")

    for row_index, row in enumerate(score_matrix):
        for col_index, value in enumerate(row):
            text_color = "white" if value >= 0.55 else THEME["ink"]
            ax.text(col_index, row_index - 0.08, f"{value:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)
            ax.text(col_index, row_index + 0.18, f"{step_matrix[row_index][col_index]} steps", ha="center", va="center", fontsize=8.5, color=text_color)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Task score", fontsize=11)
    for x in range(len(task_numbers) + 1):
        ax.axvline(x - 0.5, color="white", linewidth=1.2, alpha=0.8)
    for y in range(len(models) + 1):
        ax.axhline(y - 0.5, color="white", linewidth=1.2, alpha=0.8)

    fig.subplots_adjust(left=0.23, right=0.95, top=0.90, bottom=0.16)
    path = output_dir / "task_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def create_efficiency_scatter(models: list[ParsedModelRun], output_dir: Path) -> Path:
    sorted_models = sorted(
        models,
        key=lambda model: (
            -model.average_score,
            model.total_negative_reward_steps,
            model.total_steps,
            model.model_name,
        ),
    )
    fig, ax = plt.subplots(figsize=(15.0, 8.4))
    ax.set_axis_off()
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    max_steps = max((model.total_steps for model in sorted_models), default=1)
    max_penalties = max((model.total_negative_reward_steps for model in sorted_models), default=0)
    efficiency_scores: list[float] = []
    for model in sorted_models:
        step_penalty = (model.total_steps / max_steps) * 0.25 if max_steps else 0.0
        negative_penalty = (model.total_negative_reward_steps / max_penalties) * 0.2 if max_penalties else 0.0
        efficiency_scores.append(max(0.0, min(1.0, model.average_score - step_penalty - negative_penalty)))

    ax.text(0.03, 0.97, "Execution Summary", fontsize=30, fontweight="bold", color=THEME["ink"], va="top")
    ax.text(
        0.03,
        0.93,
        "A compact readout of score quality, step cost, and failure pressure for each model.",
        fontsize=12.5,
        color=THEME["muted"],
        va="top",
    )
    ax.text(
        0.93,
        0.935,
        "Operational view",
        ha="right",
        va="center",
        fontsize=9,
        color=THEME["teal"],
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.4", "facecolor": "#daf3ef", "edgecolor": "#b8e6dd"},
    )

    columns = ["Rank", "Model", "Status", "Avg score", "Steps", "Negative", "Takeaway"]
    rows = []
    status_colors: list[str] = []
    for index, (model, efficiency) in enumerate(zip(sorted_models, efficiency_scores), start=1):
        signal_text = "Efficient"
        if model.status != "Completed":
            signal_text = "Failed run"
        elif model.total_negative_reward_steps > 0:
            signal_text = "Penalty pressure"
        elif model.total_steps > mean(item.total_steps for item in sorted_models):
            signal_text = "Higher action cost"

        meta = model.error_type if model.error_type else f"efficiency {efficiency:.2f}"
        rows.append(
            [
                str(index),
                f"{model.model_name}\n{meta}",
                model.status,
                f"{model.average_score:.2f}",
                str(model.total_steps),
                str(model.total_negative_reward_steps),
                signal_text,
            ]
        )
        status_colors.append(STATUS_COLORS.get(model.status, THEME["muted"]))

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colWidths=[0.06, 0.40, 0.12, 0.11, 0.09, 0.10, 0.12],
        loc="upper left",
        cellLoc="left",
        bbox=[0.03, 0.10, 0.94, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(THEME["border"])
        if row == 0:
            cell.set_facecolor(THEME["bg"])
            cell.set_linewidth(0.0)
            cell.get_text().set_color(THEME["muted"])
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(10.5)
            cell.PAD = 0.18
        else:
            cell.set_facecolor(THEME["panel"])
            cell.set_linewidth(1.0)
            cell.PAD = 0.14
            if col == 0:
                cell.get_text().set_color(THEME["muted"])
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_ha("center")
            elif col in (3, 4, 5):
                cell.get_text().set_ha("center")
                if col == 3:
                    cell.get_text().set_fontweight("bold")
            elif col == 2:
                cell.get_text().set_ha("center")
                cell.get_text().set_color(status_colors[row - 1])
                cell.get_text().set_fontweight("bold")
            elif col == 6:
                cell.get_text().set_ha("center")
                cell.get_text().set_color(THEME["muted"])
            elif col == 1:
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color(THEME["ink"])
            cell.get_text().set_wrap(True)

    for row_index in range(1, len(rows) + 1):
        for col_index in range(len(columns)):
            table[(row_index, col_index)].set_height(0.13)

    ax.text(
        0.97,
        0.05,
        "Ranking prioritizes benchmark score first, then fewer penalties and fewer actions.",
        ha="right",
        va="center",
        fontsize=10,
        color=THEME["muted"],
    )
    path = output_dir / "efficiency_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_json(bundle: ReportBundle, output_dir: Path) -> Path:
    path = output_dir / "results.json"
    payload = {
        "timestamp": bundle.timestamp,
        "csv_path": bundle.csv_path,
        "logs_dir": bundle.logs_dir,
        "warnings": [asdict(item) for item in bundle.warnings],
        "models": [asdict(model) for model in bundle.models],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_summary_csv(bundle: ReportBundle, output_dir: Path) -> Path:
    path = output_dir / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_name",
                "status",
                "average_score",
                "task_1_score",
                "task_2_score",
                "task_3_score",
                "task_4_score",
                "task_5_score",
                "total_steps",
                "negative_reward_steps",
                "error_type",
                "log_path",
            ]
        )
        for model in bundle.models:
            scores = {task.task_number: task.grader_score for task in model.tasks}
            writer.writerow(
                [
                    model.model_name,
                    model.status,
                    f"{model.average_score:.4f}",
                    f"{scores.get(1, 0.0):.4f}",
                    f"{scores.get(2, 0.0):.4f}",
                    f"{scores.get(3, 0.0):.4f}",
                    f"{scores.get(4, 0.0):.4f}",
                    f"{scores.get(5, 0.0):.4f}",
                    model.total_steps,
                    model.total_negative_reward_steps,
                    model.error_type or "",
                    model.log_path,
                ]
            )
    return path


def generate_report(bundle: ReportBundle, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_dashboard = output_dir / "dashboard.html"
    if legacy_dashboard.exists():
        legacy_dashboard.unlink()
    json_path = write_json(bundle, output_dir)
    csv_path = write_summary_csv(bundle, output_dir)
    average_chart = create_average_score_chart(bundle.models, output_dir)
    heatmap_chart = create_task_heatmap(bundle.models, output_dir)
    efficiency_chart = create_efficiency_scatter(bundle.models, output_dir)
    return {
        "json": json_path,
        "csv": csv_path,
        "average_scores_chart": average_chart,
        "task_heatmap_chart": heatmap_chart,
        "efficiency_chart": efficiency_chart,
    }


def print_summary(bundle: ReportBundle, artifacts: dict[str, Path]) -> None:
    print(f"Generated report for run {bundle.timestamp}:")
    for label, path in artifacts.items():
        print(f"  - {label}: {path}")
    if bundle.warnings:
        print("\nWarnings:")
        for warning in bundle.warnings:
            print(f"  - [{warning.code}] {warning.message}")


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    timestamp, csv_path, logs_dir, output_dir = resolve_inputs(args, base_dir)
    bundle = reconcile_bundle(timestamp, csv_path, logs_dir)
    artifacts = generate_report(bundle, output_dir)
    print_summary(bundle, artifacts)
    if args.strict and bundle.warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
