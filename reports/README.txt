OpenEnv scheme_env Benchmark — Baseline Report
================================================

Files in this directory:

  leaderboard.csv
      Model rankings sorted by average score (descending).
      Columns: Model, Size, Task1, Task2, Task3, Task4, Task5, Average.

  results.json
      Full results for all models including per-task scores and standard
      deviations. Useful for programmatic downstream analysis.

  average_scores.png
      Horizontal bar chart of each model's average score across all 5 tasks.
      Bars are colour-coded: red < 0.50, orange 0.50–0.75, green > 0.75.

  task_heatmap.png
      Heatmap with models as rows and tasks as columns.
      Colour scale: red = 0.0, yellow = 0.5, green = 1.0 (RdYlGn).
      Cell values show the exact score.

  efficiency_scatter.png
      Scatter plot of average score (x) vs Task 4 score (y).
      Task 4 is the escalation-dilemma task and tests protocol adherence.
      Each point is labelled with the short model name.

  difficulty_profile.png
      Line chart showing mean score per task across all 8 models with error
      bars (±1 std). Reveals which tasks are hardest / easiest on average.

  summary.txt
      Plain-text summary: best/worst model, hardest/easiest task, and any
      model that scored 1.0 on every task.

  README.txt
      This file.

Tasks:
  Task 1 — Basic eligibility check
  Task 2 — Multi-criterion scheme selection
  Task 3 — Income-threshold boundary case
  Task 4 — Escalation dilemma (employment data conflict)
  Task 5 — Document-verification age conflict
