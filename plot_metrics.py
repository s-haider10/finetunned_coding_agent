"""Plot training metrics from a metrics.jsonl log file.

Usage:
    uv run python plot_metrics.py                          # auto-finds latest run
    uv run python plot_metrics.py code-rl-logs/2026_02_13-02_19_08/metrics.jsonl
"""
import json
import sys
import os
import glob


def find_latest_metrics() -> str:
    log_dir = os.path.expanduser("~/code-rl-logs")
    runs = sorted(glob.glob(os.path.join(log_dir, "*", "metrics.jsonl")))
    if not runs:
        print(f"No metrics.jsonl found in {log_dir}/*/")
        sys.exit(1)
    return runs[-1]


def load_metrics(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def print_summary(rows: list[dict], path: str):
    steps = [r.get("step", i) for i, r in enumerate(rows)]
    print(f"\n{'='*60}")
    print(f"  Metrics from: {path}")
    print(f"  Steps: {len(rows)} (step {steps[0]} → {steps[-1]})")
    print(f"{'='*60}\n")

    # Key metrics to track
    keys = [
        ("mean_correct",   "Pass Rate  ", "higher=better"),
        ("mean_reward",    "Mean Reward ", "higher=better"),
        ("format_rate",    "Format Rate ", "should stay ~1.0"),
        ("loss:sum",       "Loss (sum)  ", "lower=better"),
        ("n_datums",       "Datums      ", "higher=more learning"),
        ("groups_skipped", "Groups Skip ", "lower=better"),
        ("time/step_s",           "Step Time(s)", "lower=better"),
        ("time/sync_s",           "Sync Time(s)", "lower=better"),
        ("time/sample_s",         "Sample T.(s)", "lower=better"),
        ("time/score_s",          "Score T. (s)", "lower=better"),
        ("time/logprobs_s",       "Logprob  (s)", "lower=better"),
        ("time/train_s",          "Train T. (s)", "lower=better"),
    ]

    # Eval metrics
    eval_keys = [
        ("eval/pass_at_1",   "Eval Pass@1 ", "higher=better"),
        ("eval/format_rate", "Eval Format ", "should stay ~1.0"),
    ]

    for key, label, hint in keys:
        vals = [r.get(key) for r in rows if key in r]
        if not vals:
            continue
        first, last = vals[0], vals[-1]
        best = max(vals) if "higher" in hint else min(vals)
        trend = "↑" if last > first else ("↓" if last < first else "→")
        print(f"  {label} │ first: {first:>10.4f} │ last: {last:>10.4f} │ best: {best:>10.4f} │ {trend}  ({hint})")

    # Check for eval metrics
    has_eval = any(k in r for r in rows for k, _, _ in eval_keys)
    if has_eval:
        print()
        for key, label, hint in eval_keys:
            vals = [(r.get("step", i), r[key]) for i, r in enumerate(rows) if key in r]
            if not vals:
                continue
            print(f"  {label} │ " + " │ ".join(f"step {s}: {v:.4f}" for s, v in vals))

    print()


def plot(rows: list[dict], path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (install matplotlib for plots: uv pip install matplotlib)")
        print("  Showing text summary only.\n")
        return

    steps = [r.get("step", i) for i, r in enumerate(rows)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training Metrics\n{os.path.basename(os.path.dirname(path))}", fontsize=13)

    # 1. Pass rate + format rate
    ax = axes[0, 0]
    ax.plot(steps, [r.get("mean_correct", 0) for r in rows], "b-o", markersize=3, label="mean_correct")
    ax.plot(steps, [r.get("format_rate", 0) for r in rows], "g--", alpha=0.6, label="format_rate")
    eval_steps = [r.get("step", i) for i, r in enumerate(rows) if "eval/pass_at_1" in r]
    eval_vals = [r["eval/pass_at_1"] for r in rows if "eval/pass_at_1" in r]
    if eval_vals:
        ax.plot(eval_steps, eval_vals, "r-s", markersize=5, label="eval/pass_at_1")
    ax.set_ylabel("Rate")
    ax.set_title("Pass Rate & Format Rate")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 2. Loss
    ax = axes[0, 1]
    loss_vals = [r.get("loss:sum") for r in rows]
    if any(v is not None for v in loss_vals):
        valid = [(s, v) for s, v in zip(steps, loss_vals) if v is not None]
        ax.plot([s for s, _ in valid], [v for _, v in valid], "r-o", markersize=3)
    ax.set_ylabel("Loss (sum)")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # 3. Datums per step
    ax = axes[1, 0]
    ax.bar(steps, [r.get("n_datums", 0) for r in rows], color="steelblue", alpha=0.7, label="datums trained")
    max_datums = max(r.get("n_datums", 0) for r in rows) if rows else 128
    ax.axhline(y=128, color="gray", linestyle="--", alpha=0.5, label="batch_size (128)")
    ax.set_ylabel("Count")
    ax.set_title("Datums per Step (non-degenerate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Groups skipped
    ax = axes[1, 1]
    ax.bar(steps, [r.get("groups_skipped", 0) for r in rows], color="salmon", alpha=0.7)
    groups_total = rows[0].get("groups_total", 8) if rows else 8
    ax.axhline(y=groups_total, color="gray", linestyle="--", alpha=0.5, label=f"total ({groups_total})")
    ax.set_ylabel("Count")
    ax.set_xlabel("Step")
    ax.set_title("Degenerate Groups Skipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = path.replace("metrics.jsonl", "training_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"  Plot saved to: {out_path}")
    plt.show()


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_latest_metrics()

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    rows = load_metrics(path)
    if not rows:
        print("No data in metrics file yet.")
        sys.exit(1)

    print_summary(rows, path)
    plot(rows, path)


if __name__ == "__main__":
    main()
