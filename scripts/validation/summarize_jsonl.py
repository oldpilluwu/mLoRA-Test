import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def load_records(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line == "":
                continue
            records.append(json.loads(line))
    return records


def summarize(path: Path):
    records = load_records(path)
    if len(records) == 0:
        return None

    totals = [row["timing_sec"]["total"] for row in records]
    forwards = [row["timing_sec"]["forward"] for row in records]
    backwards = [row["timing_sec"]["backward"] for row in records]
    max_mem = [row["memory_bytes"]["max_allocated"] for row in records]
    reserved = [row["memory_bytes"]["max_reserved"] for row in records]
    padding = [row["batch"]["padding_waste"] for row in records if "batch" in row]
    batch_sizes = [
        row["batch"]["combined_batch_size"] for row in records if "batch" in row
    ]
    token_lens = [row["batch"]["token_len"] for row in records if "batch" in row]
    running = [row["running_count"] for row in records]

    total_duration = sum(totals)
    total_useful_tokens = sum(
        row["batch"]["useful_tokens"] for row in records if "batch" in row
    )

    return {
        "trace_file": str(path),
        "iterations": len(records),
        "total_duration_sec": total_duration,
        "avg_iteration_sec": mean(totals),
        "avg_forward_sec": mean(forwards),
        "avg_backward_sec": mean(backwards),
        "avg_running_tasks": mean(running),
        "avg_batch_size": mean(batch_sizes) if batch_sizes else 0,
        "avg_token_len": mean(token_lens) if token_lens else 0,
        "avg_padding_waste": mean(padding) if padding else 0,
        "max_memory_allocated_bytes": max(max_mem),
        "max_memory_reserved_bytes": max(reserved),
        "useful_tokens_per_sec": (
            0 if total_duration == 0 else total_useful_tokens / total_duration
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize mLoRA JSONL traces.")
    parser.add_argument("--trace-dir", default="validation_runs/results/jsonl")
    parser.add_argument("--output", default="validation_runs/results/summary.csv")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    rows = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        row = summarize(path)
        if row is not None:
            rows.append(row)

    if len(rows) == 0:
        raise SystemExit(f"No JSONL trace rows found under {trace_dir}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(output)


if __name__ == "__main__":
    main()
