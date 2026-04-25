import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_configs(config_dir: Path):
    return sorted(config_dir.glob("**/c*.yaml"))


def run_one(args, config_path: Path, output_dir: Path):
    run_name = f"{config_path.parent.name}_{config_path.stem}"
    trace_path = output_dir / "jsonl" / f"{run_name}.jsonl"
    metric_path = output_dir / "tensorboard" / run_name
    log_path = output_dir / "logs" / f"{run_name}.log"

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "mlora_train.py",
        "--base_model",
        args.base_model,
        "--config",
        str(config_path),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--metric_file",
        str(metric_path),
        "--log_file",
        str(log_path),
        "--trace_jsonl",
        str(trace_path),
    ]

    if args.trace:
        cmd.append("--trace")

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    started = time.time()
    print(f"[RUN] {run_name}")
    print(" ".join(cmd))

    result = subprocess.run(cmd, env=env)
    finished = time.time()

    summary = {
        "run_name": run_name,
        "config": str(config_path),
        "trace_jsonl": str(trace_path),
        "metric_dir": str(metric_path),
        "log_file": str(log_path),
        "returncode": result.returncode,
        "started_at": started,
        "finished_at": finished,
        "duration_sec": finished - started,
        "command": cmd,
    }

    summary_path = output_dir / "summaries" / f"{run_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if result.returncode != 0 and not args.continue_on_error:
        raise SystemExit(result.returncode)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run generated mLoRA validation configs without prompts."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--config-dir", default="validation_runs/configs")
    parser.add_argument("--output-dir", default="validation_runs/results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="int8")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    configs = find_configs(config_dir)

    if len(configs) == 0:
        raise SystemExit(f"No configs found under {config_dir}")

    all_summaries = []
    for config_path in configs:
        all_summaries.append(run_one(args, config_path, output_dir))

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(all_summaries, fp, indent=2)

    print(f"[DONE] wrote {manifest_path}")


if __name__ == "__main__":
    main()
