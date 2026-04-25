import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import yaml


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2)


def write_yaml(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(value, fp, sort_keys=False)


def make_dataset(path: Path, size: int, seq_words: int):
    rows = []
    payload = " ".join(["mLoRA"] * seq_words)
    for idx in range(size):
        rows.append(
            {
                "instruction": f"Explain adapter training case {idx}.",
                "input": payload,
                "output": f"This is synthetic validation output {idx}. {payload}",
            }
        )
    write_json(path, rows)


def make_prompt(path: Path):
    write_yaml(
        path,
        {
            "template": (
                "### Instruction:\n"
                "{{ data_point['instruction'] }}\n\n"
                "### Input:\n"
                "{{ data_point['input'] }}\n\n"
                "### Response:\n"
                "{{ data_point['output'] }}"
            )
        },
    )


def adapter_config(name: str, rank: int, output_dir: str) -> Dict:
    return {
        "name": name,
        "type": "lora",
        "path": output_dir,
        "optimizer": "adamw",
        "lr": 3e-4,
        "r": rank,
        "alpha": rank * 2,
        "dropout": 0.05,
        "target_modules": {
            "q_proj": True,
            "k_proj": True,
            "v_proj": True,
            "o_proj": True,
            "gate_proj": False,
            "down_proj": False,
            "up_proj": False,
        },
    }


def task_config(
    task_name: str,
    adapter_name: str,
    dataset_name: str,
    batch_size: int,
    mini_batch_size: int,
    num_epochs: int,
    cutoff_len: int,
    save_step: int,
) -> Dict:
    return {
        "type": "train",
        "name": task_name,
        "adapter": adapter_name,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "mini_batch_size": mini_batch_size,
        "num_epochs": num_epochs,
        "cutoff_len": cutoff_len,
        "save_step": save_step,
    }


def build_workload(args, workload: str, concurrency: int):
    out_dir = Path(args.out_dir).resolve()
    workload_dir = out_dir / workload
    data_dir = workload_dir / "data"
    prompt_path = workload_dir / "prompt.yaml"
    make_prompt(prompt_path)

    ranks = [args.rank] * args.tasks
    seq_words = [args.seq_words] * args.tasks
    mini_batch_sizes = [args.mini_batch_size] * args.tasks

    if workload == "rank_heterogeneous":
        ranks = parse_int_list(args.hetero_ranks)
    elif workload == "seq_heterogeneous":
        seq_words = parse_int_list(args.hetero_seq_words)
    elif workload == "batch_heterogeneous":
        mini_batch_sizes = parse_int_list(args.hetero_mini_batch_sizes)

    def expand(values: List[int]) -> List[int]:
        if len(values) >= args.tasks:
            return values[: args.tasks]
        return [values[idx % len(values)] for idx in range(args.tasks)]

    ranks = expand(ranks)
    seq_words = expand(seq_words)
    mini_batch_sizes = expand(mini_batch_sizes)

    datasets = []
    adapters = []
    tasks = []

    for idx in range(args.tasks):
        dataset_name = f"{workload}_data_{idx}"
        data_path = data_dir / f"{dataset_name}.json"
        make_dataset(data_path, args.data_size, seq_words[idx])

        adapter_name = f"{workload}_adapter_{idx}"
        task_name = f"{workload}_task_{idx}"
        adapter_out = str((workload_dir / "adapters" / adapter_name).resolve())

        datasets.append(
            {
                "name": dataset_name,
                "data": str(data_path),
                "prompt": str(prompt_path),
                "prompt_type": "instruction",
                "preprocess": "default",
            }
        )
        adapters.append(adapter_config(adapter_name, ranks[idx], adapter_out))
        task_batch_size = args.batch_size
        if task_batch_size < mini_batch_sizes[idx]:
            task_batch_size = mini_batch_sizes[idx]
        if task_batch_size % mini_batch_sizes[idx] != 0:
            task_batch_size = mini_batch_sizes[idx]
        tasks.append(
            task_config(
                task_name,
                adapter_name,
                dataset_name,
                task_batch_size,
                mini_batch_sizes[idx],
                args.num_epochs,
                args.cutoff_len,
                args.save_step,
            )
        )

    config = {
        "dispatcher": {"name": "default", "concurrency_num": concurrency},
        "datasets": datasets,
        "adapters": adapters,
        "tasks": tasks,
    }

    config_path = workload_dir / f"c{concurrency}.yaml"
    write_yaml(config_path, config)
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate noninteractive validation configs for mLoRA."
    )
    parser.add_argument("--out-dir", default="validation_runs/configs")
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["homogeneous", "rank_heterogeneous", "seq_heterogeneous"],
        choices=[
            "homogeneous",
            "rank_heterogeneous",
            "seq_heterogeneous",
            "batch_heterogeneous",
        ],
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--data-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--cutoff-len", type=int, default=256)
    parser.add_argument("--save-step", type=int, default=100000)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seq-words", type=int, default=64)
    parser.add_argument("--hetero-ranks", default="4,8,16,32")
    parser.add_argument("--hetero-seq-words", default="16,64,160,320")
    parser.add_argument("--hetero-mini-batch-sizes", default="1,2,4,8")
    args = parser.parse_args()

    generated = []
    for workload in args.workloads:
        for concurrency in args.concurrency:
            generated.append(build_workload(args, workload, concurrency))

    manifest = {
        "generated_configs": [str(path) for path in generated],
        "workloads": args.workloads,
        "concurrency": args.concurrency,
        "tasks": args.tasks,
        "data_size": args.data_size,
    }
    write_json(Path(args.out_dir).resolve() / "manifest.json", manifest)

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
