import os
import subprocess
import sys


def env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def split_words(value: str):
    return [item for item in value.split(" ") if item]


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    base_model = env("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    device = env("DEVICE", "cuda:0")
    precision = env("PRECISION", "int8")
    cuda_visible_devices = env("CUDA_VISIBLE_DEVICES", "0")
    out_root = env("OUT_ROOT", "validation_runs/stage2")

    tasks = env("TASKS", "32")
    data_size = env("DATA_SIZE", "128")
    num_epochs = env("NUM_EPOCHS", "1")
    batch_size = env("BATCH_SIZE", "4")
    mini_batch_size = env("MINI_BATCH_SIZE", "4")
    cutoff_len = env("CUTOFF_LEN", "1024")
    rank = env("RANK", "8")

    concurrency = split_words(env("CONCURRENCY", "1 2 4 8 16 32"))
    workloads = split_words(
        env(
            "WORKLOADS",
            "homogeneous rank_heterogeneous seq_heterogeneous batch_heterogeneous",
        )
    )
    hetero_ranks = env("HETERO_RANKS", "4,8,16,32,64,128")
    hetero_seq_words = env("HETERO_SEQ_WORDS", "16,64,256,768")
    hetero_mini_batch_sizes = env("HETERO_MINI_BATCH_SIZES", "1,2,4,8")

    run(
        [
            sys.executable,
            "scripts/validation/generate_validation_configs.py",
            "--out-dir",
            f"{out_root}/configs",
            "--workloads",
            *workloads,
            "--concurrency",
            *concurrency,
            "--tasks",
            tasks,
            "--data-size",
            data_size,
            "--num-epochs",
            num_epochs,
            "--batch-size",
            batch_size,
            "--mini-batch-size",
            mini_batch_size,
            "--cutoff-len",
            cutoff_len,
            "--rank",
            rank,
            "--seq-words",
            "64",
            "--hetero-ranks",
            hetero_ranks,
            "--hetero-seq-words",
            hetero_seq_words,
            "--hetero-mini-batch-sizes",
            hetero_mini_batch_sizes,
        ]
    )

    run(
        [
            sys.executable,
            "scripts/validation/run_fixed_baselines.py",
            "--base-model",
            base_model,
            "--config-dir",
            f"{out_root}/configs",
            "--output-dir",
            f"{out_root}/results",
            "--device",
            device,
            "--precision",
            precision,
            "--cuda-visible-devices",
            cuda_visible_devices,
            "--continue-on-error",
        ]
    )

    run(
        [
            sys.executable,
            "scripts/validation/summarize_jsonl.py",
            "--trace-dir",
            f"{out_root}/results/jsonl",
            "--output",
            f"{out_root}/results/summary.csv",
        ]
    )

    print("Stage 2 sweep complete.")
    print(f"Summary: {out_root}/results/summary.csv")


if __name__ == "__main__":
    main()
