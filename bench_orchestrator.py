"""bench_orchestrator.py — measure CPU+GPU worker mixes on REAL chunks.

Runs the orchestrator on a fixed subset of code chunks with a different worker
plan each time and reports total throughput + average GPU/CPU utilization from
the run's system_stats.csv.  Use to tune the planner defaults.

Usage:  uv run python bench_orchestrator.py --jobs 200
"""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import orjson
from orchestrator import run_llm_completions, WorkerSpec, probe

MODEL = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
TMPL = ("You are an expert programmer. Given the following {l} code, write a clear, "
        "concise instruction that a human would write to ask for exactly this code. "
        "1-3 sentences, specific, mention language. Output ONLY the instruction.\n\n"
        "CODE:\n```{l}\n{c}\n```\n\nINSTRUCTION:")

CONFIGS = [
    ("gpu-only",        [WorkerSpec(kind="gpu", name="gpu0", n_threads=2, n_parallel=8, ctx_per_seq=2048)]),
    ("gpu+1cpu",        [WorkerSpec(kind="gpu", name="gpu0", n_threads=2, n_parallel=8, ctx_per_seq=2048),
                         WorkerSpec(kind="cpu", name="cpu0", n_threads=2, ctx_per_seq=2048)]),
    ("gpu(6t)+1cpu(4t)",[WorkerSpec(kind="gpu", name="gpu0", n_threads=6, n_parallel=8, ctx_per_seq=2048),
                         WorkerSpec(kind="cpu", name="cpu0", n_threads=4, ctx_per_seq=2048)]),
    ("gpu(8t)+1cpu(2t)",[WorkerSpec(kind="gpu", name="gpu0", n_threads=8, n_parallel=8, ctx_per_seq=2048),
                         WorkerSpec(kind="cpu", name="cpu0", n_threads=2, ctx_per_seq=2048)]),
]

def avg_util(run_dir: Path) -> dict:
    p = run_dir / "system_stats.csv"
    gpu, cpu, n = [], [], 0
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                n += 1
                if row.get("gpu_util_pct") not in ("", None):
                    gpu.append(float(row["gpu_util_pct"]))
                cpu.append(float(row["cpu_pct"]))
    return {
        "avg_gpu_util": round(sum(gpu)/len(gpu), 1) if gpu else 0,
        "avg_cpu_pct": round(sum(cpu)/len(cpu), 1) if cpu else 0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=200)
    ap.add_argument("--configs", type=str, default="")
    args = ap.parse_args()

    chunks = []
    with open("data/chunks/code_chunks_raw.jsonl", "rb") as f:
        for line in f:
            if len(chunks) >= args.jobs:
                break
            chunks.append(orjson.loads(line))
    print(f"Using {len(chunks)} real chunks")

    jobs = [{"prompt": TMPL.format(l=c["language"], c=c["code"][:2000]),
             "max_tokens": 128, "temperature": 0.7, "stop": ["\n\n"]} for c in chunks]

    print(f"\n{'config':<16}{'jobs':>5}{'elapsed_s':>10}{'jobs/s':>9}{'gpu%':>6}{'cpu%':>6}{'ok%':>6}")
    print("-" * 58)
    for label, specs in CONFIGS:
        if args.configs and label not in [x.strip() for x in args.configs.split(",")]:
            continue
        out = Path(f"/tmp/bench_{label}.jsonl")
        out.unlink(missing_ok=True)
        stats = run_llm_completions(jobs, out, specs, MODEL, force_regen=True)
        util = avg_util(Path(stats["run_dir"]))
        print(f"{label:<16}{stats['jobs']:>5}{stats['elapsed_s']:>10.1f}{stats['jobs_per_s']:>9.3f}"
              f"{util['avg_gpu_util']:>6}{util['avg_cpu_pct']:>6}{stats['ok_pct']:>6}")
        print(f"    per-worker: {json.dumps({k: v['jobs_per_s'] for k, v in stats['workers'].items()})}")

if __name__ == "__main__":
    main()
