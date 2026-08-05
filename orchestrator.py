"""
orchestrator.py — barebones resource-aware orchestrator for LLM-heavy jobs
=========================================================================

The multidomain orchestrator is the shared execution runtime for every later
stage (coding agent, translation, TTS/STT, image/video, search, ...).  This
first incarnation is deliberately minimal but already owns the three things
every stage needs:

  1. Hardware probing      -> what CPU/GPU/RAM is actually available right now
  2. Resource planning     -> how to split a job across GPU *and* CPU so that
                              no bandwidth sits idle ("use every last drop")
  3. Distributed execution -> run the job over a pool of spawn-isolated
                              workers while a monitor logs real system stats
                              to a per-run log file

Why spawn-isolated workers
--------------------------
The GPU path loads LM Studio's bundled libllama.so (ctypes, RTLD_GLOBAL) and
the CPU path loads llama-cpp-python, which bundles its *own* llama.cpp.  The
two have the same exported symbols, so they can never coexist in one process.
By giving each a private process they run side-by-side and the GPU's VRAM
bandwidth and the CPU's DDR bandwidth are used simultaneously.

Why both, not just GPU
----------------------
On a laptop RTX 3050 (4 GB, ~192 GB/s VRAM) generation is memory-bandwidth
bound.  The CPU has ~50 GB/s of DDR bandwidth that stays completely unused
while the GPU grinds through 58k chunks.  Two memory buses, one job: use both.

The contract is deliberately tiny:

    from orchestrator import probe, plan_llm_resources, run_llm_completions

    res   = probe()
    specs = plan_llm_resources(res, model_path)
    jobs  = [{"prompt": ...} for chunk in chunks]     # request dicts
    stats = run_llm_completions(jobs, out_path, specs, model_path, logger=log)

`run_llm_completions` writes one JSON record per input index (same order as
`jobs`), retries failed generations internally, and logs a per-run system
stats CSV + human log under `data/logs/<run_id>/`.

Tiger Style: every resource acquisition is explicit, every worker spec is a
dataclass, and every path is asserted before use.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:  # pragma: no cover - fail explicit, tell the user the fix
    raise ImportError("Missing `psutil`. Install: uv pip install psutil")

# ─────────────────────────────────────────────────────────────────────────────
# Logging (per-run, deterministic destination)
# ─────────────────────────────────────────────────────────────────────────────

_LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING}


def _fresh_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with no inherited handlers (deterministic output)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers = []
    return logger


def _attach_file(logger: logging.Logger, path: Path, level: int = logging.INFO) -> None:
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(level)
    logger.addHandler(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware probe — the source of truth for every planning decision
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CpuInfo:
    physical: int = 0
    logical: int = 0
    base_mhz: Optional[float] = None
    turbo_mhz: Optional[float] = None
    vendor: str = ""
    model: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GpuInfo:
    present: bool = False
    index: int = -1
    name: str = ""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    vram_used_mb: int = 0
    util_pct: float = 0.0
    power_w: float = 0.0
    temp_c: float = 0.0
    cuda_version: str = ""
    lmstudio_backend: Optional[str] = None  # path to libllama.so backend dir
    lmstudio_vendor: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RamInfo:
    total_mb: int = 0
    available_mb: int = 0
    used_mb: int = 0
    swap_mb: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemResources:
    cpu: CpuInfo = field(default_factory=CpuInfo)
    ram: RamInfo = field(default_factory=RamInfo)
    gpu: GpuInfo = field(default_factory=GpuInfo)
    disk_free_mb: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _probe_cpu() -> CpuInfo:
    info = CpuInfo()
    try:
        info.physical = psutil.cpu_count(logical=False) or 0
        info.logical = psutil.cpu_count(logical=True) or 0
    except Exception:
        pass
    # Prefer lscpu for vendor/model/turbo facts when available.
    try:
        out = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Model name:"):
                info.model = line.split(":", 1)[1].strip()
            elif line.startswith("Vendor ID:"):
                info.vendor = line.split(":", 1)[1].strip()
            elif line.startswith("CPU max MHz:"):
                info.turbo_mhz = float(line.split(":", 1)[1].strip().split()[0])
            elif line.startswith("CPU min MHz:"):
                info.base_mhz = float(line.split(":", 1)[1].strip().split()[0])
    except Exception:
        pass
    if not info.base_mhz:
        try:
            import cpuinfo  # optional
            info.base_mhz = cpuinfo.get_cpu_info().get("hz_advertised_friendly", "")
            info.model = cpuinfo.get_cpu_info().get("brand_raw", "")
        except Exception:
            pass
    return info


def _nvidia_smi_query() -> Optional[List[Dict[str, Any]]]:
    """Return one dict per GPU from nvidia-smi (None if no NVIDIA GPU)."""
    if not shutil.which("nvidia-smi"):
        return None
    q = "index,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu", q, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    rows: List[Dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        def _f(x: str, default: float = 0.0) -> float:
            try:
                return float(x.split()[0]) if x else default
            except ValueError:
                return default
        rows.append({
            "index": int(parts[0]),
            "name": parts[1],
            "vram_total_mb": int(_f(parts[2])),
            "vram_used_mb": int(_f(parts[3])),
            "vram_free_mb": int(_f(parts[4])),
            "util_pct": _f(parts[5]),
            "power_w": _f(parts[6]),
            "temp_c": _f(parts[7]),
        })
    return rows or None


def _cuda_version() -> str:
    if not shutil.which("nvcc"):
        return ""
    try:
        out = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            if "release" in line:
                return line.split("release")[-1].strip()
    except Exception:
        pass
    return ""


def _lmstudio_backend() -> Optional[Dict[str, str]]:
    """Locate LM Studio's bundled llama.cpp CUDA backend (if any)."""
    try:
        from llm_backend import find_lmstudio_backend
        info = find_lmstudio_backend()
        if info is None:
            return None
        return {
            "backend": str(info["backend_dir"]),
            "vendor": str(info["vendor_dir"]) if info.get("vendor_dir") else "",
        }
    except Exception:
        return None


def probe() -> SystemResources:
    """Gather the live system state into one immutable-ish snapshot."""
    res = SystemResources()
    res.cpu = _probe_cpu()
    try:
        vm = psutil.virtual_memory()
        res.ram.total_mb = vm.total // (1024 * 1024)
        res.ram.available_mb = vm.available // (1024 * 1024)
        res.ram.used_mb = vm.used // (1024 * 1024)
        res.ram.swap_mb = psutil.swap_memory().total // (1024 * 1024)
    except Exception:
        pass
    try:
        res.disk_free_mb = psutil.disk_usage("/").free // (1024 * 1024)
    except Exception:
        pass

    rows = _nvidia_smi_query()
    lm = _lmstudio_backend()
    if rows:
        g = rows[0]
        res.gpu = GpuInfo(
            present=True,
            index=g["index"],
            name=g["name"],
            vram_total_mb=g["vram_total_mb"],
            vram_used_mb=g["vram_used_mb"],
            vram_free_mb=g["vram_free_mb"],
            util_pct=g["util_pct"],
            power_w=g["power_w"],
            temp_c=g["temp_c"],
            cuda_version=_cuda_version(),
            lmstudio_backend=(lm or {}).get("backend"),
            lmstudio_vendor=(lm or {}).get("vendor"),
        )
    elif lm:
        res.gpu = GpuInfo(present=True, name="lmstudio-backend", lmstudio_backend=lm["backend"])
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Resource planning — turn a job + hardware snapshot into a worker set
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WorkerSpec:
    """One executable unit of the job."""

    kind: str          # 'gpu' | 'cpu'
    name: str          # unique worker label for logs/stats
    n_threads: int = 0 # CPU threads owned by this worker
    n_parallel: int = 1# sequences decoded in one forward pass (GPU batching)
    ctx_per_seq: int = 1536
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    stop: List[str] = field(default_factory=lambda: ["\n\n"])
    attempts: int = 3  # internal retries per job inside the worker

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def plan_llm_resources(
    res: SystemResources,
    model_path: str,
    ctx_per_seq: int = 1536,
    max_parallel: int = 12,
    cpu_threads_each: int = 2,
    os_reserve_threads: int = 2,
    gpu_threads: int = 2,
    vram_headroom_pct: float = 0.85,
    ram_headroom_pct: float = 0.7,
    max_cpu_workers: Optional[int] = None,
    max_vram_mb: Optional[int] = None,
) -> List[WorkerSpec]:
    """Decide how many GPU + CPU workers this machine should run.

    The job is LLM token generation: memory-bandwidth bound on both devices.
    We give the GPU a batched worker sized to fit free VRAM, then use every
    remaining CPU thread (minus OS/interpreter reserve) for CPU workers.

    This is a *starting* plan — the runner reports real throughput per worker
    and you re-tune with the CLI flags / env vars below.
    """
    try:
        weights_mib = Path(model_path).stat().st_size / (1024 * 1024)
    except OSError:
        weights_mib = 1000.0
    model_path = str(model_path)

    specs: List[WorkerSpec] = []
    threads_used = 0

    gpu = res.gpu
    if gpu.present and gpu.lmstudio_backend:
        free_mb = gpu.vram_free_mb if gpu.vram_free_mb > 0 else gpu.vram_total_mb
        if max_vram_mb:
            free_mb = min(free_mb, max_vram_mb)
        budget_mb = free_mb * vram_headroom_pct
        # Digestible estimate of VRAM per sequence (KV + compute + slack).  This
        # is deliberately on the lean side so we reach a high n_parallel (the
        # GPU reuses one weight read across the whole batch).
        per_seq_mb = max(60, int(ctx_per_seq * 0.06) + 40)
        n_parallel = _clamp(int((budget_mb - weights_mib) // per_seq_mb), 1, max_parallel)
        specs.append(WorkerSpec(
            kind="gpu",
            name="gpu0",
            n_threads=gpu_threads,
            n_parallel=n_parallel,
            ctx_per_seq=ctx_per_seq,
        ))
        threads_used += gpu_threads

    avail_threads = max(0, res.cpu.logical - threads_used - os_reserve_threads)
    if cpu_threads_each > 0:
        cpu_limit = max_cpu_workers if max_cpu_workers is not None else (avail_threads // cpu_threads_each)
        # RAM check: each CPU worker holds its own ~1x weights copy + KV + python.
        ram_limit = int((res.ram.available_mb * ram_headroom_pct) // (weights_mib * 1.6))
        n_cpu = _clamp(cpu_limit, 0, ram_limit)
        for i in range(n_cpu):
            specs.append(WorkerSpec(kind="cpu", name=f"cpu{i}", n_threads=cpu_threads_each))

    return specs


def calibrate_plan(
    res: SystemResources,
    model_path: str,
    sample_jobs: List[Dict[str, Any]],
    ctx_per_seq: int = 2048,
    logger: Optional[logging.Logger] = None,
) -> List[WorkerSpec]:
    """Measure a few candidate worker mixes on a real sample; return the best.

    On a laptop GPU the CPU's DDR is far slower than VRAM bandwidth *and* CPU
    workers starve the GPU path of CPU time (the GPU's llama.cpp still needs
    threads for tokenization/batch scheduling).  So a naive "add CPU workers"
    split can be *slower* than GPU-only.  Rather than hard-code a policy, run
    gpu-only / +1cpu / +2cpu on a sample of the actual job and keep whichever
    wins — this is why the orchestrator exists, and it adapts per-machine.

    Each candidate is one full-fleet run over `sample_jobs`; `sample_jobs`
    should be a modest slice (tens) of the real workload.  Startup (model load,
    ~2-15 s) is amortised over every candidate, so use a warm sample that is
    large enough that steady-state throughput dominates.
    """
    if not sample_jobs:
        return plan_llm_resources(res, model_path, ctx_per_seq=ctx_per_seq)

    candidates = [
        ("gpu-only", plan_llm_resources(res, model_path, ctx_per_seq=ctx_per_seq, max_cpu_workers=0)),
        ("gpu+1cpu", plan_llm_resources(res, model_path, ctx_per_seq=ctx_per_seq, max_cpu_workers=1)),
        ("gpu+2cpu", plan_llm_resources(res, model_path, ctx_per_seq=ctx_per_seq, max_cpu_workers=2)),
    ]

    best: Tuple[str, List[WorkerSpec], float] = ("none", [], 0.0)
    import tempfile
    for label, specs in candidates:
        if not specs:
            continue
        tmp = Path(tempfile.gettempdir()) / f"calib_{label.replace('+','_')}.jsonl"
        tmp.unlink(missing_ok=True)
        if logger:
            logger.info("  [calib] %s (%d workers) on %d sample jobs ...",
                        label, len(specs), len(sample_jobs))
        try:
            stats = run_llm_completions(
                sample_jobs, tmp, specs, model_path,
                run_dir=Path("data/logs/calib"), logger=logger, force_regen=True,
            )
        except Exception as exc:
            if logger:
                logger.warning("  [calib] %s failed: %s", label, exc)
            continue
        rate = stats["jobs_per_s"]
        if logger:
            logger.info("  [calib] %s → %.3f jobs/s", label, rate)
        if rate > best[2]:
            best = (label, specs, rate)

    if best[0] == "none":
        if logger:
            logger.warning("  [calib] all candidates failed — falling back to GPU-only plan")
        return plan_llm_resources(res, model_path, ctx_per_seq=ctx_per_seq, max_cpu_workers=0)
    if logger:
        logger.info("  [calib] WINNER: %s (%.3f jobs/s)", best[0], best[2])
    return best[1]


# ─────────────────────────────────────────────────────────────────────────────
# System monitor — sampled real stats during the run, written to per-run files
# ─────────────────────────────────────────────────────────────────────────────


class SystemMonitor:
    """Samples CPU/RAM/GPU every `interval` seconds; writes CSV + log lines.

    All samples for one run land in the run dir so you can compare runs and
    tune the worker plan with real numbers instead of guesses.
    """

    def __init__(self, run_dir: Path, logger: logging.Logger, interval: float = 1.0) -> None:
        self.run_dir = run_dir
        self.logger = logger
        self.interval = interval
        self.csv_path = run_dir / "system_stats.csv"
        self._stop = mp.Event() if mp.get_start_method() == "spawn" else __import__("threading").Event()
        self._thread: Any = None
        self._last_proc = psutil.cpu_percent(interval=None)

    def start(self) -> None:
        import threading
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.interval + 1)
            self._thread = None

    def _sample_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "t": round(time.time() - self._t0, 2),
            "cpu_pct": psutil.cpu_percent(interval=0.1),
            "cpu_logical": psutil.cpu_count(logical=True),
            "ram_used_mb": psutil.virtual_memory().used // (1024 * 1024),
            "ram_avail_mb": psutil.virtual_memory().available // (1024 * 1024),
        }
        gpus = _nvidia_smi_query()
        if gpus:
            g = gpus[0]
            row.update({
                "gpu_util_pct": g["util_pct"],
                "gpu_vram_used_mb": g["vram_used_mb"],
                "gpu_vram_free_mb": g["vram_free_mb"],
                "gpu_power_w": g["power_w"],
                "gpu_temp_c": g["temp_c"],
            })
        return row

    def _loop(self) -> None:
        self._t0 = time.time()
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "t", "cpu_pct", "cpu_logical", "ram_used_mb", "ram_avail_mb",
                "gpu_util_pct", "gpu_vram_used_mb", "gpu_vram_free_mb",
                "gpu_power_w", "gpu_temp_c",
            ])
            writer.writeheader()
            last_log = 0.0
            while not self._stop.is_set():
                row = self._sample_row()
                writer.writerow(row)
                f.flush()
                now = row["t"]
                if now - last_log >= 10.0:
                    last_log = now
                    gpu_s = f"  GPU util {row.get('gpu_util_pct', 0):.0f}% vram {row.get('gpu_vram_used_mb', 0):.0f} MiB"
                    self.logger.info(
                        "  [sys] t=%5.0fs  cpu=%3.0f%%  ram=%d/%d MiB%s",
                        now, row["cpu_pct"], row["ram_used_mb"], row["ram_avail_mb"], gpu_s,
                    )
                self._stop.wait(self.interval)


# ─────────────────────────────────────────────────────────────────────────────
# Spawn-isolated workers (GPU batched, CPU single-seq)
# ─────────────────────────────────────────────────────────────────────────────


def _worker_logger(name: str, log_path: Path) -> logging.Logger:
    logger = _fresh_logger(f"worker:{name}")
    _attach_file(logger, log_path)
    return logger


def _redirect_stdio(log_path: str) -> None:
    """Route this worker's raw C-level stdout/stderr onto its own log file.

    llama.cpp/ggml print their verbose output (e.g. pages of
    `ggml_backend_cuda_graph_compute: CUDA graph warmup ...` and the
    `load_tensors:` tensor dump) straight to fds 1/2 via C `printf`/`fprintf`,
    bypassing Python's logging. Those fds are inherited from the parent, so they
    currently flood the user's terminal while every *real* progress line goes to
    the run log. We dup 1 and 2 onto the worker log file: the terminal stays
    readable and the output is still captured for diagnosis.
    """
    try:
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    except OSError:
        pass


def _fmt_eta(secs: float) -> str:
    secs = int(max(0.0, secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _run_gpu_worker(
    model_path: str,
    spec: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    work_q: "mp.Queue",
    result_q: "mp.Queue",
    log_path: str,
) -> None:
    """GPU worker: LM Studio llama.cpp batched decoding (n_parallel/forward pass)."""
    logger = _worker_logger(spec["name"], Path(log_path))
    _redirect_stdio(log_path)
    from llm_backend import BatchedLlama
    llm: Any = None
    try:
        llm = BatchedLlama(
            model_path,
            n_parallel=spec["n_parallel"],
            ctx_per_seq=spec["ctx_per_seq"],
            n_threads=spec["n_threads"],
            logger=logger,
        )
        llm.start()
        logger.info("worker %s ready (n_parallel=%d ctx=%d threads=%d)",
                    spec["name"], spec["n_parallel"], spec["ctx_per_seq"], spec["n_threads"])
    except Exception as exc:
        logger.error("worker %s failed to start: %s", spec["name"], exc)
        # Drain so the parent does not hang: emit None for every job we own.
        while True:
            try:
                item = work_q.get_nowait()
            except Exception:
                try:
                    item = work_q.get(timeout=0.5)
                except Exception:
                    return
            if item == "STOP":
                return
            idx, _attempts = item
            try:
                result_q.put((spec["name"], idx, None))
            except Exception:
                return
        return

    req_fields = ("max_tokens", "temperature", "top_p", "stop")
    try:
        while True:
            batch: List[Tuple[int, int, Dict[str, Any]]] = []
            stop = False
            # Fill up to n_parallel from the queue without blocking.  STOP must
            # not drop the items already collected in `batch` — process them
            # first, then exit.
            while len(batch) < spec["n_parallel"]:
                try:
                    item = work_q.get_nowait()
                except Exception:
                    break
                if item == "STOP":
                    stop = True
                    break
                idx, attempts = item
                req = {
                    k: v for k, v in jobs[idx].items()
                }
                batch.append((idx, attempts, req))
            if not batch:
                if stop:
                    return
                try:
                    item = work_q.get(timeout=2.0)
                except Exception:
                    continue
                if item == "STOP":
                    return
                idx, attempts = item
                req = {k: v for k, v in jobs[idx].items()}
                batch.append((idx, attempts, req))

            for attempt in range(spec["attempts"]):
                reqs = [r for _, _, r in batch]
                texts = llm.complete_batch(reqs)
                pending: List[Tuple[int, Dict[str, Any]]] = []
                for (idx, _a, _r), text in zip(batch, texts):
                    if text is None:
                        pending.append((idx, _r))
                    else:
                        try:
                            result_q.put((spec["name"], idx, text))
                        except Exception:
                            pass
                if not pending or attempt == spec["attempts"] - 1:
                    for idx, _r in pending:
                        try:
                            result_q.put((spec["name"], idx, None))
                        except Exception:
                            pass
                    break
                batch = [(idx, spec["attempts"] - attempt - 1, r) for idx, r in pending]
            if stop:
                return
    except Exception as exc:
        logger.error("worker %s crashed: %s", spec["name"], exc)
        raise
    finally:
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass


def _run_cpu_worker(
    model_path: str,
    spec: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    work_q: "mp.Queue",
    result_q: "mp.Queue",
    log_path: str,
) -> None:
    """CPU worker: llama.cpp CPU decode (n_gpu_layers=0) on its own threads.

    Each CPU worker is a separate process.  llama-cpp-python is typically built
    WITH CUDA, so left alone it allocates VRAM even at n_gpu_layers=0 (~0.5 GB
    of compute buffers per process) and OOMs the 4 GB GPU alongside the batched
    GPU worker.  We hide the GPU from this process (`CUDA_VISIBLE_DEVICES=""`)
    so it is pure CPU over the DDR bus and touches zero VRAM.

    Because CPU token generation is memory-bandwidth bound, a single DDR bus
    saturates quickly — the default is 2 threads per worker and the planner
    decides how many 2-thread slots to fill.
    """
    # Must be set before llama_cpp is imported by this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    logger = _worker_logger(spec["name"], Path(log_path))
    _redirect_stdio(log_path)
    llm: Any = None
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=str(model_path),
            n_ctx=spec["ctx_per_seq"],
            n_threads=spec["n_threads"],
            n_batch=512,
            n_gpu_layers=0,
            verbose=False,
        )
        logger.info("worker %s ready (cpu threads=%d ctx=%d)",
                    spec["name"], spec["n_threads"], spec["ctx_per_seq"])
    except Exception as exc:
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        logger.error("worker %s failed to start: %s", spec["name"], exc)
        while True:
            try:
                item = work_q.get(timeout=0.5)
            except Exception:
                return
            if item == "STOP":
                return
            idx, _attempts = item
            try:
                result_q.put((spec["name"], idx, None))
            except Exception:
                return
        return

    max_tokens = spec.get("max_tokens", 128)
    temperature = spec.get("temperature", 0.7)
    top_p = spec.get("top_p", 0.95)
    stop = spec.get("stop", ["\n\n"])
    try:
        while True:
            try:
                item = work_q.get(timeout=2.0)
            except Exception:
                continue
            if item == "STOP":
                return
            idx, attempts = item
            req = jobs[idx]
            text: Optional[str] = None
            for _ in range(spec["attempts"]):
                try:
                    if hasattr(llm, "complete_batch"):
                        texts = llm.complete_batch([req])
                        text = texts[0] if texts else None
                        if text:
                            break
                        text = None
                    else:
                        resp = llm(
                            req["prompt"],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stop=stop or None,
                        )
                        text = resp["choices"][0]["text"].strip()
                        if text:
                            break
                        text = None
                except Exception as exc:
                    logger.debug("cpu retry: %s", exc)
                    text = None
            try:
                result_q.put((spec["name"], idx, text))
            except Exception:
                pass
    except Exception as exc:
        logger.error("worker %s crashed: %s", spec["name"], exc)
        raise


_WORKER_FUNCS = {"gpu": _run_gpu_worker, "cpu": _run_cpu_worker}


# ─────────────────────────────────────────────────────────────────────────────
# Distributed run
# ─────────────────────────────────────────────────────────────────────────────


def _make_run_dir(logs_dir: Path) -> Path:
    run_id = uuid.uuid4().hex[:8]
    run_dir = logs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_llm_completions(
    jobs: List[Dict[str, Any]],
    out_path: Path,
    specs: List[WorkerSpec],
    model_path: str,
    run_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    results_only: bool = False,
    force_regen: bool = False,
    interval: float = 1.0,
) -> Dict[str, Any]:
    """Distribute `jobs` across `specs` (GPU + CPU workers) and write results.

    Args:
        jobs: request dicts in input order — {"prompt", "max_tokens", ...}.
        out_path: JSONL output; one JSON record per job in `jobs` order
            (index 0 → first line).  Records are the request dict merged with
            {"_idx": i, "_worker": name, "_text": text or None}.  If the file
            already has `len(jobs)` lines and force_regen is False, it returns
            early without doing any work.
        specs: planned worker set from `plan_llm_resources`.
        model_path: GGUF path shared by all workers.
        run_dir: per-run log dir; auto-created under data/logs/<run_id>/.
        logger: host logger (extra handlers are attached for this run).

    Returns:
        stats dict with timing, per-worker counts and throughput.
    """
    if logger is None:
        logger = _fresh_logger("orchestrator")
    if run_dir is None:
        run_dir = _make_run_dir(Path("data/logs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # The orchestrator is the run's single writer for system+job stats, attached
    # to the same per-run folder requested even if the caller passed its own log.
    run_logger = _fresh_logger(f"orchestrator:{run_dir.name}", logging.INFO)
    _attach_file(run_logger, run_dir / "run.log")
    # Also stream to the terminal so the user sees live progress (the worker
    # llama output no longer floods it after _redirect_stdio). Single handler so
    # nothing is duplicated.
    run_logger.addHandler(logging.StreamHandler(sys.stderr))
    logger = run_logger

    total = len(jobs)
    logger.info("=" * 64)
    logger.info("Orchestrator run %s — %d jobs, %d workers",
                run_dir.name, total, len(specs))
    for s in specs:
        logger.info("  worker %-5s kind=%-3s threads=%-2d n_parallel=%-2d ctx=%d",
                    s.name, s.kind, s.n_threads, s.n_parallel, s.ctx_per_seq)

    # ── checkpoint: resume from any partial/finished output ─────────────────
    out_path = Path(out_path)
    partial = Path(str(out_path) + ".partial")
    import orjson as _orjson

    results: Dict[int, Dict[str, Any]] = {}
    for p in (partial, out_path):
        if not p.exists():
            continue
        for line in open(p, "rb"):
            if not line.strip():
                continue
            try:
                rec = _orjson.loads(line)
            except Exception:
                continue
            idx = rec.get("_idx")
            if idx is not None and 0 <= idx < total and idx not in results:
                results[idx] = rec

    if len(results) >= total and not force_regen:
        logger.info("  Output already complete (%d/%d records) — skipping.", len(results), total)
        return {"skipped": True, "run_dir": str(run_dir)}

    needed = [i for i in range(total) if i not in results]
    if needed:
        logger.info("  Resuming from checkpoint: %d/%d jobs already done.", len(results), total)

    monitor = SystemMonitor(run_dir, logger, interval=interval)
    t0 = time.monotonic()
    procs: List[mp.Process] = []
    partial_fh: Any = None
    per_worker: Dict[str, List[Tuple[float, int]]] = {s.name: [] for s in specs}
    started_at: Dict[str, float] = {}

    try:
        if needed:
            ctx = mp.get_context("spawn")
            # Work queue must be unbounded: we pre-fill it before workers start,
            # and a bounded queue would block the parent's put() once full.
            work_q: "mp.Queue" = ctx.Queue()
            result_q: "mp.Queue" = ctx.Queue(maxsize=max(256, total // 20))

            # Seed every pending job with the worker's configured retry budget.
            attempts = specs[0].attempts if specs else 3
            for idx in needed:
                work_q.put((idx, attempts))
            for _s in specs:
                work_q.put("STOP")

            for spec in specs:
                fn = _WORKER_FUNCS[spec.kind]
                p = ctx.Process(
                    target=fn,
                    args=(model_path, spec.as_dict(), jobs, work_q, result_q,
                          str(run_dir / f"worker_{spec.name}.log")),
                    daemon=True,
                )
                p.start()
                procs.append(p)

            monitor.start()
            logger.info("  Started %d workers + system monitor.", len(procs))

            # Stream every completed result to <out>.partial so a kill or crash
            # never loses work — the next call seeds from it and resumes.
            partial_fh = open(partial, "ab")

            # ── result collection ─────────────────────────────────────────
            # Progress is time-driven (not every N jobs) so the user always
            # sees it move; ETA covers only the still-pending work.
            progress_every_s = 20.0
            last_progress = time.monotonic()

            while len(results) < total:
                try:
                    worker, idx, text = result_q.get(timeout=5.0)
                except queue.Empty:
                    # Workers may still be warming up; check liveness.  If they
                    # all exited we break and drain below (graceful exit — a
                    # worker finishes its batch, flushes, then leaves).
                    if not any(p.is_alive() for p in procs):
                        break
                    continue
                if worker not in started_at:
                    started_at[worker] = time.monotonic()
                per_worker[worker].append((time.monotonic(), len(text) if text else 0))
                record = dict(jobs[idx])
                record["_idx"] = idx
                record["_worker"] = worker
                record["_text"] = text
                record["_len"] = len(text) if text else 0
                results[idx] = record

                partial_fh.write(_orjson.dumps(record, option=_orjson.OPT_APPEND_NEWLINE))
                partial_fh.flush()

                now = time.monotonic()
                if now - last_progress >= progress_every_s or len(results) == total:
                    last_progress = now
                    elapsed = now - t0
                    rate = len(results) / max(elapsed, 1e-9)
                    eta = (total - len(results)) / rate if rate > 0 else 0.0
                    logger.info("  %d/%d jobs (%.1f%%) — %.2f jobs/s (ETA %s)",
                                len(results), total, len(results) / total * 100,
                                rate, _fmt_eta(eta))
    except KeyboardInterrupt:
        logger.error("  Interrupted — saving partial results (%d/%d done).",
                     len(results), total)
        raise
    finally:
        monitor.stop()
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        if partial_fh is not None:
            try:
                partial_fh.close()
            except Exception:
                pass

        # Drain any results still in flight.  On graceful exit the worker's
        # multiprocessing feeder thread flushes before the process returns, so
        # a short grace + repeat usually recovers everything; `terminate()`
        # above is the only path that can genuinely lose buffered items.
        def _drain() -> None:
            while True:
                try:
                    worker, idx, text = result_q.get_nowait()
                except Exception:
                    return
                if idx in results:
                    continue
                record = dict(jobs[idx])
                record["_idx"] = idx
                record["_worker"] = worker
                record["_text"] = text
                record["_len"] = len(text) if text else 0
                results[idx] = record
                if partial_fh is not None:
                    try:
                        partial_fh.write(_orjson.dumps(record, option=_orjson.OPT_APPEND_NEWLINE))
                        partial_fh.flush()
                    except Exception:
                        pass

        for _ in range(4):
            _drain()
            if len(results) >= total:
                break
            time.sleep(0.25)
        for i in range(total):
            if i not in results:
                record = dict(jobs[i])
                record["_idx"] = i
                record["_worker"] = "lost"
                record["_text"] = None
                record["_len"] = 0
                results[i] = record

        elapsed = time.monotonic() - t0
        ok = sum(1 for r in results.values() if r["_text"])
        failed = total - ok

        # ── write output in input order (even on interrupt) ───────────────
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for i in range(total):
                f.write(_orjson.dumps(results[i], option=_orjson.OPT_APPEND_NEWLINE))
        partial.unlink(missing_ok=True)

        # ── per-worker stats ───────────────────────────────────────────────
        worker_stats: Dict[str, Dict[str, Any]] = {}
        for name, samples in per_worker.items():
            n = len(samples)
            chars = sum(c for _, c in samples)
            span = time.monotonic() - started_at.get(name, t0) if samples else 0
            worker_stats[name] = {
                "jobs": n,
                "jobs_per_s": round(n / span, 3) if span > 0 else 0.0,
                "chars_per_s": round(chars / span, 1) if span > 0 else 0.0,
            }
            logger.info("  worker %-5s: %d jobs, %.2f jobs/s",
                        name, n, worker_stats[name]["jobs_per_s"])

        stats = {
            "run_dir": str(run_dir),
            "jobs": total,
            "ok": ok,
            "failed": failed,
            "resumed": total - len(needed),
            "elapsed_s": round(elapsed, 1),
            "jobs_per_s": round(ok / elapsed, 3) if elapsed else 0.0,
            "ok_pct": round(ok / total * 100, 1) if total else 0.0,
            "workers": worker_stats,
            "plan": [s.as_dict() for s in specs],
        }
        logger.info("  DONE: %d/%d ok in %.1fs (%.3f jobs/s). Output: %s",
                    ok, total, elapsed, stats["jobs_per_s"], out_path)
        logger.info("  Run dir: %s", run_dir)

        # Persist a small plan.json for the run for later comparison.
        with open(run_dir / "plan.json", "w") as f:
            json.dump(stats, f, indent=2)

    if not results_only:
        logger.handlers = []  # detach file handler after run
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI — diagnostics: show the plan and probe, no heavy work
# ─────────────────────────────────────────────────────────────────────────────


def _cli_probe(args: argparse.Namespace) -> int:
    res = probe()
    print(json.dumps(res.as_dict(), indent=2))
    if args.model_path:
        specs = plan_llm_resources(res, args.model_path, ctx_per_seq=args.ctx)
        print("PLAN:")
        for s in specs:
            print(f"  {s.name:>5} kind={s.kind:<3} threads={s.n_threads} n_parallel={s.n_parallel} ctx={s.ctx_per_seq}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Resource-aware orchestrator diagnostics")
    parser.add_argument("--model-path", default="models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--ctx", type=int, default=1536)
    args = parser.parse_args()
    return _cli_probe(args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sys.exit(main())
