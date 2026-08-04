"""
llm_backend.py — Batched local LLM inference using the llama.cpp backend bundled with LM Studio
==============================================================================================

Why this module exists
----------------------
`data_prep.py --phase instruct` (and `align`) previously called the model once per
code chunk, in a strict serial loop:

    for chunk in chunks:
        instruction = llm(prompt)          # one GPU inference, one sequence at a time

On a small GPU (RTX 3050 Laptop, 4 GB) decoding is memory-bandwidth bound, so the GPU
sat idle most of the time and 58k chunks took >30 hours.

This module instead drives llama.cpp's **low-level batch API** directly through the
**shared libraries that LM Studio already ships** (no compilation, no extra server):

    ~/.lmstudio/extensions/backends/llama.cpp-*-nvidia-cuda-*/libllama.so

Multiple sequences are decoded in ONE `llama_decode` call, so the GPU reuses the loaded
weights across N sequences each step (dynamic batching). On a bandwidth-bound GPU the
throughput grows roughly linearly with the number of parallel sequences (measured ~10x).

How it is used
--------------
    from llm_backend import find_lmstudio_backend, BatchedLlama

    info = find_lmstudio_backend()          # None if no LM Studio backend is installed
    with BatchedLlama(model_path, n_parallel=12, ctx_per_seq=1536) as llm:
        texts = llm.complete_batch([
            {"prompt": "...", "max_tokens": 128, "temperature": 0.7, "stop": ["\n\n"]},
            ...
        ])

`complete_batch` returns one text per request (None if a request failed and was retried
internally one extra time). Requests beyond `n_parallel` are chunked automatically.

Degradation: if no LM Studio backend is found, `find_lmstudio_backend()` returns None and
the caller keeps using the previous in-process llama-cpp-python path. This module never
downloads, builds, or starts a server.

API compatibility note
----------------------
The struct layouts below match llama.cpp release b8733 (commit d6f3030), which is the
build bundled with LM Studio backend package 2.13.0 (verified by reading this backend's
`display-data.json`, which records `llama.cpp release b8733 (commit d6f3030)`).

The backend directory name records the package version
(e.g. `llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.13.0`). If a future LM Studio backend
ships a different llama.cpp, re-verify the layouts against `include/llama.h` from the
matching llama.cpp commit. The runtime sanity checks in `_sanity_check_model_params` /
`_sanity_check_context_params` will refuse to run against an incompatible build instead
of crashing.
"""

# Tiger Style: explicit imports, no wildcards.
import ctypes
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# llama.cpp C types (llama.h): all fixed-width int32.
_llama_token = ctypes.c_int32
_llama_pos = ctypes.c_int32
_llama_seq_id = ctypes.c_int32

_VOID = ctypes.c_void_p
_INT32 = ctypes.c_int32
_UINT32 = ctypes.c_uint32
_INT = ctypes.c_int
_FLOAT = ctypes.c_float
_BOOL = ctypes.c_bool


class _llama_batch(ctypes.Structure):
    """Mirror of `struct llama_batch` from llama.cpp b7011."""

    _fields_ = [
        ("n_tokens", _INT32),
        ("token", ctypes.POINTER(_llama_token)),
        ("embd", ctypes.POINTER(_FLOAT)),
        ("pos", ctypes.POINTER(_llama_pos)),
        ("n_seq_id", ctypes.POINTER(_INT32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(_llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


class _llama_model_params(ctypes.Structure):
    """Mirror of `struct llama_model_params` from llama.cpp b8733."""

    _fields_ = [
        ("devices", _VOID),
        ("tensor_buft_overrides", _VOID),
        ("n_gpu_layers", _INT32),
        ("split_mode", _INT),
        ("main_gpu", _INT32),
        ("tensor_split", ctypes.POINTER(_FLOAT)),
        ("progress_callback", _VOID),
        ("progress_callback_user_data", _VOID),
        ("kv_overrides", _VOID),
        ("vocab_only", _BOOL),
        ("use_mmap", _BOOL),
        ("use_direct_io", _BOOL),
        ("use_mlock", _BOOL),
        ("check_tensors", _BOOL),
        ("use_extra_bufts", _BOOL),
        ("no_host", _BOOL),
        ("no_alloc", _BOOL),
    ]


class _llama_context_params(ctypes.Structure):
    """Mirror of `struct llama_context_params` from llama.cpp b8733."""

    _fields_ = [
        ("n_ctx", _UINT32),
        ("n_batch", _UINT32),
        ("n_ubatch", _UINT32),
        ("n_seq_max", _UINT32),
        ("n_threads", _INT32),
        ("n_threads_batch", _INT32),
        ("rope_scaling_type", _INT),
        ("pooling_type", _INT),
        ("attention_type", _INT),
        ("flash_attn_type", _INT),
        ("rope_freq_base", _FLOAT),
        ("rope_freq_scale", _FLOAT),
        ("yarn_ext_factor", _FLOAT),
        ("yarn_attn_factor", _FLOAT),
        ("yarn_beta_fast", _FLOAT),
        ("yarn_beta_slow", _FLOAT),
        ("yarn_orig_ctx", _UINT32),
        ("defrag_thold", _FLOAT),
        ("cb_eval", _VOID),
        ("cb_eval_user_data", _VOID),
        ("type_k", _INT),
        ("type_v", _INT),
        ("abort_callback", _VOID),
        ("abort_callback_data", _VOID),
        ("embeddings", _BOOL),
        ("offload_kqv", _BOOL),
        ("no_perf", _BOOL),
        ("op_offload", _BOOL),
        ("swa_full", _BOOL),
        ("kv_unified", _BOOL),
        ("samplers", _VOID),
        ("n_samplers", ctypes.c_size_t),
    ]


class _llama_sampler_chain_params(ctypes.Structure):
    """Mirror of `struct llama_sampler_chain_params` from llama.cpp b7011."""

    _fields_ = [
        ("no_perf", _BOOL),
    ]


def find_lmstudio_backend(
    backends_root: Optional[Path] = None,
) -> Optional[Dict[str, Path]]:
    """Locate the newest LM Studio NVIDIA-CUDA llama.cpp backend.

    Tiger Style: explicit search, explicit result. Returns None when nothing usable
    exists so the caller can fall back gracefully.

    Args:
        backends_root: Override for the LM Studio backends folder (defaults to
            ~/.lmstudio/extensions/backends). Useful for testing and for machines that
            keep LM Studio somewhere non-standard.

    Returns:
        {"backend_dir": Path, "vendor_dir": Path} or None.
    """
    base = backends_root or Path(os.environ.get("LMSTUDIO_BACKENDS_DIR", "~/.lmstudio/extensions/backends"))
    base = Path(base).expanduser().resolve()

    if not base.is_dir():
        return None

    # Candidate backend dirs: name pattern like
    #   llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.13.0
    candidates: List[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if "llama.cpp" not in child.name:
            continue
        if "cuda" not in child.name:
            continue
        if not (child / "libllama.so").exists():
            continue
        candidates.append(child)

    if not candidates:
        return None

    # Tiger Style: deterministic — sort by the trailing dotted version, pick the newest.
    def _version(p: Path) -> tuple:
        suffix = p.name.rsplit("-", 1)[-1]
        parts = []
        for tok in suffix.split("."):
            if tok.isdigit():
                parts.append(int(tok))
            else:
                break
        return tuple(parts) or (0,)

    candidates.sort(key=lambda p: _version(p), reverse=True)
    backend_dir = candidates[0]

    # CUDA vendor libs (libcudart.so.11, libcublas.so.11, ...) live under
    #   backends/vendor/linux-llama-cuda-vendor-v1/
    vendor_dir: Optional[Path] = None
    vendor_root = base / "vendor"
    if vendor_root.is_dir():
        for child in vendor_root.iterdir():
            if child.is_dir() and any(p.name.startswith("libcudart") for p in child.iterdir()):
                vendor_dir = child
                break

    return {"backend_dir": backend_dir, "vendor_dir": vendor_dir}


def _load_backend_lib(info: Dict[str, Path]) -> Any:
    """dlopen the LM Studio llama.cpp shared libraries.

    libggml-cuda.so needs the CUDA 11 runtime (libcudart.so.11.0, libcublas.so.11) that
    LM Studio bundles in the vendor dir. glibc reads LD_LIBRARY_PATH only at process
    start, so we preload the vendor libs globally instead — dlopen then resolves the
    NEEDED sonames against the already-loaded objects.
    """
    backend_dir: Path = info["backend_dir"]
    vendor_dir: Optional[Path] = info.get("vendor_dir")

    if vendor_dir is not None:
        # Load the vendor CUDA runtime first, in dependency-sane order.
        for name in sorted(os.listdir(vendor_dir)):
            if name.startswith("lib") and ".so" in name:
                try:
                    ctypes.CDLL(str(vendor_dir / name), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue

    # Load the core ggml libraries globally so llama_backend_init can register the
    # CUDA/CPU backends, then libllama.so last (it NEEDs the rest).
    for name in ("libggml-base.so", "libggml-cpu.so", "libggml-cuda.so", "libggml_llamacpp.so"):
        p = backend_dir / name
        if p.exists():
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

    return ctypes.CDLL(str(backend_dir / "libllama.so"), mode=ctypes.RTLD_GLOBAL)


def _bind(lib: Any) -> None:
    """Set argtypes/restype on every llama.cpp symbol this module touches.

    Tiger Style: every extern call is explicitly typed — no accidental pointer widening.
    """
    lib.llama_backend_init.restype = None
    lib.llama_backend_init.argtypes = []

    lib.llama_model_default_params.restype = _llama_model_params
    lib.llama_model_default_params.argtypes = []
    lib.llama_model_load_from_file.restype = _VOID
    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, _llama_model_params]
    lib.llama_model_get_vocab.restype = _VOID
    lib.llama_model_get_vocab.argtypes = [_VOID]
    lib.llama_model_free.restype = None
    lib.llama_model_free.argtypes = [_VOID]

    lib.llama_context_default_params.restype = _llama_context_params
    lib.llama_context_default_params.argtypes = []
    lib.llama_new_context_with_model.restype = _VOID
    lib.llama_new_context_with_model.argtypes = [_VOID, _llama_context_params]
    lib.llama_n_ctx.restype = _INT32
    lib.llama_n_ctx.argtypes = [_VOID]
    lib.llama_get_memory.restype = _VOID
    lib.llama_get_memory.argtypes = [_VOID]
    lib.llama_free.restype = None
    lib.llama_free.argtypes = [_VOID]

    lib.llama_batch_init.restype = _llama_batch
    lib.llama_batch_init.argtypes = [_INT32, _INT32, _INT32]
    lib.llama_batch_free.restype = None
    lib.llama_batch_free.argtypes = [_llama_batch]
    lib.llama_decode.restype = _INT32
    lib.llama_decode.argtypes = [_VOID, _llama_batch]

    lib.llama_tokenize.restype = _INT32
    lib.llama_tokenize.argtypes = [
        _VOID, ctypes.c_char_p, _INT32,
        ctypes.POINTER(_llama_token), _INT32, _BOOL, _BOOL,
    ]
    lib.llama_token_to_piece.restype = _INT32
    lib.llama_token_to_piece.argtypes = [_VOID, _llama_token, ctypes.c_char_p, _INT32, _INT32, _BOOL]
    lib.llama_vocab_is_eog.restype = _BOOL
    lib.llama_vocab_is_eog.argtypes = [_VOID, _llama_token]

    lib.llama_memory_seq_rm.restype = _BOOL
    lib.llama_memory_seq_rm.argtypes = [_VOID, _llama_seq_id, _llama_pos, _llama_pos]

    lib.llama_sampler_chain_default_params.restype = _llama_sampler_chain_params
    lib.llama_sampler_chain_default_params.argtypes = []
    lib.llama_sampler_chain_init.restype = _VOID
    lib.llama_sampler_chain_init.argtypes = [_llama_sampler_chain_params]
    lib.llama_sampler_chain_add.restype = None
    lib.llama_sampler_chain_add.argtypes = [_VOID, _VOID]
    lib.llama_sampler_init_top_p.restype = _VOID
    lib.llama_sampler_init_top_p.argtypes = [_FLOAT, ctypes.c_size_t]
    lib.llama_sampler_init_temp.restype = _VOID
    lib.llama_sampler_init_temp.argtypes = [_FLOAT]
    lib.llama_sampler_init_dist.restype = _VOID
    lib.llama_sampler_init_dist.argtypes = [_UINT32]
    lib.llama_sampler_sample.restype = _llama_token
    lib.llama_sampler_sample.argtypes = [_VOID, _VOID, _INT32]
    lib.llama_sampler_free.restype = None
    lib.llama_sampler_free.argtypes = [_VOID]


class BatchedLlama:
    """Batched decoder over an LM Studio llama.cpp library.

    All N sequences in a batch advance one token per `llama_decode` call, which is what
    turns a bandwidth-bound single-sequence loop into a near-linearly-faster pipeline.

    Not thread-safe: call `complete_batch` from a single thread (the batching happens
    inside the call, so phases don't need their own threads).
    """

    def __init__(
        self,
        model_path: str,
        n_parallel: int = 12,
        ctx_per_seq: int = 1536,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        assert n_parallel >= 1, "n_parallel must be >= 1"
        assert ctx_per_seq >= 256, "ctx_per_seq too small"

        self.model_path = str(model_path)
        self.n_parallel = int(n_parallel)
        self.ctx_per_seq = int(ctx_per_seq)
        self.n_ctx_total = self.n_parallel * self.ctx_per_seq
        self.n_threads = n_threads or (os.cpu_count() or 4)
        self.n_gpu_layers = n_gpu_layers
        self.logger = logger

        self._lib: Optional[Any] = None
        self._model = None
        self._vocab = None
        self._ctx = None
        self._mem = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def __enter__(self) -> "BatchedLlama":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _log(self, level: int, msg: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.log(level, msg, *args)

    def start(self) -> None:
        """Load the backend, the model, and the context.

        Raises RuntimeError if anything fails; callers should fall back to the serial
        llama-cpp-python path.
        """
        info = find_lmstudio_backend()
        if info is None:
            raise RuntimeError("No LM Studio llama.cpp backend found")
        self._log(logging.INFO, "  Loading LM Studio llama.cpp backend from %s", info["backend_dir"])
        self._lib = _load_backend_lib(info)
        _bind(self._lib)

        # Sanity-check the model-params layout before doing anything heavy.
        mparams = self._lib.llama_model_default_params()
        if not self._sanity_check_model_params(mparams):
            raise RuntimeError(
                "llama_model_params layout mismatch — this LM Studio backend uses a "
                "different llama.cpp version than llm_backend.py expects. Update the "
                "structs in llm_backend.py from the matching llama.h."
            )

        mparams.n_gpu_layers = self.n_gpu_layers
        mparams.use_mmap = True
        mparams.use_mlock = False

        self._lib.llama_backend_init()

        # LM Studio's CUDA build crashes inside `llama_model_loader::load_all_data`
        # on the very first GPU load of a process, and silently fails if the first
        # GPU load follows a plain CPU load. A warm-up sequence of (CPU load, then a
        # deliberately-failing no-mmap GPU load) deterministically produces a working
        # GPU mmap load right after. This costs a few seconds once at startup.
        self._warmup_backend(mparams)

        model_path_b = self.model_path.encode()
        self._model = self._lib.llama_model_load_from_file(model_path_b, mparams)
        if not self._model:
            raise RuntimeError(f"llama_model_load_from_file failed: {self.model_path}")
        self._vocab = self._lib.llama_model_get_vocab(self._model)
        if not self._vocab:
            raise RuntimeError("llama_model_get_vocab returned NULL")

        cparams = self._lib.llama_context_default_params()
        if not self._sanity_check_context_params(cparams):
            raise RuntimeError(
                "llama_context_params layout mismatch — see llm_backend.py header for "
                "how to update the struct definitions."
            )
        cparams.n_ctx = self.n_ctx_total
        cparams.n_batch = self.n_ctx_total
        cparams.n_ubatch = 2048
        cparams.n_seq_max = self.n_parallel
        cparams.n_threads = self.n_threads
        cparams.n_threads_batch = self.n_threads
        cparams.no_perf = True
        cparams.offload_kqv = True

        self._ctx = self._lib.llama_new_context_with_model(self._model, cparams)
        if not self._ctx:
            raise RuntimeError("llama_new_context_with_model failed")
        self._mem = self._lib.llama_get_memory(self._ctx)
        if not self._mem:
            raise RuntimeError("llama_get_memory returned NULL")

        actual_ctx = self._lib.llama_n_ctx(self._ctx)
        self._log(logging.INFO, "  Context ready: n_ctx=%d n_seq_max=%d (gpu_layers=%d)",
                  actual_ctx, self.n_parallel, self.n_gpu_layers)

    def close(self) -> None:
        if self._ctx:
            self._lib.llama_free(self._ctx)
            self._ctx = None
        if self._model:
            self._lib.llama_model_free(self._model)
            self._model = None

    def _warmup_backend(self, base: _llama_model_params) -> None:
        """Prime the LM Studio CUDA backend so the real load does not crash.

        See the comment at the call site for the observed failure modes. Both warm-up
        loads are freed immediately; failures here are expected and ignored.
        """
        model_path_b = self.model_path.encode()

        cpu = self._lib.llama_model_default_params()
        cpu.n_gpu_layers = 0
        cpu.use_mmap = True
        m = self._lib.llama_model_load_from_file(model_path_b, cpu)
        if m:
            self._lib.llama_model_free(m)

        gpu_nommap = self._lib.llama_model_default_params()
        gpu_nommap.n_gpu_layers = self.n_gpu_layers
        gpu_nommap.use_mmap = False
        m = self._lib.llama_model_load_from_file(model_path_b, gpu_nommap)
        if m:
            self._lib.llama_model_free(m)

    # ── sanity checks (ABI drift protection) ────────────────────────────────

    @staticmethod
    def _sanity_check_model_params(p: _llama_model_params) -> bool:
        return (
            p.n_gpu_layers == -1
            and p.vocab_only is False
            and p.use_mmap is True
            and p.use_direct_io is False
            and p.use_mlock is False
            and p.check_tensors is False
            and p.use_extra_bufts is True
            and p.no_host is False
            and p.no_alloc is False
        )

    @staticmethod
    def _sanity_check_context_params(p: _llama_context_params) -> bool:
        return (
            p.n_ctx == 512
            and p.n_batch == 2048
            and p.n_ubatch == 512
            and p.n_seq_max == 1
            and p.embeddings is False
            and p.offload_kqv is True
            and p.no_perf is True
            and p.op_offload is True
            and p.swa_full is True
            and p.kv_unified is False
            and p.n_samplers == 0
        )

    # ── tokenizer helpers ───────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[int]:
        data = text.encode("utf-8")
        buf = (ctypes.c_int32 * 16384)()
        n = self._lib.llama_tokenize(
            self._vocab, data, len(data), buf, 16384, True, False
        )
        if n < 0:
            raise RuntimeError(f"llama_tokenize failed (n={n})")
        return list(buf[:n])

    def _token_to_piece(self, token: int) -> bytes:
        for buf_len in (16, 256, 1024):
            buf = ctypes.create_string_buffer(buf_len)
            n = self._lib.llama_token_to_piece(self._vocab, token, buf, buf_len, 0, False)
            if n >= 0:
                return buf.raw[:n]
            if n < -buf_len:
                continue
        return b""

    def _detok(self, tokens: List[int]) -> str:
        out = b"".join(self._token_to_piece(t) for t in tokens)
        return out.decode("utf-8", errors="replace")

    def _is_eog(self, token: int) -> bool:
        return bool(self._lib.llama_vocab_is_eog(self._vocab, token))

    # ── batch helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _batch_add(batch: _llama_batch, token: int, pos: int, seq_id: int, logits: bool) -> None:
        i = batch.n_tokens
        batch.token[i] = token
        batch.pos[i] = pos
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = seq_id
        batch.logits[i] = 1 if logits else 0
        batch.n_tokens = i + 1

    def _free_seq(self, seq_id: int) -> None:
        if self._mem:
            self._lib.llama_memory_seq_rm(self._mem, seq_id, -1, -1)

    # ── public API ──────────────────────────────────────────────────────────

    def complete_batch(self, requests: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Generate a completion for each request, batching under the hood.

        Each request: {"prompt": str, "max_tokens": int, "temperature": float,
                       "top_p": float, "stop": list[str]}

        Returns one text per request (stripped), or None for a request that could not
        be generated.
        """
        if not requests:
            return []
        results: List[Optional[str]] = []
        for start in range(0, len(requests), self.n_parallel):
            sub = requests[start:start + self.n_parallel]
            results.extend(self._complete_sub_batch(sub))
        return results

    def _complete_sub_batch(self, requests: List[Dict[str, Any]]) -> List[Optional[str]]:
        # Clear KV cache for every seq_id we are about to use.  A previous sub-batch
        # may have left stale positions in the memory module for the same seq_ids,
        # and llama.cpp requires consecutive positions per sequence (Y = X + 1).
        # Without this flush any retried request hits "inconsistent sequence positions".
        n = len(requests)
        for i in range(n):
            self._free_seq(i)

        seqs = []
        for i, req in enumerate(requests):
            prompt = req["prompt"]
            tokens = self._tokenize(prompt)
            # Leave headroom for the generated tokens inside the shared context.
            budget = self.ctx_per_seq - int(req.get("max_tokens", 128)) - 16
            if len(tokens) > budget:
                tokens = tokens[:budget]
            seqs.append({
                "id": i,
                "prompt": tokens,
                "out": [],
                "max": int(req.get("max_tokens", 128)),
                "temp": float(req.get("temperature", 0.7)),
                "top_p": float(req.get("top_p", 0.95)),
                "stop": [s.encode() for s in req.get("stop", []) if s],
                "text": b"",
                "active": True,
                "failed": False,
            })

        batch = self._lib.llama_batch_init(self.n_ctx_total, 0, self.n_parallel)
        try:
            self._run_batch(batch, seqs)
        finally:
            # Always free every seq_id we used, even when _run_batch raised or a
            # sequence failed mid-way -- stale KV entries crash the next sub-batch.
            for i in range(n):
                self._free_seq(i)
            self._lib.llama_batch_free(batch)

        out: List[Optional[str]] = []
        for s in seqs:
            if s["failed"]:
                out.append(None)
            else:
                text = self._detok(s["out"]).strip()
                out.append(text if text else None)
        return out

    def _run_batch(self, batch: _llama_batch, seqs: List[Dict[str, Any]]) -> None:
        # Per-sequence sampler chains (temp + top_p), mirrors the old call:
        #   llm(prompt, max_tokens=128, temperature=0.7)
        chains = []
        for s in seqs:
            cparams = self._lib.llama_sampler_chain_default_params()
            cparams.no_perf = True
            chain = self._lib.llama_sampler_chain_init(cparams)
            self._lib.llama_sampler_chain_add(chain, self._lib.llama_sampler_init_top_p(s["top_p"], 1))
            self._lib.llama_sampler_chain_add(chain, self._lib.llama_sampler_init_temp(s["temp"]))
            # b8733 requires a terminal sampler that actually picks the token.
            self._lib.llama_sampler_chain_add(chain, self._lib.llama_sampler_init_dist(random.randint(0, 2**31 - 1)))
            chains.append(chain)

        try:
            # ── prefill: all prompts in one decode call ──────────────────────
            batch.n_tokens = 0
            for s in seqs:
                n = len(s["prompt"])
                for j, tok in enumerate(s["prompt"]):
                    self._batch_add(batch, tok, j, s["id"], logits=(j == n - 1))
            if batch.n_tokens > 0 and self._lib.llama_decode(self._ctx, batch) != 0:
                self._log(logging.WARNING,
                          "  Batch prefill decode failed (%d tokens across %d seqs)",
                          batch.n_tokens, len(seqs))
                for s in seqs:
                    s["failed"] = True
                return

            # ── sample the first generated token for every sequence ──────────
            row = 0
            for s in seqs:
                n = len(s["prompt"])
                row += n
                tok = self._lib.llama_sampler_sample(chains[s["id"]], self._ctx, row - 1)
                s["out"].append(tok)
                if self._is_eog(tok):
                    s["active"] = False

            # ── decode loop: one token per active sequence per step ──────────
            steps = 0
            while any(s["active"] for s in seqs):
                steps += 1
                if steps > 16384:
                    self._log(logging.WARNING, "  Batch exceeded max steps — aborting")
                    break

                batch.n_tokens = 0
                rows: Dict[int, int] = {}
                for s in seqs:
                    if not s["active"]:
                        continue
                    pos = len(s["prompt"]) + len(s["out"]) - 1
                    self._batch_add(batch, s["out"][-1], pos, s["id"], logits=True)
                    rows[s["id"]] = batch.n_tokens - 1

                if batch.n_tokens == 0:
                    break
                if self._lib.llama_decode(self._ctx, batch) != 0:
                    self._log(logging.WARNING,
                              "  Decode loop step %d failed (%d tokens, %d active seqs)",
                              steps, batch.n_tokens,
                              sum(1 for s in seqs if s["active"]))
                    for s in seqs:
                        if s["active"]:
                            s["failed"] = True
                            s["active"] = False
                    break

                for s in seqs:
                    if not s["active"]:
                        continue
                    tok = self._lib.llama_sampler_sample(chains[s["id"]], self._ctx, rows[s["id"]])
                    s["out"].append(tok)
                    if self._is_eog(tok):
                        s["active"] = False
                        self._free_seq(s["id"])
                        continue
                    if len(s["out"]) >= s["max"]:
                        s["active"] = False
                        self._free_seq(s["id"])
                        continue

                    s["text"] += self._token_to_piece(tok)
                    for stop in s["stop"]:
                        if stop in s["text"]:
                            idx = s["text"].rfind(stop)
                            s["text"] = s["text"][:idx]
                            s["active"] = False
                            self._free_seq(s["id"])
                            break
        finally:
            for chain in chains:
                self._lib.llama_sampler_free(chain)
