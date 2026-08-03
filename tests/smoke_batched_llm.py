"""
smoke_batched_llm.py — Quick validation of the LM Studio batched LLM backend.

Run with::

    uv run python tests/smoke_batched_llm.py

Tests:
  - find_lmstudio_backend() locates a CUDA backend
  - BatchedLlama completes a handful of prompts
  - None of the warm-up/lifecycle hacks fail
"""
import sys; sys.path.insert(0, ".")

from llm_backend import BatchedLlama, find_lmstudio_backend

info = find_lmstudio_backend()
if info is None:
    print("SKIP: no LM Studio CUDA backend found")
    raise SystemExit(0)

chunks = [
    {"language": "python", "code": "def add(a, b): return a + b", "repo": "test", "name": "add"},
    {"language": "python", "code": "def square(x): return x * x", "repo": "test", "name": "square"},
    {"language": "c", "code": "int inc(int x) { return x + 1; }", "repo": "test", "name": "inc"},
]

with BatchedLlama("models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", n_parallel=4, ctx_per_seq=1536) as llm:
    reqs = [{
        "prompt": f"You are an expert programmer. Write an instruction for the following code:\n```{c['language']}\n{c['code']}\n```\nINSTRUCTION:",
        "max_tokens": 64,
        "temperature": 0.7,
        "stop": ["\n\n"],
    } for c in chunks]
    texts = llm.complete_batch(reqs)

for c, t in zip(chunks, texts):
    status = repr(t[:80]) if t else "(FAILED)"
    print(f"  [{c['language']:>8}] {status}")
    assert t is not None, f"Generation failed for {c['name']}"
    assert len(t) > 3, f"Instruction too short for {c['name']}"

print("OK")
