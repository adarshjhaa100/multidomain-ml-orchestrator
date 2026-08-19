---

### 🧠 General Reasoning & Chain‑of‑Thought (CoT)

Best for boosting logic, problem‑solving, and multi‑step reasoning.

- **Superior-Reasoning-SFT-gpt-oss-120b** – ~435k samples from `gpt-oss-120b`. Uses a “distribution‑aligned” distillation pipeline; very data‑efficient (small student models can reach SOTA).
- **Combined Reasoning Distill (Multi‑Model)** – 1M+ samples merged from 23+ sources (Claude 4.5/4.6/4.7, etc.). Unified format, great for SFT.
- **Claude-Distills** – ~140k deduplicated, cleaned samples from Claude models. Already in `messages` format, ready to use. Covers math, code, logic.
- **Sarvam-105b-Distill-100k** – 100k reasoning prompts from Sarvam 105B, spanning 10 domains (CS, math, finance, etc.). Multiple formats (`thinking`, `sharegpt`).
- **GLM-5.1-OpenThoughts3-Distill** – Focus on science, code, maths; includes full CoT and quality scores from a judge model.
- **ZEDA** – 60k prompts with “self‑distillation” trajectories from MoE teachers (Qwen3‑30B, GLM‑4.7). Good for MoE adaptation.

---

### 🧮 Maths & Code

Specialised for numerical reasoning and programming.

- **Llama-Nemotron Post-Training Dataset** – Huge: **30M** synthetic samples, focused on math, code, reasoning, and function calling. Aggregated from Llama, DeepSeek‑R1, Qwen, etc.
- **Qwen3.5-Distillation-Dataset** – Small but high‑quality (~7.5k samples): ~70% math, 20% general instructions, 10% code. Dynamically controls teacher “thinking mode” – teaches the student when to think deeply.

---

### 🇨🇳 Chinese (if needed)

- **Chinese-DeepSeek-R1-Distill-data-110k** – 110k samples from DeepSeek‑R1, includes general topics beyond maths.
- **COIG-CQIA** – High‑quality Chinese instruction data from social media, encyclopedias, exams, finance, medicine, law, etc.

---

### 🔧 Domain‑Specific

- **Distill Expert 535k** – 535k samples for compressing shell/command output (specialised).
- **IndustryInstruction_Finance-Economics** – 122k samples focused on finance and economics.

---

### 💡 Quick Selection Guide

| Your Goal | Recommended Dataset |
| --- | --- |
| **Best all‑round reasoning** | `Superior-Reasoning-SFT` or `Combined Reasoning Distill` |
| **Maths / code heavy** | `Llama-Nemotron` (big) or `Qwen3.5-Distillation` (light, efficient) |
| **Chinese model** | `Chinese-DeepSeek-R1-Distill` or `COIG-CQIA` |
| **Limited compute, high quality** | `Superior-Reasoning-SFT` (435k) or `Qwen3.5-Distillation` (7.5k) |
| **Specific vertical** | Check the domain‑specific ones above |

---

**To help me narrow it down further:** what is your target domain? (e.g., general chat, coding assistant, mathematical solver, finance, etc.) Let me know and I can point you to the most fitting dataset for your distillation run.
