# Premise Selection RL

Train a GRPO RL agent (Qwen3-4B) to retrieve the **logical dependencies** of mathematical theorems via TheoremSearch, evaluated against prompted frontier baselines (GPT-5.5, Gemini-3.1 Pro).

## Setup

```bash
cp .env.example .env   # fill in AWS credentials and API keys
uv sync
```

## Key entry points

### Phase 2.2.b — Threshold calibration (run first, hard gate)
```bash
python scripts/calibrate_threshold.py
```
Prints true-match vs cross-pair distributions. Writes threshold to `configs/baseline.yaml` and distributions to `results/calibration.json`. **Stop if distributions overlap.**

### Phase 4/5 — Baseline evaluation
```bash
# Smoke run (one target, validate plumbing)
python -m src.eval.run_eval --config configs/baseline_gpt55.yaml  --policy api --dataset smoke
python -m src.eval.run_eval --config configs/baseline_gemini.yaml --policy api --dataset smoke

# Full smoke (100 targets)
# Then build train/val/test splits after smoke is clean:
python scripts/build_splits.py

# Frontier baselines on val
python -m src.eval.run_eval --config configs/baseline_gpt55.yaml  --policy api --dataset val
python -m src.eval.run_eval --config configs/baseline_gemini.yaml --policy api --dataset val
```

### Phase 6 — GRPO training (inside tmux)
```bash
accelerate launch -m src.train.grpo --config configs/grpo_intra.yaml
accelerate launch -m src.train.grpo --config configs/grpo_inter.yaml
```

### Phase 7 — Final evaluation
```bash
python -m src.eval.run_eval --config configs/baseline_gpt55.yaml  --policy api   --dataset test
python -m src.eval.run_eval --config configs/baseline_gemini.yaml --policy api   --dataset test
python -m src.eval.run_eval --config configs/eval_local.yaml      --policy local --dataset test --checkpoint <path>
```

## Databases

- **`v2`** — ground truth (targets, dep edges, statement bodies). All training queries go here.
- **`postgres`** — TheoremSearch API backing store. Never queried directly; accessed via `https://api.theoremsearch.com`.

Bridged only by body-similarity matching (rapidfuzz). See Phase 2.2 in `docs/PLAN.md`.

## Tests

```bash
pytest tests/
```
