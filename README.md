# AttenTrace

AttenTrace is a research project investigating whether
attention entropy trajectories provide a stronger signal
for chain-of-thought correctness than traditional output-logit entropy.

## Research Questions

1. Can entropy trajectories predict reasoning correctness?
2. Does attention entropy provide signal beyond output entropy?
3. Can internal attention dynamics act as an early-warning signal for reasoning failure?

## Current Setup

- Model: Qwen2.5-1.5B-Instruct
- Dataset: GSM8K
- Metrics:
  - Output Entropy
  - Attention Entropy
  - Kendall τ Monotonicity
- Framework:
  - PyTorch
  - Hugging Face Transformers

## Status

Phase 1 ✅ Completed
Phase 2 🚧 Attention Entropy Extraction
Phase 3 ⏳ Predictive Evaluation
