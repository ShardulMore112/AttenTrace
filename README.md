<div align="center">

# AttenTrace

### Attention Dispersion as an Early Warning Signal for Hallucinations in Large Language Models

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/Scikit--Learn-CrossValidation-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Research-Hallucination%20Detection-8A2BE2">
</p>

<p>
  <b>Mechanistic Interpretability</b> •
  <b>Uncertainty Quantification</b> •
  <b>Mathematical Reasoning</b> •
  <b>LLM Reliability</b>
</p>

</div>

---

## Key Result

> Attention dispersion becomes increasingly informative as model scale and reasoning complexity increase.

| Benchmark | Model        | Baseline AUC | AttenTrace AUC | Gain    |
| --------- | ------------ | ------------ | -------------- | ------- |
| GSM8K     | Qwen2.5-1.5B | 0.6897       | **0.6990**     | +0.0093 |
| MATH-500  | Qwen2.5-1.5B | 0.6259       | **0.6745**     | +0.0486 |
| MATH-500  | Qwen2.5-3B   | 0.5817       | **0.6575**     | +0.0759 |

---

## Pipeline

```text
Prompt
   │
   ▼
Qwen2.5 Generation (Pass 1)
   │
   ▼
Reasoning Trajectory BPE Alignment
   │
   ▼
Full Attention Extraction (Pass 2: use_cache=False)
   │
   ▼
Dispersion Feature Computation (AttenTrace)
   │
   ▼
Cross-Validation Evaluation & Hallucination Detection
```

---
## Repository Structure

<details>
<summary><b>Expand Repository Tree</b></summary>

```text
AttenTrace/
│
├── extraction_pipeline/
│   ├── extract_qwen_1.5B.ipynb    # Generation & entropy alignment
│   └── extract_qwen_3B.ipynb      # Scaled architecture extraction
│
├── data/
│   ├── gsm8k/
│   └── math500/                   # Contains final graded 3B JSON
│
├── figures/                       # Qualitative trajectories & ROC curves
│
├── final_grader.py                # Robust balanced-brace LaTeX target extraction
├── master_evaluate.py             # 50-Fold Repeated Stratified CV Evaluation
├── generate_3b_plots.py           
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

</details>

---

## Core Signal

```python
step_attn_std
```

Measures the dispersion of attention distributions across intermediate reasoning steps.

Unlike output confidence, this signal is derived entirely from internal transformer dynamics.

---

## Experimental Configuration

| Setting    | Value                      |
| ---------- | -------------------------- |
| Models     | Qwen2.5-1.5B, Qwen2.5-3B   |
| Benchmarks | GSM8K, MATH-500            |
| Metric     | ROC-AUC                    |
| Validation | Repeated Stratified K-Fold |
| Folds      | 50                         |
| Training   | None (Zero-Shot)           |

---

## Reproducibility

```bash
git clone <repo-url>

pip install -r requirements.txt

python final_grader.py

python master_evaluate.py
```

---

## Research Areas

```text
Large Language Models
├── Hallucination Detection
├── Uncertainty Quantification
├── Interpretability
└── Mathematical Reasoning
```

---

<div align="center">

Built for investigating whether internal attention dynamics can serve as a reliable uncertainty signal beyond output logits.

</div>
