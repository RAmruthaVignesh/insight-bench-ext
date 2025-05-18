#!/usr/bin/env python3
"""
frugalscore_eval.py

Compute FrugalScore (Jiang et al., 2024) for a single
(predicted_insight, ground_truth_insight) pair.

Usage:
    python frugalscore_eval.py "<predicted_insight>" "<ground_truth_insight>"
"""

import sys
import torch
from evaluate import load

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load the FrugalScore metric from HF Evaluate.
#    Here we pick the "medium" BERT→MoverScore variant
#    but you can change to tiny/small, or the BERT→BERTScore versions.
# ─────────────────────────────────────────────────────────────────────────────
# frugalscore = load(
#     "frugalscore",
#     module_id="moussaKam/frugalscore_medium_bert-base_mover-score"
# )
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_frugal = load("frugalscore", use_cpu=(DEVICE=="cpu"), process_count=1)
# ─────────────────────────────────────────────────────────────────────────────
# 2) Compute FrugalScore for one (pred, ref) pair
# ─────────────────────────────────────────────────────────────────────────────
def compute_frugalscore(predicted_insights: list[str], ground_truth_insights: list[str]) -> list[float]:
    """
    Returns a float in [0.0, 1.0].
    """
    # 🤗 Evaluate's .compute expects lists
    result = _frugal.compute(
        predictions=predicted_insights,
        references=ground_truth_insights,
        device=DEVICE
    )
    # key is "scores", a list of floats
    return result["scores"]


# ─────────────────────────────────────────────────────────────────────────────
# 3) CLI entrypoint for quick sanity‐check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python frugalscore_eval.py '<predicted_insight>' '<ground_truth_insight>'")
        sys.exit(1)

    pred, ref = sys.argv[1], sys.argv[2]
    score = compute_frugalscore(pred, ref)
    print(f"FrugalScore: {score:.4f}")