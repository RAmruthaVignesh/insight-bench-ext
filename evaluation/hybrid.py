#!/usr/bin/env python3
"""
Compute a robust hybrid score: 
  bs_weight * BERTScore 
+ qa_weight * QAEval 
+ sq_weight * SummaQA

We emphasize factual coverage: by default we give
QA‐Eval and SummaQA twice the weight of BERTScore.
We also adapt weights if the GT is short or if QA‐Eval can’t extract questions.
"""
import os
from evaluate import load
from qa_eval import compute_qa_eval
from summaqa_eval import compute_summaqa

# ────────────────────────────────────────────────────────────────────────
# 1) Initialize BERTScore once
# ────────────────────────────────────────────────────────────────────────
_bertscore = load("bertscore")

def compute_bert_score(pred: str, ref: str) -> float:
    out = _bertscore.compute(
        predictions=[pred], 
        references=[ref],
        model_type="microsoft/deberta-xlarge-mnli"
    )
    return max(0.0, min(1.0, float(out["f1"][0])))


# ────────────────────────────────────────────────────────────────────────
# 2) Robust hybrid wrapper
# ────────────────────────────────────────────────────────────────────────
def compute_hybrid_score(
    predicted_insight: str,
    ground_truth_insight: str,
    bs_weight: float = 0.2,
    qa_weight: float = 0.4,
    sq_weight: float = 0.4,
    short_threshold: int = 5
    # min_qa_pairs: int = 1
) -> float:
    """
    Returns a float in [0,1], blending three signals.
    If ref is very short, we fall back to pure BERTScore.
    If QA‐Eval extracts fewer than min_qa_pairs, we down‐weight QA/Summa.
    """
    # 1) Base semantic score
    bs = compute_bert_score(predicted_insight, ground_truth_insight)

   # 2) Short-reference fallback
    if len(ground_truth_insight.split()) < short_threshold:
        return bs

    # 3) Try factual metrics safely
    try:
        qa = compute_qa_eval(predicted_insight, ground_truth_insight)
    except Exception:
        qa = 0.0

    try:
        sq = compute_summaqa(predicted_insight, ground_truth_insight)
    except Exception:
        sq = 0.0

    # Clamp into [0,1]
    qa = max(0.0, min(1.0, qa))
    sq = max(0.0, min(1.0, sq))

    # 4) Adapt weights if QA‐Eval couldn’t extract enough pairs
    #    (i.e. it returned 1.0 by fallback or 0.0 due to failure)
    if qa == 1.0 or qa == 0.0:
        # shift weight entirely to BERTScore and SummaQA
        qa_weight = 0.0
        bs_weight = bs_weight + qa_weight * 0.5
        sq_weight = sq_weight + qa_weight * 0.5

    # 5) Normalize weights to sum=1.0
    total = bs_weight + qa_weight + sq_weight
    bs_w, qa_w, sq_w = bs_weight/total, qa_weight/total, sq_weight/total

    # 6) Weighted sum
    hybrid = bs_w*bs + qa_w*qa + sq_w*sq
    return max(0.0, min(1.0, hybrid))


# ────────────────────────────────────────────────────────────────────────
# 3) CLI for quick testing
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python hybrid.py '<predicted>' '<reference>'")
        sys.exit(1)

    pred, ref = sys.argv[1], sys.argv[2]
    score = compute_hybrid_score(pred, ref)
    print(f"HybridScore: {score:.4f}")


#!/usr/bin/env python3
# """
# Compute a hybrid score: 
#     0.5 * BERTScore 
#   + 0.25 * QAEval 
#   + 0.25 * SummaQA

# Requirements:
#     - bertscore & evaluate: pip install bert-score evaluate
#     - QAEval & SummaQA deps (openai, transformers, spacy, etc.)
# """
# import os
# from evaluate import load
# from qa_eval import compute_qa_eval
# from summaqa_eval import compute_summaqa

# # ────────────────────────────────────────────────────────────────────────
# # 1) Initialize BERTScore once
# # ────────────────────────────────────────────────────────────────────────
# _bertscore = load("bertscore")

# def compute_bert_score(pred: str, ref: str) -> float:
#     out = _bertscore.compute(
#         predictions=[pred],
#         references=[ref],
#         model_type="microsoft/deberta-xlarge-mnli"
#     )
#     return float(out["f1"][0])
# # ────────────────────────────────────────────────────────────────────────
# # 2) Hybrid wrapper
# # ────────────────────────────────────────────────────────────────────────
# def compute_hybrid_score(
#     predicted_insight: str,
#     ground_truth_insight: str,
#     bs_weight: float = 0.5,
#     qa_weight: float = 0.25,
#     sq_weight: float = 0.25,
#     short_threshold: int = 5
# ) -> float:
#     # always get a BERTScore
#     bs = compute_bert_score(predicted_insight, ground_truth_insight)
#     # fallback to BERTScore alone if GT is very short
#     if len(ground_truth_insight.split()) < short_threshold:
#         return bs

#     qa = compute_qa_eval(predicted_insight, ground_truth_insight)
#     sq = compute_summaqa(predicted_insight, ground_truth_insight)
#     return bs_weight * bs + qa_weight * qa + sq_weight * sq

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 3:
#         print("Usage: python hybrid_eval.py '<predicted>' '<reference>'")
#         sys.exit(1)
#     pred, ref = sys.argv[1], sys.argv[2]
#     print(f"HybridScore: {compute_hybrid_score(pred, ref):.4f}")
