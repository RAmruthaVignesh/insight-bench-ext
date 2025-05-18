#!/usr/bin/env python3
import sys
import spacy
from difflib import SequenceMatcher
from transformers import pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 0) Dependencies & setup
# ─────────────────────────────────────────────────────────────────────────────
# pip install spacy transformers torch
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

fill_mask = pipeline(
    "fill-mask",
    model="distilbert-base-cased",
    tokenizer="distilbert-base-cased",
    top_k=1,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1) SummaQA with fallback
# ─────────────────────────────────────────────────────────────────────────────
def compute_summaqa(predicted_insight: str, ground_truth_insight: str) -> float:
    doc = nlp(ground_truth_insight)
    ents = list(doc.ents)

    # if no entities, fallback to simple lexical overlap
    if not ents:
        ref_tokens = set(ground_truth_insight.lower().split())
        pred_tokens = set(predicted_insight.lower().split())
        if not ref_tokens:
            return 0.0
        return len(ref_tokens & pred_tokens) / len(ref_tokens)

    correct = 0
    for ent in ents:
        start, end = ent.start_char, ent.end_char
        mask_token = fill_mask.tokenizer.mask_token
        masked = ground_truth_insight[:start] + mask_token + ground_truth_insight[end:]
        combined = predicted_insight + " [SEP] " + masked

        try:
            out = fill_mask(combined)
            tok = out[0]["token_str"].strip().lower()
        except Exception:
            continue

        ratio = SequenceMatcher(None, ent.text.lower(), tok).ratio()
        if ent.text.lower() == tok or ratio >= 0.8:
            correct += 1

    return correct / len(ents)


# ─────────────────────────────────────────────────────────────────────────────
# 2) CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python summaqa_eval.py \"<predicted_insight>\" \"<ground_truth_insight>\"")
        sys.exit(1)

    pred, ref = sys.argv[1], sys.argv[2]
    score = compute_summaqa(pred, ref)
    print(f"SummaQA coverage score: {score:.3f}")


