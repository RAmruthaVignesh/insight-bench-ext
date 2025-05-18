#!/usr/bin/env python3
import os
import sys
import json
from difflib import SequenceMatcher
from openai import OpenAI, APIError
from transformers import pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 0) Bootstrap & clients
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("❌ ERROR: Please export OPENAI_API_KEY before running.")

gpt = OpenAI(api_key=OPENAI_API_KEY)

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad",
                       tokenizer="distilbert-base-cased-distilled-squad")


# ─────────────────────────────────────────────────────────────────────────────
# 1) Robust JSON extractor
# ─────────────────────────────────────────────────────────────────────────────
def extract_qa_pairs(ref_insight: str) -> list[dict]:
    prompt = f"""
Reference insight:
{ref_insight}

1) Extract each important fact as a numbered list.
2) For each fact, write a simple question asking for that fact, and give the exact answer.

Respond with a JSON list of objects, e.g.:
[
  {{ "question": "How many incidents were resolved last month?", "answer": "0" }},
  {{ "question": "What does a zero resolved incident count indicate?", "answer": "backlog or inefficiency" }}
]
"""
    try:
        resp = gpt.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content
    except APIError as e:
        print(f"⚠️ QAEval API error: {e}", file=sys.stderr)
        return []

    # strip code fences
    if raw.strip().startswith("```"):
        lines = raw.strip().splitlines()
        raw = "\n".join(lines[1:-1])

    # grab the first [...] block
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end < 0:
        print(f"⚠️ QAEval: no JSON array found in:\n{raw[:200]}", file=sys.stderr)
        return []

    snippet = raw[start:end]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        print(f"⚠️ QAEval JSON parse error on:\n{snippet}", file=sys.stderr)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 2) Compute fuzzily‐matched coverage
# ─────────────────────────────────────────────────────────────────────────────
def compute_qa_eval(pred_insight: str, ref_insight: str) -> float:
    qa_pairs = extract_qa_pairs(ref_insight)

    # if we couldn't get any questions, fall back to lexical overlap
    if not qa_pairs:
        ref_tokens = set(ref_insight.lower().split())
        pred_tokens = set(pred_insight.lower().split())
        if not ref_tokens:
            return 0.0
        overlap = len(ref_tokens & pred_tokens) / len(ref_tokens)
        return overlap

    correct = 0
    for qa in qa_pairs:
        q = qa.get("question", "").strip()
        ref_ans = str(qa.get("answer", "")).strip().lower()
        if not q or not ref_ans:
            continue

        out = qa_pipeline(question=q, context=pred_insight)
        pred_ans = out.get("answer", "").strip().lower()

        # exact or fuzzy match
        ratio = SequenceMatcher(None, ref_ans, pred_ans).ratio()
        if ref_ans == pred_ans or ref_ans in pred_ans or pred_ans in ref_ans or ratio >= 0.8:
            correct += 1

    return correct / len(qa_pairs)


# ─────────────────────────────────────────────────────────────────────────────
# 3) CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python qa_eval.py '<predicted_insight>' '<ground_truth_insight>'")
        sys.exit(1)

    pred, ref = sys.argv[1], sys.argv[2]
    score = compute_qa_eval(pred, ref)
    print(f"QAEval coverage score: {score:.3f}")
