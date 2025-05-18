#!/usr/bin/env python3
"""
Evaluate all datasets under <root_dir> with:
  - BARTScore
  - MoverScore
  - HybridScore (BERTScore + QAEval + SummaQA)

Writes a single CSV `evaluation_all_results.csv`.
"""
import os
import sys
import json
import pandas as pd

# from bartscore_eval import compute_bartscore
from frugalscore_eval import compute_frugalscore
from hybrid import compute_hybrid_score

import re
import numpy as np

from openai import OpenAI
from evaluate import load
from prompts import get_g_eval_prompt

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("❌ export OPENAI_API_KEY first")
# if not OPENROUTER_API_KEY:
#     sys.exit("❌ export OPENROUTER_API_KEY first")

# GPT‐4O Eval client (official OpenAI)
g_client = OpenAI(api_key=OPENAI_API_KEY)

# OpenRouter client (for Llama3)
# router = OpenAI(
#     api_key=OPENROUTER_API_KEY,
#     base_url="https://openrouter.ai/api/v1"
# )


def compute_g_eval(predictions: list[str], references: list[str], model_name="gpt-4o") -> list[float]:
    scores=[]
    for pred,ref in zip(predictions,references):
        tpl, system = get_g_eval_prompt(method="basic")
        prompt = tpl.format(answer=pred, gt_answer=ref)

        resp = g_client.chat.completions.create(
            model=model_name,
            messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip()
        print("▶️ G-Eval raw output:", repr(raw))
        m = re.search(r"<rating>(\d+)</rating>", raw)
        scores.append(float(m.group(1)) / 10.0 if m else 0.0)
    return scores 


def evaluate_all(root_dir: str, out_csv: str = "batch3_improved_metricresults.csv"):
    records = []

    for ds in sorted(os.listdir(root_dir)):
        ds_path   = os.path.join(root_dir, ds)
        json_path = os.path.join(ds_path, "ques_ans.json")
        if not os.path.isdir(ds_path) or not os.path.isfile(json_path):
            continue

        entries = json.load(open(json_path, "r", encoding="utf-8"))
        n = len(entries)
        print(f"\n▶️  Dataset: {ds} ({n} examples)")

        # 1) batch‐collect all preds & gts
        preds = [ e.get("predicted_insight","") or "" for e in entries ]
        gts   = [ e.get("gt_answer",        "") or "" for e in entries ]

        # 2) run each metric _once_ over the batch
        # bs_list = compute_bartscore(preds, gts)
        fg_list = compute_frugalscore(preds, gts)
        hy_list = [ compute_hybrid_score(p, g) 
                    for p, g in zip(preds, gts) ]
        gs_list = compute_g_eval(preds, gts)
        
        # 3) zip & emit
        for i, (e, fg, hy, gs) in enumerate(
                zip(entries,fg_list, hy_list, gs_list),
                start=1):
            pred = e.get("predicted_insight","")
            gt   = e.get("gt_answer","")
            print(f"  [{i:>3}/{n}]  "
                  f" Frugal={fg: .3f}  "
                  f"Hybrid={hy: .3f}  Geval={gs: .3f}")

            records.append({
                "dataset":             ds,
                "idx":                 i,
                "predicted_insight":   pred,
                "ground_truth_insight":gt,
                # "bart_score":          bs,
                "frugal_score":        fg,
                "hybrid_score":        hy,
                "geval_score":         gs
            })

    # 4) dump to CSV
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved consolidated results to {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <root_dataset_dir>")
        sys.exit(1)
    evaluate_all(sys.argv[1])