#!/usr/bin/env python3
"""Rebuild irac_outputs/updated_cot/metrics.csv from the saved prediction JSONs.

This does NOT call any LLM. It re-evaluates every pred_*.json already on disk
against data/ground_truth_irac.json using the same set-based precision / recall /
F1 logic as IRAC_notebook2.ipynb (cell 7), producing the full Q1-Q5 metric table.

Run:  python scripts/rebuild_metrics.py
"""
import json
from pathlib import Path

import pandas as pd
from nltk.metrics.scores import precision, recall, f_measure

GROUND_TRUTH_PATH = Path("ground_truth_irac.json")
OUTDIR = Path("irac_outputs/updated_cot")

# Preferred ordering for a tidy output (model order, then Q1..Q5).
MODEL_ORDER = [
    "anthropic/claude-opus-4.6",
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-pro",
    "openai/gpt-5.4",
    "moonshotai/kimi-k2.5",
]
QUESTION_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def prf(gold_set: set, pred_set: set):
    p = precision(gold_set, pred_set) or 0.0
    r = recall(gold_set, pred_set) or 0.0
    f1 = f_measure(gold_set, pred_set) or 0.0
    return p, r, f1


def eval_one(q: str, pred: dict, gt: dict, model_name: str):
    rows = []
    for concl in ["C1", "C2"]:
        pred_obj = pred[f"irac_{concl}"]
        gt_obj = gt[q][concl]

        pred_rules = {x["id"].strip() for x in pred_obj["rules_selected"]}
        gold_rules = set(gt_obj["rules_selected"])

        pred_edges = {x.strip() for x in pred_obj["edges_support"]}
        gold_edges = set(gt_obj["edges_support"])

        p_r, r_r, f1_r = prf(gold_rules, pred_rules)
        p_e, r_e, f1_e = prf(gold_edges, pred_edges)

        rows += [
            {"model": model_name, "question_number": q, "conclusion": concl,
             "metric": "rules_selected", "precision": p_r, "recall": r_r, "f1": f1_r,
             "gold_size": len(gold_rules), "pred_size": len(pred_rules)},
            {"model": model_name, "question_number": q, "conclusion": concl,
             "metric": "edges_support", "precision": p_e, "recall": r_e, "f1": f1_e,
             "gold_size": len(gold_edges), "pred_size": len(pred_edges)},
        ]
    return rows


def model_from_filename(stem: str, q_num: str) -> str:
    # "pred_anthropic_claude-opus-4.6_Q2" -> "anthropic/claude-opus-4.6"
    name = stem.replace("pred_", "").replace(f"_{q_num}", "")
    if name.endswith("_"):
        name = name[:-1]
    return name.replace("_", "/", 1)


def main():
    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    all_rows = []
    files = sorted(OUTDIR.glob("pred_*.json"))
    for fp in files:
        content = json.loads(fp.read_text(encoding="utf-8"))
        q_num = content["question_number"]
        model_name = model_from_filename(fp.stem, q_num)
        all_rows.extend(eval_one(q_num, content, gt, model_name))

    df = pd.DataFrame(all_rows)

    # Stable, readable ordering.
    df["_m"] = df["model"].apply(
        lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER))
    df["_q"] = df["question_number"].apply(
        lambda q: QUESTION_ORDER.index(q) if q in QUESTION_ORDER else len(QUESTION_ORDER))
    df["_c"] = df["conclusion"].map({"C1": 0, "C2": 1})
    df["_t"] = df["metric"].map({"rules_selected": 0, "edges_support": 1})
    df = df.sort_values(["_m", "_q", "_c", "_t"]).drop(columns=["_m", "_q", "_c", "_t"])

    out = OUTDIR / "metrics_latest.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows from {len(files)} JSON files -> {out}")
    print(df.groupby(["model"])["f1"].mean().reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
