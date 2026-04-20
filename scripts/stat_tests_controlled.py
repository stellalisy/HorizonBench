"""
Statistical significance tests for HorizonBench controlled experiments.

Computes per-model belief-update metrics (static accuracy, evolved accuracy,
gap, distractor rate) for each controlled condition, then runs paired Wilcoxon
signed-rank tests comparing conditions and within-condition tests for the two
signatures of belief-update failure (gap < 0, distractor rate > 25%).

Usage:
    python scripts/stat_tests_controlled.py

Re-run after implicit/explicit results finalize by updating the condition
config entries below (EXPLICIT_DIR, IMPLICIT_DIR, IMPLICIT_HF).

Requirements:
    pip install numpy scipy
"""

import argparse
import json
import glob as globmod
from pathlib import Path
from collections import OrderedDict
import numpy as np
from scipy import stats

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--results-dir", type=str, default="output",
                     help="Base directory containing experiment result directories")
_args, _ = _parser.parse_known_args()

BASE = Path(_args.results_dir)

MODEL_KEYS = {
    "anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude-3.7-sonnet",
    "anthropic.claude-3-5-haiku-20241022-v1:0": "Claude-3.5-haiku",
    "anthropic.claude-opus-4-5-20251101-v1:0": "Claude-opus-4.5",
    "anthropic.claude-opus-4-20250514-v1:0": "Claude-opus-4",
    "anthropic.claude-sonnet-4-20250514-v1:0": "Claude-sonnet-4",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude-sonnet-4.5",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude-3.5-son.-v1",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "Claude-haiku-4.5",
    "gemini-3.1-pro-preview": "Gemini-3.1-pro",
    "gemini-3-pro-preview": "Gemini-3-pro",
    "gemini-2.0-flash": "Gemini-2.0-flash",
    "gemini-3-flash-preview": "Gemini-3-flash",
    "gemini-2.5-pro": "Gemini-2.5-pro",
    "gemini-2.0-flash-lite": "Gemini-2.0-fl.-lite",
    "gemini-2.5-flash-lite": "Gemini-2.5-fl.-lite",
    "gemini-2.5-flash": "Gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview": "Gemini-3.1-fl.-lite",
    "gpt-4o": "gpt-4o",
    "gpt-4.1": "gpt-4.1",
    "o3": "o3",
    "o4-mini": "o4-mini",
    "o3-mini": "o3-mini",
    "o1": "o1",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5": "gpt-5",
}

# ── Condition configuration ──────────────────────────────────────────────────
# Update these paths when new experiment results come in.

CONDITIONS = OrderedDict([
    ("Neutral", {
        "result_dir": BASE / "v7_o3",
        "hf_files": [
            BASE / "v7_o3/v7_o3_filtered_5llm_hf.jsonl",
            BASE / "v7_o3/v7_o3_batch2_filtered_5llm_hf.jsonl",
        ],
        "cf_dirs": [BASE / "v7_o3"],
        "use_pass_filter": True,
        "item_filter": None,
    }),
    ("Short-horizon", {
        "result_dir": BASE / "v12_temporal_short_evolving",
        "hf_files": [BASE / "v12_temporal_short_evolving/v12_temporal_short_evolving_filtered_5llm_hf.jsonl"],
        "cf_dirs": [BASE / "v12_temporal_short_evolving"],
        "use_pass_filter": True,
        "item_filter": "short_horizon_static_ge40",
    }),
    ("Explicit", {
        "result_dir": BASE / "v19_level1v2",
        "hf_files": [BASE / "v19_level1v2/v19_level1v2_filtered_5llm_hf.jsonl"],
        "cf_dirs": [BASE / "v7_o3"],
        "use_pass_filter": True,
        "item_filter": None,
    }),
    ("Low-subtlety", {
        "result_dir": BASE / "v7_o3_majority",
        "hf_files": [BASE / "v7_o3_majority/v7_o3_majority_hf.jsonl"],
        "cf_dirs": [BASE / "v7_o3"],
        "use_pass_filter": False,
        "item_filter": None,
    }),
])


def canon(name):
    for s in ("-v1:0", "-20241022", "-20240620", "-20250219", "-20250514",
              "-20251001", "-20251101", "-20250929", "-preview"):
        name = name.replace(s, "")
    name = name.replace("anthropic.", "")
    return name.replace("_nothink", "")


def load_hf(paths):
    items = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  WARNING: HF file not found: {p}")
            continue
        with open(p) as f:
            for line in f:
                item = json.loads(line)
                items[item["id"]] = item
    return items


def load_preevo(hf_items, cf_dirs):
    preevo = {}
    for tid, hf in hf_items.items():
        if not hf.get("has_evolved"):
            continue
        for d in cf_dirs:
            matches = globmod.glob(f"{d}/user_*/benchmark/temporal_preference_recall/{tid}")
            if matches:
                with open(matches[0]) as f:
                    data = json.load(f)
                for ct in data.get("counterfactual_turns", []):
                    if ct.get("is_pre_evolution_distractor"):
                        for opt in hf["options"]:
                            if opt["value"] == ct["alternative_value"]:
                                preevo[tid] = opt["letter"]
                                break
                        break
                break
    return preevo


def short_horizon_static_ge40(tid, hf):
    if hf.get("has_evolved"):
        return True
    return hf.get("segs", 0) >= 40


ITEM_FILTERS = {
    "short_horizon_static_ge40": short_horizon_static_ge40,
}


def get_per_model_metrics(result_dir, hf_items, preevo,
                          item_filter_fn=None, use_pass_filter=True):
    valid = set(hf_items.keys())
    if item_filter_fn:
        valid = {t for t in valid if item_filter_fn(t, hf_items[t])}
    sta_ids = {t for t in valid if not hf_items[t].get("has_evolved")}
    evo_ids = {t for t in valid if hf_items[t].get("has_evolved")}

    model_metrics = {}
    for rfile in sorted(Path(result_dir).glob("results_*_nothink.jsonl")):
        raw = rfile.name.replace("results_", "").replace("_nothink.jsonl", "")
        canonical = canon(raw)
        mk_list = [k for k in MODEL_KEYS if canon(k) == canonical]
        if not mk_list:
            continue
        mname = MODEL_KEYS[mk_list[0]]

        sta_c, evo_c, ps = [], [], []
        seen = set()
        with open(rfile) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                tid = item.get("test_node_id", "")
                if tid in seen:
                    continue
                seen.add(tid)
                if use_pass_filter and not item.get("pass_filter"):
                    continue
                if not use_pass_filter and tid not in valid:
                    continue
                if use_pass_filter and tid not in valid:
                    continue
                c = 1 if item["correct"] else 0
                if tid in evo_ids:
                    evo_c.append(c)
                    if not c and tid in preevo:
                        resp = item.get("response", {})
                        ans = (resp.get("response_text", "").strip()
                               if isinstance(resp, dict) else str(resp).strip())
                        if ans:
                            ps.append(1.0 if ans[0].upper() == preevo[tid].upper() else 0.0)
                elif tid in sta_ids:
                    sta_c.append(c)

        if sta_c and evo_c:
            model_metrics[mname] = {
                "overall": np.mean(sta_c + evo_c) * 100,
                "static": np.mean(sta_c) * 100,
                "evolved": np.mean(evo_c) * 100,
                "gap": np.mean(evo_c) * 100 - np.mean(sta_c) * 100,
                "dist": np.mean(ps) * 100 if ps else float("nan"),
            }
    return model_metrics


def paired_wilcoxon(cond1, cond2, metric, alternative="two-sided"):
    common = sorted(set(cond1) & set(cond2))
    common = [m for m in common
              if not np.isnan(cond1[m][metric]) and not np.isnan(cond2[m][metric])]
    if len(common) < 3:
        return len(common), float("nan"), float("nan"), float("nan")
    v1 = np.array([cond1[m][metric] for m in common])
    v2 = np.array([cond2[m][metric] for m in common])
    diffs = v2 - v1
    stat, p = stats.wilcoxon(diffs, alternative=alternative)
    return len(common), np.mean(diffs), stat, p


def within_condition_test(cond, metric, null_value, alternative):
    models = sorted(cond)
    vals = [cond[m][metric] for m in models if not np.isnan(cond[m][metric])]
    if len(vals) < 3:
        return len(vals), float("nan"), float("nan"), float("nan")
    shifted = [v - null_value for v in vals]
    stat, p = stats.wilcoxon(shifted, alternative=alternative)
    return len(vals), np.mean(vals), stat, p


def main():
    all_conds = {}

    for cname, cfg in CONDITIONS.items():
        print(f"Loading {cname}...")
        hf = load_hf(cfg["hf_files"])

        if cname == "Short-horizon":
            for tid, item in hf.items():
                item["segs"] = item["conversation"].count("\nDate: ")

        preevo = load_preevo(hf, cfg["cf_dirs"])
        filter_fn = ITEM_FILTERS.get(cfg["item_filter"])
        metrics = get_per_model_metrics(
            cfg["result_dir"], hf, preevo,
            item_filter_fn=filter_fn,
            use_pass_filter=cfg["use_pass_filter"],
        )
        all_conds[cname] = metrics
        sta_ids = {t for t in hf if not hf[t].get("has_evolved")}
        evo_ids = {t for t in hf if hf[t].get("has_evolved")}
        if filter_fn:
            valid = {t for t in hf if filter_fn(t, hf[t])}
            sta_ids &= valid
            evo_ids &= valid
        print(f"  {len(metrics)} models, {len(sta_ids)} static + {len(evo_ids)} evolved items")

    print("\n" + "=" * 70)
    print("BETWEEN-CONDITION TESTS (paired Wilcoxon signed-rank)")
    print("=" * 70)

    neutral = all_conds["Neutral"]

    comparisons = [
        ("Neutral", "Short-horizon", "gap", "less",
         "Short-horizon gap < Neutral gap (gap widens)"),
        ("Neutral", "Short-horizon", "dist", "greater",
         "Short-horizon dist > Neutral dist"),
        ("Neutral", "Explicit", "gap", "greater",
         "Explicit gap > Neutral gap (gap narrows)"),
        ("Neutral", "Explicit", "dist", "greater",
         "Explicit dist > Neutral dist"),
        ("Neutral", "Low-subtlety", "overall", "greater",
         "Low-subtlety overall > Neutral overall"),
        ("Neutral", "Low-subtlety", "dist", "greater",
         "Low-subtlety dist > Neutral dist"),
    ]

    for c1_name, c2_name, metric, alt, description in comparisons:
        c1, c2 = all_conds.get(c1_name, {}), all_conds.get(c2_name, {})
        if not c1 or not c2:
            print(f"\n{description}: SKIPPED (condition not loaded)")
            continue
        n, mean_diff, w, p = paired_wilcoxon(c1, c2, metric, alt)
        print(f"\n{description}")
        print(f"  n={n}, mean diff={mean_diff:+.2f}, W={w:.0f}, p={p:.2e}")

    print("\n" + "=" * 70)
    print("WITHIN-CONDITION TESTS")
    print("=" * 70)

    print("\nGap < 0 (evolved-vs-static accuracy gap is negative):")
    for cname, cond in all_conds.items():
        n, mean_val, w, p = within_condition_test(cond, "gap", 0.0, "less")
        n_neg = sum(1 for m in cond if cond[m]["gap"] < 0)
        print(f"  {cname}: n={n}, mean={mean_val:+.1f}, neg={n_neg}/{n}, "
              f"W={w:.0f}, p={p:.6f}")

    print("\nDistractor > 25% (above chance):")
    for cname, cond in all_conds.items():
        n, mean_val, w, p = within_condition_test(cond, "dist", 25.0, "greater")
        n_above = sum(1 for m in cond if not np.isnan(cond[m]["dist"]) and cond[m]["dist"] > 25)
        print(f"  {cname}: n={n}, mean={mean_val:.1f}%, above={n_above}/{n}, "
              f"W={w:.0f}, p={p:.2e}")

    print("\n" + "=" * 70)
    print("CONDITION SUMMARIES")
    print("=" * 70)
    for cname, cond in all_conds.items():
        gaps = [cond[m]["gap"] for m in cond]
        dists = [cond[m]["dist"] for m in cond if not np.isnan(cond[m]["dist"])]
        stas = [cond[m]["static"] for m in cond]
        evos = [cond[m]["evolved"] for m in cond]
        ovrs = [cond[m]["overall"] for m in cond]
        print(f"\n{cname} ({len(cond)} models):")
        print(f"  Overall: {np.mean(ovrs):.1f}%")
        print(f"  Static:  {np.mean(stas):.1f}%")
        print(f"  Evolved: {np.mean(evos):.1f}%")
        print(f"  Gap:     {np.mean(gaps):+.1f}pp ({sum(1 for g in gaps if g < 0)}/{len(gaps)} neg)")
        print(f"  Dist:    {np.mean(dists):.1f}% ({sum(1 for d in dists if d > 25)}/{len(dists)} >25%)")


if __name__ == "__main__":
    main()
