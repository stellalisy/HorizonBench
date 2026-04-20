"""
Per-model belief-update metrics: O3 baseline vs controlled conditions.

Uses only O3 subset (v7_o3) as the main benchmark baseline, making comparisons
apples-to-apples with verbal (v13), behavioral (v14), majority (v7_majority),
and short-horizon (v12) conditions, all of which derive from O3-generated data.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

import argparse as _argparse
_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--results-dir", type=str, default="output",
                     help="Base directory containing experiment output dirs")
_args, _ = _parser.parse_known_args()
BASE = Path(_args.results_dir)

MODELS = {
    "anthropic.claude-3-7-sonnet-20250219-v1:0": ("Claude-3.7-son", "Claude"),
    "anthropic.claude-3-5-haiku-20241022-v1:0": ("Claude-3.5-hai", "Claude"),
    "anthropic.claude-opus-4-5-20251101-v1:0": ("Claude-op-4.5", "Claude"),
    "anthropic.claude-opus-4-20250514-v1:0": ("Claude-opus-4", "Claude"),
    "anthropic.claude-sonnet-4-20250514-v1:0": ("Claude-son-4", "Claude"),
    "anthropic.claude-sonnet-4-5-20250929-v1:0": ("Claude-son-4.5", "Claude"),
    "anthropic.claude-3-5-sonnet-20240620-v1:0": ("Claude-3.5-son", "Claude"),
    "anthropic.claude-haiku-4-5-20251001-v1:0": ("Claude-hai-4.5", "Claude"),
    "gemini-3.1-pro-preview": ("Gem-3.1-pro", "Gemini"),
    "gemini-3-pro-preview": ("Gem-3-pro", "Gemini"),
    "gemini-2.0-flash": ("Gem-2.0-fl", "Gemini"),
    "gemini-3-flash-preview": ("Gem-3-fl", "Gemini"),
    "gemini-2.5-pro": ("Gem-2.5-pro", "Gemini"),
    "gemini-2.0-flash-lite": ("Gem-2.0-fl-l", "Gemini"),
    "gemini-2.5-flash-lite": ("Gem-2.5-fl-l", "Gemini"),
    "gemini-2.5-flash": ("Gem-2.5-fl", "Gemini"),
    "gemini-3.1-flash-lite-preview": ("Gem-3.1-fl-l", "Gemini"),
    "gpt-4o": ("GPT-4o", "OpenAI"),
    "gpt-4.1": ("GPT-4.1", "OpenAI"),
    "o3": ("o3", "OpenAI"),
    "o4-mini": ("o4-mini", "OpenAI"),
    "o3-mini": ("o3-mini", "OpenAI"),
    "o1": ("o1", "OpenAI"),
    "gpt-5-mini": ("GPT-5-mini", "OpenAI"),
    "gpt-5": ("GPT-5", "OpenAI"),
}

FAMILY_ORDER = ["Claude", "Gemini", "OpenAI"]

CONDITIONS = {
    "O3 baseline": {
        "result_dirs": [BASE / "v7_o3"],
        "hf_files": [
            BASE / "v7_o3/v7_o3_filtered_5llm_hf.jsonl",
            BASE / "v7_o3/v7_o3_batch2_filtered_5llm_hf.jsonl",
        ],
        "cf_dirs": [BASE / "v7_o3"],
        "filter_pass": True,
    },
    "Short-horizon": {
        "result_dirs": [BASE / "v12_temporal_short_evolving"],
        "hf_files": [BASE / "v12_temporal_short_evolving/v12_temporal_short_evolving_filtered_5llm_hf.jsonl"],
        "cf_dirs": [BASE / "v12_temporal_short_evolving"],
        "filter_pass": True,
    },
    "Verbal": {
        "result_dirs": [BASE / "v13_explicit"],
        "hf_files": [BASE / "v13_explicit/v13_explicit_filtered_5llm_hf.jsonl"],
        "cf_dirs": [BASE / "v7_o3"],
        "filter_pass": True,
    },
    "Behavioral": {
        "result_dirs": [BASE / "v14_implicit_full"],
        "hf_files": [BASE / "v14_implicit_full/v14_implicit_full_filtered_5llm_hf.jsonl"],
        "cf_dirs": [BASE / "v7_o3"],
        "filter_pass": True,
    },
    "Low subtlety": {
        "result_dirs": [BASE / "v7_o3_majority"],
        "hf_files": [BASE / "v7_o3_majority/v7_o3_majority_hf.jsonl"],
        "cf_dirs": [BASE / "v7_o3"],
        "filter_pass": False,
    },
}


def canon(name):
    for s in ("-v1:0", "-20241022", "-20240620", "-20250219",
              "-20250514", "-20251001", "-20251101", "-20250929",
              "-preview"):
        name = name.replace(s, "")
    name = name.replace("anthropic.", "")
    return name.replace("_nothink", "")


def load_results(result_dirs, filter_pass=True):
    combined = defaultdict(list)
    for ds_path in result_dirs:
        for rfile in sorted(ds_path.glob("results_*_nothink.jsonl")):
            raw = rfile.name.replace("results_", "").replace("_nothink.jsonl", "")
            canonical = canon(raw)
            matching = [k for k in MODELS if canon(k) == canonical]
            if not matching:
                continue
            mk = matching[0]
            seen = set()
            with open(rfile) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    tid = item.get("test_node_id", "")
                    if tid in seen:
                        continue
                    seen.add(tid)
                    if filter_pass and not item.get("pass_filter", True):
                        continue
                    combined[mk].append(item)
    return combined


def load_preevo_map(hf_files, cf_dirs):
    hf_val_to_letter = {}
    for hf_path in hf_files:
        if not hf_path.exists():
            continue
        with open(hf_path) as f:
            for line in f:
                item = json.loads(line)
                mapping = {opt["value"]: opt["letter"] for opt in item.get("options", [])}
                hf_val_to_letter[item["id"]] = mapping

    preevo = {}
    for ds_path in cf_dirs:
        for ud in sorted(ds_path.glob("user_*")):
            bd = ud / "benchmark" / "temporal_preference_recall"
            if not bd.exists():
                continue
            for cf in bd.glob("*_counterfactuals.json"):
                if cf.name not in hf_val_to_letter:
                    continue
                with open(cf) as fh:
                    data = json.load(fh)
                pe = data.get("preference_evolution", {})
                if not pe.get("has_evolved"):
                    continue
                for ct in data.get("counterfactual_turns", []):
                    if ct.get("is_pre_evolution_distractor"):
                        av = str(ct["alternative_value"])
                        lm = hf_val_to_letter[cf.name]
                        if av in lm:
                            preevo[cf.name] = lm[av]
                        break
    return preevo


def compute_metrics(items, preevo):
    evo_c, sta_c, all_c = [], [], []
    dist_ind = []

    for it in items:
        c = 1 if it.get("correct", False) else 0
        evolved = it.get("has_evolved", False)
        all_c.append(c)
        if evolved:
            evo_c.append(c)
        else:
            sta_c.append(c)

        if not c and evolved:
            tid = it.get("test_node_id", "")
            if tid in preevo:
                resp = it.get("response", {})
                ans = resp.get("response_text", "").strip() if isinstance(resp, dict) else str(resp).strip()
                if len(ans) > 1:
                    ans = ans[0]
                dist_ind.append(1.0 if ans.upper() == preevo[tid].upper() else 0.0)

    return {
        "overall": np.mean(all_c) * 100 if all_c else None,
        "evolved": np.mean(evo_c) * 100 if evo_c else None,
        "static": np.mean(sta_c) * 100 if sta_c else None,
        "gap": (np.mean(evo_c) * 100 - np.mean(sta_c) * 100) if evo_c and sta_c else None,
        "dist": np.mean(dist_ind) * 100 if dist_ind else None,
        "n": len(all_c),
        "n_evo": len(evo_c),
        "n_sta": len(sta_c),
        "n_dw": len(dist_ind),
    }


def main():
    cond_data = {}
    for cname, cfg in CONDITIONS.items():
        print(f"Loading {cname}...")
        results = load_results(cfg["result_dirs"], cfg["filter_pass"])
        preevo = load_preevo_map(cfg["hf_files"], cfg["cf_dirs"])
        metrics = {}
        for mk in results:
            if mk in MODELS:
                metrics[mk] = compute_metrics(results[mk], preevo)
        cond_data[cname] = metrics
        print(f"  {len(metrics)} models, {len(preevo)} pre-evo items")

    cond_names = list(CONDITIONS.keys())

    # Per-condition detailed tables
    for cname in cond_names:
        metrics = cond_data[cname]
        print(f"\n{'='*100}")
        print(f"  {cname}")
        print(f"{'='*100}")
        print(f"{'Model':<16} {'Fam':<7} {'N':>5} {'Ovr%':>6} {'Evo%':>6} {'Sta%':>6} {'Gap':>6} {'N_e':>5} {'Dist%':>6} {'N_dw':>5}")
        print("-" * 100)

        for fam in FAMILY_ORDER:
            fam_models = [(mk, MODELS[mk][0]) for mk in metrics if MODELS[mk][1] == fam]
            fam_models.sort(key=lambda x: -(metrics[x[0]]["overall"] or 0))
            for mk, dn in fam_models:
                m = metrics[mk]
                g = f"{m['gap']:+5.1f}" if m['gap'] is not None else "  N/A"
                d = f"{m['dist']:5.1f}" if m['dist'] is not None else "  N/A"
                e = f"{m['evolved']:5.1f}" if m['evolved'] is not None else "  N/A"
                s = f"{m['static']:5.1f}" if m['static'] is not None else "  N/A"
                o = f"{m['overall']:5.1f}" if m['overall'] is not None else "  N/A"
                print(f"{dn:<16} {fam:<7} {m['n']:>5} {o} {e} {s} {g} {m['n_evo']:>5} {d} {m['n_dw']:>5}")

        # Aggregates
        vals = list(metrics.values())
        gaps = [m["gap"] for m in vals if m["gap"] is not None]
        dists = [m["dist"] for m in vals if m["dist"] is not None]
        n_neg = sum(1 for g in gaps if g < 0)
        n_d25 = sum(1 for d in dists if d > 25)
        print(f"\n  Avg: Ovr={np.mean([m['overall'] for m in vals if m['overall']]):.1f}%, "
              f"Evo={np.mean([m['evolved'] for m in vals if m['evolved']]):.1f}%, "
              f"Sta={np.mean([m['static'] for m in vals if m['static']]):.1f}%, "
              f"Gap={np.mean(gaps):+.1f}pp ({n_neg}/{len(gaps)} neg), "
              f"Dist={np.mean(dists):.1f}% ({n_d25}/{len(dists)} >25%)")

    # CROSS-CONDITION SUMMARY TABLE
    print(f"\n\n{'='*160}")
    print("  CROSS-CONDITION SUMMARY (per model)")
    print(f"{'='*160}")

    hdr = f"{'Model':<16} {'Fam':<7}"
    for cn in cond_names:
        hdr += f" | {'Ovr':>5} {'Gap':>6} {'Dst':>5}"
    print(hdr)
    print("-" * 160)

    for fam in FAMILY_ORDER:
        fam_models = [(mk, MODELS[mk][0]) for mk in MODELS if MODELS[mk][1] == fam and mk in cond_data["O3 baseline"]]
        fam_models.sort(key=lambda x: -(cond_data["O3 baseline"].get(x[0], {}).get("overall", 0) or 0))

        for mk, dn in fam_models:
            row = f"{dn:<16} {fam:<7}"
            for cn in cond_names:
                m = cond_data[cn].get(mk)
                if m and m.get("overall") is not None:
                    g = f"{m['gap']:+5.1f}" if m.get("gap") is not None else "  N/A"
                    d = f"{m['dist']:5.1f}" if m.get("dist") is not None else "  N/A"
                    row += f" | {m['overall']:5.1f} {g} {d}"
                else:
                    row += f" |   N/A   N/A   N/A"
            print(row)
        if fam_models:
            # Family averages
            row = f"  {'avg '+fam:<14} {'':7}"
            for cn in cond_names:
                fam_vals = [cond_data[cn][mk] for mk, _ in fam_models if mk in cond_data[cn]]
                if fam_vals:
                    ao = np.mean([m["overall"] for m in fam_vals if m["overall"] is not None])
                    ag = np.mean([m["gap"] for m in fam_vals if m["gap"] is not None])
                    ad = np.mean([m["dist"] for m in fam_vals if m["dist"] is not None])
                    row += f" | {ao:5.1f} {ag:+5.1f} {ad:5.1f}"
                else:
                    row += f" |   N/A   N/A   N/A"
            print(row)
            print()

    # Overall averages
    row = f"  {'ALL avg':<14} {'':7}"
    for cn in cond_names:
        vals = list(cond_data[cn].values())
        ao = np.mean([m["overall"] for m in vals if m["overall"] is not None])
        ag = np.mean([m["gap"] for m in vals if m["gap"] is not None])
        ad = np.mean([m["dist"] for m in vals if m["dist"] is not None])
        ng = sum(1 for m in vals if m["gap"] is not None and m["gap"] < 0)
        nt = sum(1 for m in vals if m["gap"] is not None)
        nd = sum(1 for m in vals if m["dist"] is not None and m["dist"] > 25)
        ndt = sum(1 for m in vals if m["dist"] is not None)
        row += f" | {ao:5.1f} {ag:+5.1f} {ad:5.1f}"
    print("-" * 160)
    print(row)

    # Sign test summary
    print(f"\n{'='*100}")
    print("  SIGN TEST SUMMARY")
    print(f"{'='*100}")
    print(f"{'Condition':<18} {'Gap<0':>8} {'Dist>25%':>10}")
    for cn in cond_names:
        vals = list(cond_data[cn].values())
        ng = sum(1 for m in vals if m["gap"] is not None and m["gap"] < 0)
        nt = sum(1 for m in vals if m["gap"] is not None)
        nd = sum(1 for m in vals if m["dist"] is not None and m["dist"] > 25)
        ndt = sum(1 for m in vals if m["dist"] is not None)
        print(f"{cn:<18} {ng:>3}/{nt:<4} {nd:>4}/{ndt:<4}")


if __name__ == "__main__":
    main()
