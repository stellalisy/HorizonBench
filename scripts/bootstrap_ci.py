"""
Bootstrap confidence intervals for all numbers cited in the paper.

Computes 95% CIs via 10,000 bootstrap resamples for:
  1. Overall per-model accuracy (strict filter)
  2. Evolved vs. static accuracy gap
  3. Temporal distance gap (medium 30-59d vs. long >=60d)
  4. Pre-evolution distractor selection rate among wrong answers
  5. Sign-test p-values for directional consistency
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np

N_BOOT = 10_000
SEED = 42
import argparse as _argparse
_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--results-dir", type=str, default="output",
                     help="Directory containing results_*.jsonl and HF JSONL files")
_args, _ = _parser.parse_known_args()
RESULTS_DIR = Path(_args.results_dir)
META_PKL = RESULTS_DIR / "v7_item_meta.pkl"
HF_FILE = RESULTS_DIR / "v7_o3_filtered_5llm_hf.jsonl"

DISPLAY_NAMES = {
    "anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude-3.7-sonnet",
    "anthropic.claude-3-5-haiku-20241022-v1:0": "Claude-3.5-haiku",
    "anthropic.claude-opus-4-5-20251101-v1:0": "Claude-opus-4.5",
    "anthropic.claude-opus-4-20250514-v1:0": "Claude-opus-4",
    "anthropic.claude-sonnet-4-20250514-v1:0": "Claude-sonnet-4",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude-sonnet-4.5",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude-3.5-sonnet-v1",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "Claude-haiku-4.5",
    "gemini-3.1-pro-preview": "Gemini-3.1-pro",
    "gemini-3-pro-preview": "Gemini-3-pro",
    "gemini-2.0-flash": "Gemini-2.0-flash",
    "gemini-3-flash-preview": "Gemini-3-flash",
    "gemini-2.5-pro": "Gemini-2.5-pro",
    "gemini-2.0-flash-lite": "Gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite": "Gemini-2.5-flash-lite",
    "gemini-2.5-flash": "Gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview": "Gemini-3.1-flash-lite",
    "gpt-4o": "gpt-4o",
    "gpt-4.1": "gpt-4.1",
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "o3": "o3",
    "o3-mini": "o3-mini",
    "o4-mini": "o4-mini",
    "o1": "o1",
}


def load_item_metadata():
    """Load per-item metadata (days_gap, has_evolved, filter status)."""
    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)

    id_to_meta = {}
    for full_path, m in meta.items():
        fname = os.path.basename(full_path)
        id_to_meta[fname] = m
    return id_to_meta


def load_preevo_distractor_map():
    """Build mapping: test_node_id -> pre-evolution distractor letter.

    Uses the HF prompts file (value->letter mapping) and the counterfactual
    JSON files (is_pre_evolution_distractor flag).
    """
    # Step 1: Load HF file to get id -> {value: letter} mapping
    # HF ids use the full filename (e.g., "..._counterfactuals.json")
    hf_value_to_letter = {}
    with open(HF_FILE) as f:
        for line in f:
            item = json.loads(line)
            hf_id = item["id"]
            mapping = {}
            for opt in item["options"]:
                mapping[opt["value"]] = opt["letter"]
            hf_value_to_letter[hf_id] = mapping

    # Step 2: For each evolved item, find the pre-evo distractor value from
    #         counterfactual files, then map to letter
    preevo_letter = {}
    for user_dir in sorted(RESULTS_DIR.glob("user_*")):
        bench_dir = user_dir / "benchmark" / "temporal_preference_recall"
        if not bench_dir.exists():
            continue
        for cf_file in bench_dir.glob("*_counterfactuals.json"):
            fname = cf_file.name

            if fname not in hf_value_to_letter:
                continue

            with open(cf_file) as fh:
                data = json.load(fh)

            pe = data.get("preference_evolution", {})
            if not pe.get("has_evolved"):
                continue

            for ct in data.get("counterfactual_turns", []):
                if ct.get("is_pre_evolution_distractor"):
                    alt_val = ct["alternative_value"]
                    letter_map = hf_value_to_letter[fname]
                    if alt_val in letter_map:
                        preevo_letter[fname] = letter_map[alt_val]
                    break

    return preevo_letter


def load_model_results():
    """Load all model results, keeping only strict-filtered items."""
    model_results = {}
    for rfile in sorted(RESULTS_DIR.glob("results_*_nothink.jsonl")):
        model_key = rfile.name.replace("results_", "").replace("_nothink.jsonl", "")
        if model_key not in DISPLAY_NAMES:
            continue

        items = []
        with open(rfile) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("pass_filter"):
                    items.append(item)
        model_results[model_key] = items
    return model_results


def bootstrap_mean(values, rng, n_boot=N_BOOT):
    """Bootstrap CI for a mean (or proportion)."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    return np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def bootstrap_diff(values_a, values_b, rng, n_boot=N_BOOT):
    """Bootstrap CI for difference in means (a - b), paired by index."""
    arr_a = np.array(values_a, dtype=float)
    arr_b = np.array(values_b, dtype=float)
    n_a, n_b = len(arr_a), len(arr_b)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        mean_a = arr_a[rng.integers(0, n_a, size=n_a)].mean()
        mean_b = arr_b[rng.integers(0, n_b, size=n_b)].mean()
        boot_diffs[i] = mean_a - mean_b
    return np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5)


def main():
    rng = np.random.default_rng(SEED)

    print("Loading item metadata...")
    item_meta = load_item_metadata()

    print("Loading pre-evolution distractor map...")
    preevo_letter = load_preevo_distractor_map()
    print(f"  Found {len(preevo_letter)} items with identifiable pre-evo distractor")

    print("Loading model results...")
    model_results = load_model_results()
    print(f"  Loaded {len(model_results)} models\n")

    print("=" * 100)
    print("1. OVERALL PER-MODEL ACCURACY (strict filter)")
    print("=" * 100)
    print(f"{'Model':<25} {'N':>5} {'Acc%':>7} {'CI_lo':>7} {'CI_hi':>7}")
    print("-" * 55)

    model_accuracies = {}
    for model_key in sorted(model_results.keys(), key=lambda k: DISPLAY_NAMES[k]):
        items = model_results[model_key]
        corrects = [1 if it["correct"] else 0 for it in items]
        acc = np.mean(corrects) * 100
        lo, hi = bootstrap_mean(corrects, rng)
        model_accuracies[model_key] = (acc, lo * 100, hi * 100)
        print(f"{DISPLAY_NAMES[model_key]:<25} {len(items):>5} {acc:>6.1f}% [{lo*100:>5.1f}, {hi*100:>5.1f}]")

    print()
    print("=" * 100)
    print("2. EVOLVED vs. STATIC ACCURACY GAP (Evo - Static)")
    print("=" * 100)
    print(f"{'Model':<25} {'Evo%':>7} {'Sta%':>7} {'Δ(pp)':>7} {'CI_lo':>7} {'CI_hi':>7} {'N_evo':>6} {'N_sta':>6}")
    print("-" * 80)

    evo_gaps = {}
    n_negative = 0
    for model_key in sorted(model_results.keys(), key=lambda k: DISPLAY_NAMES[k]):
        items = model_results[model_key]
        evo_correct = []
        sta_correct = []
        for it in items:
            evolved = it.get("has_evolved", False)
            c = 1 if it["correct"] else 0
            if evolved:
                evo_correct.append(c)
            else:
                sta_correct.append(c)

        if not evo_correct or not sta_correct:
            continue

        evo_acc = np.mean(evo_correct) * 100
        sta_acc = np.mean(sta_correct) * 100
        gap = evo_acc - sta_acc
        lo, hi = bootstrap_diff(evo_correct, sta_correct, rng)
        evo_gaps[model_key] = (gap, lo * 100, hi * 100)
        if gap < 0:
            n_negative += 1
        print(f"{DISPLAY_NAMES[model_key]:<25} {evo_acc:>6.1f}% {sta_acc:>6.1f}% {gap:>+6.1f} [{lo*100:>+6.1f}, {hi*100:>+6.1f}] {len(evo_correct):>6} {len(sta_correct):>6}")

    total_models = len(evo_gaps)
    print(f"\nSign test: {n_negative}/{total_models} models show negative gap (evo harder)")
    from scipy.stats import binomtest
    result = binomtest(n_negative, total_models, 0.5, alternative="greater")
    p_sign = result.pvalue
    print(f"Sign test p-value: {p_sign:.6f}")

    # Average gap
    avg_gap = np.mean([v[0] for v in evo_gaps.values()])
    all_evo = []
    all_sta = []
    for model_key in model_results:
        for it in model_results[model_key]:
            if model_key == list(model_results.keys())[0]:
                c = 1 if it["correct"] else 0
                if it.get("has_evolved"):
                    all_evo.append(c)
                else:
                    all_sta.append(c)
    print(f"Average gap across models: {avg_gap:+.1f} pp")

    print()
    print("=" * 100)
    print("3. TEMPORAL DISTANCE GAP (Medium 30-59d vs. Long >=60d)")
    print("=" * 100)
    print(f"{'Model':<25} {'Med%':>7} {'Long%':>7} {'Δ(pp)':>7} {'CI_lo':>7} {'CI_hi':>7} {'N_med':>6} {'N_lng':>6}")
    print("-" * 80)

    temporal_gaps = {}
    n_positive_temporal = 0
    for model_key in sorted(model_results.keys(), key=lambda k: DISPLAY_NAMES[k]):
        items = model_results[model_key]
        med_correct = []
        long_correct = []
        for it in items:
            test_id = it["test_node_id"]
            m = item_meta.get(test_id)
            if m is None:
                continue
            days = m["days_gap"]
            c = 1 if it["correct"] else 0
            if 30 <= days < 60:
                med_correct.append(c)
            elif days >= 60:
                long_correct.append(c)

        if not med_correct or not long_correct:
            continue

        med_acc = np.mean(med_correct) * 100
        long_acc = np.mean(long_correct) * 100
        gap = long_acc - med_acc
        lo, hi = bootstrap_diff(long_correct, med_correct, rng)
        temporal_gaps[model_key] = (gap, lo * 100, hi * 100)
        if gap > 0:
            n_positive_temporal += 1
        print(f"{DISPLAY_NAMES[model_key]:<25} {med_acc:>6.1f}% {long_acc:>6.1f}% {gap:>+6.1f} [{lo*100:>+6.1f}, {hi*100:>+6.1f}] {len(med_correct):>6} {len(long_correct):>6}")

    total_temporal = len(temporal_gaps)
    print(f"\nSign test: {n_positive_temporal}/{total_temporal} models show positive gap (long > medium)")

    print()
    print("=" * 100)
    print("4. PRE-EVOLUTION DISTRACTOR SELECTION RATE (among wrong answers on evolved items)")
    print("=" * 100)
    print(f"{'Model':<25} {'N_wrong':>8} {'PreEvo%':>8} {'CI_lo':>7} {'CI_hi':>7} {'p-value':>10}")
    print("-" * 75)

    from scipy.stats import binom

    preevo_rates = {}
    for model_key in sorted(model_results.keys(), key=lambda k: DISPLAY_NAMES[k]):
        items = model_results[model_key]
        selected_preevo = []

        for it in items:
            if it["correct"]:
                continue
            if not it.get("has_evolved"):
                continue
            test_id = it["test_node_id"]
            if test_id not in preevo_letter:
                continue

            model_answer = it["response"]["response_text"].strip()
            if len(model_answer) > 1:
                model_answer = model_answer[0]
            preevo_let = preevo_letter[test_id]
            selected_preevo.append(1 if model_answer == preevo_let else 0)

        if not selected_preevo:
            continue

        rate = np.mean(selected_preevo) * 100
        lo, hi = bootstrap_mean(selected_preevo, rng)
        n_selected = sum(selected_preevo)
        n_total = len(selected_preevo)
        p_val = 1 - binom.cdf(n_selected - 1, n_total, 0.25)

        preevo_rates[model_key] = (rate, lo * 100, hi * 100)
        print(f"{DISPLAY_NAMES[model_key]:<25} {n_total:>8} {rate:>7.1f}% [{lo*100:>5.1f}, {hi*100:>5.1f}] {p_val:>10.2e}")

    avg_rate = np.mean([v[0] for v in preevo_rates.values()])
    print(f"\nAverage pre-evo distractor rate: {avg_rate:.1f}%")
    print(f"Null (uniform among 4 wrong): 25.0%")

    print()
    print("=" * 100)
    print("5. SUMMARY FOR PAPER")
    print("=" * 100)

    # Best model CI
    best_key = max(model_accuracies, key=lambda k: model_accuracies[k][0])
    best_acc, best_lo, best_hi = model_accuracies[best_key]
    print(f"Best model: {DISPLAY_NAMES[best_key]} = {best_acc:.1f}% [95% CI: {best_lo:.1f}, {best_hi:.1f}]")

    print(f"Models whose CI upper bound < 20% (significantly below chance): "
          f"{sum(1 for k,v in model_accuracies.items() if v[2] < 20)}")
    print(f"Models with point estimate <= 20%: "
          f"{sum(1 for k,v in model_accuracies.items() if v[0] <= 20)}")

    # Evo-static
    print(f"\nEvolved-vs-static gap: {n_negative}/{total_models} negative (sign test p={p_sign:.4f})")
    avg_evo_gap = np.mean([v[0] for v in evo_gaps.values()])
    print(f"Average gap: {avg_evo_gap:+.1f} pp")

    # Temporal
    print(f"\nTemporal gap: {n_positive_temporal}/{total_temporal} models show long > medium")

    # Pre-evo distractor
    print(f"\nPre-evo distractor: avg {avg_rate:.1f}% (all models p < 0.001: "
          f"{all(v[1] > 25 for v in preevo_rates.values())})")

    # Table 1 caption guidance: CI width at different accuracy levels
    print("\nCI width guidance for Table 1 caption:")
    for target_p in [0.15, 0.25, 0.50, 0.60]:
        se = np.sqrt(target_p * (1 - target_p) / 831)
        hw = 1.96 * se * 100
        print(f"  p={target_p:.0%}: ±{hw:.1f} pp")


if __name__ == "__main__":
    main()
