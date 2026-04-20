"""
Generate plots for the paper on the combined multi-generator dataset (V6+V7+V8).

Figures produced:
  1. Overall accuracy bar chart per model.
  2. Conditional pre-evolution distractor rate per model (among wrong answers on
     evolved items with a pre-evo distractor, what fraction picked the distractor).
  3. Evolved vs static grouped bar chart.

Run:  python scripts/plot_model_accuracy.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

N_BOOT = 10_000
SEED = 42

import argparse as _argparse

_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--results-dir", type=str, default="output",
                     help="Base directory containing v6_sonnet45/, v7_o3/, v8_gemini3flash/ result dirs")
_parser.add_argument("--output-dir", type=str, default="figures",
                     help="Directory to write PDF figures")
_args, _ = _parser.parse_known_args()

_base = Path(_args.results_dir)
DATASETS = {
    "v6": _base / "v6_sonnet45",
    "v7": _base / "v7_o3",
    "v8": _base / "v8_gemini3flash",
}

HF_FILES = {
    "v6": [DATASETS["v6"] / "v6_sonnet45_filtered_5llm_hf.jsonl"],
    "v7": [DATASETS["v7"] / "v7_o3_filtered_5llm_hf.jsonl",
           DATASETS["v7"] / "v7_o3_batch2_filtered_5llm_hf.jsonl"],
    "v8": [DATASETS["v8"] / "v8_gemini3flash_filtered_5llm_hf.jsonl"],
}

OUT_DIR = Path(_args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "anthropic.claude-3-7-sonnet-20250219-v1:0": ("Claude-3.7-sonnet", "Claude"),
    "anthropic.claude-3-5-haiku-20241022-v1:0": ("Claude-3.5-haiku", "Claude"),
    "anthropic.claude-opus-4-5-20251101-v1:0": ("Claude-opus-4.5", "Claude"),
    "anthropic.claude-opus-4-20250514-v1:0": ("Claude-opus-4", "Claude"),
    "anthropic.claude-sonnet-4-20250514-v1:0": ("Claude-sonnet-4", "Claude"),
    "anthropic.claude-sonnet-4-5-20250929-v1:0": ("Claude-sonnet-4.5", "Claude"),
    "anthropic.claude-3-5-sonnet-20240620-v1:0": ("Claude-3.5-sonnet-v1", "Claude"),
    "anthropic.claude-haiku-4-5-20251001-v1:0": ("Claude-haiku-4.5", "Claude"),
    "gemini-3.1-pro-preview": ("Gemini-3.1-pro", "Gemini"),
    "gemini-3-pro-preview": ("Gemini-3-pro", "Gemini"),
    "gemini-2.0-flash": ("Gemini-2.0-flash", "Gemini"),
    "gemini-3-flash-preview": ("Gemini-3-flash", "Gemini"),
    "gemini-2.5-pro": ("Gemini-2.5-pro", "Gemini"),
    "gemini-2.0-flash-lite": ("Gemini-2.0-flash-lite", "Gemini"),
    "gemini-2.5-flash-lite": ("Gemini-2.5-flash-lite", "Gemini"),
    "gemini-2.5-flash": ("Gemini-2.5-flash", "Gemini"),
    "gemini-3.1-flash-lite-preview": ("Gemini-3.1-flash-lite", "Gemini"),
    "gpt-4o": ("GPT-4o", "OpenAI"),
    "gpt-4.1": ("GPT-4.1", "OpenAI"),
    "o3": ("o3", "OpenAI"),
    "o4-mini": ("o4-mini", "OpenAI"),
    "o3-mini": ("o3-mini", "OpenAI"),
    "o1": ("o1", "OpenAI"),
    "gpt-5-mini": ("GPT-5-mini", "OpenAI"),
    "gpt-5": ("GPT-5", "OpenAI"),
}

FAMILY_COLORS = {
    "Claude": "#6366f1",
    "Gemini": "#10b981",
    "OpenAI": "#f59e0b",
}

FAMILY_ORDER = ["Claude", "Gemini", "OpenAI"]


def canon(name):
    for s in ("-v1:0", "-20241022", "-20240620", "-20250219",
              "-20250514", "-20251001", "-20251101", "-20250929",
              "-preview"):
        name = name.replace(s, "")
    name = name.replace("anthropic.", "")
    return name.replace("_nothink", "")


def bootstrap_ci(values, rng, n_boot=N_BOOT, alpha=0.05):
    arr = np.array(values, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    return (np.percentile(boot_means, 100 * alpha / 2),
            np.percentile(boot_means, 100 * (1 - alpha / 2)))


# ── Load combined-dataset results across V6/V7/V8 ──────────────────────────

def load_combined_results():
    """Return {model_key: [list of result dicts]} across all generators."""
    combined = defaultdict(list)
    for ds_name, ds_path in DATASETS.items():
        for rfile in sorted(ds_path.glob("results_*_nothink.jsonl")):
            raw = rfile.name.replace("results_", "").replace("_nothink.jsonl", "")
            canonical = canon(raw)
            matching_keys = [k for k, (_, _) in MODELS.items() if canon(k) == canonical]
            if not matching_keys:
                continue
            mk = matching_keys[0]
            seen = set()
            with open(rfile) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tid = item.get("test_node_id", "")
                    if tid in seen:
                        continue
                    seen.add(tid)
                    combined[mk].append(item)
    return combined


# ── Load pre-evo distractor letter map (all datasets) ───────────────────────

def load_preevo_distractor_map():
    """Map item_id -> pre-evolution distractor letter across V6/V7/V8."""
    hf_value_to_letter = {}
    for ds_name, hf_paths in HF_FILES.items():
        for hf_path in hf_paths:
            if not hf_path.exists():
                continue
            with open(hf_path) as f:
                for line in f:
                    item = json.loads(line)
                    mapping = {opt["value"]: opt["letter"]
                               for opt in item.get("options", [])}
                    hf_value_to_letter[item["id"]] = mapping

    preevo_letter = {}
    for ds_name, ds_path in DATASETS.items():
        for user_dir in sorted(ds_path.glob("user_*")):
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
                        alt_val = str(ct["alternative_value"])
                        letter_map = hf_value_to_letter[fname]
                        if alt_val in letter_map:
                            preevo_letter[fname] = letter_map[alt_val]
                        break
    return preevo_letter


def main():
    rng = np.random.default_rng(SEED)
    combined = load_combined_results()
    preevo_letter = load_preevo_distractor_map()
    print(f"Loaded combined results for {len(combined)} models")
    print(f"Pre-evo distractor mapping: {len(preevo_letter)} items")

    # ── Figure 1: Overall accuracy bar chart ─────────────────────────────────

    family_rows = {f: [] for f in FAMILY_ORDER}
    for mk, (dname, family) in MODELS.items():
        if mk not in combined:
            continue
        items = combined[mk]
        n_total = len(items)
        if n_total == 0:
            continue
        corrects = [1.0 if it.get("correct", False) else 0.0 for it in items]
        acc = np.mean(corrects) * 100
        lo, hi = bootstrap_ci(corrects, rng)
        family_rows[family].append((dname, acc, lo * 100, hi * 100))

    for f in FAMILY_ORDER:
        family_rows[f].sort(key=lambda r: r[1], reverse=True)

    ordered = []
    for f in FAMILY_ORDER:
        ordered.extend([(dn, f, acc, lo, hi)
                        for dn, acc, lo, hi in family_rows[f]])

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(ordered))
    bar_w = 0.72

    for i, (dname, family, acc, lo, hi) in enumerate(ordered):
        color = FAMILY_COLORS[family]
        ax.bar(i, acc, width=bar_w, color=color, alpha=0.90,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(i, acc, yerr=[[acc - lo], [hi - acc]], fmt="none",
                    ecolor="#374151", elinewidth=1.0, capsize=2.5, capthick=1.0)

    ax.axhline(y=20, color="#6b7280", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(len(ordered) - 0.5, 21, "chance (20%)", fontsize=9.5, color="#6b7280",
            ha="right", va="bottom")

    cum = 0
    for f in FAMILY_ORDER:
        n = len(family_rows[f])
        if cum > 0:
            ax.axvline(x=cum - 0.5, color="#9ca3af", linewidth=0.8, linestyle="-")
        cum += n

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in ordered], fontsize=12, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 58)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlim(-0.6, len(ordered) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fam_legend = [Patch(facecolor=FAMILY_COLORS[f], alpha=0.90, label=f)
                  for f in FAMILY_ORDER]
    ax.legend(handles=fam_legend, loc="upper right", fontsize=12,
              framealpha=0.9, ncol=3, title="Model family", title_fontsize=12)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "accuracy.pdf", bbox_inches="tight", dpi=300)
    print(f"Saved {OUT_DIR / 'accuracy.pdf'}")
    plt.close()

    # ── Figure 2: Conditional pre-evo distractor rate ────────────────────────
    # Among wrong answers on evolved items with a pre-evo distractor option,
    # what fraction selected the distractor?

    dist_family_rows = {f: [] for f in FAMILY_ORDER}
    model_order_names = [r[0] for r in ordered]

    for mk, (dname, family) in MODELS.items():
        if mk not in combined:
            continue
        items = combined[mk]

        indicators = []
        for it in items:
            tid = it.get("test_node_id", "")
            if tid not in preevo_letter:
                continue
            if it.get("correct", False):
                continue
            resp = it.get("response", {})
            ans = resp.get("response_text", "").strip() if isinstance(resp, dict) else str(resp).strip()
            if len(ans) > 1:
                ans = ans[0]
            indicators.append(1.0 if ans.upper() == preevo_letter[tid].upper() else 0.0)

        if not indicators:
            continue
        rate = np.mean(indicators) * 100
        lo, hi = bootstrap_ci(indicators, rng)
        dist_family_rows[family].append((dname, rate, lo * 100, hi * 100))

    for f in FAMILY_ORDER:
        dist_family_rows[f].sort(
            key=lambda r: model_order_names.index(r[0])
            if r[0] in model_order_names else 999)

    dist_ordered = []
    for f in FAMILY_ORDER:
        dist_ordered.extend([(dn, f, rate, lo, hi)
                             for dn, rate, lo, hi in dist_family_rows[f]])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dist_ordered))
    bar_w = 0.72

    for i, (dname, family, rate, lo, hi) in enumerate(dist_ordered):
        color = FAMILY_COLORS[family]
        ax.bar(i, rate, width=bar_w, color=color, alpha=0.90,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(i, rate, yerr=[[rate - lo], [hi - rate]], fmt="none",
                    ecolor="#374151", elinewidth=1.0, capsize=2.5, capthick=1.0)

    ax.axhline(y=25, color="#ef4444", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(len(dist_ordered) - 0.5, 26, "uniform error (25%)", fontsize=14,
            color="#ef4444", ha="right", va="bottom")

    cum = 0
    for f in FAMILY_ORDER:
        n = len(dist_family_rows[f])
        if cum > 0:
            ax.axvline(x=cum - 0.5, color="#9ca3af", linewidth=0.8, linestyle="-")
        cum += n

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in dist_ordered], fontsize=14, rotation=45,
                       ha="right")
    ax.set_ylabel("Pre-evo distractor selection rate (%)", fontsize=14)
    ax.set_ylim(0, 55)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlim(-0.6, len(dist_ordered) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fam_legend = [Patch(facecolor=FAMILY_COLORS[f], alpha=0.90, label=f)
                  for f in FAMILY_ORDER]
    ax.legend(handles=fam_legend, loc="upper right", fontsize=13,
              framealpha=0.9, ncol=3, title="Model family", title_fontsize=13)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "distractor_rate.pdf", bbox_inches="tight", dpi=300)
    print(f"Saved {OUT_DIR / 'distractor_rate.pdf'}")
    plt.close()

    # ── Figure 2: Evolved vs Static – grouped vertical bars (combined) ──────

    evo_rows = {f: [] for f in FAMILY_ORDER}
    for mk, (dname, family) in MODELS.items():
        if mk not in combined:
            continue
        items = combined[mk]
        evo_c = [1.0 if it.get("correct", False) else 0.0
                 for it in items if it.get("has_evolved", False)]
        sta_c = [1.0 if it.get("correct", False) else 0.0
                 for it in items if not it.get("has_evolved", False)]
        if not evo_c or not sta_c:
            continue
        evo_acc = np.mean(evo_c) * 100
        sta_acc = np.mean(sta_c) * 100
        elo, ehi = bootstrap_ci(evo_c, rng)
        slo, shi = bootstrap_ci(sta_c, rng)
        evo_rows[family].append((dname, evo_acc, sta_acc,
                                 elo*100, ehi*100, slo*100, shi*100))

    for f in FAMILY_ORDER:
        evo_rows[f].sort(key=lambda r: (r[1] + r[2]) / 2, reverse=True)

    evo_ordered = []
    for f in FAMILY_ORDER:
        evo_ordered.extend([(dn, f, ea, sa, elo, ehi, slo, shi)
                            for dn, ea, sa, elo, ehi, slo, shi in evo_rows[f]])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(evo_ordered))
    w = 0.35

    for i, (dname, family, ea, sa, elo, ehi, slo, shi) in enumerate(evo_ordered):
        color = FAMILY_COLORS[family]
        ax.bar(i - w/2, ea, width=w, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(i - w/2, ea, yerr=[[ea-elo], [ehi-ea]], fmt="none",
                    ecolor="#374151", elinewidth=0.8, capsize=2, capthick=0.8)
        ax.bar(i + w/2, sa, width=w, color=color, alpha=0.40,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(i + w/2, sa, yerr=[[sa-slo], [shi-sa]], fmt="none",
                    ecolor="#6b7280", elinewidth=0.8, capsize=2, capthick=0.8)

    ax.axhline(y=20, color="#9ca3af", linestyle="--", linewidth=1.0, zorder=0)

    cum = 0
    for f in FAMILY_ORDER:
        n = len(evo_rows[f])
        if cum > 0:
            ax.axvline(x=cum - 0.5, color="#d1d5db", linewidth=0.8, linestyle="-")
        cum += n

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in evo_ordered], fontsize=14, rotation=45,
                       ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_ylim(0, 75)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlim(-0.6, len(evo_ordered) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor="#6b7280", alpha=0.85, label="Evolved"),
        Patch(facecolor="#6b7280", alpha=0.40, label="Static"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=16,
              framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "evo_vs_static.pdf", bbox_inches="tight", dpi=300)
    print(f"Saved {OUT_DIR / 'evo_vs_static.pdf'}")
    plt.close()


if __name__ == "__main__":
    main()
