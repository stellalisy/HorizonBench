"""Build the HorizonBench HuggingFace dataset with three configs:
  - sample: 10 curated items for quick exploration
  - benchmark: full 4,245 MCQ items enriched with metadata
  - mental_state_graphs: 360 user timelines with preference evolution
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

BASE = Path(__file__).resolve().parent.parent / "output"

SOURCES = {
    "sonnet-4.5": {
        "dir": "v6_sonnet45",
        "hf_files": ["v6_sonnet45_filtered_5llm_hf.jsonl"],
    },
    "o3": {
        "dir": "v7_o3",
        "hf_files": ["v7_o3_filtered_5llm_hf.jsonl"],
    },
    "gemini-3-flash": {
        "dir": "v8_gemini3flash",
        "hf_files": ["v8_gemini3flash_filtered_5llm_hf.jsonl"],
    },
}

SAMPLE_IDS = [
    "Marcus Trevino_event31_20260327_turn3_facilitation_style_preferences_160_counterfactuals.json",
    "Marcus Chen_event55_20260328_turn1_technology_assistance_style_preferences_291_counterfactuals.json",
    "Maya Chen_event67_20260328_turn3_communication_intimacy_58_counterfactuals.json",
    "Maya Ortiz_event114_20260325_turn17_follow_up_strategy_preferences_339_counterfactuals.json",
    "Ryan Morrison_event130_20260325_turn7_event_planning_detail_preferences_227_counterfactuals.json",
    "Jason Taylor_event79_20260326_turn3_content_length_preferences_170_counterfactuals.json",
    "Jairo Restrepo_event77_20260326_turn1_communication_medium_preferences_142_counterfactuals.json",
    "Sasha Thorne_event84_20260326_turn21_creative_collaboration_206_counterfactuals.json",
    "Julian Sterling_event145_20260326_turn13_motivation_strategy_preferences_316_counterfactuals.json",
    "Aarav Shah_event154_20260328_turn1_intergenerational_engagement_preferences_262_counterfactuals.json",
]


def parse_domain(item_id: str) -> str:
    m = re.search(r"turn\d+_(.+?)_\d+_counterfactuals", item_id)
    return m.group(1) if m else ""


def parse_user_name(item_id: str) -> str:
    return item_id.split("_event")[0]


# Cache: (source_dir, item_id) -> (cf_data, user_dir_basename)
_cf_cache: dict[tuple[str, str], tuple[dict | None, str]] = {}


def find_counterfactual_json(source_dir: Path, item_id: str) -> tuple[dict | None, str]:
    """Returns (cf_data, user_dir_basename) e.g. (data, 'user_42')."""
    key = (str(source_dir), item_id)
    if key in _cf_cache:
        return _cf_cache[key]
    pattern = str(source_dir / "user_*" / "benchmark" / "temporal_preference_recall" / item_id)
    matches = glob.glob(pattern)
    if not matches:
        _cf_cache[key] = (None, "")
        return None, ""
    path = matches[0]
    parts = path.split("/")
    user_dir = [p for p in parts if p.startswith("user_")][0]
    with open(path) as f:
        data = json.load(f)
    _cf_cache[key] = (data, user_dir)
    return data, user_dir


def extract_distractor_letter(cf_data: dict, hf_options: list) -> str | None:
    if not cf_data:
        return None
    for ct in cf_data.get("counterfactual_turns", []):
        if ct.get("is_pre_evolution_distractor"):
            alt_val = ct.get("alternative_value")
            for opt in hf_options:
                if str(opt.get("value")) == str(alt_val):
                    return opt["letter"]
    return None


def extract_preference_evolution(cf_data: dict) -> dict | None:
    if not cf_data:
        return None
    pe = cf_data.get("preference_evolution", {})
    if not pe.get("has_evolved"):
        return None
    return {
        "original_attributes": pe.get("original_expressed_attributes", {}),
        "current_attributes": pe.get("current_attributes", {}),
        "changed_attributes": pe.get("changed_attributes", {}),
        "evolution_history": pe.get("evolution_history", []),
    }


def load_hf_items() -> list[dict]:
    """Load all benchmark items from the three HF JSONL files."""
    items = []
    for generator, src in SOURCES.items():
        src_dir = BASE / src["dir"]
        for hf_file in src["hf_files"]:
            path = src_dir / hf_file
            with open(path) as f:
                for line in f:
                    item = json.loads(line)
                    if not item.get("pass_filter", True):
                        continue
                    item["generator"] = generator
                    item["_source_dir"] = str(src_dir)
                    items.append(item)
    return items


def enrich_item(item: dict) -> dict:
    """Enrich a single HF item with metadata from its counterfactual JSON."""
    item_id = item["id"]
    source_dir = Path(item["_source_dir"])
    cf_data, user_dir = find_counterfactual_json(source_dir, item_id)

    generator = item["generator"]
    user_id = f"{generator}/{user_dir}" if user_dir else f"{generator}/{parse_user_name(item_id)}"

    options = item.get("options", [])
    enriched = {
        "id": item_id,
        "generator": generator,
        "user_id": user_id,
        "conversation": item["conversation"],
        "correct_letter": item["correct_letter"],
        "options": json.dumps(options),
        "has_evolved": item.get("has_evolved", False),
        "preference_domain": parse_domain(item_id),
        "distractor_letter": extract_distractor_letter(cf_data, options) or "",
        "preference_evolution": json.dumps(extract_preference_evolution(cf_data)),
    }
    return enriched


def build_benchmark(items: list[dict]) -> Dataset:
    """Build the full benchmark dataset."""
    print(f"Enriching {len(items)} benchmark items...")
    rows = []
    for i, item in enumerate(items):
        rows.append(enrich_item(item))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)}")
    print(f"  {len(rows)}/{len(items)} done")
    return Dataset.from_list(rows)


def build_sample(items: list[dict]) -> Dataset:
    """Build the curated sample dataset (10 items)."""
    sample_set = set(SAMPLE_IDS)
    sample_items = [it for it in items if it["id"] in sample_set]
    assert len(sample_items) == len(SAMPLE_IDS), (
        f"Expected {len(SAMPLE_IDS)} sample items, found {len(sample_items)}"
    )
    rows = [enrich_item(it) for it in sample_items]
    id_order = {sid: i for i, sid in enumerate(SAMPLE_IDS)}
    rows.sort(key=lambda r: id_order[r["id"]])
    return Dataset.from_list(rows)


def build_mental_state_graphs() -> Dataset:
    """Build the mental_state_graphs dataset from all_timelines.json per user."""
    print("Building mental state graphs...")
    rows = []
    for generator, src in SOURCES.items():
        src_dir = BASE / src["dir"]
        user_dirs = sorted(glob.glob(str(src_dir / "user_*")))
        print(f"  {generator}: {len(user_dirs)} users")
        for ud in user_dirs:
            timeline_path = os.path.join(ud, "all_timelines.json")
            if not os.path.exists(timeline_path):
                continue
            with open(timeline_path) as f:
                timelines = json.load(f)
            if not timelines:
                continue
            tl = timelines[0]

            user_attrs = tl.get("user", {}).get("attributes", {})
            user_dir_name = os.path.basename(ud)
            user_id = f"{generator}/{user_dir_name}"

            tl_meta = tl.get("timeline_meta", {})

            pref_record = tl.get("preference_record", {})
            event_record = tl.get("event_record", [])

            conversations = []
            for cr in tl.get("conversation_record", []):
                conversations.append({
                    "date": cr.get("date", ""),
                    "event_id": cr.get("event_id", ""),
                    "conversation": cr.get("conversation", []),
                })

            rows.append({
                "user_id": user_id,
                "generator": generator,
                "user_profile": json.dumps(user_attrs),
                "timeline_metadata": json.dumps({
                    "start_date": tl_meta.get("start_date", ""),
                    "end_date": tl_meta.get("end_date", ""),
                    "event_count": tl_meta.get("event_count", 0),
                    "conversation_count": tl_meta.get("conversation_count", 0),
                    "preference_change_count": tl_meta.get("preference_change_count", 0),
                }),
                "preference_record": json.dumps(pref_record),
                "event_record": json.dumps(event_record),
                "conversations": json.dumps(conversations),
            })
    print(f"  Total: {len(rows)} users")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Build HorizonBench HF dataset")
    parser.add_argument("--output-dir", type=str, default="hf_release",
                        help="Directory to save the dataset")
    parser.add_argument("--push", action="store_true",
                        help="Push to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default="stellalisy/HorizonBench",
                        help="HuggingFace repo ID")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_hf_items()
    print(f"Loaded {len(items)} total benchmark items")

    sample_ds = build_sample(items)
    benchmark_ds = build_benchmark(items)
    mental_ds = build_mental_state_graphs()

    sample_dd = DatasetDict({"test": sample_ds})
    benchmark_dd = DatasetDict({"test": benchmark_ds})
    mental_dd = DatasetDict({"test": mental_ds})

    sample_dir = output_dir / "sample"
    benchmark_dir = output_dir / "benchmark"
    mental_dir = output_dir / "mental_state_graphs"

    print(f"\nSaving to {output_dir}...")
    sample_dd.save_to_disk(str(sample_dir))
    benchmark_dd.save_to_disk(str(benchmark_dir))
    mental_dd.save_to_disk(str(mental_dir))

    print(f"\nDataset summary:")
    print(f"  sample:             {len(sample_ds)} items")
    print(f"  benchmark:          {len(benchmark_ds)} items")
    print(f"  mental_state_graphs: {len(mental_ds)} users")

    if args.push:
        print(f"\nPushing to {args.repo_id}...")
        benchmark_ds.push_to_hub(args.repo_id, config_name="benchmark", split="test")
        sample_ds.push_to_hub(args.repo_id, config_name="sample", split="test")
        mental_ds.push_to_hub(args.repo_id, config_name="mental_state_graphs", split="test")
        print("Done!")


if __name__ == "__main__":
    main()
