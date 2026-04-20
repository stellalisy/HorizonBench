import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime
from typing import Dict, Any
import random
# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.api_client import LLMClient

def setup_logging(log_dir: str, verbose: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        verbose: Whether to use verbose logging
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file."""
    if config_path.endswith('.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    logging.info(f"Loaded configuration from {config_path}")
    return config


def main():
    """
    Main entry point for the framework.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Causal Preference Evolution Framework")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed")
    parser.add_argument("--api-key", type=str, help="API key for LLM service (overrides environment variable)")
    parser.add_argument("--api-account", type=str, help="API account for LLM service (overrides config)")
    parser.add_argument("--users", type=int, help="Number of users to generate (overrides config)")
    parser.add_argument("--days", type=int, help="Timeline length in days (overrides config)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Skip generation and run evaluation only on existing timelines")
    parser.add_argument("--use-cache", action="store_true", help="Use cached libraries from previous runs")
    parser.add_argument("--new-pref-prob", type=float, help="Probability (0-1) of generating new preference domains. If 0, uses only provided domains (overrides config)")
    parser.add_argument("--new-event-prob", type=float, help="Probability (0-1) of generating new event categories. If 0, uses only provided categories (overrides config)")
    parser.add_argument("--input-dir", type=str, help="Input directory containing user_* folders with benchmark data")
    parser.add_argument("--output-file", type=str, help="Output path for the consolidated prompts JSON")
    parser.add_argument("--filter-5llm", action="store_true", help="Only include items where all 5 validation LLMs got accuracy=0 (strict 5-LLM filter)")
    parser.add_argument("--filter-majority", action="store_true", help="Only include items where majority (3+/5) validation LLMs got accuracy=0")
    parser.add_argument("--start-user", type=int, help="Only process users with index >= this value")
    parser.add_argument("--end-user", type=int, help="Only process users with index <= this value")
    parser.add_argument("--benchmark-subdir", type=str, default="temporal_preference_recall",
                        help="Benchmark subdirectory name under user_X/benchmark/ (default: temporal_preference_recall)")
    parser.add_argument("--timelines-file", type=str, default="all_timelines.json",
                        help="Timeline filename to load from each user dir (default: all_timelines.json)")
    parser.add_argument("--filter-whitelist", type=str, default=None,
                        help="Path to a JSON file whose keys are item filenames to include (bypasses --filter-5llm/--filter-majority)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.random_seed:
        config["random_seed"] = args.random_seed
    if args.users:
        config["user_count"] = args.users
    if args.days:
        config["timeline_length_days"] = args.days
    if args.verbose:
        config["verbose_logging"] = True
    if args.skip_eval:
        config["evaluation"]["run_evaluation"] = False
    if args.eval_only:
        if args.skip_eval:
            logging.error("Cannot use --eval-only and --skip-eval together")
            return
        config["evaluation"]["run_evaluation"] = True
    if args.new_pref_prob is not None:
        if not 0 <= args.new_pref_prob <= 1:
            logging.error("new-pref-prob must be between 0 and 1")
            return
        config["new_preference_domain_probability"] = args.new_pref_prob
        # Validate that if probability is 0, we have initial domains
        if args.new_pref_prob == 0 and not config.get("preference_domains"):
            logging.error("When new-pref-prob is 0, preference_domains list cannot be empty")
            return
    if args.new_event_prob is not None:
        if not 0 <= args.new_event_prob <= 1:
            logging.error("new-event-prob must be between 0 and 1")
            return
        config["new_event_category_probability"] = args.new_event_prob
        # Validate that if probability is 0, we have initial categories
        if args.new_event_prob == 0 and not config.get("category_weights"):
            logging.error("When new-event-prob is 0, category_weights cannot be empty")
            return
    
    # Set up logging
    setup_logging(os.path.join(config["output_dir"], "logs"), config["verbose_logging"])
    
    random.seed(config.get("random_seed", 42))
    
    print(f"config: {config}")
    gen_benchmark_prompts(config, input_dir=args.input_dir, output_file=args.output_file, filter_5llm=args.filter_5llm,
                          filter_majority=args.filter_majority, start_user=args.start_user, end_user=args.end_user,
                          benchmark_subdir_name=args.benchmark_subdir, timelines_file=args.timelines_file,
                          filter_whitelist_path=args.filter_whitelist)






def timeline_to_history(user_timeline, event_id, turn_index):
    num_turns = 0
    history = ""
    final_event = False
    for conversation in user_timeline["conversation_record"]:
        if conversation["event_id"] == event_id:
            final_event = True
        history += f'Date: {conversation["date"]}\n'
        scenario = conversation.get("conversation", {}).get("scenario", "")
        if scenario:
            history += f'Scenario: {scenario}\n'
        for tid, turn in enumerate(conversation["conversation"]["turns"]):
            if final_event and tid >= turn_index:
                return history + "\n", num_turns
            if turn["role"] == "user":
                num_turns += 1
                history += f'User: {turn["content"]}\n'
            else:
                num_turns += 1
                history += f'Assistant: {turn["content"]}\n'
    return history + "\n", num_turns



def gen_benchmark_prompts(config, input_dir=None, output_file=None, filter_5llm=False,
                          filter_majority=False, start_user=None, end_user=None,
                          benchmark_subdir_name="temporal_preference_recall",
                          timelines_file="all_timelines.json",
                          filter_whitelist_path=None):
    all_prompts = {}
    formatted_prompts = {}

    # Load whitelist if provided (bypasses filter-5llm / filter-majority)
    whitelist = None
    if filter_whitelist_path and os.path.exists(filter_whitelist_path):
        with open(filter_whitelist_path) as f:
            whitelist = set(json.load(f).keys())
        print(f"Loaded whitelist with {len(whitelist)} items from {filter_whitelist_path}")

    if input_dir is None:
        input_dir = "output"
    if output_file is None:
        output_file = os.path.join(input_dir, os.path.basename(input_dir) + ".json")
    output_file_hf = output_file.replace(".json", "_hf.jsonl")
    
    benchmark_dir = input_dir
    user_dirs = [d for d in os.listdir(benchmark_dir) if d.startswith("user_")]
    if start_user is not None or end_user is not None:
        def _user_idx(name):
            try: return int(name.split("_")[1])
            except (ValueError, IndexError): return -1
        lo = start_user if start_user is not None else 0
        hi = end_user if end_user is not None else float('inf')
        user_dirs = [d for d in user_dirs if lo <= _user_idx(d) <= hi]

    num_turns = []
    skipped_users = []
    for user_dir in sorted(user_dirs):
        user_timelines_filepath = os.path.join(benchmark_dir, user_dir, timelines_file)
        benchmark_subdir = os.path.join(benchmark_dir, user_dir, "benchmark", benchmark_subdir_name)

        if not os.path.exists(user_timelines_filepath):
            skipped_users.append(user_dir)
            continue
        if not os.path.exists(benchmark_subdir):
            continue

        with open(user_timelines_filepath, "r") as f:
            user_timelines = json.load(f)

        user_timeline = user_timelines[0]
        test_nodes_filepaths = os.listdir(os.path.join(benchmark_dir, user_dir, "benchmark", benchmark_subdir_name))
        for test_node_filepath in test_nodes_filepaths:
            try:
                # Whitelist filter: skip items not in the whitelist
                if whitelist is not None:
                    if test_node_filepath not in whitelist:
                        continue

                with open(os.path.join(benchmark_dir, user_dir, "benchmark", benchmark_subdir_name, test_node_filepath), "r") as f:
                    test_node = json.load(f)

                # Validation-based filters (only if no whitelist)
                if whitelist is None and (filter_5llm or filter_majority):
                    hv = test_node.get("history_validation", {})
                    pm = hv.get("per_model_results", {})
                    n_zero = sum(1 for mr in pm.values() if mr.get("accuracy", 1.0) == 0.0)
                    if filter_5llm and (len(pm) < 5 or n_zero < 5):
                        continue
                    if filter_majority and (len(pm) < 5 or n_zero < 3):
                        continue

                event_id = test_node["metadata"]["event"]["id"]
                turn_index = test_node["metadata"]["turn_index"]

                history = "Conversation History:\n"

                history_temp, num_turns_temp = timeline_to_history(user_timeline, event_id, turn_index)
                history += history_temp
                num_turns.append(num_turns_temp)

                recalled_preference = test_node["metadata"]["recalled_preference"]
                counterfactual_turns = test_node["counterfactual_turns"]

                attribute = counterfactual_turns[0]["attribute"]
                original_value = str(counterfactual_turns[0]["original_value"])

                options = {str(counterfactual["alternative_value"]): counterfactual["counterfactual_turn"]["content"] for counterfactual in counterfactual_turns}
                original_turn = {original_value: test_node["original_turn"]["content"]}
                options.update(original_turn)

                question = "Which of the following responses would align with the user's preferences the most?"
                
                # Create list of (value, option) pairs and shuffle
                option_pairs = list(options.items())
                random.shuffle(option_pairs)
                
                # Track which letter corresponds to original value
                correct_letter = None
                for oid, (value, option) in enumerate(option_pairs):
                    letter = chr(ord('A') + oid)
                    if value == original_value:
                        correct_letter = letter
                    question += f"\n{letter}: {option}"

                question += "\nPlease respond with the letter of the option that aligns with the user's preferences the most and nothing else."

                input_prompt = history + "\n" + question
                
                # Check if history_validation passed (requires_history should be True)
                history_validation = test_node.get("history_validation", {})
                pass_filter = history_validation.get("requires_history", False)
                
                # Check if the target preference has evolved since first expression
                preference_evolution = test_node.get("preference_evolution", {})
                has_evolved = preference_evolution.get("has_evolved", False)
                
                all_prompts[test_node_filepath] = {
                    "prompt": input_prompt,
                    "correct_letter": correct_letter,
                    "pass_filter": pass_filter,
                    "has_evolved": has_evolved
                }

                # Create options as a list instead of dynamic keys
                options_list = []
                for oid, (value, option) in enumerate(option_pairs):
                    letter = chr(ord('A') + oid)
                    options_list.append({
                        "letter": letter,
                        "value": value,
                        "option": str(option)
                    })
                
                formatted_prompts[test_node_filepath] = {
                    "id": str(test_node_filepath),
                    "conversation": str(history),
                    "correct_letter": str(correct_letter),
                    "options": options_list,
                    "pass_filter": pass_filter,
                    "has_evolved": has_evolved
                }

            except Exception as e:
                print(f"Error processing test node: {e}")
                continue

    with open(output_file, "w") as f:
        f.write(json.dumps(all_prompts, indent=4))
    with open(output_file_hf, "w") as f:
        for line in formatted_prompts.values():
            f.write(json.dumps(line) + "\n")
    if skipped_users:
        print(f"Skipped {len(skipped_users)} users without all_timelines.json: {skipped_users[:5]}{'...' if len(skipped_users) > 5 else ''}")
    if filter_5llm:
        print(f"5-LLM strict filter applied: kept {len(all_prompts)} items")
    elif filter_majority:
        print(f"Majority filter (3+/5) applied: kept {len(all_prompts)} items")
    print(f"Saved {len(all_prompts)} prompts to {output_file}")
    print(f"Saved {len(formatted_prompts)} prompts to {output_file_hf}")
    avg_turns = sum(num_turns) / len(num_turns) if num_turns else 0
    print(f"Average number of turns: {avg_turns}")

if __name__ == "__main__":
    main()