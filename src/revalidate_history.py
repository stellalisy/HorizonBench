"""
Standalone script to re-run history validation on existing benchmark files
with configurable models.

This allows changing/adding validation models after benchmark generation is complete.
By default, new validation results are MERGED with existing results, preserving
correctness information from all previously used models.

Model names are resolved from api_info.yaml based on api_account names.

Usage:
    # Default models (gpt-5-mini, gemini-3-flash, o4-mini)
    python src/revalidate_history.py --input-dir output/my_run --user user_1 --runs 3

    # Specific models
    python src/revalidate_history.py --input-dir output/my_run \
        --api-accounts gpt-5-mini o4-mini --runs 1

    # Completely replace existing validation results
    python src/revalidate_history.py --input-dir output/my_run --user user_1 \
        --api-accounts gpt-5-mini --runs 3 --overwrite
"""

import os
import sys
import json
import re
import yaml
import logging
import argparse
import random
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_timeline(timeline_path: str) -> Dict:
    """Load the timeline JSON file."""
    with open(timeline_path, 'r') as f:
        timelines = json.load(f)
    return timelines


def load_benchmark_file(filepath: str) -> Dict:
    """Load a single benchmark file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_json_from_response(text: str) -> Dict:
    """
    Robustly extract a JSON object from an LLM response that may contain
    extra text before/after the JSON, markdown fences, or truncated content.
    """
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty response", text or "", 0)

    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (possibly with language tag)
        lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { and try to parse from there
    first_brace = text.find("{")
    if first_brace == -1:
        raise json.JSONDecodeError("No JSON object found in response", text, 0)

    # Try to find matching closing brace by tracking nesting
    depth = 0
    in_string = False
    escape_next = False
    for i in range(first_brace, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[first_brace:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # If we get here, the JSON is likely truncated (unterminated string).
    # Try to salvage by closing open strings and braces.
    partial = text[first_brace:]

    # Try to extract at least the choice field with regex
    choice_match = re.search(r'"choice"\s*:\s*"([A-Z])"', partial)
    confidence_match = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', partial)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', partial)

    if choice_match:
        return {
            "choice": choice_match.group(1),
            "confidence": confidence_match.group(1) if confidence_match else "unknown",
            "reasoning": (reasoning_match.group(1) if reasoning_match else "")[:500] + " [truncated]"
        }

    raise json.JSONDecodeError("Could not extract valid JSON from response", text, 0)


def build_validation_context(
    benchmark_data: Dict,
    conversation_record: List,
    current_conversation: Dict,
    turn_idx: int
) -> Tuple[str, int]:
    """
    Build the context string for validation, including all conversations
    AFTER the preference was expressed but NOT the one where it was expressed.

    Returns:
        Tuple of (context_string, num_intermediate_conversations)
    """
    context_parts = []

    recalled_pref = benchmark_data.get("metadata", {}).get("recalled_preference", {})
    expression_context = recalled_pref.get("expression_context", {})
    expression_event_id = expression_context.get("event_id")
    expression_date_str = expression_context.get("last_expressed_date")

    if expression_date_str:
        expression_date = datetime.fromisoformat(expression_date_str)
    else:
        expression_date = None

    conversations_after_expression = []
    for conv_entry in conversation_record:
        conv_event_id = conv_entry.get("event_id")
        conv_date_str = conv_entry.get("date")
        if conv_date_str:
            conv_date = datetime.fromisoformat(conv_date_str)
        else:
            continue

        if expression_date and conv_date > expression_date and conv_event_id != expression_event_id:
            conversations_after_expression.append(conv_entry)

    for i, conv in enumerate(conversations_after_expression):
        conv_turns = conv.get("conversation", {}).get("turns", [])
        if len(conv_turns) > 0:
            context_parts.append(f"\n=== CONVERSATION {i+1} ===")
            for turn in conv_turns:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                context_parts.append(f"{role}: {content}")

    # Add current conversation up to turn_idx
    context_parts.append("\n=== CURRENT CONVERSATION ===")
    current_turns = current_conversation.get("turns", [])
    for turn in current_turns[:turn_idx]:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        context_parts.append(f"{role}: {content}")

    context_str = "\n".join(context_parts)

    # Truncate if too long
    if len(context_str) > 100000:
        truncated_parts = []
        for i, conv in enumerate(conversations_after_expression):
            conv_turns = conv.get("conversation", {}).get("turns", [])
            if len(conv_turns) > 20:
                truncated_parts.append(f"\n=== CONVERSATION {i+1} (summary) ===")
                for turn in conv_turns[:5]:
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    truncated_parts.append(f"{role}: {content}")
                truncated_parts.append(f"[...{len(conv_turns) - 10} turns omitted...]")
                for turn in conv_turns[-5:]:
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    truncated_parts.append(f"{role}: {content}")
            else:
                truncated_parts.append(f"\n=== CONVERSATION {i+1} ===")
                for turn in conv_turns:
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    truncated_parts.append(f"{role}: {content}")

        truncated_parts.append("\n=== CURRENT CONVERSATION ===")
        for turn in current_turns[:turn_idx]:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            truncated_parts.append(f"{role}: {content}")

        context_str = "\n".join(truncated_parts)

    return context_str, len(conversations_after_expression)


def run_validation(
    benchmark_data: Dict,
    context_str: str,
    num_intermediate_convos: int,
    validation_clients: List[Tuple[str, LLMClient]],
    num_runs: int = 1,
    existing_validation: Optional[Dict] = None
) -> Dict:
    """
    Run history validation with the given LLM clients.

    Args:
        benchmark_data: The loaded benchmark file data
        context_str: The context string (conversations after preference expression)
        num_intermediate_convos: Number of intermediate conversations in context
        validation_clients: List of (model_name, LLMClient) tuples
        num_runs: Number of validation runs per model
        existing_validation: Previous validation results to preserve and merge with
        
    Returns:
        History validation results dictionary (merged with existing if provided)
    """
    # Get user info
    user_info = benchmark_data.get("metadata", {}).get("user", {}).get("attributes", {})
    user_name = user_info.get("first name", "") + " " + user_info.get("last name", "")
    user_name = user_name.strip() or "Unknown"
    user_age = user_info.get("age", "")
    user_gender = user_info.get("gender", "")

    # Get original turn content
    original_turn = benchmark_data.get("original_turn", {})
    original_response = original_turn.get("content", "")

    # Get counterfactual turns
    counterfactual_turns = benchmark_data.get("counterfactual_turns", [])
    if not counterfactual_turns:
        return {"error": "No counterfactual turns found"}

    first_cf = counterfactual_turns[0]
    original_value = first_cf.get("original_value", "")

    # Build options list: original + counterfactuals
    all_options = [{"value": original_value, "response": original_response}]
    for cf in counterfactual_turns:
        all_options.append({
            "value": cf.get("alternative_value", ""),
            "response": cf.get("counterfactual_turn", {}).get("content", "")
        })

    num_options = len(all_options)
    
    logger.info(f"Validation context: {num_intermediate_convos} intermediate conversations, {num_options} response options")
    
    # Start with existing per-model results if available
    per_model_results = {}
    if existing_validation and "per_model_results" in existing_validation:
        per_model_results = copy.deepcopy(existing_validation["per_model_results"])
        logger.info(f"Preserving existing validation results from {len(per_model_results)} model(s): {list(per_model_results.keys())}")
    
    # Track new validation runs
    total_correct = 0
    total_valid_runs = 0
    
    for model_name, validation_client in validation_clients:
        model_correct = 0
        model_details = []
        
        for run in range(num_runs):
            # Shuffle options for unbiased testing
            shuffled_options = copy.deepcopy(all_options)
            random.shuffle(shuffled_options)
            
            # Find correct label after shuffle
            correct_label = None
            for i, opt in enumerate(shuffled_options):
                if str(opt["value"]).lower().strip() == str(original_value).lower().strip():
                    correct_label = chr(ord('A') + i)
                    break
            
            # Build options string WITHOUT revealing preference values
            options_str = ""
            for i, opt in enumerate(shuffled_options):
                label = chr(ord('A') + i)
                response_preview = opt['response'][:500] + ('...' if len(opt['response']) > 500 else '')
                options_str += f"\n**Option {label}:**\n{response_preview}\n"
            
            validation_prompt = f"""You are evaluating which AI assistant response would be MOST APPROPRIATE for a user, based ONLY on the conversation context provided.

IMPORTANT: You are shown a SUBSET of this user's conversation history. There may be earlier conversations where the user expressed preferences that you CANNOT see. You must judge ONLY based on what's visible below.

### CONVERSATION CONTEXT ({num_intermediate_convos} intermediate conversation(s) + current conversation up to this point):
{context_str}

### USER PROFILE (visible information only):
- Name: {user_name}
- Age: {user_age}
- Gender: {user_gender}

### RESPONSE OPTIONS:
{options_str}

### TASK:
Based ONLY on the visible conversations above, which option seems most appropriate for the AI assistant to respond with next?
- You do NOT have access to the user's full history
- Do NOT assume any prior knowledge about this user's preferences beyond what's shown
- Do NOT use stereotypes or demographic assumptions
- Choose based ONLY on what's visible in the conversations

Return JSON only:
{{"choice": "<A/B/C/D/etc>", "confidence": "<high/medium/low>", "reasoning": "<why this seems most appropriate given ONLY the visible context>"}}"""
            
            try:
                result = validation_client.generate(validation_prompt, response_format="json")
                parsed = extract_json_from_response(result["response_text"])
                chosen = parsed.get("choice", "").strip().upper()
                is_correct = (chosen == correct_label)
                
                if is_correct:
                    model_correct += 1
                
                model_details.append({
                    "run": run + 1,
                    "chosen": chosen,
                    "correct": correct_label,
                    "is_correct": is_correct,
                    "confidence": parsed.get("confidence", "unknown"),
                    "reasoning": parsed.get("reasoning", "")
                })
            except Exception as e:
                logger.warning(f"Validation run {run + 1} with model {model_name} failed: {e}")
                model_details.append({
                    "run": run + 1,
                    "error": str(e)
                })
        
        model_valid_runs = sum(1 for d in model_details if "error" not in d)
        model_accuracy = model_correct / model_valid_runs if model_valid_runs > 0 else None
        
        per_model_results[model_name] = {
            "accuracy": model_accuracy,
            "correct_count": model_correct,
            "valid_runs": model_valid_runs,
            "total_runs": num_runs,
            "details": model_details
        }
        
        logger.info(f"  → Model {model_name}: accuracy={model_accuracy:.1%} ({model_correct}/{model_valid_runs})" if model_valid_runs > 0 else f"  → Model {model_name}: no valid runs")
        
        total_correct += model_correct
        total_valid_runs += model_valid_runs
    
    # Compute aggregate statistics across ALL models (including preserved ones)
    all_correct = sum(r.get("correct_count", 0) for r in per_model_results.values())
    all_valid_runs = sum(r.get("valid_runs", 0) for r in per_model_results.values())
    baseline_accuracy = all_correct / all_valid_runs if all_valid_runs > 0 else None
    
    # Determine if history is required
    requires_history = True
    model_accuracies = [r["accuracy"] for r in per_model_results.values() if r.get("accuracy") is not None]
    
    if model_accuracies:
        all_near_chance = all(acc <= 1.0 / num_options + 0.15 for acc in model_accuracies)
        any_above_threshold = any(acc > 0.5 for acc in model_accuracies)
        
        if any_above_threshold:
            requires_history = False
            confidence = "high"
        elif all_near_chance:
            requires_history = True
            confidence = "high"
        else:
            requires_history = baseline_accuracy <= 1.0 / num_options + 0.15
            confidence = "low" if baseline_accuracy and abs(baseline_accuracy - 1.0 / num_options) < 0.15 else "medium"
    else:
        confidence = "low"
    
    # Build per-model summary string
    model_summary = ", ".join(
        f"{name}: {r['accuracy']:.0%}" if r.get('accuracy') is not None else f"{name}: N/A"
        for name, r in per_model_results.items()
    )
    
    if requires_history:
        reasoning = f"Aggregate accuracy ({baseline_accuracy:.0%}) near chance ({1.0/num_options:.0%}). Per-model: [{model_summary}]. History IS required." if baseline_accuracy is not None else f"Per-model: [{model_summary}]. History IS required."
    else:
        reasoning = f"Aggregate accuracy ({baseline_accuracy:.0%}) above chance ({1.0/num_options:.0%}). Per-model: [{model_summary}]. Answer may be guessable WITHOUT history." if baseline_accuracy is not None else f"Per-model: [{model_summary}]. Answer may be guessable WITHOUT history."
    
    all_models_used = list(per_model_results.keys())
    
    result = {
        "requires_history": requires_history,
        "baseline_accuracy": baseline_accuracy,
        "chance_level": 1.0 / num_options,
        "num_options": num_options,
        "confidence": confidence,
        "reasoning": reasoning,
        "total_validation_runs": all_valid_runs,
        "models_used": all_models_used,
        "per_model_results": per_model_results,
        "last_updated": datetime.now().isoformat()
    }

    # Preserve original timestamp if merging
    if existing_validation and "last_updated" in existing_validation:
        result["original_timestamp"] = existing_validation.get("original_timestamp", existing_validation["last_updated"])
        result["revalidation_timestamp"] = datetime.now().isoformat()

    return result


def find_current_conversation(conversation_record: List, event_id: str) -> Optional[Dict]:
    """Find the conversation matching the given event_id."""
    for conv_entry in conversation_record:
        if conv_entry.get("event_id") == event_id:
            return conv_entry.get("conversation", {})
    return None


def revalidate_user(
    user_dir: str,
    validation_clients: List[Tuple[str, LLMClient]],
    benchmark_subdir: str = "temporal_preference_recall",
    num_runs: int = 1,
    update_files: bool = True,
    merge_results: bool = True,
    shuffle_models: bool = False
) -> Dict[str, Any]:
    """
    Re-run history validation for all benchmark files in a user directory.

    Args:
        user_dir: Path to user directory (e.g., output/v4_0112/user_1)
        validation_clients: List of (model_name, LLMClient) tuples
        num_runs: Number of validation runs per model
        update_files: Whether to update the benchmark files with new validation results
        merge_results: Whether to merge with existing validation results (True) or replace them (False)
        shuffle_models: Whether to randomize model call order per file
        
    Returns:
        Summary of validation results
    """
    timeline_path = os.path.join(user_dir, "all_timelines.json")
    if not os.path.exists(timeline_path):
        logger.error(f"Timeline file not found: {timeline_path}")
        return {"error": "Timeline not found"}

    timeline = load_timeline(timeline_path)
    if isinstance(timeline, list):
        timeline = timeline[0]
    conversation_record = timeline.get("conversation_record", [])

    benchmark_dir = os.path.join(user_dir, "benchmark", benchmark_subdir)
    if not os.path.isdir(benchmark_dir):
        logger.error(f"Benchmark directory not found: {benchmark_dir}")
        return {"error": "Benchmark directory not found"}

    benchmark_files = [f for f in os.listdir(benchmark_dir) if f.endswith("_counterfactuals.json")]
    
    results_summary = {
        "total_files": len(benchmark_files),
        "processed": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "details": []
    }

    for filename in benchmark_files:
        filepath = os.path.join(benchmark_dir, filename)
        
        try:
            logger.info(f"Processing: {filename}")
            benchmark_data = load_benchmark_file(filepath)
            
            metadata = benchmark_data.get("metadata", {})
            event = metadata.get("event", {})
            event_id = event.get("id")
            turn_idx = metadata.get("turn_index")
            
            current_conversation = find_current_conversation(conversation_record, event_id)
            if not current_conversation:
                logger.warning(f"Could not find conversation for event {event_id}")
                results_summary["errors"] += 1
                continue
            
            context_str, num_intermediate = build_validation_context(
                benchmark_data,
                conversation_record,
                current_conversation,
                turn_idx
            )
            
            # Get existing validation results to preserve (if merge_results is True)
            existing_validation = None
            if merge_results:
                existing_validation = benchmark_data.get("history_validation")
                if existing_validation:
                    existing_models = set(existing_validation.get("per_model_results", {}).keys())
                    requested_models = {name for name, _ in validation_clients}
                    if requested_models.issubset(existing_models):
                        logger.info(f"  Skipping {filename} — already has results for all requested models")
                        results_summary["processed"] += 1
                        if existing_validation.get("requires_history"):
                            results_summary["passed"] += 1
                        else:
                            results_summary["failed"] += 1
                        results_summary["details"].append({
                            "file": filename,
                            "requires_history": existing_validation.get("requires_history"),
                            "baseline_accuracy": existing_validation.get("baseline_accuracy"),
                            "confidence": existing_validation.get("confidence")
                        })
                        continue
                    logger.info(f"  Found existing validation with models: {list(existing_models)}")
            
            # Optionally shuffle model order to spread API load
            clients_for_file = list(validation_clients)
            if shuffle_models:
                random.shuffle(clients_for_file)
            
            # Run validation (will merge with existing results if merge_results=True)
            validation_result = run_validation(
                benchmark_data,
                context_str,
                num_intermediate,
                clients_for_file,
                num_runs,
                existing_validation=existing_validation
            )
            
            # Update the benchmark file
            if update_files:
                benchmark_data["history_validation"] = validation_result
                with open(filepath, 'w') as f:
                    json.dump(benchmark_data, f, indent=2)
                logger.info(f"Updated: {filename}")
            
            results_summary["processed"] += 1
            if validation_result.get("requires_history"):
                results_summary["passed"] += 1
            else:
                results_summary["failed"] += 1
            
            results_summary["details"].append({
                "file": filename,
                "requires_history": validation_result.get("requires_history"),
                "baseline_accuracy": validation_result.get("baseline_accuracy"),
                "confidence": validation_result.get("confidence")
            })
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results_summary["errors"] += 1
    
    return results_summary


def resolve_model_name_from_api_info(api_account: str, api_info_path: str) -> Optional[str]:
    """Resolve the model_name for an api_account from the api_info.yaml file."""
    with open(api_info_path, 'r') as f:
        api_info = yaml.safe_load(f)
    
    entry = api_info.get(api_account, {})
    return entry.get("model_name")


def build_validation_client(
    api_account: str,
    api_info_path: str = "src/api_info.yaml",
    base_llm_config: Optional[Dict] = None
) -> Tuple[str, LLMClient]:
    """
    Build an LLMClient for validation from an api_account name.
    Resolves model_name from api_info.yaml and constructs proper model_kwargs.
    """
    model_name = resolve_model_name_from_api_info(api_account, api_info_path)
    if not model_name:
        raise ValueError(f"Could not resolve model_name for api_account '{api_account}' in {api_info_path}")

    base_config = base_llm_config or {}
    base_model_kwargs = dict(base_config.get("model_kwargs", {}))

    model_kwargs = {
        "api_info": api_info_path,
        "api_account": api_account,
        **{k: v for k, v in base_model_kwargs.items() if k not in ("api_info", "api_account")}
    }

    client_config = {
        "model": model_name,
        "max_retries": base_config.get("max_retries", 3),
        "retry_delay": base_config.get("retry_delay", 2),
        "max_tokens": base_config.get("max_tokens", 8192),
        "temperature": base_config.get("temperature", 0.3),
        "cache_responses": base_config.get("cache_responses", True),
        "log_prompts": base_config.get("log_prompts", False),
        "log_responses": base_config.get("log_responses", False),
        "model_kwargs": model_kwargs
    }

    client = LLMClient(config=client_config)
    return model_name, client


def main():
    parser = argparse.ArgumentParser(description="Re-run history validation on existing benchmark files")
    parser.add_argument("--input-dir", type=str, required=True, help="Base output directory (e.g., output/v4_0112)")
    parser.add_argument("--user", type=str, help="Specific user directory to process (e.g., user_1). If not specified, processes all users.")
    parser.add_argument("--api-accounts", type=str, nargs="+",
                        default=["gpt-5-mini", "gemini-3-flash", "o4-mini"],
                        help="API account names from api_info.yaml (model names resolved automatically)")
    parser.add_argument("--api-info", type=str, default="src/api_info.yaml", help="Path to api_info.yaml")
    parser.add_argument("--runs", type=int, default=1, help="Number of validation runs per model")
    parser.add_argument("--no-update", action="store_true", help="Don't update benchmark files, just print results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing validation results instead of merging with them")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--shuffle-models", action="store_true", help="Randomize model call order per file to spread API load")
    parser.add_argument("--config", type=str, help="Path to config file for base LLM settings")
    parser.add_argument("--benchmark-subdir", type=str, default="temporal_preference_recall",
                        help="Benchmark subdirectory name under user_X/benchmark/ (default: temporal_preference_recall)")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Load base config if provided
    llm_config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            llm_config = config.get("llm_config", {})
    
    # Create validation clients from api_accounts (model names auto-resolved)
    validation_clients = []
    for api_account in args.api_accounts:
        try:
            model_name, client = build_validation_client(
                api_account=api_account,
                api_info_path=args.api_info,
                base_llm_config=llm_config
            )
            validation_clients.append((model_name, client))
            logger.info(f"Created validation client: {model_name} (account: {api_account})")
        except Exception as e:
            logger.error(f"Failed to create client for {api_account}: {e}")
    
    if not validation_clients:
        logger.error("No validation clients could be created. Exiting.")
        return
    
    # Determine user directories to process
    if args.user:
        user_dirs = [os.path.join(args.input_dir, args.user)]
    else:
        user_dirs = sorted([
            os.path.join(args.input_dir, d)
            for d in os.listdir(args.input_dir)
            if d.startswith("user_") and os.path.isdir(os.path.join(args.input_dir, d))
        ])
    
    all_results = {}
    for user_dir in user_dirs:
        user_name = os.path.basename(user_dir)
        print("\n" + "=" * 50)
        print(f"\nProcessing {user_name}")
        
        results = revalidate_user(
            user_dir,
            validation_clients,
            benchmark_subdir=args.benchmark_subdir,
            num_runs=args.runs,
            update_files=not args.no_update,
            merge_results=not args.overwrite,
            shuffle_models=args.shuffle_models
        )
        all_results[user_name] = results
        
        print(f"\nSummary for {user_name}:")
        print(f"  Total files: {results.get('total_files', 0)}")
        print(f"  Processed: {results.get('processed', 0)}")
        print(f"  Passed (requires history): {results.get('passed', 0)}")
        print(f"  Failed (guessable): {results.get('failed', 0)}")
        print(f"  Errors: {results.get('errors', 0)}")
    
    # Save overall summary
    summary_path = os.path.join(args.input_dir, "revalidation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "api_accounts": args.api_accounts,
            "models_used": [name for name, _ in validation_clients],
            "runs_per_model": args.runs,
            "results": all_results
        }, f, indent=2)
    
    print(f"\nSaved revalidation summary to: {summary_path}")


if __name__ == "__main__":
    main()
