"""
main.py

Main entry point for the Causal Preference Evolution Framework.
Initializes the generation pipeline and executes the data generation process.
"""

import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.generation_pipeline import GenerationPipeline
from src.evaluation.evaluator import FrameworkEvaluator
from src.llm.api_client import LLMClient
from src.benchmark_generation import analyze_history_validation

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
    """Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if config_path.endswith('.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    logging.info(f"Loaded configuration from {config_path}")
    return config

_AUTO_API_INFO_PATH = ".auto_api_info.yaml"


def _infer_api_account(model_name: str) -> str:
    """Infer the API provider account from a model name."""
    name = model_name.lower()
    if any(name.startswith(p) for p in ("gpt", "o1", "o3", "o4")):
        return "openai"
    if name.startswith("claude") or name.startswith("anthropic"):
        return "anthropic"
    if name.startswith("gemini"):
        return "gemini"
    return "openai"


def _ensure_api_info(llm_config: Dict[str, Any]) -> None:
    """Ensure model_kwargs has a valid api_info path.

    If the user hasn't set up src/api_info.yaml, auto-generate a minimal one
    from environment variables so the src/llm agents can find their keys.
    """
    model_kwargs = llm_config.setdefault("model_kwargs", {})
    api_info_path = model_kwargs.get("api_info", "src/api_info.yaml")

    if not os.path.exists(api_info_path):
        api_info_content = {
            "openai": {"api_type": "openai", "api_key": os.environ.get("OPENAI_API_KEY", "")},
            "anthropic": {"api_type": "anthropic", "api_key": os.environ.get("ANTHROPIC_API_KEY", "")},
            "gemini": {"api_type": "google", "api_key": os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))},
        }
        with open(_AUTO_API_INFO_PATH, "w") as f:
            yaml.dump(api_info_content, f)
        api_info_path = _AUTO_API_INFO_PATH

    model_kwargs["api_info"] = api_info_path
    if "api_account" not in model_kwargs:
        model_kwargs["api_account"] = _infer_api_account(llm_config.get("model", ""))


def _create_validation_clients(config: Dict[str, Any], llm_config: Dict[str, Any]):
    """Create validation LLM clients for benchmark history validation.

    Gracefully skips models whose API keys are unavailable.
    """
    benchmark_config = config.get("benchmark_generation", {})
    validation_llm_configs = benchmark_config.get("validation_llm_configs", [])
    if not validation_llm_configs:
        return None

    validation_llm_clients = []
    for val_config in validation_llm_configs:
        model_name = val_config.get("model_name", val_config.get("api_account", "unknown"))
        logging.info(f"Creating validation LLM client for {model_name}...")

        model_kwargs = dict(llm_config.get("model_kwargs", {}))
        model_kwargs["api_account"] = val_config.get("api_account", model_kwargs.get("api_account"))

        validation_client_config = {
            "model": val_config.get("model", model_name),
            "max_retries": val_config.get("max_retries", llm_config.get("max_retries", 3)),
            "retry_delay": val_config.get("retry_delay", llm_config.get("retry_delay", 2)),
            "max_tokens": val_config.get("max_tokens", llm_config.get("max_tokens", 4096)),
            "temperature": val_config.get("temperature", 0.3),
            "cache_responses": llm_config.get("cache_responses", True),
            "log_prompts": llm_config.get("log_prompts", False),
            "log_responses": llm_config.get("log_responses", False),
            "model_kwargs": model_kwargs
        }
        _ensure_api_info(validation_client_config)
        try:
            client = LLMClient(config=validation_client_config)
            validation_llm_clients.append((model_name, client))
        except Exception as e:
            logging.warning(f"Could not initialize validation client for {model_name}: {e}. Skipping.")

    if validation_llm_clients:
        logging.info(f"Created {len(validation_llm_clients)} validation LLM clients: {[name for name, _ in validation_llm_clients]}")
        return validation_llm_clients
    else:
        logging.warning("No validation LLM clients could be initialized. Benchmark validation will be skipped.")
        return None


def main():
    """
    Main entry point for the framework.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Causal Preference Evolution Framework")
    parser.add_argument("--config", type=str, default="", help="Path to configuration file")
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
    
    # Single event generation arguments
    parser.add_argument("--single-event", action="store_true", help="Generate a single event and conversation instead of full timeline")
    parser.add_argument("--expression-type", type=str, choices=["explicit", "implicit", "natural"], default="natural", help="Type of preference expression")
    parser.add_argument("--temporal-recall-days", type=int, default=30, help="Minimum days for temporal preference recall")
    parser.add_argument("--event-category", type=str, help="Category for the event to generate (optional - samples from config if not provided)")
    parser.add_argument("--event-description", type=str, help="Description for the event to generate (optional)")
    parser.add_argument("--persona-traits", type=str, help="Description of persona traits for the user (e.g., 'curious software engineer who loves problem solving')")
    parser.add_argument("--assistant-traits", type=str, help="Description of assistant traits (e.g., 'patient technical mentor with expertise in coding')")
    parser.add_argument("--generator-model", type=str, help="Model account name for the generator (overrides config)")
    parser.add_argument("--model", type=str, help="Shorthand for --generator-model (sets the generation model)")

    args = parser.parse_args()

    if args.model and not args.generator_model:
        args.generator_model = args.model
    
    # Load configuration
    if args.config == "":
        args.config = "src/config/config.yaml"
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
    
    # No validation needed for single event arguments since they're all optional
    
    # Override model in llm_config if --model or --generator-model is provided
    if args.generator_model:
        config.setdefault("llm_config", {})["model"] = args.generator_model
        config.setdefault("convo_llm_config", {})["model"] = args.generator_model

    # Set up logging
    setup_logging(os.path.join(config["output_dir"], "logs"), config.get("verbose_logging", False))
    
    logging.info(f"config: {config}")
    try:
        # Initialize LLM client with config
        llm_config = config.get("llm_config", {})
        if llm_config.get("model", "") == "":
            api_info_path = llm_config.get("model_kwargs", {}).get("api_info", "src/api_info.yaml")
            if os.path.exists(api_info_path):
                with open(api_info_path, "r") as f:
                    api_info = yaml.safe_load(f)
                llm_config["model"] = api_info.get(llm_config.get("model_kwargs", {}).get("api_account", "openai"), {}).get("model_name", "gpt-4o")
            else:
                logging.error(
                    "No model specified. Use --model MODEL, set 'model' in config yaml, "
                    "or create src/api_info.yaml (see src/api_info.yaml.example)."
                )
                return

        # Ensure model_kwargs has api_info for the src/llm agents
        llm_config.setdefault("model_kwargs", {})
        _ensure_api_info(llm_config)
        logging.info(f"llm_config: {llm_config}")
        llm_client = LLMClient(config=llm_config)

        convo_llm_config = config.get("convo_llm_config", {})
        if not convo_llm_config.get("model"):
            convo_llm_config["model"] = llm_config["model"]
        _ensure_api_info(convo_llm_config)
        logging.info(f"convo_llm_config: {convo_llm_config}")
        convo_llm_client = LLMClient(config=convo_llm_config)
        
        # Handle single event generation mode
        if args.single_event:
            logging.info("Running single event generation mode...")
            
            # Use persona and assistant traits as simple strings
            persona_traits = args.persona_traits or "friendly and helpful person"
            assistant_traits = args.assistant_traits or "knowledgeable and supportive assistant"
            
            generator_model_account = args.generator_model or llm_config.get("model", "")
            
            # Create validation LLM clients if configured (supports multiple models)
            validation_llm_clients = _create_validation_clients(config, llm_config)
            
            # Initialize generation pipeline
            pipeline = GenerationPipeline(
                llm_client=llm_client,
                convo_llm_client=convo_llm_client,
                output_dir=config["output_dir"],
                config=config,
                use_cache=args.use_cache,
                validation_llm_clients=validation_llm_clients
            )
            
            # Generate single event and conversation
            result = pipeline.generate_single_event_conversation(
                expression_type=args.expression_type,
                temporal_preference_recall_days_min=args.temporal_recall_days,
                specified_event_category=args.event_category,  # Can be None
                specified_event_description=args.event_description,  # Can be None
                specified_persona_traits=persona_traits,
                specified_assistant_traits=assistant_traits,
                generator_model_account=generator_model_account
            )
            
            logging.info("Single event generation completed successfully.")
            logging.info(f"Result saved to: {config['output_dir']}")
            return
        
        # Run generation pipeline unless eval-only mode
        generation_stats = None
        if not args.eval_only:
            # Initialize and run generation pipeline
            logging.info("Initializing generation pipeline...")
            
            validation_llm_clients = _create_validation_clients(config, llm_config)
            
            pipeline = GenerationPipeline(
                llm_client=llm_client,
                convo_llm_client=convo_llm_client,
                output_dir=config["output_dir"],
                config=config,
                use_cache=args.use_cache,
                validation_llm_clients=validation_llm_clients
            )
            
            logging.info("Running generation pipeline...")
            generation_stats = pipeline.run()
            
            logging.info(f"Generation complete. Generated {generation_stats['user_count']} user timelines.")
            
            if generation_stats.get("terminated_early"):
                logging.error(
                    f"WARNING: Generation terminated early due to infrastructure failure! "
                    f"Only {generation_stats['event_count']}/{generation_stats.get('expected_event_count', '?')} events completed. "
                    f"Re-run with --use-cache to resume from last checkpoint."
                )
            
            bench_cfg = config.get("benchmark_generation", {})
            if bench_cfg.get("enabled", False) and bench_cfg.get("validate_requires_history", True):
                logging.info("Generating history validation summary...")
                benchmark_dir = os.path.join(
                    config["output_dir"],
                    bench_cfg.get("output_dir", "benchmark")
                )
                if os.path.exists(benchmark_dir):
                    validation_results = analyze_history_validation(benchmark_dir, save_summary=True)
                    logging.info(f"History validation: {validation_results['valid_items']}/{validation_results['total_items']} items passed")
                    logging.info(f"Validation rate: {validation_results['validation_rate']:.1%}")
        else:
            logging.info("Skipping generation pipeline (eval-only mode)")
            
            # Check if timeline file exists for evaluation
            timeline_path = os.path.join(config["output_dir"], "all_timelines.json")
            if not os.path.exists(timeline_path):
                logging.error(f"Timeline file not found: {timeline_path}")
                logging.error("Run generation first or specify correct output directory with --output-dir")
                return
        
        # Run evaluation if enabled
        if config["evaluation"]["run_evaluation"]:
            logging.info("Running evaluation...")
            
            evaluator = FrameworkEvaluator(
                llm_client=llm_client,
                output_dir=os.path.join(config["output_dir"], "evaluation"),
                config=config.get("evaluation", {})
            )
            
            # Load timelines
            timeline_path = os.path.join(config["output_dir"], "all_timelines.json")
            with open(timeline_path, 'r') as f:
                timelines = json.load(f)
                
            # Evaluate timelines
            eval_results = evaluator.evaluate_batch(timelines)
            
            logging.info(f"Evaluation complete. Overall score: {eval_results['overall_score']:.2f}")
            
            # Save summary results
            summary_path = os.path.join(config["output_dir"], "evaluation_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
                
            logging.info(f"Evaluation summary saved to {summary_path}")
        
        if generation_stats and generation_stats.get("terminated_early"):
            logging.warning("Process finished but generation was INCOMPLETE due to infrastructure errors. Re-run to resume.")
            sys.exit(1)
        else:
            logging.info("Process completed successfully.")
        
    except Exception as e:
        logging.error(f"Error in main process: {e}", exc_info=True)

if __name__ == "__main__":
    main()