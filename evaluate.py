"""Evaluate a language model on HorizonBench.

Loads the benchmark from HuggingFace, sends each item to the model,
and reports accuracy broken down by evolution status and generator.

CLI usage (any litellm-supported model):
  python evaluate.py --model gpt-4o
  python evaluate.py --model claude-sonnet-4-20250514 --max-items 100
  python evaluate.py --model gemini/gemini-2.0-flash --config sample

Programmatic usage (plug in your own inference function):

    from evaluate import load_benchmark, build_prompt, extract_letter, print_results

    ds = load_benchmark(config="sample")
    results = []
    for item in ds:
        prompt = build_prompt(item)
        response_text = your_model(prompt)  # replace with your inference
        predicted = extract_letter(response_text)
        results.append({
            "id": item["id"],
            "generator": item["generator"],
            "has_evolved": item["has_evolved"],
            "correct_letter": item["correct_letter"],
            "predicted_letter": predicted,
            "correct": predicted == item["correct_letter"],
        })
    print_results(results, "your-model-name")

RAG / custom pipeline usage (access conversation as structured turns):

    from evaluate import load_benchmark, parse_conversations, build_question, extract_letter

    ds = load_benchmark(config="sample")
    item = ds[0]
    segments = parse_conversations(item["conversation"])
    # segments is a list of conversation sessions, each with date, scenario, turns
    # You can chunk/embed/retrieve from these however you like
    question = build_question(item)
    # Then combine retrieved context + question and call your model
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASET_ID = "stellalisy/HorizonBench"


def load_benchmark(config: str = "benchmark", dataset: str = DATASET_ID, max_items: int = None):
    """Load HorizonBench from HuggingFace.

    Args:
        config: Dataset config -- "benchmark" (4,245 items), "sample" (10 items).
        dataset: HuggingFace dataset ID.
        max_items: If set, return only the first N items.

    Returns:
        A HuggingFace Dataset object. Each item has fields: id, generator,
        user_id, conversation, correct_letter, options, has_evolved,
        preference_domain, distractor_letter, preference_evolution.
    """
    ds = load_dataset(dataset, config, split="test")
    if max_items and max_items < len(ds):
        ds = ds.select(range(max_items))
    return ds


def parse_conversations(conversation_text: str) -> list[dict]:
    """Parse the flat conversation text into structured segments.

    Each segment represents one dated conversation session. Useful for RAG
    systems that need to chunk, embed, or selectively retrieve parts of the
    history rather than passing the full ~163K token context.

    Args:
        conversation_text: The raw 'conversation' field from a benchmark item.

    Returns:
        List of dicts, each with keys:
          - date (str): ISO timestamp of the conversation session
          - scenario (str): The session scenario description
          - turns (list[dict]): Each turn has 'role' ('user'|'assistant') and 'content'
    """
    segments = []
    current_segment = None

    for line in conversation_text.split("\n"):
        if line.startswith("Conversation History:"):
            continue
        elif line.startswith("Date: "):
            if current_segment is not None:
                segments.append(current_segment)
            current_segment = {"date": line[6:], "scenario": "", "turns": []}
        elif line.startswith("Scenario: ") and current_segment is not None:
            current_segment["scenario"] = line[10:]
        elif line.startswith("User: ") and current_segment is not None:
            current_segment["turns"].append({"role": "user", "content": line[6:]})
        elif line.startswith("Assistant: ") and current_segment is not None:
            current_segment["turns"].append({"role": "assistant", "content": line[11:]})
        elif current_segment is not None and current_segment["turns"]:
            current_segment["turns"][-1]["content"] += "\n" + line

    if current_segment is not None:
        segments.append(current_segment)
    return segments


def build_question(item: dict) -> str:
    """Build just the MCQ question (without conversation history).

    Useful when your method handles the conversation context separately
    (e.g., RAG retrieval) and you only need the question portion.
    """
    options = json.loads(item["options"]) if isinstance(item["options"], str) else item["options"]
    question = "Which of the following responses would align with the user's preferences the most?"
    for opt in options:
        question += f"\n{opt['letter']}: {opt['option']}"
    question += "\nPlease respond with the letter of the option that aligns with the user's preferences the most and nothing else."
    return question


def build_prompt(item: dict) -> str:
    """Assemble the full evaluation prompt (conversation + question).

    This concatenates the entire conversation history with the MCQ question.
    For the default evaluation protocol, pass the output directly to an LLM.
    For custom methods (RAG, summarization, etc.), use parse_conversations()
    and build_question() separately.
    """
    return item["conversation"] + "\n" + build_question(item)


def extract_letter(response_text: str) -> str:
    """Extract the selected letter (A-E) from a model response."""
    text = response_text.strip()
    if text and text[0] in "ABCDE":
        return text[0]
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else ""


def call_litellm(model: str, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
    """Send a prompt via litellm and return the response text."""
    import litellm
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def call_llm_module(client, prompt: str) -> str:
    """Send a prompt via the src.llm LLMClient and return the response text."""
    result = client.generate(prompt)
    return result.get("response_text", "") if isinstance(result, dict) else str(result)


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    arr = np.array(values)
    means = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def run_evaluation(args):
    logger.info(f"Loading HorizonBench ({args.config} config)...")
    ds = load_benchmark(config=args.config, dataset=args.dataset, max_items=args.max_items)

    logger.info(f"Evaluating {len(ds)} items with model={args.model}")

    llm_client = None
    if args.backend == "llm":
        from src.llm.api_client import LLMClient
        llm_client = LLMClient({"model": args.model, "max_tokens": 64, "temperature": 0.0})

    results = []
    output_path = Path(args.output_dir) / f"results_{args.model.replace('/', '_')}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if output_path.exists() and args.resume:
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing_ids.add(r["id"])
                results.append(r)
        logger.info(f"Resuming: {len(existing_ids)} items already completed")

    correct_total, total = 0, 0
    for i, item in enumerate(ds):
        if item["id"] in existing_ids:
            continue

        prompt = build_prompt(item)

        try:
            if args.backend == "litellm":
                response_text = call_litellm(args.model, prompt, max_tokens=args.max_tokens, temperature=args.temperature)
            else:
                response_text = call_llm_module(llm_client, prompt)
        except Exception as e:
            logger.warning(f"Item {i} failed: {e}")
            continue

        predicted = extract_letter(response_text)
        is_correct = predicted == item["correct_letter"]

        result = {
            "id": item["id"],
            "generator": item["generator"],
            "has_evolved": item["has_evolved"],
            "correct_letter": item["correct_letter"],
            "predicted_letter": predicted,
            "correct": is_correct,
            "response_text": response_text[:500],
        }
        results.append(result)

        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        total += 1
        correct_total += int(is_correct)

        if (i + 1) % 50 == 0:
            logger.info(f"  [{i+1}/{len(ds)}] running accuracy: {correct_total/total*100:.1f}%")

    print_results(results, args.model)
    logger.info(f"Results saved to {output_path}")


def print_results(results: list, model: str):
    """Print accuracy summary."""
    if not results:
        print("No results to report.")
        return

    def acc(subset):
        if not subset:
            return 0.0, (0.0, 0.0), 0
        correct = [int(r["correct"]) for r in subset]
        mean = np.mean(correct) * 100
        lo, hi = bootstrap_ci(correct)
        return mean, (lo * 100, hi * 100), len(subset)

    all_acc, all_ci, n = acc(results)
    evolved = [r for r in results if r["has_evolved"]]
    static = [r for r in results if not r["has_evolved"]]
    evo_acc, evo_ci, n_evo = acc(evolved)
    sta_acc, sta_ci, n_sta = acc(static)

    print(f"\n{'='*60}")
    print(f"  HorizonBench Results: {model}")
    print(f"{'='*60}")
    print(f"  Overall:   {all_acc:5.1f}%  [{all_ci[0]:.1f}, {all_ci[1]:.1f}]  (n={n})")
    print(f"  Evolved:   {evo_acc:5.1f}%  [{evo_ci[0]:.1f}, {evo_ci[1]:.1f}]  (n={n_evo})")
    print(f"  Static:    {sta_acc:5.1f}%  [{sta_ci[0]:.1f}, {sta_ci[1]:.1f}]  (n={n_sta})")
    gap = evo_acc - sta_acc
    print(f"  Evo-Static gap: {gap:+.1f} pp")
    print(f"  Chance baseline: 20.0%")

    generators = sorted(set(r["generator"] for r in results))
    if len(generators) > 1:
        print(f"\n  By generator:")
        for gen in generators:
            gen_results = [r for r in results if r["generator"] == gen]
            g_acc, g_ci, g_n = acc(gen_results)
            print(f"    {gen:20s}  {g_acc:5.1f}%  [{g_ci[0]:.1f}, {g_ci[1]:.1f}]  (n={g_n})")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on HorizonBench")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o, claude-sonnet-4-20250514)")
    parser.add_argument("--backend", choices=["litellm", "llm"], default="litellm",
                        help="LLM backend: 'litellm' (default) or 'llm' (src.llm module)")
    parser.add_argument("--dataset", type=str, default=DATASET_ID, help="HuggingFace dataset ID")
    parser.add_argument("--config", type=str, default="benchmark", help="Dataset config (benchmark or sample)")
    parser.add_argument("--max-items", type=int, default=None, help="Evaluate only the first N items")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens for model response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for result files")
    parser.add_argument("--resume", action="store_true", help="Resume from existing result file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.seed)

    run_evaluation(args)


if __name__ == "__main__":
    main()
