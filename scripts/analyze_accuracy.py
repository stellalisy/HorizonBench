"""
Quick script to calculate accuracy of each model on the benchmark.
Reports accuracy for all, filtered (requires_history=True), and unfiltered subsets.
"""

import json
import os
import sys

# Ordered list of models with display names
MODEL_ORDER = [
    ("claude-opus-4-5", "anthropic.claude-opus-4-5-20251101-v1:0"),
    ("claude-opus-4", "anthropic.claude-opus-4-20250514-v1:0"),
    ("claude-sonnet-4-5", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
    ("claude-sonnet-4", "anthropic.claude-sonnet-4-20250514-v1:0"),
    ("claude-3-7-sonnet", "anthropic.claude-3-7-sonnet-20250219-v1:0"),
    ("claude-3-5-sonnet-v2", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    ("claude-3-5-sonnet-v1", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    ("claude-3-5-haiku", "anthropic.claude-3-5-haiku-20241022-v1:0"),
    ("gemini-3-pro-preview", "gemini-3-pro-preview"),
    ("gemini-3-flash-preview", "gemini-3-flash-preview"),
    ("gemini-2.5-pro", "gemini-2.5-pro"),
    ("gemini-2.5-flash", "gemini-2.5-flash"),
    ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
    ("gemini-2.0-flash", "gemini-2.0-flash"),
    ("gemini-2.0-flash-lite", "gemini-2.0-flash-lite"),
    ("gpt-5", "gpt-5"),
    ("gpt-5-mini", "gpt-5-mini"),
    ("gpt-4.1", "gpt-4.1"),
    ("gpt-4o", "gpt-4o"),
    ("o4-mini", "o4-mini"),
    ("o3", "o3"),
    ("o3-mini", "o3-mini"),
    ("o1", "o1"),
]


def analyze_model_accuracy(results_file: str):
    """Analyze accuracy for a single model results file."""
    
    all_correct = []
    filtered_correct = []  # pass_filter = True
    unfiltered_correct = []  # pass_filter = False
    
    with open(results_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            correct = data.get('correct', False)
            pass_filter = data.get('pass_filter', False)
            
            all_correct.append(correct)
            if pass_filter:
                filtered_correct.append(correct)
            else:
                unfiltered_correct.append(correct)
    
    return {
        'all': {
            'count': len(all_correct),
            'correct': sum(all_correct),
            'accuracy': sum(all_correct) / len(all_correct) if all_correct else 0
        },
        'filtered': {
            'count': len(filtered_correct),
            'correct': sum(filtered_correct),
            'accuracy': sum(filtered_correct) / len(filtered_correct) if filtered_correct else 0
        },
        'unfiltered': {
            'count': len(unfiltered_correct),
            'correct': sum(unfiltered_correct),
            'accuracy': sum(unfiltered_correct) / len(unfiltered_correct) if unfiltered_correct else 0
        }
    }


def main(benchmark_dir: str):
    """Analyze accuracy for all models in the benchmark directory."""
    
    # Find all results files
    results_files = {f.replace('results_', '').replace('.jsonl', ''): f 
                     for f in os.listdir(benchmark_dir) 
                     if f.startswith('results_') and f.endswith('.jsonl')}
    
    if not results_files:
        print(f"No results files found in {benchmark_dir}")
        return
    
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Found {len(results_files)} model result files\n")
    
    # Header
    print(f"{'Model':<25} {'All':>10} {'Filtered':>10} {'Unfiltered':>10}")
    print("=" * 60)
    
    all_results = {}
    
    for display_name, file_key in MODEL_ORDER:
        if file_key not in results_files:
            continue
            
        filepath = os.path.join(benchmark_dir, results_files[file_key])
        stats = analyze_model_accuracy(filepath)
        all_results[display_name] = stats
        
        all_acc = f"{stats['all']['accuracy']:.1%}"
        filt_acc = f"{stats['filtered']['accuracy']:.1%}"
        unfilt_acc = f"{stats['unfiltered']['accuracy']:.1%}"
        
        print(f"{display_name:<25} {all_acc:>10} {filt_acc:>10} {unfilt_acc:>10}")
    
    print("=" * 60)
    
    # Summary statistics
    print("\nSummary:")
    print("-" * 40)
    
    # Average across models
    if all_results:
        avg_all = sum(r['all']['accuracy'] for r in all_results.values()) / len(all_results)
        avg_filtered = sum(r['filtered']['accuracy'] for r in all_results.values()) / len(all_results)
        avg_unfiltered = sum(r['unfiltered']['accuracy'] for r in all_results.values()) / len(all_results)
        
        print(f"Average accuracy (all):        {avg_all:.1%}")
        print(f"Average accuracy (filtered):   {avg_filtered:.1%}")
        print(f"Average accuracy (unfiltered): {avg_unfiltered:.1%}")


if __name__ == "__main__":
    default_path = "output"
    benchmark_dir = sys.argv[1] if len(sys.argv) > 1 else default_path
    main(benchmark_dir)
