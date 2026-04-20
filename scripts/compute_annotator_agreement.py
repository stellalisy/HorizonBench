#!/usr/bin/env python3
"""
Script to compute annotator agreement and majority voted answer accuracy.
Handles empty lines as "discard" decisions and includes them in agreement calculations.
"""

import os
from collections import Counter
from typing import List, Dict
import numpy as np

def read_annotations(file_path: str) -> List[str]:
    """Read annotations from a file, treating empty lines as 'DISCARD'."""
    annotations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                annotations.append('DISCARD')
            else:
                annotations.append(line)
    return annotations

def compute_fleiss_kappa(annotations: List[List[str]]) -> Dict[str, float]:
    """Compute Fleiss Kappa agreement between annotators."""
    n_annotators = len(annotations)
    n_items = len(annotations[0])
    
    # Get all unique categories (including DISCARD)
    all_categories = set()
    for ann in annotations:
        all_categories.update(ann)
    categories = sorted(list(all_categories))
    
    # Create agreement matrix: items x categories
    agreement_matrix = np.zeros((n_items, len(categories)))
    
    for item_idx in range(n_items):
        for ann_idx in range(n_annotators):
            category = annotations[ann_idx][item_idx]
            cat_idx = categories.index(category)
            agreement_matrix[item_idx, cat_idx] += 1
    
    # Calculate Fleiss Kappa
    # P = observed agreement, Pe = expected agreement by chance
    # Kappa = (P - Pe) / (1 - Pe)
    
    # Calculate P (observed agreement)
    P = 0
    for item_idx in range(n_items):
        item_agreements = 0
        for cat_idx in range(len(categories)):
            n_ij = agreement_matrix[item_idx, cat_idx]
            item_agreements += n_ij * (n_ij - 1)
        P += item_agreements / (n_annotators * (n_annotators - 1))
    P = P / n_items
    
    # Calculate Pe (expected agreement by chance)
    Pe = 0
    for cat_idx in range(len(categories)):
        p_j = np.sum(agreement_matrix[:, cat_idx]) / (n_items * n_annotators)
        Pe += p_j ** 2
    
    # Calculate Fleiss Kappa
    if Pe == 1:
        kappa = 1.0  # Perfect agreement
    else:
        kappa = (P - Pe) / (1 - Pe)
    
    # Calculate standard error and confidence interval
    # SE = sqrt(2 / (n_items * n_annotators * (n_annotators - 1)))
    SE = np.sqrt(2 / (n_items * n_annotators * (n_annotators - 1)))
    
    # 95% confidence interval
    CI_lower = kappa - 1.96 * SE
    CI_upper = kappa + 1.96 * SE
    
    # Interpret kappa value
    if kappa < 0:
        interpretation = "Poor agreement"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    return {
        'fleiss_kappa': kappa,
        'observed_agreement': P,
        'expected_agreement': Pe,
        'standard_error': SE,
        'ci_lower': CI_lower,
        'ci_upper': CI_upper,
        'interpretation': interpretation,
        'n_categories': len(categories),
        'categories': categories
    }

def compute_pairwise_agreement(annotations: List[List[str]]) -> Dict[str, float]:
    """Compute pairwise agreement between annotators."""
    n_annotators = len(annotations)
    n_items = len(annotations[0])
    
    # Initialize agreement counters
    total_agreements = 0
    total_items = 0
    
    # For each item, check agreement between all pairs of annotators
    for item_idx in range(n_items):
        item_annotations = [annotations[ann_idx][item_idx] for ann_idx in range(n_annotators)]
        
        # Count agreements for this item
        item_agreements = 0
        item_pairs = 0
        
        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                if item_annotations[i] == item_annotations[j]:
                    item_agreements += 1
                item_pairs += 1
        
        total_agreements += item_agreements
        total_items += item_pairs
    
    pairwise_agreement = total_agreements / total_items if total_items > 0 else 0.0
    
    # Also compute overall agreement (all annotators agree)
    complete_agreements = 0
    for item_idx in range(n_items):
        item_annotations = [annotations[ann_idx][item_idx] for ann_idx in range(n_annotators)]
        if len(set(item_annotations)) == 1:  # All annotators agree
            complete_agreements += 1
    
    complete_agreement_rate = complete_agreements / n_items if n_items > 0 else 0.0
    
    return {
        'pairwise_agreement': pairwise_agreement,
        'complete_agreement_rate': complete_agreement_rate,
        'total_items': n_items,
        'complete_agreements': complete_agreements
    }

def compute_majority_vote(annotations: List[List[str]]) -> List[str]:
    """Compute majority vote for each item. In case of tie, prefer non-DISCARD answers."""
    n_items = len(annotations[0])
    majority_votes = []
    
    for item_idx in range(n_items):
        item_annotations = [annotations[ann_idx][item_idx] for ann_idx in range(len(annotations))]
        
        # Count votes for each option
        vote_counts = Counter(item_annotations)
        
        # Find the most common vote
        most_common = vote_counts.most_common(1)[0]
        
        # If there's a tie, prefer non-DISCARD answers
        if len(vote_counts) > 1:
            # Check if there are multiple options with the same count
            max_count = most_common[1]
            tied_options = [opt for opt, count in vote_counts.items() if count == max_count]
            
            if len(tied_options) > 1:
                # Prefer non-DISCARD options in case of tie
                non_discard_options = [opt for opt in tied_options if opt != 'DISCARD']
                if non_discard_options:
                    majority_votes.append(non_discard_options[0])
                else:
                    majority_votes.append('DISCARD')
            else:
                majority_votes.append(most_common[0])
        else:
            majority_votes.append(most_common[0])
    
    return majority_votes

def compute_accuracy(majority_votes: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """Compute accuracy of majority votes against ground truth."""
    if len(majority_votes) != len(ground_truth):
        print(f"Warning: Length mismatch - majority votes: {len(majority_votes)}, ground truth: {len(ground_truth)}")
        return {}
    
    correct = 0
    total = 0
    discarded = 0
    
    for i, (vote, truth) in enumerate(zip(majority_votes, ground_truth)):
        if vote == 'DISCARD':
            discarded += 1
            # Count as incorrect as per requirements
            total += 1
        else:
            total += 1
            if vote == truth:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    discard_rate = discarded / len(majority_votes) if majority_votes else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'discarded': discarded,
        'discard_rate': discard_rate
    }

def compute_consensus_accuracy(annotations: List[List[str]], ground_truth: List[str]) -> Dict[str, float]:
    """Compute accuracy for samples where all annotators agreed."""
    if not annotations or len(annotations[0]) != len(ground_truth):
        print(f"Warning: Length mismatch in consensus accuracy calculation")
        return {}
    
    n_items = len(annotations[0])
    consensus_correct = 0
    consensus_total = 0
    consensus_discarded = 0
    
    for item_idx in range(n_items):
        item_annotations = [annotations[ann_idx][item_idx] for ann_idx in range(len(annotations))]
        
        # Check if all annotators agree
        if len(set(item_annotations)) == 1:  # All annotators agree
            consensus_total += 1
            agreed_answer = item_annotations[0]
            truth = ground_truth[item_idx]
            
            if agreed_answer == 'DISCARD':
                consensus_discarded += 1
                # Count as incorrect as per requirements
            else:
                if agreed_answer == truth:
                    consensus_correct += 1
    
    consensus_accuracy = consensus_correct / consensus_total if consensus_total > 0 else 0.0
    consensus_discard_rate = consensus_discarded / consensus_total if consensus_total > 0 else 0.0
    
    return {
        'consensus_accuracy': consensus_accuracy,
        'consensus_correct': consensus_correct,
        'consensus_total': consensus_total,
        'consensus_discarded': consensus_discarded,
        'consensus_discard_rate': consensus_discard_rate
    }

def main():
    """Main function to compute and display results."""
    # File paths
    annotator_files = [
        'human_eval/annotator_1.txt',
        'human_eval/annotator_2.txt',
        'human_eval/annotator_3.txt'
    ]
    ground_truth_file = 'human_eval/ground_truth.txt'
    
    # Check if files exist
    for file_path in annotator_files + [ground_truth_file]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found!")
            return
    
    print("Reading annotation files...")
    
    # Read annotations from all annotators
    annotations = []
    for file_path in annotator_files:
        ann = read_annotations(file_path)
        annotations.append(ann)
        print(f"Read {len(ann)} annotations from {file_path}")
    
    # Read ground truth
    ground_truth = read_annotations(ground_truth_file)
    print(f"Read {len(ground_truth)} ground truth entries")
    
    # Ensure all files have the same number of annotations
    lengths = [len(ann) for ann in annotations]
    if len(set(lengths)) > 1:
        print(f"Warning: Annotation files have different lengths: {lengths}")
        # Use the minimum length
        min_length = min(lengths)
        annotations = [ann[:min_length] for ann in annotations]
        ground_truth = ground_truth[:min_length]
        print(f"Truncated to {min_length} entries for analysis")
    
    print("\n" + "="*60)
    print("ANNOTATOR AGREEMENT ANALYSIS")
    print("="*60)
    
    # Compute pairwise agreement
    agreement_stats = compute_pairwise_agreement(annotations)
    
    print(f"Total items analyzed: {agreement_stats['total_items']}")
    print(f"Pairwise agreement rate: {agreement_stats['pairwise_agreement']:.4f} ({agreement_stats['pairwise_agreement']*100:.2f}%)")
    print(f"Complete agreement rate: {agreement_stats['complete_agreement_rate']:.4f} ({agreement_stats['complete_agreement_rate']*100:.2f}%)")
    print(f"Items with complete agreement: {agreement_stats['complete_agreements']}")
    
    print("\n" + "="*60)
    print("MAJORITY VOTE ACCURACY ANALYSIS")
    print("="*60)
    
    # Compute majority votes
    majority_votes = compute_majority_vote(annotations)
    print(f"Computed majority votes for {len(majority_votes)} items")
    
    # Compute accuracy against ground truth
    accuracy_stats = compute_accuracy(majority_votes, ground_truth)
    
    print(f"Majority vote accuracy: {accuracy_stats['accuracy']:.4f} ({accuracy_stats['accuracy']*100:.2f}%)")
    print(f"Correct predictions: {accuracy_stats['correct']}")
    print(f"Total predictions: {accuracy_stats['total']}")
    print(f"Discarded items: {accuracy_stats['discarded']}")
    print(f"Discard rate: {accuracy_stats['discard_rate']:.4f} ({accuracy_stats['discard_rate']*100:.2f}%)")
    
    print("\n" + "="*60)
    print("CONSENSUS ACCURACY ANALYSIS")
    print("="*60)
    
    # Compute consensus accuracy (where all annotators agreed)
    consensus_stats = compute_consensus_accuracy(annotations, ground_truth)
    
    print(f"Consensus accuracy: {consensus_stats['consensus_accuracy']:.4f} ({consensus_stats['consensus_accuracy']*100:.2f}%)")
    print(f"Consensus correct predictions: {consensus_stats['consensus_correct']}")
    print(f"Total consensus items: {consensus_stats['consensus_total']}")
    print(f"Consensus discarded items: {consensus_stats['consensus_discarded']}")
    print(f"Consensus discard rate: {consensus_stats['consensus_discard_rate']:.4f} ({consensus_stats['consensus_discard_rate']*100:.2f}%)")
    
    # Compute Fleiss Kappa
    fleiss_kappa_stats = compute_fleiss_kappa(annotations)
    print("\n" + "="*60)
    print("FLEISS KAPPA AGREEMENT ANALYSIS")
    print("="*60)
    print(f"Fleiss Kappa: {fleiss_kappa_stats['fleiss_kappa']:.4f}")
    print(f"Observed Agreement (P): {fleiss_kappa_stats['observed_agreement']:.4f}")
    print(f"Expected Agreement (Pe): {fleiss_kappa_stats['expected_agreement']:.4f}")
    print(f"Standard Error (SE): {fleiss_kappa_stats['standard_error']:.4f}")
    print(f"95% Confidence Interval (Lower): {fleiss_kappa_stats['ci_lower']:.4f}")
    print(f"95% Confidence Interval (Upper): {fleiss_kappa_stats['ci_upper']:.4f}")
    print(f"Interpretation: {fleiss_kappa_stats['interpretation']}")
    print(f"Number of Categories: {fleiss_kappa_stats['n_categories']}")
    print(f"Categories: {', '.join(fleiss_kappa_stats['categories'])}")
    
    # Detailed breakdown
    print("\n" + "="*60)
    print("DETAILED BREAKDOWN")
    print("="*60)
    
    # Show some examples of disagreements
    disagreement_examples = []
    for i in range(len(annotations[0])):
        item_annotations = [annotations[ann_idx][i] for ann_idx in range(len(annotations))]
        if len(set(item_annotations)) > 1:  # There's a disagreement
            disagreement_examples.append((i+1, item_annotations, majority_votes[i], ground_truth[i]))
    
    print(f"Found {len(disagreement_examples)} items with annotator disagreements")
    
    if disagreement_examples:
        print("\nFirst 10 disagreement examples:")
        print("Item | Annotator1 | Annotator2 | Annotator3 | Majority | Ground Truth")
        print("-" * 70)
        for i, (item_num, anns, majority, truth) in enumerate(disagreement_examples[:10]):
            print(f"{item_num:4d} | {anns[0]:10s} | {anns[1]:10s} | {anns[2]:10s} | {majority:8s} | {truth:12s}")
    
    # Show some examples of complete agreements
    agreement_examples = []
    for i in range(len(annotations[0])):
        item_annotations = [annotations[ann_idx][i] for ann_idx in range(len(annotations))]
        if len(set(item_annotations)) == 1:  # All annotators agree
            agreement_examples.append((i+1, item_annotations[0], ground_truth[i]))
    
    print(f"\nFound {len(agreement_examples)} items with complete annotator agreement")
    
    if agreement_examples:
        print("\nFirst 10 complete agreement examples:")
        print("Item | Agreed Answer | Ground Truth | Correct?")
        print("-" * 50)
        for i, (item_num, agreed, truth) in enumerate(agreement_examples[:10]):
            correct = "✓" if agreed == truth else "✗"
            print(f"{item_num:4d} | {agreed:13s} | {truth:12s} | {correct:8s}")

if __name__ == "__main__":
    main()
