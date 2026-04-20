import argparse
import json
import yaml
import os
import sys
import random
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conversation.conversation_generator import ConversationGenerator
from src.causal_framework.models.user_model import UserModel
from src.causal_framework.models.event_model import EventModel
from src.causal_framework.models.preference_model import PreferenceModel

# --- Helper functions for working with the new conversation format ---
def extract_turn_preferences(turn):
    """Extract preferences from a conversation turn."""
    if turn.get("role") == "user":
        return turn.get("preference_expressed", [])
    elif turn.get("role") == "assistant":
        return turn.get("preference_appealed", [])
    return []

def get_turn_content(turn):
    """Get the content from a conversation turn."""
    return turn.get("content", "")

def analyze_preference_expression_patterns(conversation, preferences):
    """Analyze how preferences are expressed throughout the conversation."""
    preference_usage = {pref_id: {"expressed": 0, "appealed": 0, "turns": []} for pref_id in preferences.keys()}
    
    for i, turn in enumerate(conversation.get("turns", [])):
        turn_prefs = extract_turn_preferences(turn)
        for pref_id in turn_prefs:
            if pref_id in preference_usage:
                preference_usage[pref_id]["turns"].append(i)
                if turn.get("role") == "user":
                    preference_usage[pref_id]["expressed"] += 1
                elif turn.get("role") == "assistant":
                    preference_usage[pref_id]["appealed"] += 1
    
    return preference_usage

# --- Helper functions for loading models (stubs, to be replaced with actual model classes if available) ---
def build_user_model(conv_json):
    # Return a UserModel instance if user info is available
    user_data = conv_json.get("user", None)
    if user_data is None:
        raise ValueError("No user data found in conversation JSON.")
    # If user_data is already a dict with 'attributes', use from_dict
    if isinstance(user_data, dict) and "attributes" in user_data:
        return UserModel.from_dict(user_data)
    # Otherwise, assume it's a flat dict of attributes
    return UserModel(user_data)

def build_event_model(conv_json):
    # Return an EventModel instance if event info is available
    event_data = conv_json.get("event", None)
    if event_data is None:
        raise ValueError("No event data found in conversation JSON.")
    # If event_data is already a dict with required fields, use from_dict
    if isinstance(event_data, dict) and all(k in event_data for k in ["id", "structure_id", "attributes"]):
        return EventModel.from_dict(event_data)
    raise ValueError("Event data is not in the expected format for EventModel.")

def build_preferences_model(pref_json):
    # pref_json is expected to be a dict of {pref_id: pref_dict}
    preferences = {}
    for pref_id, pref_data in pref_json.items():
        if isinstance(pref_data, dict) and "structure_id" in pref_data and "attributes" in pref_data:
            preferences[pref_id] = PreferenceModel.from_dict(pref_data)
        else:
            raise ValueError(f"Preference data for {pref_id} is not in the expected format.")
    return preferences

# --- Argument parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark counterfactual conversation generation.")
    parser.add_argument("--conversation_dir", type=str, help="Path to conversation directory (overrides config)")
    parser.add_argument("--output_path", type=str, help="Path to save results (overrides config)")
    parser.add_argument("--mode", type=str, choices=["manual", "random", "llm"], default="llm")
    parser.add_argument("--config", type=str, help="Path to config.yaml for configuration", default="./src/config_benchmark.yaml")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test nodes for random/llm mode")
    parser.add_argument("--num_counterfactuals", type=int, default=3)
    return parser.parse_args()

# --- Data loading ---
def load_conversation_json(path):
    with open(path) as f:
        return json.load(f)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Test node selection ---
def find_test_nodes_manual(config_path, conversation_dir):
    config = load_config(config_path)
    test_nodes = []
    
    for node in config.get("manual_test_nodes", []):
        # Find user directory safely
        user_dirs = [d for d in os.listdir(conversation_dir) if node["user_id"] in d]
        if not user_dirs:
            print(f"Warning: No user directory found for user_id '{node['user_id']}' in {conversation_dir}")
            continue
        user_dir = user_dirs[0]
        
        # Get conversation files safely
        user_path = os.path.join(conversation_dir, user_dir)
        conv_files = sorted([f for f in os.listdir(user_path) if f.endswith(".json")])
        if not conv_files:
            print(f"Warning: No conversation files found in {user_path}")
            continue
        
        # Find specific conversation file safely
        target_conv_files = [f for f in conv_files if f.startswith(str(node["conv_idx"]) + "_")]
        if not target_conv_files:
            print(f"Warning: No conversation file found starting with '{node['conv_idx']}_' in {user_path}")
            print(f"Available files: {conv_files}")
            continue
        conv_file = target_conv_files[0]
        
        test_nodes.append({
            "user_dir": user_dir,
            "conv_file": conv_file,
            "turn_idx": node["turn_idx"],
            "attribute": node.get("attribute"),
            "new_value": node.get("new_value")
        })
    
    print(f"Manual mode: Found {len(test_nodes)} valid test nodes")
    return test_nodes

def find_test_nodes_random(conversation_dir, num_samples, min_test_node_turns, seed=42):
    # First, collect all meaningful attributes from assistant turns with preference appeals
    all_attributes = []
    user_dirs = [d for d in os.listdir(conversation_dir) if d.startswith("user_")]
    
    for user_dir in user_dirs:
        user_path = os.path.join(conversation_dir, user_dir)
        conv_files = [f for f in os.listdir(user_path) if f.endswith(".json")]
        
        for conv_file in conv_files:
            try:
                conv_json = load_conversation_json(os.path.join(user_path, conv_file))
                conversation = conv_json["conversation"]
                preferences = conv_json["relevant_preferences"]
                turns = conversation.get("turns", [])
                
                # Find all assistant turns with preference_appealed content
                for turn_idx, turn in enumerate(turns):
                    if (turn.get("role") == "assistant" and 
                        len(turn.get("preference_appealed", [])) > 0 and
                        turn_idx >= min_test_node_turns):
                        
                        # For each appealed preference, add all its attributes
                        for pref_id in turn.get("preference_appealed", []):
                            if pref_id in preferences:
                                pref_data = preferences[pref_id]
                                for attr_name in pref_data.get("attributes", {}):
                                    all_attributes.append({
                                        "user_dir": user_dir,
                                        "conv_file": conv_file,
                                        "turn_idx": turn_idx,
                                        "preference_id": pref_id,
                                        "attribute": attr_name,
                                        "original_value": pref_data["attributes"][attr_name]
                                    })
            except Exception as e:
                print(f"Error processing {user_dir}/{conv_file}: {e}")
                continue
    
    if not all_attributes:
        print("Warning: No meaningful attributes found in assistant turns")
        return []
    
    # Randomly sample attributes
    random.seed(seed)
    sampled_attributes = random.sample(all_attributes, min(num_samples, len(all_attributes)))
    
    print(f"Randomly selected {len(sampled_attributes)} attributes from {len(all_attributes)} total attributes")
    return sampled_attributes

def find_test_nodes_llm(conversation_dir, llm_client, num_samples, min_test_node_turns, seed=42):
    # First, collect all meaningful attributes from assistant turns with preference appeals
    all_attributes = []
    user_dirs = [d for d in os.listdir(conversation_dir) if d.startswith("user_")]
    
    for user_dir in user_dirs:
        user_path = os.path.join(conversation_dir, user_dir)
        conv_files = [f for f in os.listdir(user_path) if f.endswith(".json")]
        
        for conv_file in conv_files:
            try:
                conv_json = load_conversation_json(os.path.join(user_path, conv_file))
                conversation = conv_json["conversation"]
                preferences = conv_json["relevant_preferences"]
                turns = conversation.get("turns", [])
                
                # Find all assistant turns with preference_appealed content
                for turn_idx, turn in enumerate(turns):
                    if (turn.get("role") == "assistant" and 
                        len(turn.get("preference_appealed", [])) > 0 and
                        turn_idx >= min_test_node_turns):
                        
                        # For each appealed preference, add all its attributes
                        for pref_id in turn.get("preference_appealed", []):
                            if pref_id in preferences:
                                pref_data = preferences[pref_id]
                                for attr_name in pref_data.get("attributes", {}):
                                    all_attributes.append({
                                        "user_dir": user_dir,
                                        "conv_file": conv_file,
                                        "turn_idx": turn_idx,
                                        "preference_id": pref_id,
                                        "attribute": attr_name,
                                        "original_value": pref_data["attributes"][attr_name],
                                        "conversation": conversation,
                                        "preferences": preferences
                                    })
            except Exception as e:
                print(f"Error processing {user_dir}/{conv_file}: {e}")
                continue
    
    if not all_attributes:
        print("Warning: No meaningful attributes found in assistant turns")
        return []
    
    # Use LLM to intelligently sample the most meaningful attributes
    prompt = f"""
You are an expert in user modeling and counterfactual reasoning for AI benchmarking.

Given {len(all_attributes)} potential preference attributes from conversations, select the {min(num_samples, len(all_attributes))} most meaningful ones for counterfactual testing.

Criteria for selection:
1. Attributes that would create significant behavioral changes if modified
2. Attributes that are clearly expressed in the assistant's response
3. Diverse set covering different types of preferences
4. Attributes with clear contrasting alternatives
5. Turns that are not too early in the conversation and that are spread out, preferably don't repeatedly select the same turn or attribute

Available attributes:
""" 
    random.seed(seed)
    all_attributes = random.sample(all_attributes, min(num_samples, len(all_attributes)))
    for i, attr in enumerate(all_attributes[:50]):  # Limit to first 50 for prompt size
        prompt += f"{i+1}. {attr['user_dir']}/{attr['conv_file']} turn {attr['turn_idx']}: {attr['preference_id']}.{attr['attribute']} = {attr['original_value']}\n"
    
    if len(all_attributes) > 50:
        prompt += f"... and {len(all_attributes) - 50} more attributes\n"
    
    prompt += f"""

Return a JSON object with the selected attribute indices:
{{
  "selected_indices": [<index1>, <index2>, ...]
}}

Select exactly {min(num_samples, len(all_attributes))} indices.
"""
    
    try:
        response = llm_client.generate(prompt, response_format="json")
        selection = json.loads(response["response_text"])
        selected_indices = selection.get("selected_indices", [])
        
        # Validate indices and select attributes
        selected_attributes = []
        for idx in selected_indices:
            if 0 <= idx < len(all_attributes):
                selected_attributes.append(all_attributes[idx])
        
        if not selected_attributes:
            # Fallback to random selection
            selected_attributes = random.sample(all_attributes, min(num_samples, len(all_attributes)))
            
    except Exception as e:
        print(f"LLM selection failed: {e}. Using random selection.")
        selected_attributes = random.sample(all_attributes, min(num_samples, len(all_attributes)))
    
    print(f"LLM selected {len(selected_attributes)} attributes from {len(all_attributes)} total attributes")
    return selected_attributes



# --- Attribute perturbation ---
def get_perturbable_attributes(preferences):
    # Returns a list of (pref_id, attr) tuples for all attributes
    attrs = []
    for pref_id, pref in preferences.items():
        # Use the PreferenceModel's get_all_attributes method
        for attr in pref.get_all_attributes():
            attrs.append((pref_id, attr))
    return attrs

def get_alternative_values(attr_name, orig_value, num_alternatives=3, pref_id=None, llm_client=None):
    """
    Generate multiple semantically meaningful alternative values that create proper counterfactuals.
    This should generate values that are MAXIMALLY CONTRASTING to the user's actual preferences.
    """
    if llm_client and pref_id:
        # Use LLM to generate semantically meaningful alternatives
        prompt = f"""Generate {num_alternatives} MAXIMALLY CONTRASTING alternative values for this preference attribute.

Preference: {pref_id}
Attribute: {attr_name}
Original Value: {orig_value}
Original Type: {type(orig_value).__name__}

### CRITICAL REQUIREMENTS:
Generate values that are as DIFFERENT as possible from each other AND from the original:
1. If original is on one end of a spectrum, include the OPPOSITE end
2. Include QUALITATIVELY different categories, not just degree variations
3. Each alternative should lead to a NOTICEABLY DIFFERENT assistant response
4. Alternatives should be easy to distinguish in text (not subtle variations)

### BAD EXAMPLES (too similar - AVOID THESE):
- Original: "detailed" → "comprehensive", "thorough", "in-depth" (all mean basically the same thing!)
- Original: "positive" → "upbeat", "optimistic", "cheerful" (all positive sentiments!)
- Original: "high" → "very high", "extremely high", "quite high" (just degree variations!)
- Original: 3 → 4, 5, 6 (too close numerically!)

### GOOD EXAMPLES (maximally contrasting - DO THIS):
- Original: "detailed" → "brief", "bullet-points-only", "visual-diagram-only"
- Original: "positive" → "neutral-factual", "devil's-advocate", "tough-love"
- Original: "high" → "low", "none", "moderate-with-caveats"
- Original: 3 → 1, 5, 10 (spread across the range)
- Original: "solo" → "large-group", "partner", "team-based"
- Original: "formal" → "casual", "playful", "technical-jargon"

### FOR NUMERICAL VALUES:
- Spread alternatives across the full plausible range
- Don't cluster near the original value
- Include extremes if they make sense

Return a JSON object with this structure:
{{
  "alternatives": [
    {{"value": <alternative1>, "contrast_type": "opposite|different_category|different_approach", "explanation": "Why this creates a meaningfully different response"}},
    {{"value": <alternative2>, "contrast_type": "opposite|different_category|different_approach", "explanation": "Why this creates a meaningfully different response"}},
    {{"value": <alternative3>, "contrast_type": "opposite|different_category|different_approach", "explanation": "Why this creates a meaningfully different response"}}
  ]
}}

IMPORTANT: Each alternative must be OBVIOUSLY different from the others. If you find yourself generating similar values, STOP and think of more contrasting options.
"""
        
        try:
            response = llm_client.generate(prompt, response_format="json")
            result = json.loads(response["response_text"])
            alternatives = result.get("alternatives", [])
            if alternatives:
                # Return all the alternative values
                return [alt["value"] for alt in alternatives if alt["value"] != orig_value]
        except Exception as e:
            print(f"Warning: LLM failed to generate alternatives for {attr_name}: {e}")
            # Fall back to heuristic
    
    # Fallback heuristic for when LLM is not available or fails
    alternatives = []
    for i in range(num_alternatives):
        alt = _get_heuristic_alternative(attr_name, orig_value)
        if alt != orig_value and alt not in alternatives:
            alternatives.append(alt)
        else:
            # Try some variations for additional alternatives
            # First check if it's a numeric value (even if stored as string)
            try:
                numeric_val = float(orig_value)
                # Generate variations based on the range of the value
                if 0 <= numeric_val <= 1:
                    # For 0-1 range, use smaller increments and stay in bounds
                    variations = [
                        max(0, numeric_val - 0.1),
                        min(1, numeric_val + 0.1),
                        max(0, numeric_val - 0.3),
                        min(1, numeric_val + 0.3),
                        0.5 if numeric_val != 0.5 else 0.2
                    ]
                else:
                    # For other ranges, use standard variations
                    variations = [numeric_val + 1, numeric_val - 1, numeric_val * 2, numeric_val / 2]
                
                # Convert back to original type if needed
                if isinstance(orig_value, int) or (isinstance(orig_value, str) and '.' not in str(orig_value)):
                    variations = [int(v) for v in variations if v == int(v) and v >= 0]
                
                for var in variations:
                    if var != numeric_val and var not in alternatives:
                        alternatives.append(var)
                        break
            except (ValueError, TypeError):
                # If not numeric, try string variations
                if isinstance(orig_value, str):
                    variations = [f"{alt}_variant", f"alternative_{alt}", f"modified_{alt}"]
                    for var in variations:
                        if var not in alternatives:
                            alternatives.append(var)
                            break
                else:
                    # If not numeric and not string, just use the original alternative
                    if alt not in alternatives:
                        alternatives.append(alt)
    
    return alternatives[:num_alternatives]  # Return only the requested number

def _get_heuristic_alternative(attr_name, orig_value):
    """Fallback heuristic-based alternative value generation."""
    
    # Define some common contrasting mappings
    if isinstance(orig_value, str):
        contrasts = {
            # Risk levels
            "high": "low", "low": "high", "medium": "high",
            "conservative": "aggressive", "aggressive": "conservative",
            
            # Experience levels  
            "expert": "beginner", "beginner": "expert", "intermediate": "expert",
            "advanced": "novice", "novice": "advanced",
            
            # Preferences
            "online": "in_person", "in_person": "online", "digital": "physical",
            "virtual": "physical", "remote": "local",
            
            # Sizes/amounts
            "large": "small", "small": "large", "big": "tiny",
            
            # Time horizons
            "short_term": "long_term", "long_term": "short_term",
            "immediate": "delayed", "quick": "slow",
            
            # Locations
            "hybrid": "in_person", "indoor": "outdoor", "urban": "rural",
            "nature": "urban", "outdoor": "indoor", "rural": "urban"
        }
        
        # Try exact match first
        if orig_value.lower() in contrasts:
            return contrasts[orig_value.lower()]
        
        # Try partial matches
        for key, value in contrasts.items():
            if key in orig_value.lower():
                return orig_value.lower().replace(key, value)
    
    elif isinstance(orig_value, bool):
        return not orig_value
    
    elif isinstance(orig_value, int):
        # For scales, try to invert them
        if 1 <= orig_value <= 5:  # Typical 1-5 scale
            return 6 - orig_value
        elif 1 <= orig_value <= 10:  # 1-10 scale
            return 11 - orig_value
        else:
            return max(1, orig_value // 2) if orig_value > 2 else orig_value * 2
    
    elif isinstance(orig_value, float):
        # For float scales, invert if it looks like a 0-1 scale
        if 0 <= orig_value <= 1:
            return 1.0 - orig_value
        else:
            return orig_value * 0.5 if orig_value > 1 else orig_value * 2
    
    elif isinstance(orig_value, list):
        # For lists, try to find contrasting options
        financial_contrasts = {
            "stocks": ["bonds", "savings_accounts"],
            "cryptocurrency": ["government_bonds", "certificates_of_deposit"],
            "real_estate": ["stocks", "mutual_funds"],
            "bonds": ["stocks", "cryptocurrency"]
        }
        
        for item in orig_value:
            if item.lower() in financial_contrasts:
                return financial_contrasts[item.lower()]
        
        # If no specific contrasts found, return a general opposite
        return ["conservative_options"] if any("crypto" in str(x).lower() or "stock" in str(x).lower() for x in orig_value) else ["growth_investments"]
    
    # If all else fails, return the original value (no change)
    return orig_value

def perturb_graph(user, event, preferences, attribute, new_value):
    user_cf = deepcopy(user)
    event_cf = deepcopy(event)
    preferences_cf = deepcopy(preferences)
    pref_id, attr = attribute.split(".", 1)
    
    if pref_id in preferences_cf:
        preferences_cf[pref_id].set_attribute(attr, new_value)
    else:
        raise ValueError(f"Preference ID '{pref_id}' not found in preferences")
    
    return user_cf, event_cf, preferences_cf

# --- Difference metric ---
def compute_difference_metric(original_prefs, perturbed_prefs):
    diff = 0.0
    for pref_id in original_prefs:
        orig_attrs = original_prefs[pref_id].get_all_attributes()
        pert_attrs = perturbed_prefs[pref_id].get_all_attributes()
        meta = original_prefs[pref_id].metadata
        weights = meta.get("weights", {})
        for attr in orig_attrs:
            orig = orig_attrs[attr]
            pert = pert_attrs.get(attr, None)
            # Use weight if present, else 1
            w = weights.get(attr, 1)
            if orig != pert:
                diff += w
    return diff

def compute_preference_expression_difference(original_conversation, cf_conversation, original_preferences, cf_preferences, perturbed_turn_idx, original_patterns, cf_patterns):
    """
    Compute the difference in preference expression patterns, accounting for attribute value changes.
    
    Args:
        original_conversation: Original conversation dict
        cf_conversation: Counterfactual conversation dict  
        original_preferences: Original preference models
        cf_preferences: Counterfactual preference models
        perturbed_turn_idx: Index of the turn that was regenerated
    """
    diff_metrics = {
        "preference_shift": 0,  # Number of preferences that shifted expression/appeal counts
        "turn_impact": 0,      # Number of turns where preferences changed
        "total_expression_change": 0,  # Total change in expression counts
        "attribute_value_changes": 0,  # Number of preference attributes that changed values
        "content_reflects_changes": False  # Whether the regenerated content reflects the attribute changes
    }
    
    # Analyze structural changes (same as before)
    
    all_prefs = set(original_patterns.keys()) | set(cf_patterns.keys())
    
    for pref_id in all_prefs:
        orig_pattern = original_patterns.get(pref_id, {"expressed": 0, "appealed": 0, "turns": []})
        cf_pattern = cf_patterns.get(pref_id, {"expressed": 0, "appealed": 0, "turns": []})
        
        # Check if expression/appeal counts changed
        if (orig_pattern["expressed"] != cf_pattern["expressed"] or 
            orig_pattern["appealed"] != cf_pattern["appealed"]):
            diff_metrics["preference_shift"] += 1
            
        # Count total expression changes
        diff_metrics["total_expression_change"] += abs(orig_pattern["expressed"] - cf_pattern["expressed"])
        diff_metrics["total_expression_change"] += abs(orig_pattern["appealed"] - cf_pattern["appealed"])
        
        # Count turn impact differences
        orig_turns = set(orig_pattern["turns"])
        cf_turns = set(cf_pattern["turns"])
        if orig_turns != cf_turns:
            diff_metrics["turn_impact"] += len(orig_turns.symmetric_difference(cf_turns))
    
    # NEW: Analyze attribute value changes
    for pref_id in original_preferences:
        if pref_id in cf_preferences:
            orig_attrs = original_preferences[pref_id].get_all_attributes()
            cf_attrs = cf_preferences[pref_id].get_all_attributes()
            
            for attr_name, orig_value in orig_attrs.items():
                cf_value = cf_attrs.get(attr_name)
                if orig_value != cf_value:
                    diff_metrics["attribute_value_changes"] += 1
    
    # NEW: Check if the regenerated turn content reflects the attribute changes
    if perturbed_turn_idx is not None:
        orig_turn = original_conversation["turns"][perturbed_turn_idx]
        cf_turn = cf_conversation["turns"][perturbed_turn_idx]
        
        orig_content = orig_turn.get("content", "").lower()
        cf_content = cf_turn.get("content", "").lower()
        
        # Simple heuristic: if content changed and we have attribute changes, assume it reflects changes
        if orig_content != cf_content and diff_metrics["attribute_value_changes"] > 0:
            diff_metrics["content_reflects_changes"] = True
            
            # More sophisticated check: look for evidence of the new attribute values in the content
            for pref_id in original_preferences:
                if pref_id in cf_preferences:
                    orig_attrs = original_preferences[pref_id].get_all_attributes()
                    cf_attrs = cf_preferences[pref_id].get_all_attributes()
                    
                    for attr_name, orig_value in orig_attrs.items():
                        cf_value = cf_attrs.get(attr_name)
                        if orig_value != cf_value:
                            # Check if the new value appears in the counterfactual content
                            if str(cf_value).lower() in cf_content and str(orig_value).lower() not in cf_content:
                                diff_metrics["content_reflects_changes"] = True
                            # Check if old value is gone and new value is present
                            elif str(orig_value).lower() in orig_content and str(cf_value).lower() in cf_content:
                                diff_metrics["content_reflects_changes"] = True
    
    return diff_metrics

def compute_turn_content_difference(original_turn, cf_turn):
    """Compute the difference between original and counterfactual turn content."""
    orig_content = original_turn.get("content", "")
    cf_content = cf_turn.get("content", "")
    
    # Simple metrics for content difference
    diff_metrics = {
        "content_changed": orig_content != cf_content,
        "content_length_diff": abs(len(cf_content) - len(orig_content)),
        "word_count_diff": abs(len(cf_content.split()) - len(orig_content.split())),
        "character_similarity": len(set(orig_content.lower()) & set(cf_content.lower())) / max(len(set(orig_content.lower())), len(set(cf_content.lower())), 1)
    }
    
    return diff_metrics

def generate_diverse_counterfactuals(conversation, turn_idx, user, event, preferences, attribute, orig_value, alternative_values, conv_gen):
    """
    Generate diverse counterfactual responses for multiple attribute values in a single prompt.
    
    Args:
        conversation: Original conversation
        turn_idx: Turn index to regenerate
        user: User model
        event: Event model  
        preferences: Original preferences
        attribute: Attribute being perturbed (e.g., "spirituality_230.group_size_preference")
        orig_value: Original attribute value
        alternative_values: List of alternative values to generate responses for
        conv_gen: Conversation generator
    
    Returns:
        Dict mapping alternative values to their counterfactual data
    """
    pref_id, attr_name = attribute.split(".", 1)
    
    # Create counterfactual preference models for each alternative
    cf_models = {}
    for alt_value in alternative_values:
        user_cf, event_cf, preferences_cf = perturb_graph(user, event, preferences, attribute, alt_value)
        cf_models[alt_value] = {
            "user_cf": user_cf,
            "event_cf": event_cf, 
            "preferences_cf": preferences_cf
        }
    
    # Build the prompt for diverse generation
    original_turn = conversation["turns"][turn_idx]
    context_turns = conversation["turns"][:turn_idx]
    
    # Format context
    context_str = ""
    for i, turn in enumerate(context_turns[-5:]):  # Last 5 turns for context
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        context_str += f"{role.capitalize()}: {content}\n"
    
    # Create the prompt
    prompt = f"""You are generating diverse alternative responses for a conversation based on different user preference configurations.

CONTEXT:
{context_str}

ORIGINAL RESPONSE:
{original_turn.get("content", "")}

PREFERENCE BEING VARIED:
- Attribute: {attr_name}
- Original Value: {orig_value}

TASK: Generate {len(alternative_values)} DIVERSE alternative responses that each reflect a different preference value. Each response should:
1. Be significantly different from the others in style, approach, and content
2. Clearly reflect the specific preference value assigned to it
3. Be natural and conversational
4. Maintain the same helpful, friendly tone
5. Be appropriate for the conversation context

ALTERNATIVE PREFERENCE VALUES AND REQUIRED RESPONSES:
"""
    
    for i, alt_value in enumerate(alternative_values):
        prompt += f"\nOption {chr(66 + i)} (Preference: {attr_name} = {alt_value}):\n"
        prompt += f"Generate a response that clearly reflects this preference value and is distinct from other options.\n"
    
    prompt += f"""

Return your response in the following JSON format:
{{
  "B": "Response for {alternative_values[0] if len(alternative_values) > 0 else 'first alternative'}",
  "C": "Response for {alternative_values[1] if len(alternative_values) > 1 else 'second alternative'}",
  "D": "Response for {alternative_values[2] if len(alternative_values) > 2 else 'third alternative'}"
}}

Make sure each response is:
- Substantially different from the others, one should be able to map the preference value to the response
- Clearly reflects its assigned preference value
- Natural and contextually appropriate
- Engaging and helpful
- Important: Do not directly mention the original or alternative values in the response!
"""
    
    try:
        # Generate diverse responses
        response = conv_gen.convo_llm_client.generate(prompt, response_format="json")
        generated_responses = json.loads(response["response_text"])
        
        # Create results for each alternative
        results = {}
        option_keys = ["B", "C", "D"]
        
        for i, alt_value in enumerate(alternative_values):
            if i >= len(option_keys):
                break
                
            option_key = option_keys[i]
            if option_key in generated_responses:
                # Create counterfactual conversation
                cf_conversation = deepcopy(conversation)
                cf_conversation["turns"][turn_idx]["content"] = generated_responses[option_key]
                
                # Update turn preferences if needed
                cf_models_data = cf_models[alt_value]
                
                results[alt_value] = {
                    "user_cf": cf_models_data["user_cf"],
                    "event_cf": cf_models_data["event_cf"],
                    "preferences_cf": cf_models_data["preferences_cf"],
                    "conversation": cf_conversation
                }
            else:
                print(f"Warning: No response generated for option {option_key} (value: {alt_value})")
                
    except Exception as e:
        print(f"Error generating diverse counterfactuals: {e}")
        # Fallback to individual generation
        results = {}
        for alt_value in alternative_values:
            user_cf, event_cf, preferences_cf = perturb_graph(user, event, preferences, attribute, alt_value)
            cf_conversation = conv_gen.regenerate_conversation_turn(
                conversation, turn_idx, user_cf, event_cf, preferences_cf, original_preferences=preferences
            )
            results[alt_value] = {
                "user_cf": user_cf,
                "event_cf": event_cf,
                "preferences_cf": preferences_cf,
                "conversation": cf_conversation
            }
    
    return results

# --- Main workflow ---
def main():
    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        # Override config with command line arguments
        if args.mode:
            config["mode"] = args.mode
        if args.num_samples:
            config["num_samples"] = args.num_samples
        if args.num_counterfactuals:
            config["num_counterfactuals"] = args.num_counterfactuals
    else:
        # Use command line arguments as config
        config = {
            "mode": args.mode,
            "num_samples": args.num_samples,
            "num_counterfactuals": args.num_counterfactuals
        }
    
    # Determine conversation_dir - command line arg takes precedence over config
    conversation_dir = args.conversation_dir or config.get("conversation_dir")
    if not conversation_dir:
        raise ValueError("No conversation_dir specified. Provide it via --conversation_dir argument or in the config file.")
    
    # Determine output path - command line arg takes precedence over config
    output_path = args.output_path or config.get("output_path")
    if not output_path:
        raise ValueError("No output path specified. Provide it via --output_path argument or in the config file.")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize LLMClient and ConversationGenerator
    from src.llm.api_client import LLMClient
    llm_config = config.get("llm_config", {})
    if llm_config.get("model", "") == "" and os.path.exists(llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml")):
        with open(llm_config.get("model_kwargs", {}).get("api_info", "api_info.yaml"), "r") as f:
            api_info = yaml.safe_load(f)
        llm_config["model"] = api_info.get(llm_config.get("model_kwargs", {}).get("api_account", "openai"), {}).get("model_name", "gpt-4o")
        
    llm_client = LLMClient(config=llm_config)

    conv_gen = ConversationGenerator(llm_client, config)
    
    # Find test nodes based on mode
    if config["mode"] == "manual":
        test_nodes = find_test_nodes_manual(args.config, conversation_dir)
    elif config["mode"] == "random":
        test_nodes = find_test_nodes_random(conversation_dir, config["num_samples"], config["min_test_node_turns"], config["random_seed"])
    else:  # llm mode
        test_nodes = find_test_nodes_llm(conversation_dir, llm_client, config["num_samples"], config["min_test_node_turns"], config["random_seed"])
    # Group results by test node
    grouped_results = {}
    print(f"Processing {len(test_nodes)} test nodes...")
    
    for node_idx, node in enumerate(test_nodes):
        if node_idx % 10 == 0:
            print(f"Progress: {node_idx}/{len(test_nodes)} test nodes processed")
            
        conv_path = os.path.join(conversation_dir, node["user_dir"], node["conv_file"])
        conv_json = load_conversation_json(conv_path)
        user = build_user_model(conv_json)
        event = build_event_model(conv_json)
        preferences = build_preferences_model(conv_json["relevant_preferences"])
        conversation = conv_json["conversation"]
        original_turn = conversation["turns"][node["turn_idx"]]
        
        # Extract attribute information from the test node
        pref_id = node["preference_id"]
        attr_name = node["attribute"]
        orig_value = node["original_value"]
        attribute = f"{pref_id}.{attr_name}"
        
        # Create unique key for this test node
        node_key = f"{node['user_dir']}_{node['conv_file']}_{node['turn_idx']}_{pref_id}_{attr_name}"
        
        if node_key not in grouped_results:
            # Extract context (previous 10 turns)
            context_turns = []
            start_idx = max(0, node["turn_idx"] - 10)
            for i in range(start_idx, node["turn_idx"]):
                turn = conversation["turns"][i]
                context_turns.append({
                    "role": turn.get("role"),
                    "content": turn.get("content", ""),
                    "turn_index": i
                })
            
            # Initialize the grouped result structure
            grouped_results[node_key] = {
                "test_node": {
                    "user_dir": node["user_dir"],
                    "conv_file": node["conv_file"],
                    "turn_idx": node["turn_idx"],
                    "preference_id": pref_id,
                    "attribute": attr_name,
                    "perturbed_attribute": attribute,
                    "original_value": orig_value
                },
                "context": context_turns,
                "options": {},
                "original_response": "A",
                "additional_information": {}
            }
            
            # Add original response as option A
            original_patterns = analyze_preference_expression_patterns(conversation, preferences)
            original_turn_prefs = [preferences[p].to_dict() for p in extract_turn_preferences(original_turn) if p in preferences]
            
            grouped_results[node_key]["options"]["A"] = original_turn.get("content", "")
            grouped_results[node_key]["additional_information"]["A"] = {
                "new_value": orig_value,
                "alternative_index": "original",
                "turn_preferences": original_turn_prefs,
                "preference_patterns": original_patterns,
                "attribute_difference_metric": 0.0,
                "expression_difference_metric": {
                    "preference_shift": 0,
                    "turn_impact": 0,
                    "total_expression_change": 0,
                    "attribute_value_changes": 0,
                    "content_reflects_changes": False
                },
                "content_difference_metric": {
                    "content_changed": False,
                    "content_length_diff": 0,
                    "word_count_diff": 0,
                    "character_similarity": 1.0
                },
                "user_cf": user.to_dict() if hasattr(user, 'to_dict') else user,
                "event_cf": event.to_dict() if hasattr(event, 'to_dict') else event,
                "preferences_cf": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in preferences.items()}
            }
        
        # Generate alternative values for this attribute
        if config["mode"] == "manual":
            # For manual mode, we expect specific values in the config
            alternative_values = [node.get("new_value", orig_value)]
        else:
            # Generate multiple alternative values using LLM or heuristics
            alternative_values = get_alternative_values(
                attr_name, orig_value, 
                num_alternatives=config["num_counterfactuals"], 
                pref_id=pref_id, 
                llm_client=llm_client
            )
        
        # Filter out alternatives that are the same as original
        valid_alternatives = [val for val in alternative_values if val != orig_value]
        if not valid_alternatives:
            print(f"Warning: No valid alternatives for {attribute}. Skipping.")
            continue
            
        print(f"Generating diverse counterfactuals for {attribute}: {orig_value} -> {valid_alternatives}")
        
        # Generate all counterfactuals together for diversity
        cf_results = generate_diverse_counterfactuals(
            conversation, node["turn_idx"], user, event, preferences, 
            attribute, orig_value, valid_alternatives, conv_gen
        )
        
        # Process results for each alternative
        option_labels = ["B", "C", "D"]  # Start from B since A is original
        original_patterns = analyze_preference_expression_patterns(conversation, preferences)
        
        for alt_idx, (new_value, cf_data) in enumerate(cf_results.items()):
            if alt_idx >= len(option_labels):
                print(f"Warning: Too many alternatives for {attribute}. Skipping alternative {alt_idx + 1}")
                break
                
            option_label = option_labels[alt_idx]
            
            user_cf = cf_data["user_cf"]
            event_cf = cf_data["event_cf"] 
            preferences_cf = cf_data["preferences_cf"]
            cf_conversation = cf_data["conversation"]
            cf_turn = cf_conversation["turns"][node["turn_idx"]]
            
            cf_patterns = analyze_preference_expression_patterns(cf_conversation, preferences_cf)
            diff_val = compute_difference_metric(preferences, preferences_cf)
            expression_diff = compute_preference_expression_difference(
                conversation, cf_conversation, preferences, preferences_cf, node["turn_idx"], original_patterns, cf_patterns
            )
            content_diff = compute_turn_content_difference(original_turn, cf_turn)
            cf_turn_prefs = [preferences_cf[p].to_dict() for p in extract_turn_preferences(cf_turn) if p in preferences_cf]
            
            # Add to grouped results
            grouped_results[node_key]["options"][option_label] = cf_turn.get("content", "")
            grouped_results[node_key]["additional_information"][option_label] = {
                "new_value": new_value,
                "alternative_index": alt_idx,
                "turn_preferences": cf_turn_prefs,
                "preference_patterns": cf_patterns,
                "attribute_difference_metric": diff_val,
                "expression_difference_metric": expression_diff,
                "content_difference_metric": content_diff,
                "user_cf": user_cf.to_dict() if hasattr(user_cf, 'to_dict') else user_cf,
                "event_cf": event_cf.to_dict() if hasattr(event_cf, 'to_dict') else event_cf,
                "preferences_cf": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in preferences_cf.items()} if isinstance(preferences_cf, dict) else preferences_cf
            }
    
    # Convert grouped results to list format
    results = list(grouped_results.values())
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark generation completed!")
    print(f"Total test nodes processed: {len(test_nodes)}")
    print(f"Total grouped benchmark results generated: {len(results)}")
    
    # Calculate total alternatives generated
    total_alternatives = 0
    for result in results:
        total_alternatives += len(result["options"]) - 1  # Subtract 1 for original response
    
    print(f"Total counterfactual alternatives generated: {total_alternatives}")
    print(f"Results saved to: {output_path}")


def analyze_history_validation(benchmark_dir: str, save_summary: bool = True) -> Dict[str, Any]:
    """
    Analyze history validation results across all benchmark items.
    
    This helps assess the quality of the generated benchmark by checking
    how many items truly require historical context to answer correctly.
    
    Args:
        benchmark_dir: Path to the benchmark output directory
        save_summary: If True, writes a summary JSON file to the benchmark directory
        
    Returns:
        Dict with analysis results including:
            - total_items: Total number of benchmark items
            - valid_items: Items that require history (good benchmark items)
            - invalid_items: Items that don't require history (potentially problematic)
            - validation_rate: Percentage of items that passed validation
            - baseline_accuracy_distribution: Distribution of baseline accuracies
            - passed_items: List of filenames that passed validation
            - failed_items: List of filenames that failed validation
            - details: Per-item validation results
    """
    import glob

    results = {
        "total_items": 0,
        "valid_items": 0,
        "invalid_items": 0,
        "not_validated": 0,
        "validation_rate": 0.0,
        "baseline_accuracy_distribution": [],
        "passed_items": [],
        "failed_items": [],
        "not_validated_items": [],
        "details": []
    }
    
    # Find all counterfactual files
    pattern = os.path.join(benchmark_dir, "**", "*_counterfactuals.json")
    files = glob.glob(pattern, recursive=True)
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            results["total_items"] += 1
            
            validation = data.get("history_validation", {})
            requires_history = validation.get("requires_history")
            baseline_accuracy = validation.get("baseline_accuracy")
            
            filename = os.path.basename(filepath)
            relative_path = os.path.relpath(filepath, benchmark_dir)
            
            item_detail = {
                "file": filename,
                "path": relative_path,
                "requires_history": requires_history,
                "baseline_accuracy": baseline_accuracy,
                "chance_level": validation.get("chance_level"),
                "num_options": validation.get("num_options"),
                "confidence": validation.get("confidence"),
                "reasoning": validation.get("reasoning", ""),
                "models_used": validation.get("models_used", []),
                "per_model_results": validation.get("per_model_results", {}),
                "temporal_preference_id": data.get("metadata", {}).get("temporal_preference_id"),
                "user": data.get("metadata", {}).get("user", {}).get("name", "unknown")
            }
            results["details"].append(item_detail)
            
            if requires_history is None:
                results["not_validated"] += 1
                results["not_validated_items"].append(relative_path)
            elif requires_history:
                results["valid_items"] += 1
                results["passed_items"].append(relative_path)
                if baseline_accuracy is not None:
                    results["baseline_accuracy_distribution"].append(baseline_accuracy)
            else:
                results["invalid_items"] += 1
                results["failed_items"].append(relative_path)
                if baseline_accuracy is not None:
                    results["baseline_accuracy_distribution"].append(baseline_accuracy)
                    
        except Exception as e:
            print(f"Warning: Failed to process {filepath}: {e}")
            continue
    
    # Calculate validation rate
    validated_items = results["valid_items"] + results["invalid_items"]
    if validated_items > 0:
        results["validation_rate"] = results["valid_items"] / validated_items
    
    # Calculate baseline accuracy statistics
    if results["baseline_accuracy_distribution"]:
        accuracies = results["baseline_accuracy_distribution"]
        results["baseline_accuracy_stats"] = {
            "mean": sum(accuracies) / len(accuracies),
            "min": min(accuracies),
            "max": max(accuracies),
            "count": len(accuracies)
        }
    
    # Aggregate per-model statistics
    per_model_stats = {}
    for item in results["details"]:
        per_model = item.get("per_model_results", {})
        for model_name, model_result in per_model.items():
            if model_name not in per_model_stats:
                per_model_stats[model_name] = {
                    "total_correct": 0,
                    "total_runs": 0,
                    "items_evaluated": 0
                }
            per_model_stats[model_name]["total_correct"] += model_result.get("correct_count", 0)
            per_model_stats[model_name]["total_runs"] += model_result.get("valid_runs", 0)
            per_model_stats[model_name]["items_evaluated"] += 1
    
    # Calculate per-model accuracy
    for model_name, stats in per_model_stats.items():
        if stats["total_runs"] > 0:
            stats["accuracy"] = stats["total_correct"] / stats["total_runs"]
        else:
            stats["accuracy"] = None
    
    results["per_model_stats"] = per_model_stats
    
    # Save summary JSON if requested
    if save_summary and results["total_items"] > 0:
        summary_data = {
            "generated_at": datetime.now().isoformat(),
            "benchmark_directory": benchmark_dir,
            "summary": {
                "total_items": results["total_items"],
                "valid_items": results["valid_items"],
                "invalid_items": results["invalid_items"],
                "not_validated": results["not_validated"],
                "validation_rate": results["validation_rate"],
                "baseline_accuracy_stats": results.get("baseline_accuracy_stats", {}),
                "per_model_stats": per_model_stats
            },
            "passed_items": results["passed_items"],
            "failed_items": results["failed_items"],
            "not_validated_items": results["not_validated_items"],
            "item_details": results["details"]
        }
        
        summary_path = os.path.join(benchmark_dir, "history_validation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Validation summary saved to: {summary_path}")
    
    return results


def print_history_validation_report(benchmark_dir: str) -> None:
    """
    Print a formatted report of history validation results.
    
    Args:
        benchmark_dir: Path to the benchmark output directory
    """
    results = analyze_history_validation(benchmark_dir)
    
    print("\n" + "="*60)
    print("HISTORY VALIDATION REPORT")
    print("="*60)
    print(f"\nTotal benchmark items: {results['total_items']}")
    print(f"  ✓ Valid (requires history): {results['valid_items']}")
    print(f"  ✗ Invalid (guessable): {results['invalid_items']}")
    print(f"  ? Not validated: {results['not_validated']}")
    
    if results['total_items'] > 0:
        print(f"\nValidation rate: {results['validation_rate']:.1%}")
        
    if "baseline_accuracy_stats" in results:
        stats = results["baseline_accuracy_stats"]
        print(f"\nBaseline accuracy (without history):")
        print(f"  Mean: {stats['mean']:.1%}")
        print(f"  Min:  {stats['min']:.1%}")
        print(f"  Max:  {stats['max']:.1%}")
    
    # Show per-model statistics
    if "per_model_stats" in results and results["per_model_stats"]:
        print(f"\nPer-model validation accuracy:")
        for model_name, stats in results["per_model_stats"].items():
            acc = stats.get("accuracy")
            acc_str = f"{acc:.1%}" if acc is not None else "N/A"
            print(f"  {model_name}: {acc_str} ({stats['total_correct']}/{stats['total_runs']} runs across {stats['items_evaluated']} items)")
    
    # Show problematic items
    invalid_items = [d for d in results["details"] if d["requires_history"] == False]
    if invalid_items:
        print(f"\n⚠ ITEMS THAT MAY NOT REQUIRE HISTORY ({len(invalid_items)}):")
        for item in invalid_items[:10]:  # Show first 10
            print(f"  - {item['file']}: baseline_accuracy={item['baseline_accuracy']:.1%}")
            print(f"    {item['reasoning']}")
        if len(invalid_items) > 10:
            print(f"  ... and {len(invalid_items) - 10} more")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()