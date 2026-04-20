"""
evaluator.py

Implements evaluation metrics and validation for the Causal Preference Evolution Framework.
Provides methods for assessing the quality, consistency, and plausibility of generated data.
"""

import json
import logging
import os
import random
import statistics
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from src.causal_framework.models.user_model import UserModel
from src.causal_framework.models.event_model import EventModel
from src.causal_framework.models.preference_model import PreferenceModel
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class FrameworkEvaluator:
    """
    Evaluates the quality, consistency, and plausibility of data generated
    by the Causal Preference Evolution Framework.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the framework evaluator.
        
        Args:
            llm_client: Client for accessing LLM services
            output_dir: Directory for saving evaluation results
            config: Optional configuration parameters
        """
        self.llm_client = llm_client
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set default configuration or update with provided config
        self.config = {
            "evaluation_samples": 100,  # Number of samples to evaluate
            "consistency_threshold": 0.7,  # Threshold for consistency score
            "plausibility_threshold": 0.7,  # Threshold for plausibility score
            "diversity_threshold": 0.5,  # Threshold for diversity score
            "random_seed": 42,
            "llm_based_evaluation": True,  # Whether to use LLM for subjective evaluation
            "save_detailed_results": True,  # Whether to save detailed results
            "metrics": [
                "temporal_consistency",
                "preference_stability",
                "psychological_plausibility",
                "conversation_naturalness",
                "event_sequence_plausibility",
                "diversity"
            ]
        }
        
        if config:
            self.config.update(config)
            
        # Set random seed
        random.seed(self.config["random_seed"])
        
        logger.info("FrameworkEvaluator initialized")
    
    def evaluate_timeline(self, timeline: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single timeline.
        
        Args:
            timeline: Generated timeline data
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating timeline for user: {timeline.get('user', {}).get('attributes', {}).get('first_name', 'unknown')}")
        
        # Initialize results
        results = {
            "timeline_id": timeline.get("user", {}).get("id", "unknown"),
            "user_name": f"{timeline.get('user', {}).get('attributes', {}).get('first_name', '')} {timeline.get('user', {}).get('attributes', {}).get('last_name', '')}",
            "metrics": {},
            "overall_score": 0.0,
            "passed": False
        }
        
        # Extract timeline components
        user_data = timeline.get("user", {})
        events = timeline.get("event_record", [])
        preference_record = timeline.get("preference_record", {})
        conversations = timeline.get("conversation_record", [])
        
        # Skip if timeline is empty
        if not events or not preference_record:
            logger.warning("Timeline is empty, skipping evaluation")
            results["overall_score"] = 0.0
            results["passed"] = False
            return results
        
        # Convert user data to UserModel
        user = UserModel(user_data.get("attributes", {}), user_id=user_data.get("id"))
        
        # Compute metrics
        metrics_to_compute = self.config["metrics"]
        metric_scores = {}
        
        try:
            # Temporal consistency
            if "temporal_consistency" in metrics_to_compute:
                temporal_score = self._evaluate_temporal_consistency(events, preference_record)
                metric_scores["temporal_consistency"] = temporal_score
            
            # Preference stability
            if "preference_stability" in metrics_to_compute:
                stability_score = self._evaluate_preference_stability(preference_record)
                metric_scores["preference_stability"] = stability_score
            
            # Psychological plausibility
            if "psychological_plausibility" in metrics_to_compute:
                plausibility_score = self._evaluate_psychological_plausibility(user, events, preference_record)
                metric_scores["psychological_plausibility"] = plausibility_score
            
            # Conversation naturalness
            if "conversation_naturalness" in metrics_to_compute:
                naturalness_score = self._evaluate_conversation_naturalness(conversations, user)
                metric_scores["conversation_naturalness"] = naturalness_score
            
            # Event sequence plausibility
            if "event_sequence_plausibility" in metrics_to_compute:
                sequence_score = self._evaluate_event_sequence_plausibility(events, user)
                metric_scores["event_sequence_plausibility"] = sequence_score
            
            # Diversity
            if "diversity" in metrics_to_compute:
                diversity_score = self._evaluate_diversity(timeline)
                metric_scores["diversity"] = diversity_score
            
            # Store metrics
            results["metrics"] = metric_scores
            
            # Calculate overall score (average of all metrics)
            if metric_scores:
                overall_score = statistics.mean(metric_scores.values())
                results["overall_score"] = overall_score
                
                # Consider passed if overall score meets threshold
                results["passed"] = overall_score >= self.config["plausibility_threshold"]
            
        except Exception as e:
            logger.error(f"Error evaluating timeline: {e}")
            # Return partial results if any metrics were computed
            if metric_scores:
                results["metrics"] = metric_scores
                results["overall_score"] = statistics.mean(metric_scores.values())
            
        logger.info(f"Timeline evaluation complete. Overall score: {results['overall_score']:.2f}")
        return results
    
    def evaluate_batch(self, timelines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of timelines.
        
        Args:
            timelines: List of generated timelines
            
        Returns:
            Dictionary of aggregate evaluation metrics
        """
        logger.info(f"Evaluating batch of {len(timelines)} timelines")
        
        # Sample timelines if too many
        if len(timelines) > self.config["evaluation_samples"]:
            logger.info(f"Sampling {self.config['evaluation_samples']} timelines for evaluation")
            eval_timelines = random.sample(timelines, self.config["evaluation_samples"])
        else:
            eval_timelines = timelines
        
        # Evaluate each timeline
        individual_results = []
        
        for timeline in eval_timelines:
            try:
                result = self.evaluate_timeline(timeline)
                individual_results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate timeline: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Compute aggregate metrics
        aggregate_results = self._compute_aggregate_metrics(individual_results)
        
        # Save detailed results if enabled
        if self.config["save_detailed_results"]:
            self._save_evaluation_results(individual_results, aggregate_results)
            
        logger.info(f"Batch evaluation complete. Overall score: {aggregate_results['overall_score']:.2f}")
        return aggregate_results
    
    def _evaluate_temporal_consistency(
        self,
        events: List[Dict[str, Any]],
        preference_record: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> float:
        """
        Evaluate temporal consistency of events and preferences.
        
        Args:
            events: List of events
            preference_record: Dictionary of preference records
            
        Returns:
            Temporal consistency score (0-1)
        """
        # Check if there are enough events and preference records
        if len(events) < 2 or len(preference_record) < 2:
            logger.warning("Not enough events or preference records for temporal consistency evaluation")
            return 0.5  # Neutral score
        
        # Sort events by date
        sorted_events = sorted(events, key=lambda e: e.get("date", ""))
        
        # Check chronological order
        chronological_errors = 0
        for i in range(1, len(sorted_events)):
            prev_date = sorted_events[i-1].get("date", "")
            curr_date = sorted_events[i].get("date", "")
            
            if prev_date > curr_date:
                chronological_errors += 1
        
        # Compute chronological order score
        chronological_score = 1.0 - (chronological_errors / len(sorted_events) if sorted_events else 0)
        
        # Sort preference records by date
        sorted_prefs = sorted([(k, v) for k, v in preference_record.items()], key=lambda x: x[0])
        
        # Check for abrupt preference changes
        preference_change_scores = []
        
        for i in range(1, len(sorted_prefs)):
            prev_prefs = sorted_prefs[i-1][1]
            curr_prefs = sorted_prefs[i][1]
            
            # For each preference type
            for pref_id in set(prev_prefs.keys()).intersection(curr_prefs.keys()):
                prev_pref = prev_prefs.get(pref_id, {})
                curr_pref = curr_prefs.get(pref_id, {})
                
                # Compare attributes
                if isinstance(prev_pref, dict) and isinstance(curr_pref, dict):
                    prev_attrs = prev_pref.get("attributes", {})
                    curr_attrs = curr_pref.get("attributes", {})
                    
                    for attr in set(prev_attrs.keys()).intersection(curr_attrs.keys()):
                        prev_val = prev_attrs.get(attr)
                        curr_val = curr_attrs.get(attr)
                        
                        # Check for change
                        if prev_val != curr_val:
                            # Calculate change magnitude for numeric values
                            if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                                max_val = max(abs(prev_val), abs(curr_val))
                                if max_val > 0:
                                    change_mag = abs(prev_val - curr_val) / max_val
                                    # Penalize large changes
                                    if change_mag > 0.5:
                                        preference_change_scores.append(1.0 - change_mag)
                                    else:
                                        preference_change_scores.append(1.0)
                                else:
                                    preference_change_scores.append(1.0)
                            # For non-numeric values, just record that a change happened
                            else:
                                preference_change_scores.append(0.8)  # Small penalty for any change
        
        # Compute preference change score
        preference_score = statistics.mean(preference_change_scores) if preference_change_scores else 1.0
        
        # Combine scores
        consistency_score = 0.5 * chronological_score + 0.5 * preference_score
        
        logger.debug(f"Temporal consistency score: {consistency_score:.2f}")
        return consistency_score
    
    def _evaluate_preference_stability(
        self,
        preference_record: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> float:
        """
        Evaluate preference stability over time.
        
        Args:
            preference_record: Dictionary of preference records
            
        Returns:
            Preference stability score (0-1)
        """
        # Check if there are enough preference records
        if len(preference_record) < 2:
            logger.warning("Not enough preference records for stability evaluation")
            return 0.5  # Neutral score
        
        # Sort preference records by date
        sorted_prefs = sorted([(k, v) for k, v in preference_record.items()], key=lambda x: x[0])
        
        # Calculate stability for each preference type
        stability_scores = []
        
        # Track preference types
        pref_types = set()
        for _, prefs in sorted_prefs:
            pref_types.update(prefs.keys())
        
        for pref_id in pref_types:
            # Extract values for this preference type across time
            pref_values = []
            
            for _, prefs in sorted_prefs:
                if pref_id in prefs:
                    pref_data = prefs[pref_id]
                    if isinstance(pref_data, dict) and "attributes" in pref_data:
                        pref_values.append(pref_data["attributes"])
            
            # Skip if not enough values
            if len(pref_values) < 2:
                continue
                
            # Calculate attribute stability
            for attr in set().union(*[p.keys() for p in pref_values]):
                attr_values = [p.get(attr) for p in pref_values if attr in p]
                
                # Skip if not enough values
                if len(attr_values) < 2:
                    continue
                
                # Calculate stability for numeric attributes
                if all(isinstance(v, (int, float)) for v in attr_values if v is not None):
                    non_none_values = [v for v in attr_values if v is not None]
                    if not non_none_values:
                        continue
                        
                    # Calculate variance
                    mean_val = statistics.mean(non_none_values)
                    max_val = max(abs(v) for v in non_none_values)
                    
                    if max_val > 0:
                        normalized_variance = statistics.variance(non_none_values) / (max_val ** 2)
                        
                        # Convert to stability score (higher variance = lower stability)
                        attr_stability = max(0.0, 1.0 - min(1.0, normalized_variance * 10))
                        stability_scores.append(attr_stability)
                    else:
                        # All values are zeros, perfectly stable
                        stability_scores.append(1.0)
                
                # For categorical attributes, count changes
                else:
                    changes = sum(1 for i in range(1, len(attr_values)) if attr_values[i] != attr_values[i-1])
                    change_rate = changes / (len(attr_values) - 1) if len(attr_values) > 1 else 0
                    
                    # Convert to stability score (higher change rate = lower stability)
                    attr_stability = 1.0 - change_rate
                    stability_scores.append(attr_stability)
        
        # Compute overall stability score
        stability_score = statistics.mean(stability_scores) if stability_scores else 0.5
        
        logger.debug(f"Preference stability score: {stability_score:.2f}")
        return stability_score
    
    def _evaluate_psychological_plausibility(
        self,
        user: UserModel,
        events: List[Dict[str, Any]],
        preference_record: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> float:
        """
        Evaluate psychological plausibility of preference evolution.
        
        Args:
            user: User model
            events: List of events
            preference_record: Dictionary of preference records
            
        Returns:
            Psychological plausibility score (0-1)
        """
        # For more subjective evaluation, use LLM if enabled
        if self.config["llm_based_evaluation"]:
            return self._llm_evaluate_psychological_plausibility(user, events, preference_record)
        
        # Otherwise, use heuristic evaluation
        # Check for preference-event correlation
        # Preferences should change in response to relevant events
        
        # Sort events and preference records by date
        sorted_events = sorted(events, key=lambda e: e.get("date", ""))
        sorted_prefs = sorted([(k, v) for k, v in preference_record.items()], key=lambda x: x[0])
        
        # Skip if not enough data
        if len(sorted_events) < 2 or len(sorted_prefs) < 2:
            logger.warning("Not enough events or preference records for psychological plausibility evaluation")
            return 0.5  # Neutral score
        
        # Track preference changes around events
        plausibility_scores = []
        
        # For each preference change
        for i in range(1, len(sorted_prefs)):
            prev_date = sorted_prefs[i-1][0]
            curr_date = sorted_prefs[i][0]
            prev_prefs = sorted_prefs[i-1][1]
            curr_prefs = sorted_prefs[i][1]
            
            # Find events between these preference states
            relevant_events = [
                e for e in sorted_events 
                if prev_date <= e.get("date", "") <= curr_date
            ]
            
            # If no events, preference change is less plausible
            if not relevant_events:
                plausibility_scores.append(0.3)  # Penalize changes without events
                continue
                
            # Check if changes are consistent with events
            for pref_id in set(prev_prefs.keys()).intersection(curr_prefs.keys()):
                prev_pref = prev_prefs.get(pref_id, {})
                curr_pref = curr_prefs.get(pref_id, {})
                
                # Compare attributes
                if isinstance(prev_pref, dict) and isinstance(curr_pref, dict):
                    prev_attrs = prev_pref.get("attributes", {})
                    curr_attrs = curr_pref.get("attributes", {})
                    
                    # Check for changes
                    changed_attrs = [
                        attr for attr in set(prev_attrs.keys()).intersection(curr_attrs.keys())
                        if prev_attrs.get(attr) != curr_attrs.get(attr)
                    ]
                    
                    if changed_attrs:
                        # Check if any event is relevant to this preference
                        pref_domain = prev_pref.get("domain", "unknown")
                        
                        relevant_event_found = False
                        for event in relevant_events:
                            event_data = event.get("event", {})
                            event_category = event_data.get("category", "unknown")
                            
                            # Simplistic relevance check based on category-domain matching
                            # Could be much more sophisticated in a real implementation
                            if self._is_category_relevant_to_domain(event_category, pref_domain):
                                relevant_event_found = True
                                break
                                
                        if relevant_event_found:
                            plausibility_scores.append(0.9)  # High plausibility for changes with relevant events
                        else:
                            plausibility_scores.append(0.5)  # Medium plausibility for changes with any events
        
        # Compute overall plausibility score
        plausibility_score = statistics.mean(plausibility_scores) if plausibility_scores else 0.5
        
        logger.debug(f"Psychological plausibility score: {plausibility_score:.2f}")
        return plausibility_score
    
    def _llm_evaluate_psychological_plausibility(
        self,
        user: UserModel,
        events: List[Dict[str, Any]],
        preference_record: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> float:
        """
        Use LLM to evaluate psychological plausibility.
        
        Args:
            user: User model
            events: List of events
            preference_record: Dictionary of preference records
            
        Returns:
            Psychological plausibility score (0-1)
        """
        try:
            # Prepare simplified user data
            user_info = {
                "age": user.get_attribute("age"),
                "gender": user.get_attribute("gender"),
                "interests": user.get_interests()[:5] if user.get_interests() else [],
                "personality_traits": user.get_personality_traits()[:5] if user.get_personality_traits() else [],
                "life_stage": user.get_life_stage()
            }
            
            # Prepare simplified event data
            event_data = []
            
            for event in sorted(events, key=lambda e: e.get("date", ""))[:10]:  # Limit to 10 events
                event_info = event.get("event", {})
                event_data.append({
                    "date": event.get("date", ""),
                    "name": event_info.get("name", "Event"),
                    "category": event_info.get("category", "unknown"),
                    "key_attributes": {
                        k: v for k, v in event_info.get("attributes", {}).items()
                        if k in ["location", "participants", "emotional_impact", "outcome"]
                    }
                })
            
            # Prepare simplified preference data
            pref_data = []
            for date, prefs in sorted(preference_record.items(), key=lambda x: x[0])[:5]:  # Limit to 5 timestamps
                pref_entry = {"date": date, "preferences": {}}
                
                for pref_id, pref_info in prefs.items():
                    if isinstance(pref_info, dict) and "attributes" in pref_info:
                        pref_entry["preferences"][pref_id] = {
                            "domain": pref_info.get("domain", "unknown"),
                            "key_attributes": {
                                k: v for k, v in pref_info.get("attributes", {}).items()
                                if k in list(pref_info.get("attributes", {}).keys())[:3]  # Limit to top 3 attributes
                            }
                        }
                
                pref_data.append(pref_entry)
            
            # Create prompt for LLM
            prompt = """
            Evaluate the psychological plausibility of preference evolution in the following user timeline.
            Consider principles like stability bias, contextual adaptation, experience-driven change, and preference interdependence.
            
            User information:
            """
            prompt += json.dumps(user_info, indent=2)
            
            prompt += "\n\nEvents (chronological order):\n"
            prompt += json.dumps(event_data, indent=2)
            
            prompt += "\n\nPreference evolution (chronological order):\n"
            prompt += json.dumps(pref_data, indent=2)
            
            prompt += """
            
            Evaluate the psychological plausibility of how the user's preferences evolve over time in relation to events and user characteristics.
            Consider the following:
            1. Do preferences change in response to relevant events?
            2. Are changes consistent with the user's characteristics?
            3. Is there an appropriate balance between stability and change?
            4. Are there any implausible or unjustified preference shifts?
            
            Provide a plausibility score from 0.0 to 1.0, where:
            - 0.0: Completely implausible
            - 0.5: Moderately plausible
            - 1.0: Highly plausible and psychologically realistic
            
            Return a JSON object with the following structure:
            {
              "plausibility_score": float,
              "reasoning": "brief explanation"
            }
            """
            
            # Query LLM
            response = self.llm_client.generate(prompt, response_format="json")
            response_data = json.loads(response["response_text"])
            
            # Extract score
            plausibility_score = response_data.get("plausibility_score", 0.5)
            reasoning = response_data.get("reasoning", "")
            
            logger.debug(f"LLM psychological plausibility score: {plausibility_score} - {reasoning}")
            return plausibility_score
            
        except Exception as e:
            logger.error(f"Failed to evaluate psychological plausibility with LLM: {e}")
            return 0.5  # Fallback to neutral score
    
    def _evaluate_conversation_naturalness(
        self,
        conversations: List[Dict[str, Any]],
        user: UserModel
    ) -> float:
        """
        Evaluate naturalness of conversations.
        
        Args:
            conversations: List of conversations
            user: User model
            
        Returns:
            Conversation naturalness score (0-1)
        """
        # Skip if no conversations
        if not conversations:
            logger.warning("No conversations for naturalness evaluation")
            return 0.5  # Neutral score
        
        # For more subjective evaluation, use LLM if enabled
        if self.config["llm_based_evaluation"]:
            return self._llm_evaluate_conversation_naturalness(conversations, user)
        
        # Otherwise, use heuristic evaluation
        naturalness_scores = []
        
        for conversation in conversations:
            # Get conversation turns
            turns = conversation.get("turns", [])
            
            # Skip if not enough turns
            if len(turns) < 2:
                continue
                
            # Check basic metrics
            
            # 1. Turn length variability (more variable = more natural)
            user_lengths = [len(turn.get("user", "").split()) for turn in turns]
            assistant_lengths = [len(turn.get("assistant", "").split()) for turn in turns]
            
            # Coefficient of variation for user turns
            user_mean = statistics.mean(user_lengths) if user_lengths else 0
            user_stdev = statistics.stdev(user_lengths) if len(user_lengths) > 1 else 0
            user_cv = user_stdev / user_mean if user_mean > 0 else 0
            
            # Score higher if coefficient of variation is in a natural range (0.2-0.6)
            length_var_score = min(1.0, max(0.0, 1.0 - abs(user_cv - 0.4) / 0.4))
            
            # 2. Question frequency (natural conversations have questions)
            question_marks = sum(1 for turn in turns for msg in [turn.get("user", ""), turn.get("assistant", "")] if "?" in msg)
            question_ratio = question_marks / (2 * len(turns))  # Two messages per turn
            
            # Score higher if question ratio is in a natural range (0.1-0.5)
            question_score = min(1.0, max(0.0, 1.0 - abs(question_ratio - 0.3) / 0.3))
            
            # 3. Message length ratio (assistant typically responds with proportional length)
            length_ratios = []
            for turn in turns:
                user_len = len(turn.get("user", "").split())
                assistant_len = len(turn.get("assistant", "").split())
                
                if user_len > 0:
                    length_ratios.append(assistant_len / user_len)
            
            # Score higher if ratio is in a natural range (0.8-2.5)
            mean_ratio = statistics.mean(length_ratios) if length_ratios else 0
            ratio_score = min(1.0, max(0.0, 1.0 - abs(mean_ratio - 1.5) / 1.5))
            
            # Combine scores
            turn_naturalness = (length_var_score + question_score + ratio_score) / 3
            naturalness_scores.append(turn_naturalness)
        
        # Compute overall naturalness score
        naturalness_score = statistics.mean(naturalness_scores) if naturalness_scores else 0.5
        
        logger.debug(f"Conversation naturalness score: {naturalness_score:.2f}")
        return naturalness_score
    
    def _llm_evaluate_conversation_naturalness(
        self,
        conversations: List[Dict[str, Any]],
        user: UserModel
    ) -> float:
        """
        Use LLM to evaluate conversation naturalness.
        
        Args:
            conversations: List of conversations
            user: User model
            
        Returns:
            Conversation naturalness score (0-1)
        """
        try:
            # Sample a conversation to evaluate
            conversation = random.choice(conversations)["conversation"]
            
            # Get conversation turns
            turns = conversation.get("turns", [])
            
            # Skip if not enough turns
            if len(turns) < 2:
                logger.warning("Not enough turns for conversation naturalness evaluation")
                return 0.5  # Neutral score
            
            # Limit to first 5 turns for brevity
            turns = turns[:5]
            
            # Prepare user info
            user_info = {
                "name": user.get_formatted_name(),
                "age": user.get_attribute("age"),
                "gender": user.get_attribute("gender")
            }
            
            # Create prompt for LLM
            prompt = (
            "Evaluate the naturalness and realism of the following conversation between a A and B.\n"
            "A's information:\n"
            )
            
            prompt += json.dumps(user_info, indent=2)
            
            prompt += "\n\nConversation:\n"
            for i, turn in enumerate(turns):
                # prompt += f"Turn {i+1}:\n"
                role = "A" if turn.get('role', '') == "user" else "B"
                prompt += f"{role}: {turn.get('content', '')}\n"
            
            prompt += (
            "\nEvaluate the naturalness and realism of this conversation. Consider the following:\n"
            "1. Does the user's language seem authentic and human-like?\n"
            "2. Are the turns well-connected and coherent as a conversation?\n"
            "3. Is there appropriate variation in message length and style?\n"
            "4. Does the conversation flow naturally or feel scripted/artificial?\n\n"
            "Provide a naturalness score from 0.0 to 1.0, where:\n"
            "- 0.0: Completely artificial/scripted\n"
            "- 0.5: Moderately natural\n"
            "- 1.0: Highly natural, indistinguishable from real human conversation\n\n"
            
            "Return a JSON object with the following structure:\n"
            "{{"
            '  "naturalness_score": float,'
            '  "reasoning": "brief explanation"'
            "}}"
            )
            
            # Query LLM
            response = self.llm_client.generate(prompt, response_format="json")
            response_data = json.loads(response["response_text"])
            
            # Extract score
            naturalness_score = response_data.get("naturalness_score", 0.5)
            reasoning = response_data.get("reasoning", "")
            
            logger.debug(f"LLM conversation naturalness score: {naturalness_score} - {reasoning}")
            return naturalness_score
            
        except Exception as e:
            logger.error(f"Failed to evaluate conversation naturalness with LLM: {e}")
            return 0.5  # Fallback to neutral score
    
    def _evaluate_event_sequence_plausibility(
        self,
        events: List[Dict[str, Any]],
        user: UserModel
    ) -> float:
        """
        Evaluate plausibility of event sequences.
        
        Args:
            events: List of events
            user: User model
            
        Returns:
            Event sequence plausibility score (0-1)
        """
        # Skip if not enough events
        if len(events) < 3:
            logger.warning("Not enough events for sequence plausibility evaluation")
            return 0.5  # Neutral score
        
        # For more subjective evaluation, use LLM if enabled
        if self.config["llm_based_evaluation"]:
            return self._llm_evaluate_event_sequence_plausibility(events, user)
        
        # Otherwise, use heuristic evaluation
        
        # Sort events by date
        sorted_events = sorted(events, key=lambda e: e.get("date", ""))
        
        # Check event frequency
        date_format = "%Y-%m-%dT%H:%M:%S"
        event_dates = []
        
        for event in sorted_events:
            date_str = event.get("date", "")
            if date_str:
                try:
                    event_dates.append(datetime.strptime(date_str.split(".")[0], date_format))
                except (ValueError, TypeError):
                    continue
        
        # Skip if not enough dates
        if len(event_dates) < 2:
            logger.warning("Not enough event dates for frequency evaluation")
            return 0.5  # Neutral score
        
        # Calculate time deltas between events
        time_deltas = [(event_dates[i] - event_dates[i-1]).total_seconds() / 86400 for i in range(1, len(event_dates))]
        
        # Check for unusually short deltas (multiple events in same day)
        short_deltas = [d for d in time_deltas if d < 1]
        short_ratio = len(short_deltas) / len(time_deltas) if time_deltas else 0
        
        # Penalize if too many events happen on the same day
        freq_score = 1.0 - min(1.0, short_ratio * 2)
        
        # Check for event type diversity
        event_categories = []
        for event in sorted_events:
            event_data = event.get("event", {})
            category = event_data.get("category", "unknown")
            event_categories.append(category)
        
        # Calculate category diversity
        unique_categories = set(event_categories)
        category_diversity = len(unique_categories) / len(event_categories) if event_categories else 0
        
        # Score higher if good category diversity (not too repetitive)
        diversity_score = min(1.0, category_diversity * 3)  # Scale up, reward diversity
        
        # Check for event type transitioning
        transition_scores = []
        for i in range(1, len(event_categories)):
            prev_cat = event_categories[i-1]
            curr_cat = event_categories[i]
            
            # Detect rapid category switching
            if prev_cat != curr_cat:
                # Check time between events
                if i < len(time_deltas)+1 and time_deltas[i-1] < 3:
                    # Penalize rapid switching between different categories
                    transition_scores.append(0.5)
                else:
                    # Reward normal transitions
                    transition_scores.append(1.0)
            else:
                # Same category is fine
                transition_scores.append(0.8)
        
        transition_score = statistics.mean(transition_scores) if transition_scores else 0.5
        
        # Combine scores
        plausibility_score = (freq_score + diversity_score + transition_score) / 3
        
        logger.debug(f"Event sequence plausibility score: {plausibility_score:.2f}")
        return plausibility_score
    
    def _llm_evaluate_event_sequence_plausibility(
        self,
        events: List[Dict[str, Any]],
        user: UserModel
    ) -> float:
        """
        Use LLM to evaluate event sequence plausibility.
        
        Args:
            events: List of events
            user: User model
            
        Returns:
            Event sequence plausibility score (0-1)
        """
        try:
            # Sort events by date
            sorted_events = sorted(events, key=lambda e: e.get("date", ""))
            
            # Limit to first 10 events for brevity
            sorted_events = sorted_events[:10]
            
            # Prepare user info
            user_info = {
                "age": user.get_attribute("age"),
                "gender": user.get_attribute("gender"),
                "occupation": user.get_attribute("occupation", "Unknown"),
                "life_stage": user.get_life_stage()
            }
            
            # Prepare event sequence data
            event_sequence = []
            for event in sorted_events:
                event_data = event.get("event", {})
                event_sequence.append({
                    "date": event.get("date", ""),
                    "name": event_data.get("name", "Event"),
                    "category": event_data.get("category", "unknown"),
                    "key_attributes": {
                        k: v for k, v in event_data.get("attributes", {}).items()
                        if k in ["location", "participants", "duration", "outcome"]
                    }
                })
            
            # Create prompt for LLM
            prompt = """
            Evaluate the plausibility of the following event sequence for a user.
            Consider how realistic and coherent the sequence is given the user's characteristics.
            
            User information:
            """
            prompt += json.dumps(user_info, indent=2)
            
            prompt += "\n\nEvent sequence (chronological order):\n"
            prompt += json.dumps(event_sequence, indent=2)
            
            prompt += """
            
            Evaluate the plausibility of this event sequence. Consider the following:
            1. Is the frequency of events realistic for a normal person?
            2. Do the events follow a logical progression?
            3. Are there any unlikely or implausible transitions between events?
            4. Is there appropriate diversity in event types?
            5. Is the sequence consistent with the user's characteristics?
            
            Provide a plausibility score from 0.0 to 1.0, where:
            - 0.0: Completely implausible
            - 0.5: Moderately plausible
            - 1.0: Highly plausible and realistic
            
            Return a JSON object with the following structure:
            {
              "plausibility_score": float,
              "reasoning": "brief explanation"
            }
            """
            
            # Query LLM
            response = self.llm_client.generate(prompt, response_format="json")
            response_data = json.loads(response["response_text"])
            
            # Extract score
            plausibility_score = response_data.get("plausibility_score", 0.5)
            reasoning = response_data.get("reasoning", "")
            
            logger.debug(f"LLM event sequence plausibility score: {plausibility_score} - {reasoning}")
            return plausibility_score
            
        except Exception as e:
            logger.error(f"Failed to evaluate event sequence plausibility with LLM: {e}")
            return 0.5  # Fallback to neutral score
    
    def _evaluate_diversity(self, timeline: Dict[str, Any]) -> float:
        """
        Evaluate diversity of the generated timeline.
        
        Args:
            timeline: Generated timeline data
            
        Returns:
            Diversity score (0-1)
        """
        # Extract components
        events = timeline.get("event_record", [])
        conversation_record = timeline.get("conversation_record", [])
        
        # Skip if not enough data
        if not events:
            logger.warning("Not enough data for diversity evaluation")
            return 0.5  # Neutral score
        
        # Calculate diversity metrics
        
        # 1. Event category diversity
        event_categories = []
        for event in events:
            event_data = event.get("event", {})
            category = event_data.get("category", "unknown")
            event_categories.append(category)
        
        unique_categories = set(event_categories)
        category_diversity = len(unique_categories) / min(len(event_categories), 10) if event_categories else 0
        
        # 2. Event attribute diversity
        attribute_values = {}
        for event in events:
            event_data = event.get("event", {})
            attributes = event_data.get("attributes", {})
            
            for attr, value in attributes.items():
                if attr not in attribute_values:
                    attribute_values[attr] = set()
                
                # Convert to string for set operations
                attribute_values[attr].add(str(value))
        
        # Calculate average attribute diversity
        attr_diversity_scores = []
        for attr, values in attribute_values.items():
            # Normalize by the number of events
            attr_diversity = len(values) / min(len(events), 20)
            attr_diversity_scores.append(min(1.0, attr_diversity))
        
        attribute_diversity = statistics.mean(attr_diversity_scores) if attr_diversity_scores else 0.5
        
        # 3. Conversation diversity (if available)
        conversation_diversity = 0.5  # Default
        
        if conversation_record:
            # Analyze conversation lengths
            conv_lengths = [len(conv.get("turns", [])) for conv in conversation_record]
            
            # Coefficient of variation (higher = more diverse)
            mean_length = statistics.mean(conv_lengths) if conv_lengths else 0
            stdev_length = statistics.stdev(conv_lengths) if len(conv_lengths) > 1 else 0
            
            conv_cv = stdev_length / mean_length if mean_length > 0 else 0
            
            # Score higher for moderate variation (not too uniform, not too extreme)
            conversation_diversity = min(1.0, max(0.0, 1.5 * conv_cv))
        
        # Combine diversity scores
        diversity_score = (category_diversity + attribute_diversity + conversation_diversity) / 3
        
        logger.debug(f"Diversity score: {diversity_score:.2f}")
        return diversity_score
    
    def _is_category_relevant_to_domain(self, category: str, domain: str) -> bool:
        """
        Determine if an event category is relevant to a preference domain.
        
        Args:
            category: Event category
            domain: Preference domain
            
        Returns:
            True if relevant, False otherwise
        """
        # Simplified relevance mapping
        relevance_map = {
            "social": ["entertainment", "communication", "lifestyle"],
            "professional": ["productivity", "communication"],
            "recreational": ["entertainment", "lifestyle", "food"],
            "educational": ["productivity", "entertainment"],
            "health": ["food", "lifestyle"],
            "travel": ["lifestyle", "food", "entertainment"]
        }
        
        # Check if domain is in the list for this category
        if category in relevance_map:
            return domain in relevance_map[category]
            
        return False
    
    def _compute_aggregate_metrics(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate metrics from individual timeline results.
        
        Args:
            individual_results: List of individual timeline evaluation results
            
        Returns:
            Dictionary of aggregate metrics
        """
        # Initialize aggregate results
        aggregate_results = {
            "overall_score": 0.0,
            "metrics": {},
            "passed_count": 0,
            "total_count": len(individual_results),
            "pass_rate": 0.0
        }
        
        # Skip if no results
        if not individual_results:
            return aggregate_results
        
        # Count passed timelines
        passed_count = sum(1 for result in individual_results if result.get("passed", False))
        
        # Calculate pass rate
        pass_rate = passed_count / len(individual_results) if individual_results else 0.0
        
        # Aggregate metrics
        all_metrics = {}
        
        for result in individual_results:
            metrics = result.get("metrics", {})
            
            for metric_name, metric_value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                    
                all_metrics[metric_name].append(metric_value)
        
        # Calculate average for each metric
        aggregate_metrics = {}
        
        for metric_name, values in all_metrics.items():
            aggregate_metrics[metric_name] = statistics.mean(values)
        
        # Calculate overall score
        overall_score = statistics.mean([result.get("overall_score", 0.0) for result in individual_results])
        
        # Set aggregate results
        aggregate_results["overall_score"] = overall_score
        aggregate_results["metrics"] = aggregate_metrics
        aggregate_results["passed_count"] = passed_count
        aggregate_results["pass_rate"] = pass_rate
        
        return aggregate_results
    
    def _save_evaluation_results(
        self,
        individual_results: List[Dict[str, Any]],
        aggregate_results: Dict[str, Any]
    ) -> None:
        """
        Save evaluation results to files.
        
        Args:
            individual_results: List of individual timeline evaluation results
            aggregate_results: Dictionary of aggregate metrics
        """
        try:
            # Save individual results
            individual_path = os.path.join(self.output_dir, "evaluation_individual.json")
            with open(individual_path, 'w', encoding='utf-8') as f:
                json.dump(individual_results, f, indent=2)
                
            logger.debug(f"Saved individual evaluation results to {individual_path}")
            
            # Save aggregate results
            aggregate_path = os.path.join(self.output_dir, "evaluation_aggregate.json")
            with open(aggregate_path, 'w', encoding='utf-8') as f:
                json.dump(aggregate_results, f, indent=2)
                
            logger.debug(f"Saved aggregate evaluation results to {aggregate_path}")
            
            # Generate evaluation report
            report = self._generate_evaluation_report(individual_results, aggregate_results)
            
            report_path = os.path.join(self.output_dir, "evaluation_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
            logger.info(f"Saved evaluation report to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    def _generate_evaluation_report(
        self,
        individual_results: List[Dict[str, Any]],
        aggregate_results: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            individual_results: List of individual timeline evaluation results
            aggregate_results: Dictionary of aggregate metrics
            
        Returns:
            Formatted report string
        """
        report = "Causal Preference Evolution Framework Evaluation Report\n"
        report += "=" * 60 + "\n\n"
        
        # Add generation time
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add overall results
        report += "Overall Results:\n"
        report += "-" * 30 + "\n"
        report += f"Overall Score: {aggregate_results['overall_score']:.2f}\n"
        report += f"Pass Rate: {aggregate_results['pass_rate']:.2%} ({aggregate_results['passed_count']}/{aggregate_results['total_count']})\n\n"
        
        # Add metric breakdowns
        report += "Metric Breakdown:\n"
        report += "-" * 30 + "\n"
        
        for metric_name, metric_value in aggregate_results.get("metrics", {}).items():
            report += f"{metric_name.replace('_', ' ').title()}: {metric_value:.2f}\n"
            
        report += "\n"
        
        # Add individual timeline summary
        report += "Individual Timeline Summary:\n"
        report += "-" * 30 + "\n"
        
        for i, result in enumerate(individual_results[:10]):  # Limit to first 10 for brevity
            user_name = result.get("user_name", f"User {i+1}")
            overall_score = result.get("overall_score", 0.0)
            passed = "PASSED" if result.get("passed", False) else "FAILED"
            
            report += f"{user_name}: {overall_score:.2f} - {passed}\n"
            
        if len(individual_results) > 10:
            report += f"... and {len(individual_results) - 10} more\n"
            
        report += "\n"
        
        # Add recommendations based on scores
        report += "Recommendations:\n"
        report += "-" * 30 + "\n"
        
        metrics = aggregate_results.get("metrics", {})
        
        if "temporal_consistency" in metrics and metrics["temporal_consistency"] < 0.7:
            report += "- Improve temporal consistency of events and preferences\n"
            
        if "preference_stability" in metrics and metrics["preference_stability"] < 0.7:
            report += "- Adjust preference evolution parameters to increase stability\n"
            
        if "psychological_plausibility" in metrics and metrics["psychological_plausibility"] < 0.7:
            report += "- Enhance psychological plausibility of preference changes\n"
            
        if "conversation_naturalness" in metrics and metrics["conversation_naturalness"] < 0.7:
            report += "- Improve conversation generation for more natural dialogues\n"
            
        if "event_sequence_plausibility" in metrics and metrics["event_sequence_plausibility"] < 0.7:
            report += "- Enhance event sampling for more realistic sequences\n"
            
        if "diversity" in metrics and metrics["diversity"] < 0.5:
            report += "- Increase diversity in generated data\n"
        
        return report