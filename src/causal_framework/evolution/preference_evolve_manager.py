"""
preference_evolve_manager.py

Manages preference evolution for the Causal Preference Evolution Framework.
Orchestrates stability bias, contextual adaptation, experience-driven change,
and relational interdependence.
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime

from src.causal_framework.models.user_model import UserModel
from src.causal_framework.models.event_model import EventModel
from src.causal_framework.models.preference_model import PreferenceModel
from src.causal_framework.evolution.stability_processor import StabilityProcessor
from src.causal_framework.evolution.context_adapter import ContextAdapter
from src.causal_framework.evolution.experience_tracker import ExperienceTracker
from src.causal_framework.evolution.relation_propagator import RelationPropagator
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class PreferenceEvolveManager:
    """
    Orchestrates preference evolution based on psychological principles.
    Manages the four key mechanisms for preference evolution: stability bias,
    contextual adaptation, experience-driven change, and relational interdependence.
    """
    
    def __init__(
        self, 
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the preference evolution manager.
        
        Args:
            llm_client: Client for accessing LLM services
            config: Optional configuration parameters
        """
        self.llm_client = llm_client
        
        # Set default configuration or update with provided config
        self.config = {
            "stability": {
                "default_stability_weight": 0.7,
                "max_change_magnitude": 0.3,
                "stability_variance": 0.2,
                "random_seed": 42
            },
            "context": {
                "context_categories": [
                    "professional", "personal", "social", "recreational", "family", "health"
                ],
                "default_context_influence": 0.5,
                "random_seed": 42
            },
            "experience": {
                "first_experience_weight": 0.8,
                "experience_decay_rate": 0.7,
                "recency_half_life_days": 30,
                "max_tracked_experiences": 20,
                "positive_impact_multiplier": 1.2,
                "negative_impact_multiplier": 1.5,
                "random_seed": 42
            },
            "relation": {
                "propagation_depth": 2,
                "min_strength_threshold": 0.2,
                "propagation_decay": 0.7,
                "max_propagation_changes": 5,
                "random_seed": 42
            },
            "interdependence_threshold": 0.5,
            "random_seed": 42
        }
        
        if config:
            self.config.update(config)
            
        random.seed(self.config["random_seed"])
        
        # Initialize specialized components
        self.stability_processor = StabilityProcessor(self.config["stability"])
        self.context_adapter = ContextAdapter(llm_client=llm_client, config=self.config["context"])
        self.experience_tracker = ExperienceTracker(self.config["experience"])
        self.relation_propagator = RelationPropagator(self.config["relation"])
        
        # Cache for preference dependencies
        self._preference_dependencies = {}
        
        logger.info("PreferenceEvolveManager initialized")
    
    def evolve_preference(
        self,
        preference: PreferenceModel,
        user: UserModel,
        event: EventModel,
        relevance_score: float,
        preference_history: Optional[List[PreferenceModel]] = None,
        event_context: Optional[str] = None
    ) -> PreferenceModel:
        """
        Evolve a preference based on an event, user characteristics, and preference history.
        
        Args:
            preference: Current preference state
            user: User model
            event: Current event triggering the evolution
            relevance_score: How relevant this preference is to the event (0-1)
            preference_history: Optional history of this preference over time
            event_context: Optional context of the event (e.g., professional, personal)
            
        Returns:
            Updated preference model
        """
        # Skip evolution if relevance is too low
        if relevance_score < 0.2:
            return preference
            
        # Clone the preference to avoid modifying the original
        updated_preference = preference.clone()
        
        # Detect event context if not provided
        context = event_context
        if context is None:
            # Extract from event category or metadata
            event_metadata = event.metadata or {}
            if "category" in event_metadata:
                context = event_metadata["category"]
            elif event.has_attribute("category"):
                context = event.get_attribute("category")
            else:
                context = "personal"  # Default context
        
        # 1. Apply contextual adaptation if appropriate
        user_attributes = user.get_all_attributes()
        logger.debug(f"base_preference: {preference.get_all_attributes()}")
        try:
            adapted_attributes = self.context_adapter.adapt_to_context(
                base_preference=preference.get_all_attributes(),
                event=event.to_dict(),
                agent=user.to_dict(),
                user_attributes=user_attributes
            )
        except Exception as e:
            logger.warning(f"Failed to adapt preference to context: {e}")
            # Use base preference attributes as fallback
            adapted_attributes = preference.get_all_attributes()
        
        # 2. Calculate experience factor
        preference_id = preference.structure_id
        logger.debug(f"preference_id: {preference_id}")
        
        # Record this experience if it doesn't exist yet
        if not self.experience_tracker.get_experience_count(preference_id):
            try:
                event_data = event.to_dict()
                affected_attributes = list(preference.get_all_attributes().keys())
                
                self.experience_tracker.record_experience(
                    preference_id=preference_id,
                    event_data=event_data,
                    experience_date=event.date,
                    affected_attributes=affected_attributes
                )
            except Exception as e:
                logger.warning(f"Failed to record experience: {e}")
                # Continue without recording experience
            
        try:
            # Get experience factor for preference evolution
            experience_factor = self.experience_tracker.calculate_experience_factor(
                preference_id=preference_id,
                current_date=event.date
            )
        except Exception as e:
            logger.warning(f"Failed to calculate experience factor: {e}")
            # Use default experience factor
            experience_factor = 0.1
        
        try:
            # Get attribute-specific experience factors
            attribute_experience_factors = self.experience_tracker.get_attribute_experience_factors(
                preference_id=preference_id,
                current_date=event.date
            )
        except Exception as e:
            logger.warning(f"Failed to get attribute experience factors: {e}")
            # Use default attribute experience factors
            attribute_experience_factors = {}

        # 3. For each attribute, determine if and how it should evolve
        for attr_name, current_value in preference.get_all_attributes().items():
            # Get adapted value as target
            target_value = adapted_attributes.get(attr_name, current_value)
            
            # Skip if values are identical
            if target_value == current_value:
                continue
                
            # Get attribute-specific experience factor
            attr_experience_factor = attribute_experience_factors.get(attr_name, experience_factor)
            
            # Get stability weight for this attribute
            stability_weight = self._get_attribute_stability_weight(
                preference=preference,
                attribute_name=attr_name,
                user_attributes=user_attributes
            )
            
            # Adjust stability weight based on relevance and experience
            adjusted_stability = stability_weight * (1 - relevance_score * attr_experience_factor)
            
            # Apply stability bias
            new_value = self.stability_processor.apply_stability_bias(
                current_value=current_value,
                target_value=target_value,
                stability_weight=adjusted_stability
            )
            
            # Apply change if different
            if new_value != current_value:
                updated_preference.set_attribute(attr_name, new_value)
                logger.debug(
                    f"Evolved preference {preference.structure_id} attribute {attr_name} "
                    f"from {current_value} to {new_value}"
                )
        
        # Record this experience
        self._record_evolution_experience(
            preference=preference,
            updated_preference=updated_preference,
            event=event
        )
        
        return updated_preference
    
    def apply_interdependence(
        self,
        updated_preferences: Dict[str, PreferenceModel],
        original_preferences: Dict[str, PreferenceModel],
        relevant_preferences: Dict[str, float]
    ) -> Dict[str, PreferenceModel]:
        """
        Apply interdependence between preference attributes - changes in one
        preference can affect related preferences.
        
        Args:
            updated_preferences: Preferences after primary evolution
            original_preferences: Original preferences before evolution
            relevant_preferences: Map of preference IDs to relevance scores
            
        Returns:
            Preferences after applying interdependence effects
        """
        # Ensure we have dependency graph
        preference_ids = list(updated_preferences.keys())
        if not self._preference_dependencies:
            self._generate_preference_dependencies(preference_ids)
            
        # Convert preferences to the format expected by relation_propagator
        source_changes = {}
        for pref_id, updated_pref in updated_preferences.items():
            if pref_id in original_preferences:
                original_pref = original_preferences[pref_id]
                changes = {}
                
                # Detect changes
                for attr_name, updated_value in updated_pref.get_all_attributes().items():
                    original_value = original_pref.get_attribute(attr_name)
                    if original_value != updated_value:
                        changes[attr_name] = {"from": original_value, "to": updated_value}
                        
                if changes:
                    source_changes[pref_id] = changes
        
        # If no changes, return updated preferences as is
        if not source_changes:
            return updated_preferences
            
        # Convert preferences to the format expected by relation_propagator
        all_preferences = {}
        for pref_id, pref in original_preferences.items():
            all_preferences[pref_id] = pref.get_all_attributes()
            
        # Calculate propagated changes
        propagated_changes = self.relation_propagator.propagate_changes(
            dependency_graph=self._preference_dependencies,
            source_changes=source_changes,
            all_preferences=all_preferences
        )
        
        # Apply propagated changes to result preferences
        result_preferences = {k: v.clone() for k, v in updated_preferences.items()}
        
        for pref_id, attr_changes in propagated_changes.items():
            # Skip if preference isn't in our set
            if pref_id not in result_preferences:
                continue
                
            # Get preference model
            pref_model = result_preferences[pref_id]
            
            # Apply each attribute change
            for attr_name, change in attr_changes.items():
                if pref_model.has_attribute(attr_name):
                    pref_model.set_attribute(attr_name, change["to"])
                    logger.debug(
                        f"Applied interdependence: {pref_id}.{attr_name} changed "
                        f"from {change['from']} to {change['to']}"
                    )
                    
        return result_preferences
    
    def _get_attribute_stability_weight(
        self,
        preference: PreferenceModel,
        attribute_name: str,
        user_attributes: Dict[str, Any]
    ) -> float:
        """
        Get stability weight for a preference attribute.
        
        Args:
            preference: Preference model
            attribute_name: Name of the attribute
            user_attributes: User attributes for personalized stability
            
        Returns:
            Stability weight (0-1)
        """
        # Get metadata for this attribute if available
        preference_id = preference.structure_id
        attribute_metadata = self._get_attribute_metadata(preference_id, attribute_name)
        
        # Use explicitly defined stability if available
        if attribute_metadata and "stability" in attribute_metadata:
            base_stability = attribute_metadata["stability"]
        else:
            base_stability = self.config["stability"]["default_stability_weight"]
            
        # Calculate appropriate stability weight
        return self.stability_processor.calculate_stability_weight(
            base_stability=base_stability,
            attribute_name=attribute_name,
            user_attributes=user_attributes
        )
    
    def _get_attribute_metadata(
        self,
        preference_id: str,
        attribute_name: str
    ) -> Dict[str, Any]:
        """
        Get metadata for a preference attribute using LLM if needed.
        
        Args:
            preference_id: ID of the preference
            attribute_name: Name of the attribute
            
        Returns:
            Attribute metadata dictionary
        """
        # Use LLM to generate attribute metadata
        # This would normally be stored in the preference library, but we
        # generate it here as a fallback
        
        # Create a simple prompt for the LLM
        prompt = f"""
        Generate appropriate metadata for the preference attribute:
        
        Preference: {preference_id}
        Attribute: {attribute_name}
        
        The metadata should include:
        - description: Brief description of what this attribute represents
        - type: Data type (string, integer, float, boolean, list)
        - stability: How resistant this attribute is to change (0-1 scale)
        - context_influence: How much context affects this attribute (0-1 scale)
        
        Return a JSON object with the metadata.
        """
        
        response = self.llm_client.generate(prompt, response_format="json")
        metadata = json.loads(response["response_text"])
        return metadata
    
    def _generate_preference_dependencies(self, preference_ids: List[str]) -> None:
        """
        Generate dependency graph for preferences using LLM.
        
        Args:
            preference_ids: List of preference IDs to include in the graph
        """
        prompt = f"""
        For the following preference structures, create a dependency graph that indicates how changes in
        one preference are likely to affect others. Assign a dependency strength between 0.0 (no effect)
        and 1.0 (strong effect) for each directed relationship.
        
        Preference structures: {", ".join(preference_ids)}
        
        Return a JSON object mapping source preference IDs to objects containing target preference IDs
        and their dependency strengths. Only include relationships with strength > 0.2:
        
        Example:
        {{
          "music_preference": {{
            "fashion_preference": 0.7,
            "social_activity_preference": 0.4
          }},
          "food_preference": {{
            "health_preference": 0.8
          }}
        }}
        """

        response = self.llm_client.generate(prompt, response_format="json")
        dependencies = json.loads(response["response_text"])

        valid_dependencies = {}
        for source_id, targets in dependencies.items():
            if source_id in preference_ids:
                valid_targets = {}
                for target_id, strength in targets.items():
                    if target_id in preference_ids and isinstance(strength, (int, float)) and 0 <= strength <= 1:
                        valid_targets[target_id] = strength
                if valid_targets:
                    valid_dependencies[source_id] = valid_targets

        self._preference_dependencies = valid_dependencies
        logger.debug(f"Generated dependency graph with {len(valid_dependencies)} source preferences")
    
    def _record_evolution_experience(
        self,
        preference: PreferenceModel,
        updated_preference: PreferenceModel,
        event: EventModel
    ) -> None:
        """
        Record an evolution experience for future reference.
        
        Args:
            preference: Original preference
            updated_preference: Updated preference
            event: Event that triggered the evolution
        """
        # Check if there were any changes
        differences = preference.compare(updated_preference)
        if not differences:
            return
            
        # Extract changed attributes
        changed_attributes = list(differences.keys())
        
        # Record experience
        preference_id = preference.structure_id
        event_data = event.to_dict()
        
        # Calculate impact based on changes
        impact_sum = 0.0
        for attr in changed_attributes:
            diff = differences[attr]
            if diff["status"] == "different":
                # For numeric values, calculate relative change
                if isinstance(diff["self"], (int, float)) and isinstance(diff["other"], (int, float)):
                    base = max(1.0, abs(diff["self"]))  # Avoid division by zero
                    change = (diff["other"] - diff["self"]) / base
                    impact_sum += change
                else:
                    # For non-numeric values, assume moderate impact
                    impact_sum += 0.5 if diff["other"] is not None else -0.5
                    
        # Average impact across all changed attributes
        avg_impact = impact_sum / len(changed_attributes) if changed_attributes else 0.0
        
        # Ensure impact is in valid range
        impact = max(-1.0, min(1.0, avg_impact))
        
        # Record experience
        self.experience_tracker.record_experience(
            preference_id=preference_id,
            event_data=event_data,
            experience_date=event.date,
            experience_impact=impact,
            affected_attributes=changed_attributes
        )
    
    def get_preference_history(
        self, 
        preference_id: str,
        attribute_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical experience data for a preference.
        
        Args:
            preference_id: ID of the preference
            attribute_name: Optional attribute name to filter by
            
        Returns:
            List of experience records
        """
        return self.experience_tracker.get_experience_impact_history(
            preference_id=preference_id,
            attribute_name=attribute_name
        )
    
    def clear_evolution_history(self, preference_id: Optional[str] = None) -> None:
        """
        Clear evolution history data.
        
        Args:
            preference_id: Optional preference ID to clear only specific history
        """
        self.experience_tracker.clear_experiences(preference_id)
        logger.debug(f"Cleared evolution history for {'all preferences' if preference_id is None else preference_id}")