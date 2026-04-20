"""
preference_sampler.py

Implements preference sampling for the Causal Preference Evolution Framework.
Samples preference values based on user characteristics and event context.
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional, Set, Tuple, Union

from src.causal_framework.models.user_model import UserModel
from src.causal_framework.models.event_model import EventModel
from src.libraries.preference_library import PreferenceLibrary
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class PreferenceSampler:
    """
    Samples preference values based on user characteristics and event context.
    Implements conditional probability for preference attribute instantiation.
    """
    
    def __init__(
        self, 
        preference_library: PreferenceLibrary, 
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the preference sampler.
        
        Args:
            preference_library: Library of preference structures
            llm_client: Client for accessing LLM services
            config: Optional configuration parameters
        """
        self.preference_library = preference_library
        self.llm_client = llm_client
        
        # Set default configuration or update with provided config
        self.config = {
            "demographic_influence_weight": 0.7,  # Weight of demographic influence
            "personality_influence_weight": 0.8,  # Weight of personality influence
            "cache_preference_distributions": True,  # Whether to cache distributions
            "random_seed": 42
        }
        
        if config:
            self.config.update(config)
            
        random.seed(self.config["random_seed"])
        
        # Cache for preference distributions
        self._distribution_cache = {}
        
        logger.info("PreferenceSampler initialized")
    
    def sample_initial_preference(
        self,
        user: UserModel,
        preference_structure: Dict[str, Any],
        existing_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sample initial preference values for a user.
        
        Args:
            user: User model
            preference_structure: Preference structure from library
            existing_preferences: Optional dictionary of existing preference values to maintain consistency with
            
        Returns:
            Dictionary of sampled preference attribute values
        """
        # Extract information about the preference
        pref_id = preference_structure.get("name", "unknown_preference")
        attributes = preference_structure.get("attributes", {})
        attribute_metadata = preference_structure.get("attribute_metadata", {})
        
        # Initialize sampled values with defaults
        sampled_values = {}
        
        # Try to sample using LLM
        llm_values = self._sample_using_llm(user, preference_structure, existing_preferences)
        
        if llm_values:
            # LLM sampling succeeded, use those values
            sampled_values.update(llm_values)
            
            # Fill in any missing attributes with defaults
            for attr_name, default_value in attributes.items():
                if attr_name not in sampled_values:
                    sampled_values[attr_name] = default_value
                    
            logger.debug(f"Sampled initial {pref_id} preference using LLM")
            return sampled_values
        
        # LLM sampling failed, use fallback sampling
        logger.warning(f"LLM sampling failed for {pref_id}, using fallback sampling")
        
        # Sample each attribute
        for attr_name, default_value in attributes.items():
            metadata = attribute_metadata.get(attr_name, {})
            
            # Get possible values for categorical attributes
            possible_values = metadata.get("possible_values")
            
            if possible_values and isinstance(possible_values, list):
                # Sample from possible values
                sampled_values[attr_name] = random.choice(possible_values)
            elif isinstance(default_value, bool):
                # Sample boolean value
                sampled_values[attr_name] = random.choice([True, False])
            elif isinstance(default_value, int):
                # Sample integer value
                min_val = metadata.get("min_value", default_value - 10)
                max_val = metadata.get("max_value", default_value + 10)
                sampled_values[attr_name] = random.randint(min_val, max_val)
            elif isinstance(default_value, float):
                # Sample float value
                min_val = metadata.get("min_value", default_value * 0.5)
                max_val = metadata.get("max_value", default_value * 1.5)
                sampled_values[attr_name] = random.uniform(min_val, max_val)
            elif isinstance(default_value, list):
                # Sample list elements
                if default_value and all(isinstance(x, str) for x in default_value):
                    # List of strings, sample a subset
                    subset_size = random.randint(1, len(default_value))
                    sampled_values[attr_name] = random.sample(default_value, subset_size)
                else:
                    # Unknown list type, use default
                    sampled_values[attr_name] = default_value
            else:
                # Use default value for other types
                sampled_values[attr_name] = default_value
                
        logger.debug(f"Sampled initial {pref_id} preference using fallback method")
        return sampled_values
    
    def sample_event_relevant_preference(
        self,
        user: UserModel,
        event: EventModel,
        preference_structure: Dict[str, Any],
        existing_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sample preference values relevant to an event.
        
        Args:
            user: User model
            event: Event model
            preference_structure: Preference structure from library
            existing_preferences: Optional dictionary of existing preference values
            
        Returns:
            Dictionary of sampled preference attribute values
        """
        # Extract information about the preference
        pref_id = preference_structure.get("name", "unknown_preference")
        attributes = preference_structure.get("attributes", {})
        
        # Initialize with existing values if provided
        sampled_values = existing_preferences.copy() if existing_preferences else {}
        
        # Try to sample using LLM
        llm_values = self._sample_using_llm_for_event(user, event, preference_structure, existing_preferences)
        
        if llm_values:
            # LLM sampling succeeded, use those values
            sampled_values.update(llm_values)
            
            # Fill in any missing attributes with defaults
            for attr_name, default_value in attributes.items():
                if attr_name not in sampled_values:
                    sampled_values[attr_name] = default_value
                    
            logger.debug(f"Sampled event-relevant {pref_id} preference using LLM")
            return sampled_values
        
        # LLM sampling failed, use fallback to initial sampling
        logger.warning(f"LLM sampling failed for event-relevant {pref_id}, using fallback to initial sampling")
        
        # If we have existing preferences, use those
        if existing_preferences:
            return existing_preferences
            
        # Otherwise, sample initial preferences
        return self.sample_initial_preference(user, preference_structure)
    
    def batch_sample_preferences(
        self,
        user: UserModel,
        preference_structures: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Sample multiple preferences in a batch.
        
        Args:
            user: User model
            preference_structures: Dictionary mapping preference IDs to structures
            
        Returns:
            Dictionary mapping preference IDs to sampled attribute values
        """
        sampled_preferences = {}
        
        # Try to batch sample using LLM
        batch_values = self._batch_sample_using_llm(user, preference_structures)
        
        if batch_values:
            # LLM batch sampling succeeded
            sampled_preferences.update(batch_values)
            
            # Check for any missing preferences
            missing_prefs = set(preference_structures.keys()) - set(batch_values.keys())
            
            # Sample missing preferences individually
            for pref_id in missing_prefs:
                sampled_preferences[pref_id] = self.sample_initial_preference(
                    user,
                    preference_structures[pref_id]
                )
                
            logger.debug(f"Batch sampled {len(sampled_preferences)} preferences using LLM")
            return sampled_preferences
        
        # LLM batch sampling failed, sample individually
        logger.warning("LLM batch sampling failed, sampling preferences individually")
        
        for pref_id, pref_structure in preference_structures.items():
            sampled_preferences[pref_id] = self.sample_initial_preference(
                user,
                pref_structure
            )
            
        return sampled_preferences
    
    def sample_related_preferences(
        self,
        base_preference_id: str,
        base_preference_values: Dict[str, Any],
        user: UserModel,
        related_preference_structures: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Sample related preferences based on a base preference.
        
        Args:
            base_preference_id: ID of the base preference
            base_preference_values: Values of the base preference
            user: User model
            related_preference_structures: Dictionary of related preference structures
            
        Returns:
            Dictionary mapping preference IDs to sampled attribute values
        """
        sampled_preferences = {}
        
        # Try to sample related preferences using LLM
        related_values = self._sample_related_using_llm(
            base_preference_id,
            base_preference_values,
            user,
            related_preference_structures
        )
        
        if related_values:
            # LLM sampling succeeded
            sampled_preferences.update(related_values)
            
            # Check for any missing preferences
            missing_prefs = set(related_preference_structures.keys()) - set(related_values.keys())
            
            # Sample missing preferences individually
            for pref_id in missing_prefs:
                sampled_preferences[pref_id] = self.sample_initial_preference(
                    user,
                    related_preference_structures[pref_id]
                )
                
            logger.debug(f"Sampled {len(sampled_preferences)} related preferences using LLM")
            return sampled_preferences
        
        # LLM sampling failed, sample individually
        logger.warning("LLM related preference sampling failed, sampling individually")
        
        for pref_id, pref_structure in related_preference_structures.items():
            sampled_preferences[pref_id] = self.sample_initial_preference(
                user,
                pref_structure
            )
            
        return sampled_preferences
    
    def _sample_using_llm(
        self,
        user: UserModel,
        preference_structure: Dict[str, Any],
        existing_preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Sample preference values using LLM.
        
        Args:
            user: User model
            preference_structure: Preference structure from library
            existing_preferences: Optional dictionary of existing preference values to maintain consistency with
            
        Returns:
            Dictionary of sampled attribute values or None if sampling fails
        """
        # Check if we have a cached distribution
        cache_key = f"{preference_structure.get('name', 'unknown')}-{user.get_life_stage()}-{user.get_attribute('gender')}"
        
        if self.config["cache_preference_distributions"] and cache_key in self._distribution_cache:
            logger.debug(f"Using cached distribution for {cache_key}")
            return self._sample_from_cached_distribution(
                self._distribution_cache[cache_key],
                preference_structure
            )
        
        # Extract information about the preference
        pref_id = preference_structure.get("name", "unknown_preference")
        attributes = preference_structure.get("attributes", {})
        attribute_metadata = preference_structure.get("attribute_metadata", {})
        
        # Prepare user information
        user_info = {
            "age": user.get_attribute("age"),
            "gender": user.get_attribute("gender"),
            "personality_traits": user.get_personality_traits(),
            "interests": user.get_interests(),
            "occupation": user.get_attribute("occupation", "Unknown"),
            "education_level": user.get_attribute("education_level", "Unknown"),
            "life_stage": user.get_life_stage()
        }
        
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following user information, generate plausible preference values for {pref_id}.
        
        User information:
        {json.dumps(user_info, indent=2)}
        
        Preference structure:
        Name: {pref_id}
        Description: {preference_structure.get('description', '')}
        """
        
        # Add existing preferences if available
        if existing_preferences:
            prompt += "\nExisting preference values to maintain consistency with:\n"
            for attr_name, value in existing_preferences.items():
                prompt += f"- {attr_name}: {value}\n"
        
        prompt += "\nGenerate values for each of the following attributes:\n"
        
        # Add attribute descriptions
        for attr_name, default_value in attributes.items():
            metadata = attribute_metadata.get(attr_name, {})
            attr_type = metadata.get("type", type(default_value).__name__)
            description = metadata.get("description", attr_name)
            
            prompt += f"\n- {attr_name} ({attr_type}): {description}"
            
            # Add possible values for categorical attributes
            possible_values = metadata.get("possible_values")
            if possible_values and isinstance(possible_values, list):
                prompt += f"\n  Possible values: {possible_values}"
            elif attr_type == "boolean":
                prompt += "\n  Possible values: [true, false]"
            elif attr_type in ["integer", "int"]:
                min_val = metadata.get("min_value", "any")
                max_val = metadata.get("max_value", "any")
                prompt += f"\n  Range: {min_val} to {max_val}"
            elif attr_type in ["float", "number"]:
                min_val = metadata.get("min_value", "any")
                max_val = metadata.get("max_value", "any")
                prompt += f"\n  Range: {min_val} to {max_val}"
            elif attr_type == "list":
                item_type = metadata.get("item_type", "string")
                prompt += f"\n  List of {item_type}s"
                
        prompt += """
        
        Return a JSON object mapping attribute names to their generated values.
        Make sure the values are appropriate for the attribute types and the user's characteristics.
        If existing preference values are provided, maintain reasonable consistency with those values.
        """
        
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            sampled_values = json.loads(response["response_text"])
            
            # Validate and convert values
            validated_values = {}
            
            for attr_name, expected_value in attributes.items():
                if attr_name not in sampled_values:
                    # Use default value if missing
                    validated_values[attr_name] = expected_value
                    continue
                    
                sampled_value = sampled_values[attr_name]
                
                # Convert to expected type
                if isinstance(expected_value, bool):
                    if isinstance(sampled_value, bool):
                        validated_values[attr_name] = sampled_value
                    elif isinstance(sampled_value, str):
                        validated_values[attr_name] = sampled_value.lower() in ["true", "yes", "1"]
                    else:
                        validated_values[attr_name] = bool(sampled_value)
                elif isinstance(expected_value, int):
                    try:
                        validated_values[attr_name] = int(float(sampled_value))
                    except (ValueError, TypeError):
                        validated_values[attr_name] = expected_value
                elif isinstance(expected_value, float):
                    try:
                        validated_values[attr_name] = float(sampled_value)
                    except (ValueError, TypeError):
                        validated_values[attr_name] = expected_value
                elif isinstance(expected_value, list):
                    if isinstance(sampled_value, list):
                        validated_values[attr_name] = sampled_value
                    elif isinstance(sampled_value, str) and "," in sampled_value:
                        # Try to parse comma-separated string
                        validated_values[attr_name] = [item.strip() for item in sampled_value.split(",")]
                    else:
                        validated_values[attr_name] = [sampled_value]
                else:
                    # String or other type
                    validated_values[attr_name] = str(sampled_value)
            
            # Cache the result if enabled
            if self.config["cache_preference_distributions"]:
                self._distribution_cache[cache_key] = validated_values
                
            return validated_values
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            return None
    
    def _sample_using_llm_for_event(
        self,
        user: UserModel,
        event: EventModel,
        preference_structure: Dict[str, Any],
        existing_preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Sample preference values for an event using LLM.
        
        Args:
            user: User model
            event: Event model
            preference_structure: Preference structure from library
            existing_preferences: Optional dictionary of existing preference values
            
        Returns:
            Dictionary of sampled attribute values or None if sampling fails
        """
        # Extract information about the preference
        pref_id = preference_structure.get("name", "unknown_preference")
        attributes = preference_structure.get("attributes", {})
        attribute_metadata = preference_structure.get("attribute_metadata", {})
        
        # Prepare user information (simplified)
        user_info = {
            "age": user.get_attribute("age"),
            "gender": user.get_attribute("gender"),
            "interests": user.get_interests()[:5] if user.get_interests() else []  # Limit to top 5
        }
        
        # Prepare event information
        event_info = {
            "name": event.get_attribute("name", "event"),
            "category": event.metadata.get("category") if event.metadata else None
        }
        
        # Add key event attributes
        for attr_name in ["location", "participants", "activity", "topic"]:
            if event.has_attribute(attr_name):
                event_info[attr_name] = event.get_attribute(attr_name)
                
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following user and event information, generate plausible preference values for {pref_id}.
        
        User information:
        {json.dumps(user_info, indent=2)}
        
        Event information:
        {json.dumps(event_info, indent=2)}
        
        Preference structure:
        Name: {pref_id}
        Description: {preference_structure.get('description', '')}
        """
        
        # Add existing preferences if available
        if existing_preferences:
            prompt += "\nExisting preference values:\n"
            for attr_name, value in existing_preferences.items():
                prompt += f"- {attr_name}: {value}\n"
                
        prompt += "\nGenerate values for each of the following attributes:\n"
        
        # Add attribute descriptions
        for attr_name, default_value in attributes.items():
            metadata = attribute_metadata.get(attr_name, {})
            attr_type = metadata.get("type", type(default_value).__name__)
            description = metadata.get("description", attr_name)
            
            prompt += f"\n- {attr_name} ({attr_type}): {description}"
            
            # Add existing value if available
            if existing_preferences and attr_name in existing_preferences:
                prompt += f"\n  Current value: {existing_preferences[attr_name]}"
            
            # Add possible values for categorical attributes
            possible_values = metadata.get("possible_values")
            if possible_values and isinstance(possible_values, list):
                prompt += f"\n  Possible values: {possible_values}"
                
        prompt += """
        
        Return a JSON object mapping attribute names to their generated values.
        Make sure the values are appropriate for the attribute types, the user's characteristics, and the event context.
        If existing preference values are provided, maintain reasonable consistency with those values.
        """
        
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            sampled_values = json.loads(response["response_text"])
            
            # Validate and convert values (similar to _sample_using_llm)
            validated_values = {}
            
            for attr_name, expected_value in attributes.items():
                if attr_name not in sampled_values:
                    # Use existing value if available, otherwise default
                    if existing_preferences and attr_name in existing_preferences:
                        validated_values[attr_name] = existing_preferences[attr_name]
                    else:
                        validated_values[attr_name] = expected_value
                    continue
                    
                sampled_value = sampled_values[attr_name]
                
                # Convert to expected type (same as in _sample_using_llm)
                if isinstance(expected_value, bool):
                    if isinstance(sampled_value, bool):
                        validated_values[attr_name] = sampled_value
                    elif isinstance(sampled_value, str):
                        validated_values[attr_name] = sampled_value.lower() in ["true", "yes", "1"]
                    else:
                        validated_values[attr_name] = bool(sampled_value)
                elif isinstance(expected_value, int):
                    try:
                        validated_values[attr_name] = int(float(sampled_value))
                    except (ValueError, TypeError):
                        validated_values[attr_name] = expected_value
                elif isinstance(expected_value, float):
                    try:
                        validated_values[attr_name] = float(sampled_value)
                    except (ValueError, TypeError):
                        validated_values[attr_name] = expected_value
                elif isinstance(expected_value, list):
                    if isinstance(sampled_value, list):
                        validated_values[attr_name] = sampled_value
                    elif isinstance(sampled_value, str) and "," in sampled_value:
                        # Try to parse comma-separated string
                        validated_values[attr_name] = [item.strip() for item in sampled_value.split(",")]
                    else:
                        validated_values[attr_name] = [sampled_value]
                else:
                    # String or other type
                    validated_values[attr_name] = str(sampled_value)
                
            return validated_values
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            return None
    
    def _batch_sample_using_llm(
        self,
        user: UserModel,
        preference_structures: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Sample multiple preferences in a batch using LLM.
        
        Args:
            user: User model
            preference_structures: Dictionary mapping preference IDs to structures
            
        Returns:
            Dictionary mapping preference IDs to sampled attribute values
            or None if sampling fails
        """
        # If too many preferences, batch in smaller groups
        if len(preference_structures) > 5:
            # Process in batches of 5
            all_results = {}
            for i in range(0, len(preference_structures), 30):
                batch = {k: preference_structures[k] for k in list(preference_structures.keys())[i:i+30]}
                batch_results = self._batch_sample_using_llm(user, batch)
                if batch_results:
                    all_results.update(batch_results)
            
            # If we got results for all preferences, return them
            if len(all_results) == len(preference_structures):
                return all_results
            elif all_results:
                # We got partial results, better than nothing
                return all_results
            else:
                # No results at all
                return None
        
        # Prepare user information
        user_info = {
            "age": user.get_attribute("age"),
            "gender": user.get_attribute("gender"),
            "personality_traits": user.get_personality_traits(),
            "interests": user.get_interests(),
            "occupation": user.get_attribute("occupation", "Unknown"),
            "education_level": user.get_attribute("education_level", "Unknown"),
            "life_stage": user.get_life_stage()
        }
        
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following user information, generate plausible preference values for multiple preference structures.
        
        User information:
        {json.dumps(user_info, indent=2)}
        
        Generate values for each attribute in the following preference structures:
        """
        
        # Add preference structures
        for pref_id, pref_structure in preference_structures.items():
            prompt += f"\n\n{pref_id}:"
            prompt += f"\nDescription: {pref_structure.get('description', '')}"
            prompt += "\nAttributes:"
            
            attributes = pref_structure.get("attributes", {})
            attribute_metadata = pref_structure.get("attribute_metadata", {})
            
            for attr_name, default_value in attributes.items():
                metadata = attribute_metadata.get(attr_name, {})
                attr_type = metadata.get("type", type(default_value).__name__)
                description = metadata.get("description", attr_name)
                
                prompt += f"\n- {attr_name} ({attr_type}): {description}"
                
                # Add possible values for categorical attributes
                possible_values = metadata.get("possible_values")
                if possible_values and isinstance(possible_values, list):
                    prompt += f"\n  Possible values: {possible_values}"
                
        prompt += """
        
        Return a JSON object with preference IDs as keys, each mapping to an object of attribute values.
        
        CRITICAL REQUIREMENTS:
        1. Make sure the values are appropriate for the attribute types and the user's characteristics
        2. Ensure STRONG CONSISTENCY between all preferences - they should tell a coherent story about this user
        3. Consider psychological coherence - preferences should reflect a unified personality and value system
        4. Avoid contradictions between preferences in different domains
        5. All preferences should be plausible for the same individual
        
        Example format:
        {{
          "preference_id1": {{
            "attribute1": value1,
            "attribute2": value2,
            ...
          }},
          "preference_id2": {{
            ...
          }}
        }}
        """
        
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            batch_results = json.loads(response["response_text"])
            
            # Validate results structure
            if not isinstance(batch_results, dict):
                logger.error("LLM response is not a dictionary")
                return None
                
            # Validate and process each preference
            validated_results = {}
            
            for pref_id, sampled_values in batch_results.items():
                if pref_id not in preference_structures:
                    logger.warning(f"LLM returned unknown preference ID: {pref_id}")
                    continue
                    
                if not isinstance(sampled_values, dict):
                    logger.warning(f"LLM returned non-dictionary values for {pref_id}")
                    continue
                    
                # Get expected attributes and types
                pref_structure = preference_structures[pref_id]
                attributes = pref_structure.get("attributes", {})
                
                # Validate and convert values
                validated_values = {}
                
                for attr_name, expected_value in attributes.items():
                    if attr_name not in sampled_values:
                        # Use default value if missing
                        validated_values[attr_name] = expected_value
                        continue
                        
                    sampled_value = sampled_values[attr_name]
                    
                    # Convert to expected type (same as in _sample_using_llm)
                    if isinstance(expected_value, bool):
                        if isinstance(sampled_value, bool):
                            validated_values[attr_name] = sampled_value
                        elif isinstance(sampled_value, str):
                            validated_values[attr_name] = sampled_value.lower() in ["true", "yes", "1"]
                        else:
                            validated_values[attr_name] = bool(sampled_value)
                    elif isinstance(expected_value, int):
                        try:
                            validated_values[attr_name] = int(float(sampled_value))
                        except (ValueError, TypeError):
                            validated_values[attr_name] = expected_value
                    elif isinstance(expected_value, float):
                        try:
                            validated_values[attr_name] = float(sampled_value)
                        except (ValueError, TypeError):
                            validated_values[attr_name] = expected_value
                    elif isinstance(expected_value, list):
                        if isinstance(sampled_value, list):
                            validated_values[attr_name] = sampled_value
                        elif isinstance(sampled_value, str) and "," in sampled_value:
                            # Try to parse comma-separated string
                            validated_values[attr_name] = [item.strip() for item in sampled_value.split(",")]
                        else:
                            validated_values[attr_name] = [sampled_value]
                    else:
                        # String or other type
                        validated_values[attr_name] = str(sampled_value)
                
                validated_results[pref_id] = validated_values
            
            return validated_results
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            return None
    
    def _sample_related_using_llm(
        self,
        base_preference_id: str,
        base_preference_values: Dict[str, Any],
        user: UserModel,
        related_preference_structures: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Sample related preferences based on a base preference using LLM.
        
        Args:
            base_preference_id: ID of the base preference
            base_preference_values: Values of the base preference
            user: User model
            related_preference_structures: Dictionary of related preference structures
            
        Returns:
            Dictionary mapping preference IDs to sampled attribute values
            or None if sampling fails
        """
        # Prepare user information (simplified)
        user_info = {
            "age": user.get_attribute("age"),
            "gender": user.get_attribute("gender")
        }
        
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following user information and base preference values, 
        generate plausible values for related preference structures.
        
        User information:
        {json.dumps(user_info, indent=2)}
        
        Base preference ({base_preference_id}):
        {json.dumps(base_preference_values, indent=2)}
        
        Generate values for each attribute in the following related preference structures:
        """
        
        # Add related preference structures
        for pref_id, pref_structure in related_preference_structures.items():
            prompt += f"\n\n{pref_id}:"
            prompt += f"\nDescription: {pref_structure.get('description', '')}"
            prompt += "\nAttributes:"
            
            attributes = pref_structure.get("attributes", {})
            attribute_metadata = pref_structure.get("attribute_metadata", {})
            
            for attr_name, default_value in attributes.items():
                metadata = attribute_metadata.get(attr_name, {})
                attr_type = metadata.get("type", type(default_value).__name__)
                description = metadata.get("description", attr_name)
                
                prompt += f"\n- {attr_name} ({attr_type}): {description}"
                
                # Add possible values for categorical attributes
                possible_values = metadata.get("possible_values")
                if possible_values and isinstance(possible_values, list):
                    prompt += f"\n  Possible values: {possible_values}"
                
        prompt += f"""
        
        The values for the related preferences should be consistent with the base preference ({base_preference_id}).
        Consider how preferences in different domains might correlate or influence each other.
        
        Return a JSON object with preference IDs as keys, each mapping to an object of attribute values.
        Make sure the values are appropriate for the attribute types and consistent with the base preference.
        
        Example format:
        {{
          "preference_id1": {{
            "attribute1": value1,
            "attribute2": value2,
            ...
          }},
          "preference_id2": {{
            ...
          }}
        }}
        """
        
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            related_results = json.loads(response["response_text"])
            
            # Validate results structure
            if not isinstance(related_results, dict):
                logger.error("LLM response is not a dictionary")
                return None
                
            # Validate and process each preference (similar to _batch_sample_using_llm)
            validated_results = {}
            
            for pref_id, sampled_values in related_results.items():
                if pref_id not in related_preference_structures:
                    logger.warning(f"LLM returned unknown preference ID: {pref_id}")
                    continue
                    
                if not isinstance(sampled_values, dict):
                    logger.warning(f"LLM returned non-dictionary values for {pref_id}")
                    continue
                    
                # Get expected attributes and types
                pref_structure = related_preference_structures[pref_id]
                attributes = pref_structure.get("attributes", {})
                
                # Validate and convert values (same as in _batch_sample_using_llm)
                validated_values = {}
                
                for attr_name, expected_value in attributes.items():
                    if attr_name not in sampled_values:
                        # Use default value if missing
                        validated_values[attr_name] = expected_value
                        continue
                        
                    sampled_value = sampled_values[attr_name]
                    
                    # Convert to expected type
                    if isinstance(expected_value, bool):
                        if isinstance(sampled_value, bool):
                            validated_values[attr_name] = sampled_value
                        elif isinstance(sampled_value, str):
                            validated_values[attr_name] = sampled_value.lower() in ["true", "yes", "1"]
                        else:
                            validated_values[attr_name] = bool(sampled_value)
                    elif isinstance(expected_value, int):
                        try:
                            validated_values[attr_name] = int(float(sampled_value))
                        except (ValueError, TypeError):
                            validated_values[attr_name] = expected_value
                    elif isinstance(expected_value, float):
                        try:
                            validated_values[attr_name] = float(sampled_value)
                        except (ValueError, TypeError):
                            validated_values[attr_name] = expected_value
                    elif isinstance(expected_value, list):
                        if isinstance(sampled_value, list):
                            validated_values[attr_name] = sampled_value
                        elif isinstance(sampled_value, str) and "," in sampled_value:
                            # Try to parse comma-separated string
                            validated_values[attr_name] = [item.strip() for item in sampled_value.split(",")]
                        else:
                            validated_values[attr_name] = [sampled_value]
                    else:
                        # String or other type
                        validated_values[attr_name] = str(sampled_value)
                
                validated_results[pref_id] = validated_values
            
            return validated_results
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            return None
    
    def _sample_from_cached_distribution(
        self,
        distribution: Dict[str, Any],
        preference_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sample preference values from a cached distribution.
        
        Args:
            distribution: Cached distribution of attribute values
            preference_structure: Preference structure from library
            
        Returns:
            Dictionary of sampled preference attribute values
        """
        # Get expected attributes
        attributes = preference_structure.get("attributes", {})
        
        # Copy distribution values for attributes that exist in the structure
        sampled_values = {}
        
        for attr_name, expected_value in attributes.items():
            if attr_name in distribution:
                sampled_values[attr_name] = distribution[attr_name]
            else:
                # Use default value for missing attributes
                sampled_values[attr_name] = expected_value
                
        return sampled_values