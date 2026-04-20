"""
context_adapter.py

Implements dynamic contextual adaptation for the Causal Preference Evolution Framework.
Uses LLM-based reasoning to handle how preferences manifest differently across contexts
while maintaining underlying coherence.
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class ContextAdapter:
    """
    Handles dynamic contextual adaptation in preference evolution, using LLM reasoning
    to ensure preferences manifest appropriately across different contexts.
    """
    
    def __init__(self, llm_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the context adapter.
        
        Args:
            llm_client: LLM client for dynamic context reasoning
            config: Optional configuration parameters
        """
        self.llm_client = llm_client
        
        # Set default configuration or update with provided config
        self.config = {
            "default_context_influence": 0.5,  # How strongly context influences preference expression
            "max_contexts_to_consider": 5,      # Maximum number of contexts to analyze
            "context_adaptation_threshold": 0.1, # Minimum change needed to apply adaptation
            "random_seed": 42                   # Seed for reproducibility
        }
        
        if config:
            self.config.update(config)
            
        random.seed(self.config["random_seed"])
        
        # Cache for context analysis and modifiers
        self._context_cache = {}
        self._modifier_cache = {}
        
        logger.debug("ContextAdapter initialized with dynamic LLM-based reasoning")
    
    def adapt_to_context(
        self,
        base_preference: Dict[str, Any],
        event: Dict[str, Any],
        agent: Dict[str, Any],
        other_agent: Optional[Dict[str, Any]] = None,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dynamically adapt a preference to the specific context of an event and agents.
        
        Args:
            base_preference: Base preference attributes
            event: Event information that provides context
            agent: Primary agent (whose preference is being adapted)
            other_agent: Optional other agent in the interaction
            user_attributes: Optional additional user attributes for personalized adaptation
            
        Returns:
            Preference attributes adapted to the context
        """
        # First, identify relevant contexts for this specific situation
        relevant_contexts = self._identify_relevant_contexts(
            event=event,
            agent=agent,
            preference=base_preference,
            other_agent=other_agent
        )
        
        if not relevant_contexts:
            logger.debug("No relevant contexts identified, returning base preference")
            return base_preference.copy()
        
        # Get context modifiers using LLM reasoning
        context_modifiers = self._generate_context_modifiers(
            base_preference=base_preference,
            relevant_contexts=relevant_contexts,
            event=event,
            agent=agent,
            other_agent=other_agent,
            user_attributes=user_attributes
        )
        
        # Apply modifiers to base preference
        adapted_preference = self._apply_modifiers(base_preference, context_modifiers)
        
        logger.debug(f"Adapted preference for contexts: {relevant_contexts}")
        return adapted_preference
    
    def _identify_relevant_contexts(
        self,
        event: Dict[str, Any],
        agent: Dict[str, Any],
        preference: Dict[str, Any],
        other_agent: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Use LLM to dynamically identify relevant contexts for this specific situation.
        
        Args:
            event: Event information
            agent: Primary agent
            other_agent: Optional other agent
            preference: Preference being adapted
            
        Returns:
            List of relevant context categories
        """
        # Create cache key
        event_id = event.get('id', '')
        agent_id = agent.get('id', '')
        cache_key = f"contexts_{hash(str(event_id))}_{hash(str(agent_id))}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        # Prepare event and agent information
        event_info = {
            "name": event.get("name", ""),
            "category": event.get("category", ""),
            "description": event.get("metadata", {}).get("description", ""),
            "attributes": event.get("attributes", {}),
            "metadata": event.get("metadata", {})
        }
        
        agent_info = {
            "role": agent.get("role", ""),
            "attributes": agent.get("attributes", {}),
            "personality": agent.get("personality_traits", [])
        }
        
        other_agent_info = {}
        if other_agent:
            other_agent_info = {
                "role": other_agent.get("role", ""),
                "attributes": other_agent.get("attributes", {}),
                "personality": other_agent.get("personality_traits", [])
            }
        
        preference_info = {
            "domain": preference.get("domain", ""),
            "name": preference.get("name", ""),
            "attributes": list(preference.get("attributes", {}).keys())
        }
        
        prompt = f"""
        Analyze this situation and identify the most relevant contextual factors that would influence how preferences are expressed or modified.

        EVENT CONTEXT:
        {json.dumps(event_info, indent=2)}

        PRIMARY AGENT:
        {json.dumps(agent_info, indent=2)}

        OTHER AGENT (if applicable):
        {json.dumps(other_agent_info, indent=2)}

        PREFERENCE BEING CONSIDERED:
        {json.dumps(preference_info, indent=2)}

        Based on this specific situation, identify 3-5 contextual factors that would most influence how this preference manifests. Consider factors like:
        - Social dynamics (formal vs informal, public vs private)
        - Relationship context (professional, personal, romantic, family)
        - Environmental factors (setting, time, circumstances)
        - Cultural/social expectations
        - Emotional state or intensity
        - Power dynamics or hierarchy
        - Task/goal orientation

        Respond with a JSON list of specific, relevant context categories for this situation:
        {{
            "contextual_factors": ["context1", "context2", "context3"]
        }}

        Be creative and specific to this exact situation rather than using generic categories.
        """
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            contexts = json.loads(response["response_text"])
            
            # Validate response
            if isinstance(contexts, dict) and "contextual_factors" in contexts and all(isinstance(c, str) for c in contexts["contextual_factors"]):
                # Limit to max contexts
                relevant_contexts = contexts["contextual_factors"][:self.config["max_contexts_to_consider"]]
                self._context_cache[cache_key] = relevant_contexts
                return relevant_contexts
            else:
                logger.warning(f"Invalid context response format: {response}")
                return []
                
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error identifying contexts: {e}")
            return []
    
    def _generate_context_modifiers(
        self,
        base_preference: Dict[str, Any],
        relevant_contexts: List[str],
        event: Dict[str, Any],
        agent: Dict[str, Any],
        other_agent: Optional[Dict[str, Any]] = None,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to generate appropriate context modifiers for the preference.
        
        Args:
            base_preference: Base preference to modify
            relevant_contexts: Identified relevant contexts
            event: Event information
            agent: Primary agent
            other_agent: Optional other agent
            user_attributes: Optional user attributes
            
        Returns:
            Dictionary of context modifiers
        """
        # Create cache key
        cache_key = f"modifiers_{hash(str(base_preference))}_{hash(str(relevant_contexts))}"
        if cache_key in self._modifier_cache:
            return self._modifier_cache[cache_key]
        
        # Prepare information for LLM
        preference_details = {
            "name": base_preference.get("name", ""),
            "domain": base_preference.get("domain", ""),
            "description": base_preference.get("description", ""),
            "attributes": base_preference.get("attributes", {}),
            "attribute_metadata": base_preference.get("attribute_metadata", {})
        }
        
        event_summary = {
            "type": event.get("name", ""),
            "category": event.get("category", ""),
            "setting": event.get("attributes", {}).get("setting", ""),
            "formality": event.get("attributes", {}).get("formality_level", "")
        }
        
        agent_summary = {
            "role": agent.get("role", ""),
            "personality": agent.get("attributes", {}).get("personality_traits", []),
            "age": agent.get("attributes", {}).get("age", ""),
            "background": agent.get("attributes", {}).get("background", "")
        }
        
        prompt = f"""
        Given this preference and the identified contextual factors, determine how the preference attributes should be modified to appropriately reflect the context.

        PREFERENCE TO ADAPT:
        {json.dumps(preference_details, indent=2)}

        RELEVANT CONTEXTUAL FACTORS:
        {json.dumps(relevant_contexts, indent=2)}

        EVENT CONTEXT:
        {json.dumps(event_summary, indent=2)}

        AGENT CONTEXT:
        {json.dumps(agent_summary, indent=2)}

        For each attribute in the preference, determine if and how it should be modified given these contexts. Consider:
        - How would this context make someone express this preference differently?
        - What social expectations or norms apply in this context?
        - How would the agent's role/personality interact with this context?
        - What would be appropriate vs inappropriate in this specific situation?

        Respond with JSON containing modifiers for each attribute that should change:
        {{
            "attribute_name": {{
                "type": "numeric|boolean|list|explicit",
                "modifier": <modification_value>,
                "reasoning": "explanation of why this change makes sense"
            }},
            ...
        }}

        For numeric attributes: modifier is a relative change (-1.0 to 1.0, where 0.2 = 20% increase)
        For boolean attributes: modifier is probability of flipping (0.0 to 1.0)
        For list attributes: modifier is {{"add": [...], "remove": [...], "prioritize": [...]}}
        For explicit: modifier is the new value to use

        Only include attributes that should actually change - if no change is needed, omit the attribute.
        """
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            modifiers = json.loads(response["response_text"])
            
            # Validate and cache
            if isinstance(modifiers, dict):
                self._modifier_cache[cache_key] = modifiers
                return modifiers
            else:
                logger.warning(f"Invalid modifier response format: {response}")
                return {}
                
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error generating context modifiers: {e}")
            return {}
    
    def _apply_modifiers(
        self,
        base_preference: Dict[str, Any],
        context_modifiers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply the generated context modifiers to the base preference.
        
        Args:
            base_preference: Base preference attributes
            context_modifiers: Generated context modifiers
            
        Returns:
            Preference with context modifications applied
        """
        adapted_preference = base_preference.copy()
        base_attributes = adapted_preference.get("attributes", {}).copy()
        
        for attr_name, modifier_info in context_modifiers.items():
            if attr_name not in base_attributes:
                continue
                
            attr_value = base_attributes[attr_name]
            modifier_type = modifier_info.get("type", "")
            modifier_value = modifier_info.get("modifier")
            
            # Apply modification based on type
            if modifier_type == "numeric" and isinstance(attr_value, (int, float)):
                new_value = self._apply_numeric_modifier(attr_value, modifier_value)
                base_attributes[attr_name] = new_value
                
            elif modifier_type == "boolean" and isinstance(attr_value, bool):
                new_value = self._apply_boolean_modifier(attr_value, modifier_value)
                base_attributes[attr_name] = new_value
                
            elif modifier_type == "list" and isinstance(attr_value, list):
                new_value = self._apply_list_modifier(attr_value, modifier_value)
                base_attributes[attr_name] = new_value
                
            elif modifier_type == "explicit":
                base_attributes[attr_name] = modifier_value
                
        adapted_preference["attributes"] = base_attributes
        return adapted_preference
    
    def _apply_numeric_modifier(self, value: Union[int, float], modifier: Union[int, float]) -> Union[int, float]:
        """Apply a relative numeric modifier to a value."""
        if abs(modifier) < 1.0:
            # Relative modifier
            new_value = value * (1.0 + modifier)
        else:
            # Absolute modifier
            new_value = value + modifier
            
        # Return same type as input
        if isinstance(value, int):
            return int(round(new_value))
        return new_value
    
    def _apply_boolean_modifier(self, value: bool, modifier: Union[int, float]) -> bool:
        """Apply a probability-based modifier to a boolean value."""
        if modifier <= 0:
            return value
            
        # Probability of flipping the value
        if random.random() < modifier:
            return not value
        return value
    
    def _apply_list_modifier(self, value: List[Any], modifier: Dict[str, Any]) -> List[Any]:
        """Apply list modifications (add, remove, prioritize)."""
        new_value = value.copy()
        
        # Apply additions
        if "add" in modifier and isinstance(modifier["add"], list):
            new_value.extend(modifier["add"])
            
        # Apply removals
        if "remove" in modifier and isinstance(modifier["remove"], list):
            new_value = [item for item in new_value if item not in modifier["remove"]]
            
        # Apply prioritization (move items to front)
        if "prioritize" in modifier and isinstance(modifier["prioritize"], list):
            for item in reversed(modifier["prioritize"]):
                if item in new_value:
                    new_value.remove(item)
                    new_value.insert(0, item)
                    
        return new_value
    
    def create_context_modifiers(
        self,
        preference_structure: Dict[str, Any],
        event: Dict[str, Any],
        agent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create context modifiers for a preference structure using dynamic analysis.
        
        Args:
            preference_structure: Preference structure with attributes
            event: Event providing context
            agent: Agent providing context
            
        Returns:
            Dictionary of context modifiers
        """
        return self.adapt_to_context(
            base_preference=preference_structure,
            event=event,
            agent=agent
        )