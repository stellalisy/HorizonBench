"""
stability_processor.py

Implements the stability bias mechanism for the Causal Preference Evolution Framework.
Handles resistance to change for preference attributes based on stability weights.
"""

import logging
import random
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class StabilityProcessor:
    """
    Handles stability bias in preference evolution, ensuring preferences
    demonstrate appropriate resistance to change over time.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the stability processor.
        
        Args:
            config: Optional configuration parameters
        """
        # Set default configuration or update with provided config
        self.config = {
            "default_stability_weight": 0.7,  # Default stability weight (higher = more resistance to change)
            "max_change_magnitude": 0.3,      # Maximum magnitude of a single preference change
            "stability_variance": 0.2,        # Variance in stability weights for randomization
            "random_seed": 42                 # Seed for reproducibility
        }
        
        if config:
            self.config.update(config)
            
        random.seed(self.config["random_seed"])
        
        logger.debug("StabilityProcessor initialized")
    
    def apply_stability_bias(
        self,
        current_value: Any,
        target_value: Any,
        stability_weight: Optional[float] = None,
        attribute_type: Optional[str] = None
    ) -> Any:
        """
        Apply stability bias to limit the magnitude of preference change.
        
        Args:
            current_value: Current preference attribute value
            target_value: Target preference attribute value (before stability bias)
            stability_weight: Stability weight for this attribute (0-1 scale)
            attribute_type: Type of the attribute (string, int, float, boolean, list)
            
        Returns:
            New preference attribute value after applying stability bias
        """
        # If stability weight not provided, use default
        if stability_weight is None:
            stability_weight = self.config["default_stability_weight"]
            
        # Ensure stability weight is in valid range
        stability_weight = max(0.0, min(1.0, stability_weight))
        
        # If values are the same, no change needed
        if current_value == target_value:
            return current_value
            
        # Apply stability bias based on attribute type
        if isinstance(current_value, (int, float)) and isinstance(target_value, (int, float)):
            return self._apply_numeric_stability(current_value, target_value, stability_weight)
        elif isinstance(current_value, bool) and isinstance(target_value, bool):
            return self._apply_boolean_stability(current_value, target_value, stability_weight)
        elif isinstance(current_value, list) and isinstance(target_value, list):
            return self._apply_list_stability(current_value, target_value, stability_weight)
        else:
            # For strings and other types, use probability-based stability
            return self._apply_categorical_stability(current_value, target_value, stability_weight)
    
    def _apply_numeric_stability(
        self,
        current_value: Union[int, float],
        target_value: Union[int, float],
        stability_weight: float
    ) -> Union[int, float]:
        """
        Apply stability bias to numeric values by limiting change magnitude.
        
        Args:
            current_value: Current numeric value
            target_value: Target numeric value
            stability_weight: Stability weight (0-1)
            
        Returns:
            New numeric value after applying stability bias
        """
        # Calculate change magnitude
        change = target_value - current_value
        
        # Apply stability weight to limit change magnitude
        adjusted_change = change * (1 - stability_weight)
        
        # Further limit to maximum change magnitude
        max_change = self.config["max_change_magnitude"] * abs(current_value) if current_value != 0 else 1.0
        
        if abs(adjusted_change) > max_change:
            adjusted_change = max_change if adjusted_change > 0 else -max_change
            
        # Calculate new value
        new_value = current_value + adjusted_change
        
        # If original value was integer, convert back
        if isinstance(current_value, int):
            new_value = int(round(new_value))
            
        logger.debug(f"Applied numeric stability: {current_value} -> {target_value} -> {new_value} (stability: {stability_weight:.2f})")
        return new_value
    
    def _apply_boolean_stability(
        self,
        current_value: bool,
        target_value: bool,
        stability_weight: float
    ) -> bool:
        """
        Apply stability bias to boolean values based on change probability.
        
        Args:
            current_value: Current boolean value
            target_value: Target boolean value
            stability_weight: Stability weight (0-1)
            
        Returns:
            New boolean value after applying stability bias
        """
        # If values are the same, no change needed
        if current_value == target_value:
            return current_value
            
        # Calculate change probability based on stability weight
        change_probability = 1.0 - stability_weight
        
        # Decide whether to change based on probability
        if random.random() < change_probability:
            logger.debug(f"Applied boolean stability: {current_value} -> {target_value} (stability: {stability_weight:.2f}, change allowed)")
            return target_value
        else:
            logger.debug(f"Applied boolean stability: {current_value} -/-> {target_value} (stability: {stability_weight:.2f}, change blocked)")
            return current_value
    
    def _apply_categorical_stability(
        self,
        current_value: Any,
        target_value: Any,
        stability_weight: float
    ) -> Any:
        """
        Apply stability bias to categorical values based on change probability.
        
        Args:
            current_value: Current categorical value
            target_value: Target categorical value
            stability_weight: Stability weight (0-1)
            
        Returns:
            New categorical value after applying stability bias
        """
        # If values are the same, no change needed
        if current_value == target_value:
            return current_value
            
        # Calculate change probability based on stability weight
        change_probability = 1.0 - stability_weight
        
        # Decide whether to change based on probability
        if random.random() < change_probability:
            logger.debug(f"Applied categorical stability: {current_value} -> {target_value} (stability: {stability_weight:.2f}, change allowed)")
            return target_value
        else:
            logger.debug(f"Applied categorical stability: {current_value} -/-> {target_value} (stability: {stability_weight:.2f}, change blocked)")
            return current_value
    
    def _apply_list_stability(
        self,
        current_value: List[Any],
        target_value: List[Any],
        stability_weight: float
    ) -> List[Any]:
        """
        Apply stability bias to list values by limiting the number of changes.
        
        Args:
            current_value: Current list value
            target_value: Target list value
            stability_weight: Stability weight (0-1)
            
        Returns:
            New list value after applying stability bias
        """
        # If lists are identical, no change needed
        if current_value == target_value:
            return current_value
            
        # Convert to sets for comparison
        current_set = set(str(item) for item in current_value)
        target_set = set(str(item) for item in target_value)
        
        # Find additions and removals
        additions = [item for item in target_value if str(item) not in current_set]
        removals = [item for item in current_value if str(item) not in target_set]
        
        # Apply stability bias to limit number of changes
        allowed_changes = max(1, int(len(additions) * (1 - stability_weight)))
        allowed_additions = random.sample(additions, min(allowed_changes, len(additions))) if additions else []
        
        allowed_changes = max(1, int(len(removals) * (1 - stability_weight)))
        allowed_removals = random.sample(removals, min(allowed_changes, len(removals))) if removals else []
        
        # Create new list by applying allowed changes
        new_value = [item for item in current_value if item not in allowed_removals]
        new_value.extend(allowed_additions)
        
        logger.debug(f"Applied list stability: {len(current_value)} items -> {len(target_value)} items -> {len(new_value)} items (stability: {stability_weight:.2f})")
        return new_value
    
    def calculate_stability_weight(
        self,
        base_stability: float,
        attribute_name: str,
        user_attributes: Optional[Dict[str, Any]] = None,
        preference_importance: Optional[float] = None
    ) -> float:
        """
        Calculate an appropriate stability weight for a preference attribute.
        
        Args:
            base_stability: Base stability weight for this attribute
            attribute_name: Name of the attribute
            user_attributes: Optional user attributes to inform stability
            preference_importance: Optional importance of this preference to the user
            
        Returns:
            Calculated stability weight (0-1)
        """
        stability = base_stability
        
        # Adjust based on attribute name (some attributes are inherently more stable)
        if any(stable_term in attribute_name.lower() for stable_term in ["core", "identity", "fundamental"]):
            stability += 0.1
        elif any(fluid_term in attribute_name.lower() for fluid_term in ["temporary", "current", "situational"]):
            stability -= 0.1
            
        # Adjust based on preference importance if provided
        if preference_importance is not None:
            # More important preferences are generally more stable
            stability += 0.1 * preference_importance
            
        # Adjust based on user attributes if provided
        if user_attributes:
            # Older users typically have more stable preferences
            age = user_attributes.get("age")
            if age is not None:
                if age > 50:
                    stability += 0.1
                elif age < 25:
                    stability -= 0.1
                    
            # Some personality traits affect preference stability
            personality_traits = user_attributes.get("personality_traits", [])
            if "openness" in personality_traits:
                stability -= 0.05
            if "conscientiousness" in personality_traits:
                stability += 0.05
                
        # Add small random variation for more natural evolution
        stability += random.uniform(-self.config["stability_variance"], self.config["stability_variance"])
        
        # Ensure result is in valid range
        stability = max(0.0, min(1.0, stability))
        
        return stability