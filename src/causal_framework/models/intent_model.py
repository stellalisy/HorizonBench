"""
intent_model.py

Defines the intent model representation within the Causal Preference Evolution Framework.
Intents represent short-term, goal-oriented motivations for specific interactions.
"""

import copy
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class IntentModel:
    """
    Represents an agent's intent with specific attribute values.
    Intents are short-term, goal-oriented motivations for specific interactions,
    distinct from long-term preferences.
    """
    
    def __init__(
        self,
        intent_type: str,
        attributes: Dict[str, Any],
        intent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an intent model instance.
        
        Args:
            intent_type: Type of intent (e.g., "seek_support", "provide_guidance")
            attributes: Dictionary of attribute name to value mappings
            intent_id: Unique identifier for this intent instance
            metadata: Optional metadata about this intent
        """
        self.intent_type = intent_type
        self._attributes = attributes.copy()
        self.id = intent_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        
        logger.debug(f"Initialized intent model {intent_type} with ID {self.id}")
    
    def get_attribute(self, attribute_name: str, default: Any = None) -> Any:
        """
        Get the value of an intent attribute.
        
        Args:
            attribute_name: Name of the attribute to retrieve
            default: Default value to return if attribute doesn't exist
            
        Returns:
            Value of the attribute, or default if not found
        """
        return self._attributes.get(attribute_name, default)
    
    def set_attribute(self, attribute_name: str, value: Any) -> None:
        """
        Set the value of an intent attribute.
        
        Args:
            attribute_name: Name of the attribute to set
            value: New value for the attribute
        """
        self._attributes[attribute_name] = value
        logger.debug(f"Set attribute '{attribute_name}' to {value} for intent {self.id}")
    
    def get_all_attributes(self) -> Dict[str, Any]:
        """
        Get all attributes of this intent.
        
        Returns:
            Dictionary of all attribute name to value mappings
        """
        return self._attributes.copy()
    
    def has_attribute(self, attribute_name: str) -> bool:
        """
        Check if this intent has a specific attribute.
        
        Args:
            attribute_name: Name of the attribute to check
            
        Returns:
            True if the attribute exists, False otherwise
        """
        return attribute_name in self._attributes
    
    def clone(self) -> 'IntentModel':
        """
        Create a deep copy of this intent model.
        
        Returns:
            New IntentModel instance with identical attributes
        """
        return IntentModel(
            intent_type=self.intent_type,
            attributes=copy.deepcopy(self._attributes),
            intent_id=None,  # Generate new ID for clone
            metadata=copy.deepcopy(self.metadata)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this intent model to a dictionary representation.
        
        Returns:
            Dictionary representation of the intent model
        """
        return {
            "id": self.id,
            "intent_type": self.intent_type,
            "attributes": self._attributes.copy(),
            "metadata": self.metadata.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentModel':
        """
        Create an IntentModel instance from a dictionary representation.
        
        Args:
            data: Dictionary containing intent model data
            
        Returns:
            IntentModel instance
        """
        return cls(
            intent_type=data["intent_type"],
            attributes=data["attributes"],
            intent_id=data.get("id"),
            metadata=data.get("metadata", {})
        )
    
    def __str__(self) -> str:
        """String representation of the intent model."""
        return f"IntentModel(type={self.intent_type}, attributes={len(self._attributes)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the intent model."""
        return f"IntentModel(id={self.id}, type={self.intent_type}, attributes={self._attributes})" 