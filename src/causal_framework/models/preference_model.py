"""
preference_model.py

Defines the preference model representation within the Causal Preference Evolution Framework.
This model represents instantiated preference entities with attributes that can evolve over time.
"""

import copy
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class PreferenceModel:
    """
    Represents an instantiated preference entity with specific attribute values.
    Preferences are a core component of the Causal Preference Evolution Framework
    and capture user preferences that can evolve over time.
    """
    
    def __init__(
        self, 
        structure_id: str, 
        attributes: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a preference model instance.
        
        Args:
            structure_id: Identifier for the type of preference structure
            attributes: Dictionary of attribute name to value mappings
            metadata: Optional metadata about this preference (e.g., creation time, version)
        """
        self.structure_id = structure_id
        self._attributes = attributes.copy()  # Create a copy to avoid external modification
        self.metadata = metadata or {}
        
        # Temporal tracking for preference expressions
        self.last_expressed_date = None  # When this preference was last expressed/updated
        self.expression_context = None  # Context where it was expressed (event info)
        
        # Evolution tracking
        self.evolution_history: List[Dict[str, Any]] = []
        self.original_expressed_attributes: Optional[Dict[str, Any]] = None
        
        # Validate attributes
        self._validate_attributes()
        
        logger.debug(f"Initialized preference model {structure_id} with {len(attributes)} attributes")
    
    def to_json(self) -> str:
        """
        Convert the preference model to a JSON string.
        
        Returns:
            JSON string representation of the preference model
        """
        return json.dumps(self.to_dict())
    
    def _validate_attributes(self) -> None:
        """
        Validate attribute values for consistency and correctness.
        Raises ValueError if validation fails.
        """
        # Check for None values (valid types are all other primitives and lists/dicts)
        for attr_name, value in self._attributes.items():
            if value is None:
                raise ValueError(f"Attribute '{attr_name}' cannot have None value")
    
    def get_attribute(self, attribute_name: str, default: Any = None) -> Any:
        """
        Get the value of a preference attribute.
        
        Args:
            attribute_name: Name of the attribute to retrieve
            default: Default value to return if attribute doesn't exist
            
        Returns:
            Value of the attribute, or default if not found
        """
        return self._attributes.get(attribute_name, default)
    
    def set_attribute(self, attribute_name: str, value: Any) -> None:
        """
        Set the value of a preference attribute.
        
        Args:
            attribute_name: Name of the attribute to set
            value: New value for the attribute
            
        Raises:
            ValueError: If value is None
        """
        if value is None:
            raise ValueError(f"Cannot set attribute '{attribute_name}' to None value")
            
        self._attributes[attribute_name] = value
        logger.debug(f"Set attribute '{attribute_name}' to {value} for preference {self.structure_id}")
    
    def get_all_attributes(self) -> Dict[str, Any]:
        """
        Get all attributes of this preference.
        
        Returns:
            Dictionary of all attribute name to value mappings
        """
        return self._attributes.copy()  # Return a copy to prevent external modification
    
    def has_attribute(self, attribute_name: str) -> bool:
        """
        Check if this preference has a specific attribute.
        
        Args:
            attribute_name: Name of the attribute to check
            
        Returns:
            True if the attribute exists, False otherwise
        """
        return attribute_name in self._attributes
    
    def remove_attribute(self, attribute_name: str) -> bool:
        """
        Remove an attribute from this preference.
        
        Args:
            attribute_name: Name of the attribute to remove
            
        Returns:
            True if attribute was removed, False if it didn't exist
        """
        if attribute_name in self._attributes:
            del self._attributes[attribute_name]
            logger.debug(f"Removed attribute '{attribute_name}' from preference {self.structure_id}")
            return True
        return False
    
    def update_expression_tracking(self, date: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update when and where this preference was last expressed.
        
        Args:
            date: Date when the preference was expressed (datetime object)
            context: Optional context about where it was expressed (event info, etc.)
        """
        
        # Accept both datetime objects and ISO strings
        if isinstance(date, str):
            self.last_expressed_date = datetime.fromisoformat(date)
        elif isinstance(date, datetime):
            self.last_expressed_date = date
        else:
            raise ValueError(f"Date must be datetime object or ISO string, got {type(date)}")
        
        self.expression_context = context
        logger.debug(f"Updated expression tracking for preference {self.structure_id} to {self.last_expressed_date}")
    
    def snapshot_original_attributes(self) -> None:
        """Capture current attribute values as the 'originally expressed' baseline."""
        if self.original_expressed_attributes is None:
            self.original_expressed_attributes = self._attributes.copy()

    def record_evolution(
        self,
        attribute_changes: Dict[str, Dict[str, Any]],
        event_date: Optional[Any] = None,
        event_name: Optional[str] = None,
        event_category: Optional[str] = None,
        trigger: Optional[str] = None
    ) -> None:
        """
        Record an evolution event that changed one or more attributes.
        
        Args:
            attribute_changes: Dict mapping attr_name -> {'from': old_val, 'to': new_val}
            event_date: When the evolution occurred
            event_name: Name of the triggering event
            event_category: Category of the triggering event
            trigger: What triggered the evolution (e.g., 'evolution_event', 'contextual')
        """
        actual_changes = {
            k: v for k, v in attribute_changes.items()
            if v.get('from') != v.get('to')
        }
        if not actual_changes:
            return

        entry = {
            "date": event_date.isoformat() if isinstance(event_date, datetime) else str(event_date) if event_date else None,
            "event_name": event_name,
            "event_category": event_category,
            "trigger": trigger,
            "attribute_changes": actual_changes
        }
        self.evolution_history.append(entry)
        logger.debug(f"Recorded evolution for {self.structure_id}: {list(actual_changes.keys())}")

    def get_evolution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of how this preference has evolved from its original state.
        
        Returns:
            Dict with original values, current values, total changes, and history
        """
        has_evolved = len(self.evolution_history) > 0 and self.original_expressed_attributes is not None
        
        if not has_evolved:
            return {
                "has_evolved": False,
                "num_evolution_events": 0,
                "original_attributes": self.original_expressed_attributes or self._attributes.copy(),
                "current_attributes": self._attributes.copy(),
                "changed_attributes": {},
                "history": []
            }
        
        changed = {}
        for attr_name, orig_val in self.original_expressed_attributes.items():
            curr_val = self._attributes.get(attr_name)
            if orig_val != curr_val:
                changed[attr_name] = {"original": orig_val, "current": curr_val}
        
        return {
            "has_evolved": True,
            "num_evolution_events": len(self.evolution_history),
            "original_attributes": self.original_expressed_attributes.copy(),
            "current_attributes": self._attributes.copy(),
            "changed_attributes": changed,
            "history": copy.deepcopy(self.evolution_history)
        }

    def get_days_since_last_expressed(self, current_date: Any) -> Optional[int]:
        """
        Get the number of days since this preference was last expressed.
        
        Args:
            current_date: Current date (datetime object)
            
        Returns:
            Number of days since last expression, or None if never expressed
        """
        
        if not self.last_expressed_date:
            return None
            
        # Accept both datetime objects and ISO strings
        if isinstance(current_date, str):
            current_date = datetime.fromisoformat(current_date)
        elif not isinstance(current_date, datetime):
            raise ValueError(f"Current date must be datetime object or ISO string, got {type(current_date)}")
        
        delta = current_date - self.last_expressed_date
        return delta.days
    
    def compare(self, other: 'PreferenceModel') -> Dict[str, Dict[str, Any]]:
        """
        Compare this preference with another and identify differences.
        
        Args:
            other: Another PreferenceModel instance to compare with
            
        Returns:
            Dictionary of attribute differences, with 'self' and 'other' values
        """
        if not isinstance(other, PreferenceModel) or self.structure_id != other.structure_id:
            raise ValueError("Can only compare preferences of the same structure_id")
            
        differences = {}
        
        # Check attributes in self
        for attr_name, self_value in self._attributes.items():
            if attr_name not in other._attributes:
                differences[attr_name] = {
                    "self": self_value,
                    "other": None,
                    "status": "missing_in_other"
                }
            elif self_value != other._attributes[attr_name]:
                differences[attr_name] = {
                    "self": self_value,
                    "other": other._attributes[attr_name],
                    "status": "different"
                }
                
        # Check for attributes in other but not in self
        for attr_name in other._attributes:
            if attr_name not in self._attributes:
                differences[attr_name] = {
                    "self": None,
                    "other": other._attributes[attr_name],
                    "status": "missing_in_self"
                }
                
        return differences
    
    def merge(self, other: 'PreferenceModel', overwrite: bool = False) -> 'PreferenceModel':
        """
        Merge attributes from another preference into this one.
        
        Args:
            other: Another PreferenceModel instance to merge from
            overwrite: If True, overwrites existing attributes with values from other
            
        Returns:
            A new PreferenceModel with merged attributes
            
        Raises:
            ValueError: If preferences have different structure_ids
        """
        if not isinstance(other, PreferenceModel) or self.structure_id != other.structure_id:
            raise ValueError("Can only merge preferences of the same structure_id")
            
        # Create merged attributes
        merged_attributes = self._attributes.copy()
        
        for attr_name, other_value in other._attributes.items():
            if attr_name not in merged_attributes or overwrite:
                merged_attributes[attr_name] = other_value
                
        # Create merged metadata
        merged_metadata = {
            **self.metadata,
            "merged_from": other.structure_id,
            "merge_overwrite": overwrite
        }
        
        # Return a new instance with merged attributes
        return PreferenceModel(
            structure_id=self.structure_id,
            attributes=merged_attributes,
            metadata=merged_metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the preference model to a dictionary representation.
        
        Returns:
            Dictionary representation of the preference model
        """
        return {
            "structure_id": self.structure_id,
            "attributes": self._attributes,
            "metadata": self.metadata,
            "last_expressed_date": self.last_expressed_date.isoformat() if self.last_expressed_date else None,
            "expression_context": self.expression_context,
            "evolution_history": copy.deepcopy(self.evolution_history),
            "original_expressed_attributes": copy.deepcopy(self.original_expressed_attributes)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreferenceModel':
        """
        Create a preference model from a dictionary representation.
        
        Args:
            data: Dictionary representation of a preference model
            
        Returns:
            New PreferenceModel instance
            
        Raises:
            ValueError: If required fields are missing from the data
        """
        from datetime import datetime
        
        # Validate required fields
        if "structure_id" not in data or "attributes" not in data:
            raise ValueError("Missing required fields in preference data")
            
        instance = cls(
            structure_id=data["structure_id"],
            attributes=data["attributes"],
            metadata=data.get("metadata", {})
        )
        
        # Restore temporal tracking
        if data.get("last_expressed_date"):
            instance.last_expressed_date = datetime.fromisoformat(data["last_expressed_date"])
        instance.expression_context = data.get("expression_context")
        
        # Restore evolution tracking
        instance.evolution_history = data.get("evolution_history", [])
        instance.original_expressed_attributes = data.get("original_expressed_attributes")
        
        return instance
    
    def clone(self) -> 'PreferenceModel':
        """
        Create a deep copy of this preference model.
        
        Returns:
            New PreferenceModel instance with identical attributes
        """
        cloned = PreferenceModel(
            structure_id=self.structure_id,
            attributes=copy.deepcopy(self._attributes),
            metadata=copy.deepcopy(self.metadata)
        )
        
        # Copy temporal tracking
        cloned.last_expressed_date = self.last_expressed_date
        cloned.expression_context = copy.deepcopy(self.expression_context) if self.expression_context else None
        
        # Copy evolution tracking
        cloned.evolution_history = copy.deepcopy(self.evolution_history)
        cloned.original_expressed_attributes = copy.deepcopy(self.original_expressed_attributes)
        
        return cloned

    @classmethod
    def from_json(cls, json_str: str) -> 'PreferenceModel':
        """
        Create a preference model from a JSON string.
        
        Args:
            json_str: JSON string representation of a preference model
            
        Returns:
            New PreferenceModel instance
            
        Raises:
            ValueError: If JSON is invalid or required fields are missing
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    def __eq__(self, other: object) -> bool:
        """
        Compare equality with another preference model.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if not isinstance(other, PreferenceModel):
            return False
            
        return (
            self.structure_id == other.structure_id and
            self._attributes == other._attributes
        )
    
    def __repr__(self) -> str:
        """
        Get string representation of the preference model.
        
        Returns:
            String representation
        """
        attr_count = len(self._attributes)
        return f"PreferenceModel(structure_id='{self.structure_id}', {attr_count} attributes)"
    
    def calculate_attribute_change(
        self, 
        other: 'PreferenceModel', 
        attribute_name: str
    ) -> Optional[float]:
        """
        Calculate the magnitude of change in a numeric attribute between two preference states.
        
        Args:
            other: Another PreferenceModel instance to compare with
            attribute_name: Name of the numeric attribute to compare
            
        Returns:
            Change magnitude as a fraction (0-1) or None if not comparable
            
        Raises:
            ValueError: If preferences have different structure_ids or attribute is not numeric
        """
        if not isinstance(other, PreferenceModel) or self.structure_id != other.structure_id:
            raise ValueError("Can only compare preferences of the same structure_id")
            
        # Check if both have the attribute
        if not (self.has_attribute(attribute_name) and other.has_attribute(attribute_name)):
            return None
            
        self_value = self.get_attribute(attribute_name)
        other_value = other.get_attribute(attribute_name)
        
        # Check if both values are numeric
        if not (isinstance(self_value, (int, float)) and isinstance(other_value, (int, float))):
            return None
            
        # Calculate absolute change magnitude
        abs_change = abs(self_value - other_value)
        
        # Normalize to a 0-1 scale based on the larger value
        max_value = max(abs(self_value), abs(other_value))
        if max_value == 0:
            return 0.0  # No change if both values are 0
            
        normalized_change = abs_change / max_value
        return min(1.0, normalized_change)  # Cap at 1.0
    
    def calculate_overall_change(self, other: 'PreferenceModel') -> float:
        """
        Calculate the overall change magnitude across all numeric attributes.
        
        Args:
            other: Another PreferenceModel instance to compare with
            
        Returns:
            Average change magnitude across all comparable attributes
            
        Raises:
            ValueError: If preferences have different structure_ids
        """
        if not isinstance(other, PreferenceModel) or self.structure_id != other.structure_id:
            raise ValueError("Can only compare preferences of the same structure_id")
            
        # Get all attribute names from both preferences
        all_attrs = set(list(self._attributes.keys()) + list(other._attributes.keys()))
        
        change_values = []
        for attr_name in all_attrs:
            change = self.calculate_attribute_change(other, attr_name)
            if change is not None:
                change_values.append(change)
                
        if not change_values:
            return 0.0  # No comparable attributes
            
        return sum(change_values) / len(change_values)