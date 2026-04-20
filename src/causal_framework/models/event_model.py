"""
event_model.py

Defines the event model representation within the Causal Preference Evolution Framework.
Events are key components that trigger preference changes and provide contexts
for human-AI interactions.
"""

import copy
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Union

logger = logging.getLogger(__name__)

class EventModel:
    """
    Represents an instantiated event with specific attribute values.
    Events are occurrences in a user's timeline that may trigger preference
    changes and provide contexts for conversations.
    """
    
    def __init__(
        self, 
        structure_id: str, 
        name: str,
        category: str,
        base_attributes: Dict[str, Any],
        preference_dependent_attributes: Dict[str, bool] = None,
        date: Optional[datetime] = None,
        event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an event model instance.
        
        Args:
            structure_id: Identifier for the type of event structure
            base_attributes: Dictionary of attribute name to value mappings
                            for preference-independent attributes
            preference_dependent_attributes: Dictionary indicating which attributes
                                           depend on preferences (True) or not (False)
            date: Date/time when the event occurs
            event_id: Unique identifier for this event instance
            metadata: Optional metadata about this event
        """
        self.structure_id = structure_id
        self.name = name
        self.category = category
        self._attributes = base_attributes.copy()  # Create a copy to avoid external modification
        self._preference_dependent = preference_dependent_attributes or {}
        self.date = date or datetime.now()
        self.id = event_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        
        logger.debug(f"Initialized event model {structure_id} with ID {self.id}")
    
    def get_attribute(self, attribute_name: str, default: Any = None) -> Any:
        """
        Get the value of an event attribute.
        
        Args:
            attribute_name: Name of the attribute to retrieve
            default: Default value to return if attribute doesn't exist
            
        Returns:
            Value of the attribute, or default if not found
        """
        return self._attributes.get(attribute_name, default)
    
    def set_attribute(self, attribute_name: str, value: Any, is_preference_dependent: Optional[bool] = None) -> None:
        """
        Set the value of an event attribute.
        
        Args:
            attribute_name: Name of the attribute to set
            value: New value for the attribute
            is_preference_dependent: Optionally specify if this attribute depends on preferences
        """
        self._attributes[attribute_name] = value
        
        # Update preference dependency status if provided
        if is_preference_dependent is not None:
            self._preference_dependent[attribute_name] = is_preference_dependent
            
        logger.debug(f"Set attribute '{attribute_name}' to {value} for event {self.id}")
    
    def get_all_attributes(self) -> Dict[str, Any]:
        """
        Get all attributes of this event.
        
        Returns:
            Dictionary of all attribute name to value mappings
        """
        return self._attributes.copy()  # Return a copy to prevent external modification
    
    def has_attribute(self, attribute_name: str) -> bool:
        """
        Check if this event has a specific attribute.
        
        Args:
            attribute_name: Name of the attribute to check
            
        Returns:
            True if the attribute exists, False otherwise
        """
        return attribute_name in self._attributes
    
    def remove_attribute(self, attribute_name: str) -> bool:
        """
        Remove an attribute from this event.
        
        Args:
            attribute_name: Name of the attribute to remove
            
        Returns:
            True if attribute was removed, False if it didn't exist
        """
        if attribute_name in self._attributes:
            del self._attributes[attribute_name]
            # Also remove from preference dependent tracking
            if attribute_name in self._preference_dependent:
                del self._preference_dependent[attribute_name]
            logger.debug(f"Removed attribute '{attribute_name}' from event {self.id}")
            return True
        return False
    
    def is_preference_dependent(self, attribute_name: str) -> bool:
        """
        Check if an attribute is preference-dependent.
        
        Args:
            attribute_name: Name of the attribute to check
            
        Returns:
            True if the attribute depends on preferences, False otherwise
        """
        return self._preference_dependent.get(attribute_name, False)
    
    def get_preference_dependent_attributes(self) -> Dict[str, Any]:
        """
        Get all preference-dependent attributes.
        
        Returns:
            Dictionary of attribute names to values for preference-dependent attributes
        """
        return {
            attr: value for attr, value in self._attributes.items()
            if self.is_preference_dependent(attr)
        }
    
    def get_preference_independent_attributes(self) -> Dict[str, Any]:
        """
        Get all preference-independent attributes.
        
        Returns:
            Dictionary of attribute names to values for preference-independent attributes
        """
        return {
            attr: value for attr, value in self._attributes.items()
            if not self.is_preference_dependent(attr)
        }
    
    def has_preference(self, preference_id: str) -> bool:
        """
        Check if this event has an associated preference.
        Used for tracking which preferences were relevant to this event.
        
        Args:
            preference_id: ID of the preference to check
            
        Returns:
            True if the event has this preference, False otherwise
        """
        return preference_id in self.metadata.get("associated_preferences", {})
    
    def get_preference(self, preference_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific preference associated with this event.
        
        Args:
            preference_id: ID of the preference to retrieve
            
        Returns:
            Preference data if found, None otherwise
        """
        return self.metadata.get("associated_preferences", {}).get(preference_id)
    
    def set_preference(self, preference_id: str, preference_data: Dict[str, Any]) -> None:
        """
        Associate a preference with this event.
        Used to track which preferences were relevant and their state at event time.
        
        Args:
            preference_id: ID of the preference to associate
            preference_data: Preference data to store
        """
        if "associated_preferences" not in self.metadata:
            self.metadata["associated_preferences"] = {}
            
        self.metadata["associated_preferences"][preference_id] = preference_data
        logger.debug(f"Associated preference {preference_id} with event {self.id}")
    
    def get_all_preferences(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all preferences associated with this event.
        
        Returns:
            Dictionary mapping preference IDs to preference data
        """
        return self.metadata.get("associated_preferences", {}).copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event model to a dictionary representation.
        
        Returns:
            Dictionary representation of the event model
        """
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "structure_id": self.structure_id,
            "attributes": self._attributes,
            "preference_dependent_attributes": self._preference_dependent,
            "date": self.date.isoformat() if self.date else None,
            "date_unix": int(self.date.timestamp()) if self.date else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventModel':
        """
        Create an event model from a dictionary representation.
        
        Args:
            data: Dictionary representation of an event model
            
        Returns:
            New EventModel instance
            
        Raises:
            ValueError: If required fields are missing from the data
        """
        # Validate required fields
        required_fields = ["id", "structure_id", "attributes"]
        if not all(key in data for key in required_fields):
            missing_fields = [key for key in required_fields if key not in data]
            raise ValueError(f"Missing required fields in event data: {missing_fields}")
            
        # Parse date if it exists
        date = None
        if "date" in data and data["date"]:
            try:
                date = datetime.fromisoformat(data["date"])
            except ValueError:
                logger.warning(f"Could not parse date '{data['date']}', using current time")
                date = datetime.now()
        elif "date_unix" in data and data["date_unix"]:
            try:
                date = datetime.fromtimestamp(data["date_unix"])
            except (ValueError, OSError):
                logger.warning(f"Could not parse unix timestamp '{data['date_unix']}', using current time")
                date = datetime.now()
                
        return cls(
            structure_id=data["structure_id"],
            name=data.get("name", data["structure_id"]),  # Use structure_id as fallback for name
            category=data.get("category", "unknown"),     # Use "unknown" as fallback for category
            base_attributes=data["attributes"],
            preference_dependent_attributes=data.get("preference_dependent_attributes", {}),
            date=date,
            event_id=data["id"],
            metadata=data.get("metadata", {})
        )
    
    def clone(self) -> 'EventModel':
        """
        Create a deep copy of this event model.
        
        Returns:
            New EventModel instance with identical attributes
        """
        return EventModel(
            structure_id=self.structure_id,
            name=self.name,
            category=self.category,
            base_attributes=copy.deepcopy(self._attributes),
            preference_dependent_attributes=copy.deepcopy(self._preference_dependent),
            date=self.date,
            event_id=self.id,
            metadata=copy.deepcopy(self.metadata)
        )
    
    def to_json(self) -> str:
        """
        Convert the event model to a JSON string.
        
        Returns:
            JSON string representation of the event model
        """
        data = self.to_dict()
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EventModel':
        """
        Create an event model from a JSON string.
        
        Args:
            json_str: JSON string representation of an event model
            
        Returns:
            New EventModel instance
            
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
        Compare equality with another event model.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if not isinstance(other, EventModel):
            return False
            
        return (
            self.id == other.id and
            self.structure_id == other.structure_id and
            self._attributes == other._attributes and
            self.date == other.date
        )
    
    def __repr__(self) -> str:
        """
        Get string representation of the event model.
        
        Returns:
            String representation
        """
        attr_count = len(self._attributes)
        return f"EventModel(id='{self.id}', structure_id='{self.structure_id}', {attr_count} attributes)"
    
    def is_similar_to(self, other: 'EventModel', threshold: float = 0.7) -> bool:
        """
        Check if this event is similar to another event based on shared attributes.
        
        Args:
            other: Another EventModel instance to compare with
            threshold: Similarity threshold (0-1) for considering events similar
            
        Returns:
            True if events are similar, False otherwise
        """
        if not isinstance(other, EventModel) or self.structure_id != other.structure_id:
            return False
            
        # Count matching attributes
        matching_attrs = 0
        total_attrs = 0
        
        # Check attributes in self
        for attr_name, self_value in self._attributes.items():
            if attr_name in other._attributes and self_value == other._attributes[attr_name]:
                matching_attrs += 1
            total_attrs += 1
            
        # Add attributes unique to other
        for attr_name in other._attributes:
            if attr_name not in self._attributes:
                total_attrs += 1
                
        # Calculate similarity score
        similarity = matching_attrs / total_attrs if total_attrs > 0 else 0
        
        return similarity >= threshold
    
    def get_time_proximity(self, other: 'EventModel') -> float:
        """
        Calculate the time proximity between two events (0-1 scale).
        
        Args:
            other: Another EventModel instance to compare with
            
        Returns:
            Proximity score where 1 is very close in time and 0 is far apart
            
        Raises:
            ValueError: If either event has no date
        """
        if not self.date or not other.date:
            raise ValueError("Both events must have valid dates")
            
        # Calculate time difference in hours
        time_diff = abs((self.date - other.date).total_seconds()) / 3600
        
        # Scale to 0-1 proximity (using exponential decay)
        # Proximity of 0.5 at 24 hours, 0.25 at 48 hours, etc.
        proximity = 2 ** (-time_diff / 24)
        
        return proximity
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to this event for categorization.
        
        Args:
            tag: Tag string to add
        """
        if "tags" not in self.metadata:
            self.metadata["tags"] = []
            
        if tag not in self.metadata["tags"]:
            self.metadata["tags"].append(tag)
            logger.debug(f"Added tag '{tag}' to event {self.id}")
    
    def has_tag(self, tag: str) -> bool:
        """
        Check if this event has a specific tag.
        
        Args:
            tag: Tag string to check
            
        Returns:
            True if the event has this tag, False otherwise
        """
        return tag in self.metadata.get("tags", [])
    
    def get_tags(self) -> List[str]:
        """
        Get all tags associated with this event.
        
        Returns:
            List of tag strings
        """
        return self.metadata.get("tags", []).copy()