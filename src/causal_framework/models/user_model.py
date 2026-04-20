"""
user_model.py

Defines the user model representation within the Causal Preference Evolution Framework.
This model represents user personas with attributes that serve as a foundation for
preference sampling and evolution.
"""

import copy
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Set, Union

logger = logging.getLogger(__name__)

class UserModel:
    """
    Represents a user persona with specific attribute values.
    User personas are the foundation of the Causal Preference Evolution Framework,
    influencing how preferences are sampled and evolve over time.
    """
    
    # Define common attribute groups for validation and access
    DEMOGRAPHIC_ATTRIBUTES = {
        "age", "gender", "ethnicity", "education_level", "income_bracket",
        "occupation", "location", "relationship_status", "has_children"
    }
    
    IDENTITY_ATTRIBUTES = {
        "first_name", "last_name", "username", "email", "phone"
    }
    
    PERSONALITY_ATTRIBUTES = {
        "personality_traits", "values", "interests", "hobbies", 
        "communication_style", "tech_savviness"
    }
    
    REQUIRED_ATTRIBUTES = {
        "first_name", "last_name", "age", "gender"
    }
    
    def __init__(
        self, 
        attributes: Dict[str, Any],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a user model instance.
        
        Args:
            attributes: Dictionary of attribute name to value mappings
            user_id: Unique identifier for this user
            metadata: Optional metadata about this user
        """
        self._attributes = attributes.copy()  # Create a copy to avoid external modification
        self.id = user_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        
        logger.debug(f"Initialized user model with ID {self.id}")
    
    def validate(self) -> bool:
        """
        Validate that the user model has all required attributes.
        
        Returns:
            True if valid, False otherwise
        """
        for attr in self.REQUIRED_ATTRIBUTES:
            if attr not in self._attributes:
                logger.warning(f"User {self.id} is missing required attribute: {attr}")
                return False
        return True
    
    def get_attribute(self, attribute_name: str, default: Any = None) -> Any:
        """
        Get the value of a user attribute.
        
        Args:
            attribute_name: Name of the attribute to retrieve
            default: Default value to return if attribute doesn't exist
            
        Returns:
            Value of the attribute, or default if not found
        """
        return self._attributes.get(attribute_name, default)
    
    def set_attribute(self, attribute_name: str, value: Any) -> None:
        """
        Set the value of a user attribute.
        
        Args:
            attribute_name: Name of the attribute to set
            value: New value for the attribute
        """
        self._attributes[attribute_name] = value
        logger.debug(f"Set attribute '{attribute_name}' to {value} for user {self.id}")
    
    def get_all_attributes(self) -> Dict[str, Any]:
        """
        Get all attributes of this user.
        
        Returns:
            Dictionary of all attribute name to value mappings
        """
        return self._attributes.copy()  # Return a copy to prevent external modification
    
    def has_attribute(self, attribute_name: str) -> bool:
        """
        Check if this user has a specific attribute.
        
        Args:
            attribute_name: Name of the attribute to check
            
        Returns:
            True if the attribute exists, False otherwise
        """
        return attribute_name in self._attributes
    
    def remove_attribute(self, attribute_name: str) -> bool:
        """
        Remove an attribute from this user.
        
        Args:
            attribute_name: Name of the attribute to remove
            
        Returns:
            True if attribute was removed, False if it didn't exist
        """
        if attribute_name in self._attributes:
            del self._attributes[attribute_name]
            logger.debug(f"Removed attribute '{attribute_name}' from user {self.id}")
            return True
        return False
    
    def get_demographic_attributes(self) -> Dict[str, Any]:
        """
        Get demographic attributes of this user.
        
        Returns:
            Dictionary of demographic attribute name to value mappings
        """
        return {
            attr: self._attributes[attr] for attr in self.DEMOGRAPHIC_ATTRIBUTES
            if attr in self._attributes
        }
    
    def get_identity_attributes(self) -> Dict[str, Any]:
        """
        Get identity attributes of this user.
        
        Returns:
            Dictionary of identity attribute name to value mappings
        """
        return {
            attr: self._attributes[attr] for attr in self.IDENTITY_ATTRIBUTES
            if attr in self._attributes
        }
    
    def get_personality_attributes(self) -> Dict[str, Any]:
        """
        Get personality attributes of this user.
        
        Returns:
            Dictionary of personality attribute name to value mappings
        """
        return {
            attr: self._attributes[attr] for attr in self.PERSONALITY_ATTRIBUTES
            if attr in self._attributes
        }
    
    def get_missing_attributes(self) -> List[str]:
        """
        Get a list of required attributes that are missing from this user.
        
        Returns:
            List of missing attribute names
        """
        return [attr for attr in self.REQUIRED_ATTRIBUTES if attr not in self._attributes]
    
    def get_life_stage(self) -> str:
        """
        Determine the user's life stage based on age and other attributes.
        
        Returns:
            String representing the life stage (child, teenager, young_adult, adult, middle_aged, senior)
        """
        age = int(self.get_attribute("age")) if self.get_attribute("age") else None
        
        if age is None:
            return "unknown"
        
        if age < 13:
            return "child"
        elif age < 20:
            return "teenager"
        elif age < 30:
            return "young_adult"
        elif age < 50:
            return "adult"
        elif age < 65:
            return "middle_aged"
        else:
            return "senior"
    
    def get_formatted_name(self, include_title: bool = False) -> str:
        """
        Get the user's formatted name.
        
        Args:
            include_title: Whether to include title (Mr., Ms., etc.)
            
        Returns:
            Formatted name string
        """
        first_name = self.get_attribute("first_name", "")
        last_name = self.get_attribute("last_name", "")
        
        if include_title:
            title = self.get_attribute("title", "")
            if title:
                return f"{title} {first_name} {last_name}"
                
        return f"{first_name} {last_name}"
    
    def get_age_group(self) -> str:
        """
        Get the user's age group for demographic categorization.
        
        Returns:
            String representing age group (e.g., "18-24", "25-34", etc.)
        """
        age = self.get_attribute("age")
        
        if age is None:
            return "unknown"
            
        if age < 18:
            return "under_18"
        elif age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65_plus"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the user model to a dictionary representation.
        
        Returns:
            Dictionary representation of the user model
        """
        return {
            "id": self.id,
            "attributes": self._attributes,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserModel':
        """
        Create a user model from a dictionary representation.
        
        Args:
            data: Dictionary representation of a user model
            
        Returns:
            New UserModel instance
            
        Raises:
            ValueError: If required fields are missing from the data
        """
        # Validate required fields
        if "attributes" not in data:
            raise ValueError("Missing required 'attributes' field in user data")
            
        return cls(
            attributes=data["attributes"],
            user_id=data.get("id"),
            metadata=data.get("metadata", {})
        )
    
    def clone(self) -> 'UserModel':
        """
        Create a deep copy of this user model.
        
        Returns:
            New UserModel instance with identical attributes
        """
        return UserModel(
            attributes=copy.deepcopy(self._attributes),
            user_id=self.id,
            metadata=copy.deepcopy(self.metadata)
        )
    
    def to_json(self) -> str:
        """
        Convert the user model to a JSON string.
        
        Returns:
            JSON string representation of the user model
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UserModel':
        """
        Create a user model from a JSON string.
        
        Args:
            json_str: JSON string representation of a user model
            
        Returns:
            New UserModel instance
            
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
        Compare equality with another user model.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if not isinstance(other, UserModel):
            return False
            
        return (
            self.id == other.id and
            self._attributes == other._attributes
        )
    
    def __repr__(self) -> str:
        """
        Get string representation of the user model.
        
        Returns:
            String representation
        """
        name = self.get_formatted_name()
        attr_count = len(self._attributes)
        return f"UserModel(id='{self.id}', name='{name}', {attr_count} attributes)"
    
    def similarity_score(self, other: 'UserModel') -> float:
        """
        Calculate a similarity score between two users based on their attributes.
        
        Args:
            other: Another UserModel instance to compare with
            
        Returns:
            Similarity score from 0.0 (completely different) to 1.0 (identical)
        """
        if not isinstance(other, UserModel):
            return 0.0
            
        # Demographic similarity (weighted more heavily)
        demo_attrs = self.DEMOGRAPHIC_ATTRIBUTES.intersection(
            set(self._attributes.keys()).intersection(other._attributes.keys())
        )
        
        demo_score = 0.0
        for attr in demo_attrs:
            if self._attributes[attr] == other._attributes[attr]:
                demo_score += 1.0
                
        demo_score = demo_score / len(demo_attrs) if demo_attrs else 0.0
        
        # Personality similarity
        pers_attrs = self.PERSONALITY_ATTRIBUTES.intersection(
            set(self._attributes.keys()).intersection(other._attributes.keys())
        )
        
        pers_score = 0.0
        for attr in pers_attrs:
            # Handle lists of traits
            if isinstance(self._attributes[attr], list) and isinstance(other._attributes[attr], list):
                common = set(self._attributes[attr]).intersection(set(other._attributes[attr]))
                total = set(self._attributes[attr]).union(set(other._attributes[attr]))
                pers_score += len(common) / len(total) if total else 0.0
            # Handle scalar values
            elif self._attributes[attr] == other._attributes[attr]:
                pers_score += 1.0
                
        pers_score = pers_score / len(pers_attrs) if pers_attrs else 0.0
        
        # Combine scores with demographic attributes weighted more heavily
        combined_score = (demo_score * 0.7) + (pers_score * 0.3)
        
        return combined_score
    
    def add_interest(self, interest: str) -> None:
        """
        Add an interest to the user's interests list.
        
        Args:
            interest: Interest to add
        """
        if "interests" not in self._attributes:
            self._attributes["interests"] = []
            
        if interest not in self._attributes["interests"]:
            self._attributes["interests"].append(interest)
            logger.debug(f"Added interest '{interest}' to user {self.id}")
    
    def remove_interest(self, interest: str) -> bool:
        """
        Remove an interest from the user's interests list.
        
        Args:
            interest: Interest to remove
            
        Returns:
            True if interest was removed, False if it wasn't in the list
        """
        if "interests" in self._attributes and interest in self._attributes["interests"]:
            self._attributes["interests"].remove(interest)
            logger.debug(f"Removed interest '{interest}' from user {self.id}")
            return True
        return False
    
    def has_interest(self, interest: str) -> bool:
        """
        Check if the user has a specific interest.
        
        Args:
            interest: Interest to check
            
        Returns:
            True if the user has this interest, False otherwise
        """
        return "interests" in self._attributes and interest in self._attributes["interests"]
    
    def get_interests(self) -> List[str]:
        """
        Get the user's interests.
        
        Returns:
            List of interest strings
        """
        return self._attributes.get("interests", []).copy()
    
    def add_personality_trait(self, trait: str) -> None:
        """
        Add a personality trait to the user.
        
        Args:
            trait: Personality trait to add
        """
        if "personality_traits" not in self._attributes:
            self._attributes["personality_traits"] = []
            
        if trait not in self._attributes["personality_traits"]:
            self._attributes["personality_traits"].append(trait)
            logger.debug(f"Added personality trait '{trait}' to user {self.id}")
    
    def get_personality_traits(self) -> List[str]:
        """
        Get the user's personality traits.
        
        Returns:
            List of personality trait strings
        """
        return self._attributes.get("personality_traits", []).copy()