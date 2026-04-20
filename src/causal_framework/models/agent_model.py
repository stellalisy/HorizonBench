"""
agent_model.py

Defines the agent model representation within the Causal Preference Evolution Framework.
This model represents both user and assistant agents with attributes and roles.
"""

import copy
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Set, Union

logger = logging.getLogger(__name__)

class AgentModel:
    """
    Represents an agent (user or assistant) with specific attribute values and role.
    Agent models are the foundation of the symmetric conversation system.
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
    
    ROLE_ATTRIBUTES = {
        "role", "expertise", "experience_level", "specialization",
        "professional_background", "interaction_style"
    }
    
    REQUIRED_ATTRIBUTES = {
        "first_name", "last_name", "age", "gender"
    }
    
    def __init__(
        self, 
        attributes: Dict[str, Any],
        agent_id: Optional[str] = None,
        role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an agent model instance.
        
        Args:
            attributes: Dictionary of attribute name to value mappings
            agent_id: Unique identifier for this agent
            role: Role of the agent (e.g., "user", "therapist", "tutor", "friend")
            metadata: Optional metadata about this agent
        """
        self._attributes = attributes.copy()
        self.id = agent_id or str(uuid.uuid4())
        self.role = role or "user"
        self.metadata = metadata or {}
        
        logger.debug(f"Initialized agent model with ID {self.id} and role {self.role}")
    
    def get_attribute(self, attribute_name: str, default: Any = None) -> Any:
        """
        Get the value of an agent attribute.
        
        Args:
            attribute_name: Name of the attribute to retrieve
            default: Default value to return if attribute doesn't exist
            
        Returns:
            Value of the attribute, or default if not found
        """
        return self._attributes.get(attribute_name, default)
    
    def set_attribute(self, attribute_name: str, value: Any) -> None:
        """
        Set the value of an agent attribute.
        
        Args:
            attribute_name: Name of the attribute to set
            value: New value for the attribute
        """
        self._attributes[attribute_name] = value
        logger.debug(f"Set attribute '{attribute_name}' to {value} for agent {self.id}")
    
    def get_all_attributes(self) -> Dict[str, Any]:
        """
        Get all attributes of this agent.
        
        Returns:
            Dictionary of all attribute name to value mappings
        """
        return self._attributes.copy()
    
    def has_attribute(self, attribute_name: str) -> bool:
        """
        Check if this agent has a specific attribute.
        
        Args:
            attribute_name: Name of the attribute to check
            
        Returns:
            True if the attribute exists, False otherwise
        """
        return attribute_name in self._attributes
    
    def get_formatted_name(self) -> str:
        """
        Get formatted name for display.
        
        Returns:
            Formatted name string
        """
        first_name = self.get_attribute("first_name", "")
        last_name = self.get_attribute("last_name", "")
        
        if first_name and last_name:
            return f"{first_name} {last_name}"
        elif first_name:
            return first_name
        elif last_name:
            return last_name
        else:
            return f"Agent {self.id[:8]}"
    
    def get_interests(self) -> List[str]:
        """
        Get list of interests.
        
        Returns:
            List of interest strings
        """
        interests = self.get_attribute("interests", [])
        if isinstance(interests, str):
            return [interests]
        elif isinstance(interests, list):
            return interests
        else:
            return []
    
    def get_personality_traits(self) -> List[str]:
        """
        Get list of personality traits.
        
        Returns:
            List of personality trait strings
        """
        traits = self.get_attribute("personality_traits", [])
        if isinstance(traits, str):
            return [traits]
        elif isinstance(traits, list):
            return traits
        else:
            return []
    
    def get_age_group(self) -> str:
        """
        Get age group category.
        
        Returns:
            Age group string
        """
        age = self.get_attribute("age", 0)
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
    
    def get_life_stage(self) -> str:
        """
        Determine life stage based on age and other attributes.
        
        Returns:
            Life stage string
        """
        age = int(self.get_attribute("age", 0))
        has_children = self.get_attribute("has_children", False)
        relationship_status = self.get_attribute("relationship_status", "single")
        
        if age < 18:
            return "adolescent"
        elif age < 25:
            return "young_adult"
        elif age < 35:
            if has_children:
                return "young_parent"
            elif relationship_status in ["married", "partnered"]:
                return "young_coupled"
            else:
                return "young_professional"
        elif age < 55:
            if has_children:
                return "established_parent"
            else:
                return "established_adult"
        else:
            return "mature_adult"
    
    def is_user(self) -> bool:
        """
        Check if this agent is a user (not an assistant).
        
        Returns:
            True if agent is a user, False if assistant
        """
        return self.role == "user"
    
    def is_assistant(self) -> bool:
        """
        Check if this agent is an assistant.
        
        Returns:
            True if agent is an assistant, False if user
        """
        return self.role != "user"
    
    def clone(self) -> 'AgentModel':
        """
        Create a deep copy of this agent model.
        
        Returns:
            New AgentModel instance with identical attributes
        """
        return AgentModel(
            attributes=copy.deepcopy(self._attributes),
            agent_id=None,  # Generate new ID for clone
            role=self.role,
            metadata=copy.deepcopy(self.metadata)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this agent model to a dictionary representation.
        
        Returns:
            Dictionary representation of the agent model
        """
        return {
            "id": self.id,
            "role": self.role,
            "attributes": self._attributes.copy(),
            "metadata": self.metadata.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentModel':
        """
        Create an AgentModel instance from a dictionary representation.
        
        Args:
            data: Dictionary containing agent model data
            
        Returns:
            AgentModel instance
        """
        return cls(
            attributes=data["attributes"],
            agent_id=data.get("id"),
            role=data.get("role", "user"),
            metadata=data.get("metadata", {})
        )
    
    def __str__(self) -> str:
        """String representation of the agent model."""
        return f"AgentModel(name={self.get_formatted_name()}, role={self.role})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent model."""
        return f"AgentModel(id={self.id}, name={self.get_formatted_name()}, role={self.role}, attributes={len(self._attributes)})" 