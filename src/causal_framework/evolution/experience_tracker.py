"""
experience_tracker.py

Implements experience-driven change for the Causal Preference Evolution Framework.
Tracks how accumulated experiences affect preference evolution over time.
"""

import logging
import random
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ExperienceTracker:
    """
    Tracks and manages how experiences drive preference changes over time.
    Implements experience-based learning and preference refinement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the experience tracker.
        
        Args:
            config: Optional configuration parameters
        """
        # Set default configuration or update with provided config
        self.config = {
            "first_experience_weight": 0.8,    # Weight for first exposure to an experience
            "experience_decay_rate": 0.7,      # Rate at which additional experiences have diminishing impact
            "recency_half_life_days": 30,      # Half-life for recency effects (in days)
            "max_tracked_experiences": 20,     # Maximum number of experiences to track per preference
            "positive_impact_multiplier": 1.2, # Multiplier for positive experiences
            "negative_impact_multiplier": 1.5, # Multiplier for negative experiences (stronger impact)
            "random_seed": 42                  # Seed for reproducibility
        }
        
        if config:
            self.config.update(config)
            
        random.seed(self.config["random_seed"])
        
        # Store experience records for preferences
        self._experience_records = {}
        
        logger.debug("ExperienceTracker initialized")
    
    def record_experience(
        self,
        preference_id: str,
        event_data: Dict[str, Any],
        experience_date: Optional[datetime] = None,
        experience_impact: Optional[float] = None,
        affected_attributes: Optional[List[str]] = None
    ) -> None:
        """
        Record an experience that could affect a preference.
        
        Args:
            preference_id: ID of the affected preference
            event_data: Data about the event causing the experience
            experience_date: Date of the experience
            experience_impact: Impact value (-1 to 1, negative to positive)
            affected_attributes: List of preference attributes affected
        """
        # Use current time if not provided
        if experience_date is None:
            experience_date = datetime.now()
            
        # Initialize experience record for this preference if needed
        if preference_id not in self._experience_records:
            self._experience_records[preference_id] = []
            
        # Create experience record
        experience = {
            "event_id": event_data.get("id", "unknown"),
            "event_type": event_data.get("name", "unknown"),
            "date": experience_date,
            "impact": experience_impact if experience_impact is not None else self._estimate_impact(event_data),
            "affected_attributes": affected_attributes or [],
            "metadata": {
                "category": event_data.get("category", "unknown"),
                "location": event_data.get("location", "unknown"),
                "emotional_impact": event_data.get("emotional_impact", "neutral")
            }
        }
        
        # Add to records
        self._experience_records[preference_id].append(experience)
        
        # Trim if too many records
        if len(self._experience_records[preference_id]) > self.config["max_tracked_experiences"]:
            # Sort by impact (keep most impactful experiences)
            self._experience_records[preference_id].sort(key=lambda x: abs(x["impact"]), reverse=True)
            self._experience_records[preference_id] = self._experience_records[preference_id][:self.config["max_tracked_experiences"]]
            
        logger.debug(f"Recorded experience for preference {preference_id} from event {experience['event_id']}")
    
    def get_experience_count(self, preference_id: str) -> int:
        """
        Get the number of recorded experiences for a preference.
        
        Args:
            preference_id: ID of the preference
            
        Returns:
            Number of recorded experiences
        """
        return len(self._experience_records.get(preference_id, []))
    
    def calculate_experience_factor(
        self,
        preference_id: str,
        current_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate an experience factor for preference evolution.
        
        Args:
            preference_id: ID of the preference
            current_date: Current date for recency calculations
            
        Returns:
            Experience factor (0-1 scale)
        """
        # Use current time if not provided
        if current_date is None:
            current_date = datetime.now()
            
        # Get experiences for this preference
        experiences = self._experience_records.get(preference_id, [])
        
        # If no experiences, return maximum factor (first experience has full impact)
        if not experiences:
            return 1.0
            
        # Calculate decay factor based on experience count
        decay_rate = self.config["experience_decay_rate"]
        count_factor = decay_rate ** len(experiences)
        
        # Calculate recency factor (more recent experiences have greater impact)
        recency_factors = []
        half_life = self.config["recency_half_life_days"]
        
        for exp in experiences:
            exp_date = exp["date"]
            days_ago = (current_date - exp_date).total_seconds() / (24 * 3600)
            recency_factor = 2 ** (-days_ago / half_life)
            recency_factors.append(recency_factor)
            
        avg_recency = sum(recency_factors) / len(recency_factors) if recency_factors else 0.5
        
        # Calculate impact factor (experiences with stronger impact have greater effect)
        impact_factors = []
        
        for exp in experiences:
            impact = exp["impact"]
            # Apply multipliers for positive/negative experiences
            if impact > 0:
                impact_factor = abs(impact) * self.config["positive_impact_multiplier"]
            else:
                impact_factor = abs(impact) * self.config["negative_impact_multiplier"]
                
            impact_factors.append(min(1.0, impact_factor))
            
        avg_impact = sum(impact_factors) / len(impact_factors) if impact_factors else 0.5
        
        # Combine factors (count, recency, impact)
        experience_factor = 0.4 * count_factor + 0.3 * avg_recency + 0.3 * avg_impact
        
        # Ensure result is in valid range
        experience_factor = max(0.2, min(1.0, experience_factor))
        
        logger.debug(f"Calculated experience factor for preference {preference_id}: {experience_factor:.2f}")
        return experience_factor
    
    def get_attribute_experience_factors(
        self,
        preference_id: str,
        current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate experience factors for individual preference attributes.
        
        Args:
            preference_id: ID of the preference
            current_date: Current date for recency calculations
            
        Returns:
            Dictionary mapping attribute names to experience factors
        """
        # Get experiences for this preference
        experiences = self._experience_records.get(preference_id, [])
        
        # Initialize attribute factors with default value
        attribute_factors = {}
        
        # If no experiences, return empty dictionary
        if not experiences:
            return attribute_factors
            
        # Group experiences by affected attributes
        attribute_experiences = {}
        
        for exp in experiences:
            for attr in exp["affected_attributes"]:
                logger.debug(f"Processing experience for attr: {attr}")
                if attr not in attribute_experiences:
                    attribute_experiences[attr] = []
                    
                attribute_experiences[attr].append(exp)
                
        # Calculate factor for each attribute
        for attr, attr_experiences in attribute_experiences.items():
            # Similar calculation to calculate_experience_factor but for specific attribute
            
            # Calculate decay factor
            decay_rate = self.config["experience_decay_rate"]
            count_factor = decay_rate ** len(attr_experiences)
            
            # Calculate recency factor
            recency_factors = []
            half_life = self.config["recency_half_life_days"]
            
            for exp in attr_experiences:
                exp_date = exp["date"]
                days_ago = (current_date - exp_date).total_seconds() / (24 * 3600) if current_date else 0
                recency_factor = 2 ** (-days_ago / half_life)
                recency_factors.append(recency_factor)
                
            avg_recency = sum(recency_factors) / len(recency_factors) if recency_factors else 0.5
            
            # Calculate impact factor
            impact_factors = []
            
            for exp in attr_experiences:
                impact = exp["impact"]
                # Apply multipliers for positive/negative experiences
                if impact > 0:
                    impact_factor = abs(impact) * self.config["positive_impact_multiplier"]
                else:
                    impact_factor = abs(impact) * self.config["negative_impact_multiplier"]
                    
                impact_factors.append(min(1.0, impact_factor))
                
            avg_impact = sum(impact_factors) / len(impact_factors) if impact_factors else 0.5
            
            # Combine factors
            attribute_factor = 0.4 * count_factor + 0.3 * avg_recency + 0.3 * avg_impact
            
            # Ensure result is in valid range
            attribute_factor = max(0.2, min(1.0, attribute_factor))
            
            attribute_factors[attr] = attribute_factor
            
        return attribute_factors
    
    def _estimate_impact(self, event_data: Dict[str, Any]) -> float:
        """
        Estimate the impact of an event on preferences.
        
        Args:
            event_data: Data about the event
            
        Returns:
            Estimated impact value (-1 to 1, negative to positive)
        """
        # Default to neutral impact
        impact = 0.0
        
        # Extract emotional impact if available
        emotional_impact = event_data.get("emotional_impact", "neutral")
        
        # Convert textual emotional impact to numeric value
        if isinstance(emotional_impact, str):
            if emotional_impact.lower() in ["very positive", "excellent", "amazing"]:
                impact = 0.8
            elif emotional_impact.lower() in ["positive", "good", "enjoyable"]:
                impact = 0.5
            elif emotional_impact.lower() in ["slightly positive", "somewhat good"]:
                impact = 0.2
            elif emotional_impact.lower() in ["neutral", "mixed", "average"]:
                impact = 0.0
            elif emotional_impact.lower() in ["slightly negative", "somewhat bad"]:
                impact = -0.2
            elif emotional_impact.lower() in ["negative", "bad", "unpleasant"]:
                impact = -0.5
            elif emotional_impact.lower() in ["very negative", "terrible", "awful"]:
                impact = -0.8
        elif isinstance(emotional_impact, (int, float)):
            # Ensure value is in range [-1, 1]
            impact = max(-1.0, min(1.0, float(emotional_impact)))
            
        # Add slight randomization
        impact += random.uniform(-0.1, 0.1)
        impact = max(-1.0, min(1.0, impact))  # Ensure still in range
        
        return impact
    
    def get_experience_impact_history(
        self,
        preference_id: str,
        attribute_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the impact history of experiences for a preference.
        
        Args:
            preference_id: ID of the preference
            attribute_name: Optional attribute name to filter by
            
        Returns:
            List of experience impact records
        """
        # Get experiences for this preference
        experiences = self._experience_records.get(preference_id, [])
        
        # Filter by attribute if specified
        if attribute_name is not None:
            experiences = [
                exp for exp in experiences
                if attribute_name in exp["affected_attributes"]
            ]
            
        # Create impact history
        impact_history = []
        
        for exp in sorted(experiences, key=lambda x: x["date"]):
            impact_record = {
                "date": exp["date"],
                "event_type": exp["event_type"],
                "impact": exp["impact"],
                "category": exp["metadata"]["category"]
            }
            
            impact_history.append(impact_record)
            
        return impact_history
    
    def clear_experiences(self, preference_id: Optional[str] = None) -> None:
        """
        Clear experience records.
        
        Args:
            preference_id: Optional preference ID to clear only specific records
        """
        if preference_id is not None:
            if preference_id in self._experience_records:
                del self._experience_records[preference_id]
                logger.debug(f"Cleared experiences for preference {preference_id}")
        else:
            self._experience_records = {}
            logger.debug("Cleared all experience records")