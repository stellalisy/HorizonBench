"""
event_sampler.py

Implements event sampling for the Causal Preference Evolution Framework.
Samples events based on user characteristics, preferences, and event history.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import math

from src.causal_framework.models.user_model import UserModel
from src.causal_framework.models.event_model import EventModel
from src.causal_framework.models.preference_model import PreferenceModel
from src.libraries.event_library import EventLibrary
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class EventSampler:
    """
    Samples events based on user characteristics, preferences, and event history.
    Implements conditional probability to determine the most likely next event.
    """
    
    def __init__(
        self, 
        event_library: EventLibrary, 
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the event sampler.
        
        Args:
            event_library: Library of event structures
            llm_client: Client for accessing LLM services
            config: Optional configuration parameters
        """
        self.event_library = event_library
        self.llm_client = llm_client
        
        # Set default configuration or use provided config
        self.config = config or {}
        
        # Set defaults for any missing keys
        defaults = {
            "max_context_events": 10,  # Number of previous events to consider
            "recency_weight": 0.7,    # Weight for recency in event probability
            "category_transition_smoothing": 0.3,  # Smoothing for category transitions
            "random_seed": 42,
            "realistic_category_weight": 0.7,  # Weight for realistic categories vs novel categories
            "category_weights": {
                "emotional_social_support": 0.2638,           # 26.38% - advice, comfort, emotional guidance
                "romantic_interaction": 0.2234,               # 22.34% - romantic conversations, dating scenarios
                "risky_dark_roleplay": 0.1009,                # 10.09% - intense, complex roleplay scenarios
                "collaborative_storytelling": 0.2557,         # 25.57% - collaborative writing, character development
                "philosophical_moral_inquiry": 0.0753,        # 7.53% - ethics, meaning, moral discussions
                "critical_debates_strategic_analysis": 0.0808 # 8.08% - debates, strategic thinking, analysis
            }
        }
        
        # Update defaults with provided config
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
            
        random.seed(self.config["random_seed"])
        
        # Cache for category transition probabilities
        self._category_transition_cache = {}
        
        logger.info("EventSampler initialized with AI assistant interaction categories")
    
    def sample_event(
        self,
        user: UserModel,
        preferences: Dict[str, PreferenceModel],
        previous_events: List[EventModel],
        temporal_pref_id: str,
        selected_preference: PreferenceModel,
        date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Sample the next event based on user, preferences, and event history.
        
        Args:
            user: User model
            preferences: Dictionary mapping preference IDs to preference models
            previous_events: List of previous events
            date: Optional date for the event
            
        Returns:
            Sampled event structure
        """
        # Step 1: prepare additional context

        # Get or generate event date
        event_date = date or datetime.now()
        
        # Limit context to most recent events
        context_events = previous_events[-self.config["max_context_events"]:] if previous_events else []
        
        # Use weighted sampling to get a realistic or novel event
        realistic_weight = self.config.get("realistic_category_weight", 0.7)
        category_weights = user.metadata['event_category_weights'] if 'event_category_weights' in user.metadata else self.config.get("category_weights", {})

        # Step 2: sample relevant event category
        likely_category = self._determine_likely_category(user, preferences, context_events, event_date, realistic_weight, category_weights, temporal_pref_id, selected_preference)

        if temporal_pref_id:
            sampled_event, metadata = self._generate_event_from_preference(
                user=user,
                context_events=context_events,
                current_date=date,
                temporal_pref_id=temporal_pref_id,
                selected_preference=selected_preference,
                category=likely_category
            )
            event_id = f"{likely_category}_{self.event_library.get_structure_count() + 1}"
            self.event_library.add_event(event_id, sampled_event, metadata)
        else:
            candidate_events = self.event_library.get_events_by_category(likely_category)
            if not candidate_events:
                logger.warning(f"No events found in category {likely_category}, generating synthetic event")
                sampled_event = self.event_library.generate_synthetic_event(category=likely_category)
                event_id = f"{likely_category}_{self.event_library.get_structure_count() + 1}"
                self.event_library.add_event(event_id, sampled_event)
            else:
                # Calculate event probabilities using LLM
                sampled_event = self._calculate_event_probabilities(
                    candidate_events,
                    user,
                    preferences,
                    context_events,
                    event_date
                )   
            # If LLM-based sampling fails, fall back to random selection
            if not sampled_event:
                logger.warning("LLM-based event sampling failed, falling back to random selection")
                event_id = random.choice(list(candidate_events.keys()))
                sampled_event = candidate_events[event_id]
                if "id" not in sampled_event:
                    sampled_event["id"] = event_id
        
        logger.debug(f"Sampled event: {sampled_event.get('name', 'Unknown')} (category: {sampled_event.get('category', 'Unknown')})")
        return sampled_event
    
    def _determine_likely_category(
        self,
        user: UserModel,
        preferences: Dict[str, PreferenceModel],
        context_events: List[EventModel],
        date: datetime,
        realistic_weight: float,
        category_weights: Dict[str, float],
        temporal_pref_id: str,
        selected_preference: PreferenceModel,
    ) -> str:
        """
        Determine the most likely event category based on user, preferences, and context.
        
        Args:
            user: User model
            preferences: Dictionary of preference models
            context_events: List of recent events
            date: Event date
            realistic_weight: Weight for realistic vs novel categories
            category_weights: Base category weights
            temporal_pref_id: Temporal preference ID
            selected_preference: Selected preference model
            days_since: Days since last event
            
        Returns:
            Most likely event category
        
        1. decide whether use new category or realistic category

        2. If new category, prompt the model to generate 5 new categories different from category_weights keys based on this user's info, 
        each with probability, then sample a category
        3. If old category, prompt the model to update the category_weights probability based on this user's info, then sample from this 
        list
        """

        past_events_str = ""
        for past_event in context_events:
            info = {
                "event_name": past_event.name,
                "event_category": past_event.category,
                "event_info": past_event.get_all_attributes(),
                "event_date": past_event.date.isoformat() if past_event.date else None,
            }
            past_events_str += json.dumps(info, indent=2)

        # Calculate days since last event
        if context_events:
            most_recent_event = max(context_events, key=lambda e: e.date or datetime.min)
            days_delta = (date - most_recent_event.date).days if most_recent_event.date else 0
        else:
            days_delta = 0

        context=f"""User Information - {user.get_formatted_name()}:
{json.dumps(user.get_all_attributes(), indent=2)}

Past Events: 
{past_events_str}

Days Since Last Event: {days_delta}

TASK:
Based on this user's demographics, interests, personality traits, and life stage, as well as the past events that have happened to them, predict how their likelihood to engage with each category differs from the base probabilities.

Consider factors like:
- Age and life stage preferences
- Interest alignment with category themes
- Personality trait compatibility
- Occupation/education relevance
- Cultural and social factors

"""
        if temporal_pref_id:
            context += f"IMPORTANT: The next event should be designed so that the assistant's knowledge on the user's preferences on {temporal_pref_id} can be used or tested. The preference of interest: {json.dumps(selected_preference.to_dict(), indent=2)}"
        
        # Step 1: Decide whether to use new category or realistic category
        use_realistic = random.random() > (1 - realistic_weight) / math.log(len(category_weights))
        
        if use_realistic:
            # Step 3: Use realistic categories with updated probabilities
            updated_weights = self._update_realistic_category_weights(context, category_weights)
            categories = list(updated_weights.keys())
            weights = list(updated_weights.values())
            user.metadata['event_category_weights'] = updated_weights
        else:
            # Step 2: Generate new categories
            new_categories = self._generate_new_categories(context, category_weights.keys())
            categories = list(new_categories.keys())
            weights = list(new_categories.values())
        
        # Sample category based on probabilities
        if categories and weights:
            return random.choices(categories, weights=weights, k=1)[0]
        else:
            # Fallback to default category
            return list(category_weights.keys())[0] if category_weights else "emotional_social_support"
    
    def _update_realistic_category_weights(
        self,
        context: str,
        category_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Update realistic category weights based on user info and temporal factors."""

        
        try:
            prompt = f"""
Update category probabilities based on user profile and temporal factors.

{context}

Prior probability: {json.dumps(category_weights, indent=2)}

Return updated probabilities as JSON:
{{"category1": 0.3, "category2": 0.2, ...}}
"""
            response = self.llm_client.generate(prompt, response_format="json")
            updated_weights = json.loads(response["response_text"])
            
            # Validate and normalize
            valid_weights = {k: v for k, v in updated_weights.items() if k in category_weights and isinstance(v, (int, float))}
            if valid_weights:
                total = sum(valid_weights.values())
                return {k: v/total for k, v in valid_weights.items()}
        except Exception as e:
            logger.warning(f"Failed to update category weights: {e}")
        
        return category_weights
    
    def _generate_new_categories(
        self,
        context: str,
        existing_categories: Set[str]
    ) -> Dict[str, float]:
        """Generate 5 new categories different from existing ones."""
        try:
            prompt = f"""Your job is to figure our what's a good event category for the user described below. 

{context}

Generate 5 new event categories different from: {list(existing_categories)}. Remember, the list of new event categories needs to be fundamentally different, minor rewrites with little semantic differences DO NOT count.

Return as JSON with probabilities:
{{"new_category1": 0.3, "new_category2": 0.2, ...}}
"""
            response = self.llm_client.generate(prompt, response_format="json")
            new_categories = json.loads(response["response_text"])
            
            # Validate and normalize
            valid_categories = {k: v for k, v in new_categories.items() 
                              if k not in existing_categories and isinstance(v, (int, float))}
            if valid_categories:
                total = sum(valid_categories.values())
                return {k: v/total for k, v in valid_categories.items()}
        except Exception as e:
            logger.warning(f"Failed to generate new categories: {e}")
        
        # Fallback: return one new category
        return {"novel_interaction": 1.0}
    

    

    def _generate_event_from_preference(
        self,
        user: UserModel,
        context_events: List[EventModel],
        current_date: datetime,
        temporal_pref_id: str,
        selected_preference: PreferenceModel,
        category: str,
    ) -> EventModel:
        """
        Generate an event specifically relevant to a given preference.
        
        Args:
            user_agent: User agent model
            preference: Preference to generate event for
            current_date: Current date
            is_temporal_recall: Whether this is for temporal preference recall
            
        Returns:
            EventModel instance relevant to the preference
        """
        # Get preference structure details
        # pref_structure = self.preference_library.get_structure(preference.structure_id)
        
        # Create prompt for generating relevant event
        past_events_str = ""
        for past_event in context_events:
            info = {
                "event_name": past_event.name,
                "event_category": past_event.category,
                "event_info": past_event.get_all_attributes(),
                "event_date": past_event.date.isoformat() if past_event.date else None,
            }
            past_events_str += json.dumps(info, indent=2)

        if context_events:
            most_recent_event = max(context_events, key=lambda e: e.date or datetime.min)
            days_delta = (current_date - most_recent_event.date).days if most_recent_event.date else 0
        else:
            days_delta = 0

        if category:
            category_instruction = f"The scenario should be in the category: {category}"
        else:
            category_instruction = """Choose an appropriate category for the scenario. You can use one of these common categories based on real AI usage patterns:
            - emotional_social_support, creative_storytelling, relationship_guidance, productivity_assistance, entertainment_interaction, philosophical_inquiry, analytical_discussion
            
            OR create a NEW category that reflects other realistic AI assistant usage scenarios such as:
            - educational_tutoring, technical_support, health_wellness, financial_planning, travel_planning, 
            - legal_advice, parenting_guidance, career_coaching, hobby_exploration, language_learning,
            - cooking_assistance, fitness_coaching, mental_health_support, creative_writing, academic_research,
            - personal_development, decision_making, conflict_resolution, time_management, goal_setting,
            - social_skills_practice, cultural_exploration, news_analysis, product_recommendations, etc.
            
            Be creative and think about diverse scenarios where people realistically seek AI assistance."""

        prompt = f"""Your Task: Generate an event that would be specifically relevant to this user's preference.

User Information - {user.get_formatted_name()}:
{json.dumps(user.get_all_attributes(), indent=2)}

Past Events: 
{past_events_str}

Days Since Last Event: {days_delta}

{category_instruction}
        
Preference of Interest: {json.dumps(selected_preference.to_dict(), indent=2)}
        
{"This is for temporal preference recall - the user expressed this preference a while ago." if temporal_pref_id else ""}
        
Generate an event that would naturally trigger this preference. The event should:
    - Be realistic and fit the user's life context
    - Directly involve the preference domain
    - Create a natural opportunity for the preference to be expressed or recalled
        
Return a JSON object with this structure:
{{
    "name": "Event name",
    "category": "event_category",
    "description": "Extremely concise description of the event",
    "attributes": {{
        "attribute1": "value1",
        "attribute2": "value2",
        ...
    }},
    "preference_dependent": {{
        "attribute_name_1": true/false,
        "attribute_name_1": true/false,
        ...
    }}
}}
"""
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            generated_event = json.loads(response["response_text"])
            
            # Ensure required fields exist
            if "attributes" not in generated_event:
                generated_event["attributes"] = {}
                
            if "preference_dependent" not in generated_event:
                generated_event["preference_dependent"] = {}
                
            # Ensure category is set
            if "category" in generated_event and not category:
                category = generated_event["category"]
                
            if "category" not in generated_event and category:
                generated_event["category"] = category
            
            metadata = {
                "generated_for_temporal_preference": temporal_pref_id is not None,
                "temporal_recall": temporal_pref_id is not None,
            }
            
            logger.debug("Generated synthetic event structure using LLM")
            return generated_event, metadata

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to generate temporal preference event: {e}")
            return None
        


    def _calculate_event_probabilities(
        self,
        candidate_events: Dict[str, Dict[str, Any]],
        user: UserModel,
        preferences: Dict[str, PreferenceModel],
        context_events: List[EventModel],
        date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate probabilities for candidate events using both systematic compatibility scoring
        and LLM-based inference.
        
        Args:
            candidate_events: Dictionary of candidate event structures
            user: User model
            preferences: Dictionary of preference models
            context_events: List of recent events
            date: Event date
            
        Returns:
            Selected event structure or None if selection fails
        """
        # If too many candidates, sample a subset
        if len(candidate_events) > 10:
            sample_size = min(10, len(candidate_events))
            sampled_ids = random.sample(list(candidate_events.keys()), sample_size)
            candidate_subset = {eid: candidate_events[eid] for eid in sampled_ids}
        else:
            candidate_subset = candidate_events
            
        # Calculate systematic compatibility scores
        compatibility_scores = self._calculate_preference_event_compatibility(
            candidate_subset, preferences
        )
        
        # Calculate LLM-based scores for additional context
        llm_scores = self._calculate_llm_event_probabilities(
            candidate_subset, user, preferences, context_events, date
        )
        
        # Combine scores (70% compatibility, 30% LLM insight)
        final_scores = {}
        for event_id in candidate_subset:
            compat_score = compatibility_scores.get(event_id, 0.1)  # Minimum baseline
            llm_score = llm_scores.get(event_id, 0.1)  # Minimum baseline
            
            # Weighted combination
            final_scores[event_id] = (0.7 * compat_score) + (0.3 * llm_score)
        
        # Normalize final scores
        total = sum(final_scores.values())
        if total > 0:
            final_scores = {k: v / total for k, v in final_scores.items()}
        else:
            # Fallback to uniform distribution
            final_scores = {k: 1.0 / len(candidate_subset) for k in candidate_subset}
        
        # Sample event based on probabilities
        event_ids = list(final_scores.keys())
        weights = list(final_scores.values())
        
        selected_id = random.choices(event_ids, weights=weights, k=1)[0]
        if "id" not in candidate_subset[selected_id]:
            candidate_subset[selected_id]["id"] = selected_id
        return candidate_subset[selected_id]
    
    def _calculate_preference_event_compatibility(
        self,
        candidate_events: Dict[str, Dict[str, Any]],
        preferences: Dict[str, PreferenceModel]
    ) -> Dict[str, float]:
        """
        Calculate systematic compatibility scores between user preferences and events using LLM.
        
        Args:
            candidate_events: Dictionary of candidate event structures
            preferences: Dictionary of preference models
            
        Returns:
            Dictionary mapping event IDs to compatibility scores (0-1)
        """
        if not preferences:
            # No preferences, return neutral scores
            return {event_id: 0.5 for event_id in candidate_events}
        
        # Batch process events for efficiency
        compatibility_scores = {}
        
        # Create a comprehensive compatibility assessment prompt
        prompt = self._create_compatibility_assessment_prompt(candidate_events, preferences)
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            scores = json.loads(response["response_text"])
            
            # Validate and normalize scores
            for event_id in candidate_events:
                if event_id in scores and isinstance(scores[event_id], (int, float)):
                    # Ensure score is in [0, 1] range
                    compatibility_scores[event_id] = max(0.0, min(1.0, float(scores[event_id])))
                else:
                    # Default neutral score if not provided or invalid
                    compatibility_scores[event_id] = 0.5
                    
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse compatibility scores: {e}")
            # Fallback to neutral scores
            compatibility_scores = {event_id: 0.5 for event_id in candidate_events}
            
        return compatibility_scores
    
    def _create_compatibility_assessment_prompt(
        self,
        candidate_events: Dict[str, Dict[str, Any]],
        preferences: Dict[str, PreferenceModel]
    ) -> str:
        """
        Create a comprehensive prompt for assessing preference-event compatibility.
        
        Args:
            candidate_events: Dictionary of candidate event structures
            preferences: Dictionary of preference models
            
        Returns:
            Formatted prompt for LLM assessment
        """
        prompt = """
        Assess the compatibility between user preferences and candidate events. 
        For each event, calculate how well it aligns with the user's preferences on a scale of 0.0 to 1.0.
        
        Consider:
        - How well event characteristics match preference values
        - Alignment between preference domains and event categories
        - Specific attribute compatibility (e.g., social preferences vs event size, environmental preferences vs location)
        - Overall fit between user's psychological profile and event experience
        
        User Preferences:
        """
        
        # Add preference details
        for pref_id, pref_model in preferences.items():
            pref_domain = pref_model.metadata.get("domain", "unknown")
            pref_description = pref_model.metadata.get("description", "No description")
            
            prompt += f"\n{pref_id} (Domain: {pref_domain}):\n"
            prompt += f"  Description: {pref_description}\n"
            prompt += "  Attributes:\n"
            
            for attr_name, attr_value in pref_model.get_all_attributes().items():
                prompt += f"    - {attr_name}: {attr_value}\n"
        
        prompt += "\nCandidate Events:\n"
        
        # Add event details
        for event_id, event_struct in candidate_events.items():
            event_name = event_struct.get("name", event_id)
            event_category = event_struct.get("category", "unknown")
            event_description = event_struct.get("metadata", {}).get("description", "No description")
            
            prompt += f"ID: \n{event_id}:\n"
            prompt += f"  Name: {event_name}\n"
            prompt += f"  Category: {event_category}\n"
            prompt += f"  Description: {event_description}\n"
            
            # Add event attributes
            if "attributes" in event_struct:
                prompt += "  Attributes:\n"
                for attr_name, attr_value in event_struct["attributes"].items():
                    prompt += f"    - {attr_name}: {attr_value}\n"
        
        prompt += """
        
        Calculate compatibility scores (0.0 to 1.0) for each event:
        - 0.0-0.2: Very poor fit (conflicts with preferences)
        - 0.2-0.4: Poor fit (doesn't align well)
        - 0.4-0.6: Neutral fit (neither good nor bad)
        - 0.6-0.8: Good fit (aligns well with preferences)
        - 0.8-1.0: Excellent fit (perfect or near-perfect alignment)
        
        Return a JSON object mapping event IDs to compatibility scores:
        {
          "event_id1": 0.75,
          "event_id2": 0.45,
          ...
        }
        
        Base your scores on:
        1. How well event characteristics satisfy user preferences
        2. Whether the event type aligns with preference domains
        3. Specific attribute matches (e.g., social group size, activity level, environment)
        4. Overall experiential fit for this user's psychological profile
        """
        
        return prompt
    
    def _calculate_single_preference_compatibility(
        self,
        pref_attrs: Dict[str, Any],
        pref_domain: str,
        event_attrs: Dict[str, Any],
        event_category: str
    ) -> float:
        """
        Calculate compatibility between a single preference and an event using LLM.
        This method is now deprecated in favor of batch compatibility assessment.
        
        Returns:
            Neutral compatibility score (0.5)
        """
        # This method is deprecated - all compatibility is now calculated via LLM batch processing
        return 0.5
    
    def _check_attribute_compatibility(
        self,
        pref_attr: str,
        pref_value: Any,
        event_attrs: Dict[str, Any]
    ) -> Optional[float]:
        """
        This method is now deprecated in favor of LLM-based compatibility assessment.
        
        Returns:
            None to indicate no hardcoded compatibility rules
        """
        # All compatibility checking is now done via LLM to avoid hardcoded rules
        return None
    
    def _get_preference_stability_weight(self, pref_model: PreferenceModel) -> float:
        """
        Get the stability weight for a preference (more stable preferences have more influence).
        
        Args:
            pref_model: Preference model
            
        Returns:
            Stability weight (0.1 to 1.0)
        """
        # Check if preference has stability metadata
        stability = pref_model.metadata.get("average_stability", 0.7)
        
        # Ensure weight is in reasonable range
        return max(0.1, min(1.0, stability))
    
    def _calculate_llm_event_probabilities(
        self,
        candidate_events: Dict[str, Dict[str, Any]],
        user: UserModel,
        preferences: Dict[str, PreferenceModel],
        context_events: List[EventModel],
        date: datetime
    ) -> Dict[str, float]:
        """
        Calculate event probabilities using LLM (original method, renamed for clarity).
        
        Args:
            candidate_events: Dictionary of candidate event structures
            user: User model
            preferences: Dictionary of preference models
            context_events: List of recent events
            date: Event date
            
        Returns:
            Dictionary mapping event IDs to probability scores
        """
        # Create a prompt for the LLM
        prompt = """
        Based on the user information, preferences, and recent event history, calculate the probability
        of each candidate event being the next event in the user's timeline.
        
        User information:
        """
        
        # Add simplified user info
        user_info = {
            "age": user.get_attribute("age"),
            "gender": user.get_attribute("gender"),
            "interests": user.get_interests()[:5] if user.get_interests() else [],  # Limit to top 5
            "occupation": user.get_attribute("occupation", "Unknown"),
            "life_stage": user.get_life_stage()
        }
        
        prompt += json.dumps(user_info, indent=2)
        
        # Add preferences summary
        prompt += "\n\nUser preferences:\n"
        for pref_id, pref_model in preferences.items():
            prompt += f"- {pref_id}:\n"
            top_attrs = list(pref_model.get_all_attributes().items())[:3]  # Limit to top 3 attributes
            for attr, value in top_attrs:
                prompt += f"  - {attr}: {value}\n"
        
        # Add context events (simplified)
        if context_events:
            prompt += "\n\nRecent event history:\n"
            for event in context_events:
                event_name = event.get_attribute("name", "event")
                if event.date:
                    days_ago = (date - event.date).days
                    if days_ago == 0:
                        time_str = "earlier today"
                    elif days_ago == 1:
                        time_str = "yesterday"
                    else:
                        time_str = f"{days_ago} days ago"
                else:
                    time_str = "recently"
                    
                prompt += f"- {event_name} {time_str}\n"
                
                # Add key attributes
                for attr_name in ["location", "participants", "outcome", "emotional_impact"]:
                    if event.has_attribute(attr_name):
                        prompt += f"  - {attr_name}: {event.get_attribute(attr_name)}\n"
        
        # Add candidate events
        prompt += "\n\nCandidate events:\n"
        for i, (event_id, event_struct) in enumerate(candidate_events.items()):
            prompt += f"{i+1}. ID: {event_id}\n"
            prompt += f"   Name: {event_struct.get('name', event_id)}\n"
            
            # Add key attributes
            if "attributes" in event_struct:
                for attr_name, attr_value in list(event_struct["attributes"].items())[:3]:  # Limit to top 3
                    prompt += f"   {attr_name}: {attr_value}\n"
        
        # Request probabilities
        prompt += """
        Calculate the probability of each candidate event being the next event in the user's timeline.
        Consider how well each event aligns with the user's characteristics, preferences, and recent history.
        
        Return a JSON object mapping event IDs to probabilities (values between 0 and 1 that sum to 1):
        {
          "event_id1": 0.3,
          "event_id2": 0.2,
          ...
        }
        """
        
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            event_probs = json.loads(response["response_text"])
            
            # Validate probabilities
            valid_probs = {}
            for event_id, prob in event_probs.items():
                if event_id in candidate_events and isinstance(prob, (int, float)) and 0 <= prob <= 1:
                    valid_probs[event_id] = prob
            
            # If no valid probabilities, return uniform distribution
            if not valid_probs:
                return {event_id: 1.0 / len(candidate_events) for event_id in candidate_events}
                
            # Normalize probabilities
            total = sum(valid_probs.values())
            if total > 0:
                valid_probs = {k: v / total for k, v in valid_probs.items()}
                
            return valid_probs
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            # Return uniform distribution as fallback
            return {event_id: 1.0 / len(candidate_events) for event_id in candidate_events}