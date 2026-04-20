"""
event_library.py

Implements the Universal Event Library for the Causal Preference Evolution Framework.
Stores and provides access to event structures representing possible occurrences in a user's life.
"""

import json
import logging
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple, Set, Union

from src.libraries.base_library import BaseLibrary
from src.llm.api_client import LLMClient
from src.libraries.prompts.generate_event_prompt import generate_event_prompt_template
from src.libraries.prompts.generate_event_prompt_batch import generate_batch_event_prompt_template


logger = logging.getLogger(__name__)

class EventLibrary(BaseLibrary[Dict[str, Any]]):
    """
    Universal Event Library for storing and accessing event structures.
    Event structures represent possible occurrences in a user's life, with 
    attributes that may depend on user preferences.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the event library.
        
        Args:
            llm_client: Optional LLM client for augmenting event structures
        """
        super().__init__("Universal Event Library")
        self.llm_client = llm_client
        self._event_categories = {}
        logger.info("Initialized Universal Event Library")
    
    def add_event(
        self, 
        event_id: str, 
        event_structure: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an event structure to the library.
        
        Args:
            event_id: Unique identifier for the event
            event_structure: Dictionary of event attributes
            metadata: Optional metadata about the event
        """
        # Ensure event structure has required fields
        if "attributes" not in event_structure:
            event_structure["attributes"] = {}
            
        if "preference_dependent" not in event_structure:
            event_structure["preference_dependent"] = {}

        if "category" not in event_structure:
            event_structure["category"] = "emotional_social_support"
            
        # Auto-classify attributes if not explicitly set
        event_structure["preference_dependent"] = self._auto_classify_attributes(
            event_structure["attributes"], 
            event_structure.get("preference_dependent", {})
        )
        
        # Add category to metadata if present
        if metadata is not None:
            metadata["category"] = event_structure["category"]
                
            # Track categories for efficient filtering
            category = event_structure["category"]
            if category not in self._event_categories:
                self._event_categories[category] = set()
            self._event_categories[category].add(event_id)
        else:
            metadata = {"category": event_structure["category"]}
        
        for k, v in event_structure.items():
            if k not in ['id', 'category', 'attributes', 'preference_dependent']:
                metadata[k] = v
            
        # Add to library
        self.add_structure(event_id, event_structure, metadata)
        
    def _auto_classify_attributes(
        self, 
        attributes: Dict[str, Any], 
        existing_classification: Dict[str, bool]
    ) -> Dict[str, bool]:
        """
        Auto-classify event attributes as preference-dependent or preference-independent.
        
        Args:
            attributes: Dictionary of event attributes
            existing_classification: Existing classification to preserve
            
        Returns:
            Updated classification dictionary
        """
        # Start with existing classification
        classification = existing_classification.copy()
        
        # Define patterns for preference-independent attributes (fixed scenario context)
        preference_independent_patterns = [
            # Task and scenario context
            'task_type', 'communication_type', 'project_type', 'topic_complexity',
            'subject_area', 'creative_domain', 'urgency_level', 'complexity',
            # Relationships and audience (fixed context)
            'audience', 'relationship', 'recipient', 'stakeholder',
            # Situation context (fixed facts)
            'situation_context', 'deadline', 'constraints', 'requirements',
            # Content and subject matter
            'topic', 'domain', 'field', 'area', 'subject',
            # Technical details (fixed)
            'platform', 'format', 'medium', 'channel', 'technology',
            # External constraints
            'budget', 'timeline', 'scope', 'limitation'
        ]
        
        # Define patterns for preference-dependent attributes (AI interaction style preferences)
        preference_dependent_patterns = [
            # Communication style preferences
            'tone', 'formality', 'humor', 'directness', 'politeness',
            'response_style', 'communication_style', 'approach_style',
            # Detail and explanation preferences
            'detail_level', 'explanation_style', 'teaching_approach',
            'feedback_approach', 'depth', 'verbosity',
            # Collaboration preferences
            'collaboration_style', 'interaction_style', 'working_style',
            'guidance_level', 'autonomy_level', 'support_level',
            # Learning and processing preferences
            'learning_style', 'processing_style', 'thinking_style',
            'example_preference', 'structure_preference'
        ]
        
        # Classify attributes that haven't been explicitly classified
        for attr_name in attributes:
            if attr_name not in classification:
                attr_lower = attr_name.lower()
                
                # Check if it matches preference-independent patterns
                is_preference_independent = any(
                    pattern in attr_lower for pattern in preference_independent_patterns
                )
                
                # Check if it matches preference-dependent patterns
                is_preference_dependent = any(
                    pattern in attr_lower for pattern in preference_dependent_patterns
                )
                
                if is_preference_independent and not is_preference_dependent:
                    classification[attr_name] = False  # Preference-independent
                elif is_preference_dependent and not is_preference_independent:
                    classification[attr_name] = True   # Preference-dependent
                else:
                    # Default to preference-independent for ambiguous cases
                    # This ensures event facts are preserved by default
                    classification[attr_name] = False
                    logger.warning(f"Ambiguous attribute '{attr_name}' classified as preference-independent by default")
        
        return classification
    
    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an event structure by ID.
        
        Args:
            event_id: ID of the event to retrieve
            
        Returns:
            Event structure if found, None otherwise
        """
        return self.get_structure(event_id)
    
    def _sample_event(
        self, 
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sample a random event from the library.
        
        Args:
            filter_func: Optional function to filter events
            category: Optional category to filter by
            
        Returns:
            Sampled event structure, or empty dict if no events match
        """
        # If category is specified, restrict to events in that category
        if category is not None:
            category_events = self._event_categories.get(category, set())
            if not category_events:
                logger.warning(f"No events found in category: {category}")
                return {}
                
            # Create a filter function that checks both category and user-provided filter
            def combined_filter(event: Dict[str, Any]) -> bool:
                category_match = event.get("category") == category
                filter_match = filter_func(event) if filter_func else True
                return category_match and filter_match
                
            filter_to_use = combined_filter
        else:
            filter_to_use = filter_func
        
        # Sample an event
        result = self.sample_structure(filter_to_use)
        
        if result is None:
            logger.warning("No matching events found for sampling")
            return {}
            
        event_id, event_structure = result
        
        # Ensure the structure has its ID field set
        if "id" not in event_structure:
            event_structure["id"] = event_id
            
        logger.debug(f"Sampled event {event_id}")
        return event_structure
    
    def sample_event_weighted(
        self,
        category_weights: Optional[Dict[str, float]] = None,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        realistic_weight: float = 0.7  # Weight for realistic categories vs novel categories
    ) -> Dict[str, Any]:
        """
        Sample an event using hybrid approach: weighted realistic categories + diverse novel categories.
        
        Args:
            category_weights: Dictionary mapping categories to weights (probabilities)
            filter_func: Optional function to filter events
            realistic_weight: Probability of sampling from realistic categories vs novel categories
            
        Returns:
            Sampled event structure, or empty dict if no events match
        """
        # Use default weights if none provided (based on Character.AI usage data)
        if category_weights is None:
            category_weights = {
                "emotional_social_support": 0.2638,           # 26.38% - advice, comfort, emotional guidance
                "romantic_interaction": 0.2234,               # 22.34% - romantic conversations, dating scenarios
                "risky_dark_roleplay": 0.1009,                # 10.09% - intense, complex roleplay scenarios
                "collaborative_storytelling": 0.2557,         # 25.57% - collaborative writing, character development
                "philosophical_moral_inquiry": 0.0753,        # 7.53% - ethics, meaning, moral discussions
                "critical_debates_strategic_analysis": 0.0808 # 8.08% - debates, strategic thinking, analysis
            }
        
        # Get all available categories
        all_categories = self.get_event_categories()
        
        # Separate realistic (weighted) categories from novel categories
        realistic_categories = [cat for cat in all_categories 
                              if cat in category_weights and len(self._event_categories.get(cat, set())) > 0]
        novel_categories = [cat for cat in all_categories 
                           if cat not in category_weights and len(self._event_categories.get(cat, set())) > 0]
        
        # Decide whether to sample from realistic or novel categories
        if realistic_categories and novel_categories:
            use_realistic = random.random() < realistic_weight
        elif realistic_categories:
            use_realistic = True
        elif novel_categories:
            use_realistic = False
        else:
            logger.warning("No events found in any categories, falling back to uniform sampling")
            return self._sample_event(filter_func)
        
        if use_realistic:
            # Sample from realistic categories using weights
            available_weights = [category_weights[cat] for cat in realistic_categories]
            total_weight = sum(available_weights)
            
            if total_weight == 0:
                logger.warning("All realistic category weights are zero, falling back to uniform sampling")
                return self._sample_event(filter_func)
                
            # Normalize weights
            normalized_weights = [w / total_weight for w in available_weights]
            
            # Sample a category based on weights
            selected_category = random.choices(realistic_categories, weights=normalized_weights)[0]
            logger.debug(f"Sampled from realistic category: {selected_category}")
        else:
            # Sample from novel categories uniformly
            selected_category = random.choice(novel_categories)
            logger.debug(f"Sampled from novel category: {selected_category}")
        
        # Sample an event from the selected category
        return self._sample_event(filter_func, selected_category)
    
    def get_event_categories(self) -> List[str]:
        """
        Get all event categories in the library.
        
        Returns:
            List of category names
        """
        return list(self._event_categories.keys())
    
    def get_events_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all events in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary mapping event IDs to event structures
        """
        if category not in self._event_categories:
            return {}
            
        result = {}
        for event_id in self._event_categories[category]:
            event_structure = self.get_structure(event_id)
            if event_structure is not None:
                # Ensure the structure has its ID field set
                if "id" not in event_structure:
                    event_structure["id"] = event_id
                result[event_id] = event_structure
                
        return result
    
    def generate_synthetic_event(
        self, 
        category: Optional[str] = None,
        base_attributes: Optional[Dict[str, Any]] = None,
        user: Optional['UserModel'] = None  # Add user parameter for social graph access
    ) -> Dict[str, Any]:
        """
        Generate a synthetic event structure using LLM.
        
        Args:
            category: Optional event category
            base_attributes: Optional base attributes to include
            user: Optional user model to access social graph for relevant scenarios
            
        Returns:
            Generated event structure
            
        Raises:
            ValueError: If LLM client is not available
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for synthetic event generation")
            
        # Start with base attributes or empty dict
        event_structure = {
            "attributes": {},
            "preference_dependent": {}
        }
        
        if base_attributes:
            event_structure["attributes"].update(base_attributes)
            
        if category:
            event_structure["category"] = category

        # Determine if we should include social graph context
        include_social_context = False
        social_graph_info = ""
        
        if user and hasattr(user, 'get_attribute') and user.get_attribute('social_graph'):
            # Only include social context for certain categories and with 25% probability
            social_relevant_categories = {
                'emotional_social_support', 'relationship_guidance', 'productivity_assistance', 
                'creative_storytelling'  # collaborative projects
            }
            
            if (not category or category in social_relevant_categories) and random.random() < 0.25:
                include_social_context = True
                social_graph = user.get_attribute('social_graph', {})
                
                if social_graph and 'nodes' in social_graph and 'edges' in social_graph:
                    # Extract a few relevant social connections
                    nodes = social_graph['nodes']
                    edges = social_graph['edges']
                    
                    # Sample a few people from the social graph
                    node_names = list(nodes.keys())
                    if len(node_names) > 1:  # Exclude the user themselves
                        other_people = [name for name in node_names if name != user.get_formatted_name()]
                        if other_people:
                            sampled_people = random.sample(other_people, min(3, len(other_people)))
                            
                            social_graph_info = f"""
        
        OPTIONAL SOCIAL CONTEXT: The user has the following people in their social/professional network who could potentially be relevant to some scenarios:
        """
                            for person in sampled_people:
                                description = nodes.get(person, "No description")
                                social_graph_info += f"- {person}: {description}\n"
                                
                            # Find relevant relationships
                            relevant_relations = []
                            for edge_id, edge_info in edges.items():
                                if isinstance(edge_info, dict) and 'Nodes' in edge_info:
                                    edge_nodes = edge_info['Nodes']
                                    if any(person in edge_nodes for person in sampled_people):
                                        relation = edge_info.get('Relation', 'Connected')
                                        relevant_relations.append(f"- {' & '.join(edge_nodes)}: {relation}")
                            
                            if relevant_relations:
                                social_graph_info += f"\nSome relationships: {'; '.join(relevant_relations[:2])}\n"
                                
                            social_graph_info += """
        NOTE: Social context is OPTIONAL - only use if it naturally fits the scenario. Many scenarios should remain individual-focused (learning, personal reflection, technical help, etc.). When social context IS relevant, you could create scenarios like:
        - Seeking advice about a workplace conflict with a colleague
        - Getting help planning a group event or project
        - Asking for perspective on a relationship issue
        - Drafting messages to reconnect with someone
        - Collaborative creative projects with others
        """
            
        # Add category instruction if specified
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
            
        # Add base attributes instruction if provided
        if base_attributes:
            base_attr_str = json.dumps(base_attributes, indent=2)
            base_attributes_instruction = f"Include these base attributes: {base_attr_str}"
        else:
            base_attributes_instruction = ""
            
        prompt = generate_event_prompt_template.format(
            category_instruction=category_instruction,
            social_graph_info=social_graph_info,
            base_attributes_instruction=base_attributes_instruction
        )
        
        # Generate event structure using LLM
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
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
            
            # if "id" not in generated_event and "category":
            #     event_id = f"{category}_{self.get_structure_count() + 1}"
            #     generated_event["id"] = event_id
            
            # self.add_event(generated_event["id"], generated_event)
                
            logger.debug("Generated synthetic event structure using LLM")
            return generated_event
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            # Return basic structure if parsing fails
            return event_structure
    
    def batch_generate_events(
        self, 
        categories: List[str], 
        count_per_category: int = 10,
        user: Optional['UserModel'] = None  # Add user parameter
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a batch of synthetic events across multiple categories.
        
        Args:
            categories: List of event categories to generate
            count_per_category: Number of events to generate per category
            user: Optional user model to access social graph for relevant scenarios
            
        Returns:
            Dictionary mapping categories to lists of generated event structures
            
        Raises:
            ValueError: If LLM client is not available
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for batch event generation")
            
        generated_events = {}
        
        for category in categories:
            category_events = []
            
            # Determine if we should include social graph context for this category
            social_graph_info = ""
            if user and hasattr(user, 'get_attribute') and user.get_attribute('social_graph'):
                # Only include social context for certain categories and with 30% probability for batch generation
                social_relevant_categories = {
                    'emotional_social_support', 'relationship_guidance', 'productivity_assistance', 
                    'creative_storytelling'  # collaborative projects
                }
                
                if category in social_relevant_categories and random.random() < 0.3:
                    social_graph = user.get_attribute('social_graph', {})
                    
                    if social_graph and 'nodes' in social_graph and 'edges' in social_graph:
                        # Extract a few relevant social connections
                        nodes = social_graph['nodes']
                        edges = social_graph['edges']
                        
                        # Sample a few people from the social graph
                        node_names = list(nodes.keys())
                        if len(node_names) > 1:  # Exclude the user themselves
                            other_people = [name for name in node_names if name != user.get_formatted_name()]
                            if other_people:
                                sampled_people = random.sample(other_people, min(3, len(other_people)))
                                
                                social_graph_info = (
                                    "OPTIONAL SOCIAL CONTEXT: The user has the following people in their "
                                    "social/professional network who could potentially be relevant to some scenarios:\n"
                                    )
                                for person in sampled_people:
                                    description = nodes.get(person, "No description")
                                    social_graph_info += f"- {person}: {description}\n"
                                    
                                # Find relevant relationships
                                relevant_relations = []
                                for edge_id, edge_info in edges.items():
                                    if isinstance(edge_info, dict) and 'Nodes' in edge_info:
                                        edge_nodes = edge_info['Nodes']
                                        if any(person in edge_nodes for person in sampled_people):
                                            relation = edge_info.get('Relation', 'Connected')
                                            relevant_relations.append(f"- {' & '.join(edge_nodes)}: {relation}")
                                
                                if relevant_relations:
                                    social_graph_info += f"\nSome relationships: {'; '.join(relevant_relations[:2])}\n"
                                    
                                social_graph_info += (
            "NOTE: Social context is OPTIONAL - only use when it naturally fits the scenario. "
            "Most scenarios should remain individual-focused. "
            "When social context IS relevant, you could create scenarios involving the people above."
                                )
            
            # Generate a batch prompt for this category
            batch_prompt = generate_batch_event_prompt_template.format(
                social_graph_info=social_graph_info,
                count_per_category=count_per_category,
                category=category
            )
            
            response = self.llm_client.generate(batch_prompt, response_format="json")
            
            try:
                batch_events = json.loads(response["response_text"])["events"]
                
                # Validate and process each event
                for event in batch_events:
                    # Ensure required fields exist
                    if "attributes" not in event:
                        event["attributes"] = {}
                        
                    if "preference_dependent" not in event:
                        event["preference_dependent"] = {}
                        
                    # Ensure category is set
                    if "category" not in event:
                        event["category"] = category
                    
                    # if "id" not in event and "category":
                    #     event_id = f"{category}_{self.get_structure_count() + 1}"
                    #     event["id"] = event_id
                        
                    category_events.append(event)
                    
            except json.JSONDecodeError:
                logger.error(f"LLM response for category {category} could not be parsed as JSON")
                # Fallback: generate events one by one
                for i in range(count_per_category):
                    event = self.generate_synthetic_event(category=category, user=user)
                    category_events.append(event)
            
            generated_events[category] = category_events
            logger.info(f"Generated {len(category_events)} events for category {category}")
            
        return generated_events
    
    def generate_diverse_events(self, total_count: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate diverse events across both known and novel categories.
        
        Args:
            total_count: Total number of events to generate
            
        Returns:
            Dictionary mapping categories to lists of generated event structures
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for diverse event generation")
            
        # Generate some novel categories first
        novel_categories = self.generate_novel_categories(5)  # Generate 5 new categories
        
        # Combine with existing categories
        all_categories = self.get_event_categories() + novel_categories
        
        # Distribute events across categories
        events_per_category = max(1, total_count // len(all_categories))
        remaining_events = total_count % len(all_categories)
        
        generated_events = {}
        
        for i, category in enumerate(all_categories):
            # Add extra events to first few categories for remainder
            count_for_category = events_per_category + (1 if i < remaining_events else 0)
            
            if count_for_category > 0:
                category_events = self.batch_generate_events([category], count_for_category)
                generated_events.update(category_events)
                
        return generated_events
    
    def generate_novel_categories(self, count: int = 5) -> List[str]:
        """
        Generate novel event categories that don't exist in the library.
        
        Args:
            count: Number of novel categories to generate
            
        Returns:
            List of novel category names
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for novel category generation")
            
        existing_categories = self.get_event_categories()
        
        prompt = f"""
        Generate {count} completely NEW and diverse event categories for AI assistant interactions.
        
        Existing categories to avoid duplicating:
        {existing_categories}
        
        Create categories that represent realistic scenarios where people seek AI assistance, but are NOT covered by existing categories.
        
        Examples of diverse novel categories could include:
        - pet_care_guidance, gardening_advice, home_improvement_help, fashion_styling, 
        - music_composition, photography_tips, art_creation, dance_instruction,
        - meditation_guidance, spiritual_exploration, astrology_discussion, 
        - sports_coaching, gaming_strategies, board_game_rules, puzzle_solving,
        - genealogy_research, historical_exploration, science_experiments, 
        - invention_brainstorming, business_planning, investment_strategies,
        - event_planning, party_organization, gift_suggestions, interior_design,
        - language_translation, cultural_etiquette, travel_safety, local_customs,
        - automotive_repair, electronics_troubleshooting, appliance_maintenance,
        - sustainable_living, environmental_awareness, climate_action, conservation_tips,
        - elderly_care, disability_support, accessibility_planning, inclusive_design
        
        Be creative and think of scenarios that are distinct from existing categories!
        
        Return ONLY a JSON object:
        {{
          "categories": ["category1", "category2", "category3", ...]
        }}
        """
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            generated = json.loads(response["response_text"])
            
            if isinstance(generated, dict) and "categories" in generated:
                novel_categories = generated["categories"][:count]  # Limit to requested count
                logger.info(f"Generated {len(novel_categories)} novel categories: {novel_categories}")
                return novel_categories
            else:
                logger.error("Invalid response format from LLM for novel categories")
                return []
                
        except json.JSONDecodeError:
            logger.error("LLM response could not be parsed as JSON for novel categories")
            return []
    
    def register_event_template(self, template_id: str, template: Dict[str, Any]) -> None:
        """
        Register an event template for future instantiation.
        
        Args:
            template_id: Unique identifier for the template
            template: Template structure with placeholders
        """
        if "metadata" not in template:
            template["metadata"] = {}
            
        template["metadata"]["is_template"] = True
        self.add_structure(template_id, template, template.get("metadata"))
        logger.debug(f"Registered event template {template_id}")
    
    def instantiate_from_template(
        self,
        template_id: str,
        attribute_values: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create an event instance from a registered template.
        
        Args:
            template_id: ID of the template to use
            attribute_values: Values to populate template placeholders
            
        Returns:
            Instantiated event structure, or None if template not found
        """
        template = self.get_structure(template_id)
        
        if template is None:
            logger.warning(f"Event template {template_id} not found")
            return None
            
        # Clone the template
        instance = json.loads(json.dumps(template))
        
        # Remove template flag from metadata
        if "metadata" in instance and "is_template" in instance["metadata"]:
            del instance["metadata"]["is_template"]
            
        # Replace placeholders in attributes
        if "attributes" in instance:
            for attr_name, attr_value in instance["attributes"].items():
                if isinstance(attr_value, str) and attr_value.startswith("{{") and attr_value.endswith("}}"):
                    # Extract placeholder name
                    placeholder = attr_value[2:-2].strip()
                    if placeholder in attribute_values:
                        instance["attributes"][attr_name] = attribute_values[placeholder]
        
        logger.debug(f"Instantiated event from template {template_id}")
        return instance
    
    def _parse_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse an event structure from its serialized form.
        
        Args:
            data: Serialized event data
            
        Returns:
            Parsed event structure
        """
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data
    
    def _serialize_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize an event structure to a dictionary.
        
        Args:
            structure: Event structure to serialize
            
        Returns:
            Serialized event data
        """
        return structure

    def add_structure(self, structure_id: str, structure: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a structure to the library.
        
        Args:
            structure_id: Unique identifier for the structure
            structure: The structure to add
            metadata: Optional metadata about the structure
        """
        self._structures[structure_id] = structure
        self._metadata[structure_id] = metadata or {}
        if "category" in structure:
            category = structure["category"]
            if category not in self._event_categories:
                self._event_categories[category] = set()
            self._event_categories[category].add(structure_id)
        logger.debug(f"Added structure {structure_id} to {self.library_name}")

    def generate_and_augment_categories(self, existing_categories: Optional[List[str]] = None) -> List[str]:
        """
        Generate new event categories or augment existing ones using LLM.
        Ensures semantic uniqueness from existing categories.
        
        Args:
            existing_categories: Optional list of existing categories to augment
            
        Returns:
            List of all categories (existing + new)
            
        Raises:
            ValueError: If LLM client is not available
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for category generation")

        # Base prompt for category generation
        prompt = """
        You are a domain expert designing a comprehensive taxonomy of event categories for user interactions.
        
        Output a valid parsable JSON object containing only a list of major event categories.
        Each category should:
        - Have a distinguishing and easily understandable name
        - Be broad but distinct from others
        - Focus on a major type of user activity or interaction
        - Be semantically unique (not just different wording for the same concept)
        
        Constraints:
        1. Generate at least 20 distinct category names
        2. The categories should be on the same abstract level, cannot be a subcategory of another
        3. Make sure the categories are semantically different, not just different phrasings
        4. Try NOT to use the format "A and B" unless absolutely necessary
        
        {existing_categories_instruction}
        
        Return ONLY a valid JSON object with the following structure:
        {{
          "categories": ["Category1", "Category2", "Category3", ...]
        }}
        """
        
        # Add existing categories instruction if provided
        if existing_categories:
            categories_str = json.dumps(existing_categories)
            existing_categories_instruction = f"""
            Here are the existing categories that need to be augmented:
            {categories_str}
            
            Please:
            1. Review these categories for quality and consistency
            2. Keep any good categories
            3. Remove or revise problematic categories
            4. Add new categories that are SEMANTICALLY DIFFERENT from existing ones
               (not just different wording for similar concepts)
            5. Ensure no conceptual overlap with existing categories
            """
        else:
            existing_categories_instruction = "Generate a comprehensive list of event categories from scratch."
            
        # Format the prompt
        formatted_prompt = prompt.format(existing_categories_instruction=existing_categories_instruction)
        
        try:
            response = self.llm_client.generate(formatted_prompt, response_format="json")
            generated = json.loads(response["response_text"])
            
            if not isinstance(generated, dict) or "categories" not in generated:
                logger.error("Invalid response format from LLM")
                return existing_categories or []
                
            # Get the generated categories
            new_categories = generated["categories"]
            
            # If we had existing categories, merge them
            if existing_categories:
                # Convert to sets for efficient operations
                existing_set = set(existing_categories)
                new_set = set(new_categories)
                
                # Keep all unique categories
                all_categories = list(existing_set.union(new_set))
                
                # Log changes
                added = new_set - existing_set
                removed = existing_set - new_set
                if added:
                    logger.info(f"Added new categories: {added}")
                if removed:
                    logger.info(f"Removed categories: {removed}")
                    
                return all_categories
            else:
                logger.info(f"Generated {len(new_categories)} event categories")
                return new_categories
                
        except Exception as e:
            logger.error(f"Failed to generate/augment categories: {e}")
            return existing_categories or []

    # def get_attribute_classification_guide(self) -> Dict[str, Any]:
    #     """
    #     Get a guide for classifying event attributes.
        
    #     Returns:
    #         Dictionary with classification guidance
    #     """
    #     return {
    #         "preference_independent": {
    #             "description": "Fixed ground truth about the event that exists regardless of user preferences",
    #             "examples": [
    #                 "venue location (indoor/outdoor)",
    #                 "event date and time",
    #                 "entry fee or ticket price",
    #                 "performer or speaker names",
    #                 "duration and schedule",
    #                 "venue capacity",
    #                 "age restrictions",
    #                 "dress code requirements",
    #                 "event format (workshop, concert, etc.)"
    #             ],
    #             "principle": "These are facts about the event that exist independently of any user"
    #         },
    #         "preference_dependent": {
    #             "description": "Aspects that depend on user choices, participation, or personal approach",
    #             "examples": [
    #                 "level of participation (active vs passive)",
    #                 "what to bring or prepare",
    #                 "who to attend with",
    #                 "personal goals for the event",
    #                 "focus areas or interests",
    #                 "transportation choice",
    #                 "accommodation preferences",
    #                 "engagement style"
    #             ],
    #             "principle": "These are aspects controlled by or dependent on the specific user"
    #         }
    #     }

    def __len__(self):
        return len(self._structures)