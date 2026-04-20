"""
generation_pipeline.py

Implements the end-to-end generation pipeline for the Causal Preference Evolution Framework.
Orchestrates the entire data generation process, from user sampling to conversation generation.
"""

import os
import json
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Set

from src.causal_framework.models.user_model import UserModel
from src.causal_framework.models.event_model import EventModel
from src.causal_framework.models.preference_model import PreferenceModel
from src.causal_framework.models.agent_model import AgentModel
from src.causal_framework.sampling.event_sampler import EventSampler
from src.causal_framework.sampling.preference_sampler import PreferenceSampler
from src.causal_framework.evolution.preference_evolve_manager import PreferenceEvolveManager
from src.conversation.conversation_generator import ConversationGenerator
from src.libraries.user_library import UserLibrary
from src.libraries.event_library import EventLibrary
from src.libraries.preference_library import PreferenceLibrary
from src.timeline_generator import TimelineGenerator
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class GenerationPipeline:
    """
    Orchestrates the end-to-end data generation process for the Causal Preference Evolution Framework.
    Handles initialization, configuration, and execution of the full pipeline.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        convo_llm_client: LLMClient,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        validation_llm_clients: Optional[List[Tuple[str, LLMClient]]] = None
    ):
        """
        Initialize the generation pipeline.
        
        Args:
            llm_client: Client for accessing LLM services
            convo_llm_client: Client for generating conversations
            output_dir: Directory for saving generated data
            config: Optional configuration parameters
            use_cache: Whether to load and use existing cached libraries
            validation_llm_clients: Optional list of (model_name, LLMClient) tuples for 
                history validation. Multiple models can be used for robust validation.
        """
        self.llm_client = llm_client
        self.convo_llm_client = convo_llm_client
        self.validation_llm_clients = validation_llm_clients
        self.output_dir = output_dir
        self.use_cache = use_cache
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set default configuration or update with provided config
        self.config = {
            "timeline_length_days": 180,  # Default 6 months
            "avg_events_per_week": 3,
            "relevant_preferences_per_event": 3,
            "conversation_probability": 0.7,
            "max_preference_shifts": 5,
            "user_count": 10,
            "preference_domains": ["entertainment", "food", "communication", "productivity", "lifestyle"],
            "category_weights": {
                "social": 0.3, "professional": 0.25, "recreational": 0.2, 
                "educational": 0.1, "health": 0.1, "travel": 0.05
            },
            "random_seed": 42,
            "cache_dir": os.path.join(output_dir, "cache"),
            
            # New fields for single event generation
            "specified_event_category": None,
            "specified_event_description": None,
            "specified_persona_traits": None,
            "specified_assistant_traits": None,

            "verbose_logging": True
        }
        self.config["libraries"] = {
            "user_library_path": os.path.join(self.config["cache_dir"], "users.json"),
            "event_library_path": os.path.join(self.config["cache_dir"], "events.json"),
            "preference_library_path": os.path.join(self.config["cache_dir"], "preferences.json"),
            "append_to_existing": True
        }
        
        if config:
            self.config.update(config)
            
        # Set up logging
        self._setup_logging()
            
        # Create cache directory if enabled
        if self.config.get("use_cache", True):
            os.makedirs(self.config["cache_dir"], exist_ok=True)
            
        # Set random seed
        random.seed(self.config["random_seed"])
        
        # Initialize libraries with the provided llm_client
        self.user_library = self._initialize_user_library()
        self.event_library = self._initialize_event_library()
        self.preference_library = self._initialize_preference_library()
        
        # Initialize components
        self.event_sampler = EventSampler(self.event_library, self.llm_client, self.config)
        self.preference_sampler = PreferenceSampler(self.preference_library, self.llm_client)
        self.preference_evolve_manager = PreferenceEvolveManager(self.llm_client)
        # self.conversation_generator = ConversationGenerator(self.llm_client, self.config)
        
        # Initialize timeline generator with appropriate config
        # timeline_config = {
        #     "timeline_length_days": self.config["timeline_length_days"],
        #     "avg_events_per_week": self.config["avg_events_per_week"],
        #     "relevant_preferences_per_event": self.config["relevant_preferences_per_event"],
        #     "conversation_probability": self.config["conversation_probability"],
        #     "max_preference_shifts": self.config["max_preference_shifts"],
        #     "random_seed": self.config["random_seed"]
        # }
        
        self.timeline_generator = TimelineGenerator(
            user_library=self.user_library,
            event_library=self.event_library,
            preference_library=self.preference_library,
            llm_client=self.llm_client,
            convo_llm_client=self.convo_llm_client,
            config=self.config,
            validation_llm_clients=self.validation_llm_clients
        )
        
        logger.info("GenerationPipeline initialized")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full generation pipeline.
        
        Returns:
            Dictionary with generation statistics
        """
        start_time = time.time()
        
        logger.info("Starting the generation pipeline")
        
        # Prepare libraries
        self._prepare_libraries()
        
        # Generate timelines
        generated_timelines = self._generate_timelines()
        
        # Save results
        self._save_results(generated_timelines)
        
        # Collect statistics
        stats = self._collect_stats(generated_timelines)
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Generation pipeline completed in {elapsed_time:.2f} seconds")
        logger.info(f"Generated {stats['user_count']} user timelines with {stats['event_count']} events and {stats['conversation_count']} conversations")
        
        return stats
    
    def _setup_logging(self) -> None:
        """
        Set up logging for the pipeline.
        """
        # Configure logging
        log_level = logging.DEBUG if self.config["verbose_logging"] else logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_file = os.path.join(self.output_dir, "generation.log")
        
        # Create a file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Add new handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logger.debug("Logging configured")
    
    def _initialize_llm_client(self) -> LLMClient:
        """
        Initialize the LLM client.
        
        Returns:
            Initialized LLM client
        """
        client = LLMClient(
            api_key=self.api_key,
            model=self.config["model"],
            config={
                "cache_responses": True,
                "log_prompts": self.config["verbose_logging"],
                "log_responses": self.config["verbose_logging"],
                "random_seed": self.config["random_seed"]
            }
        )
        
        logger.debug(f"LLM client initialized with model: {self.config['model']}")
        return client
    
    def _initialize_user_library(self) -> UserLibrary:
        """
        Initialize the user library with option to load from existing file.
        
        Returns:
            Initialized user library
        """
        user_library = UserLibrary(self.llm_client, self.config)
        
        # Check for path to existing library
        library_path = self.config.get("libraries", {}).get("user_library_path")
        append_to_existing = self.config.get("libraries", {}).get("append_to_existing", True)
        
        if self.use_cache and library_path and os.path.exists(library_path):
            try:
                count = user_library.load_from_file(library_path)
                logger.info(f"Loaded {count} users from existing library: {library_path}")
                
                # If not appending, return the loaded library as is
                if not append_to_existing and count > 0:
                    return user_library
                    
            except Exception as e:
                logger.warning(f"Failed to load user library from {library_path}: {e}")
        elif not self.use_cache:
            logger.info("Skipping user library cache as use_cache=False")
        
        logger.debug("User library initialized")
        return user_library

    def _initialize_event_library(self) -> EventLibrary:
        """
        Initialize the event library with option to load from existing file.
        
        Returns:
            Initialized event library
        """
        event_library = EventLibrary(self.llm_client)
        
        # Check for path to existing library
        library_path = self.config.get("libraries", {}).get("event_library_path")
        append_to_existing = self.config.get("libraries", {}).get("append_to_existing", True)
        
        if self.use_cache and library_path and os.path.exists(library_path):
            try:
                count = event_library.load_from_file(library_path)
                logger.info(f"Loaded {count} events from existing library: {library_path}")
                
                # If not appending, return the loaded library as is
                if not append_to_existing and count > 0:
                    return event_library
                    
            except Exception as e:
                logger.warning(f"Failed to load event library from {library_path}: {e}")
        elif not self.use_cache:
            logger.info("Skipping event library cache as use_cache=False")
        
        logger.debug("Event library initialized")
        return event_library

    def _initialize_preference_library(self) -> PreferenceLibrary:
        """
        Initialize the preference library with option to load from existing file.
        
        Returns:
            Initialized preference library
        """
        preference_library = PreferenceLibrary(self.llm_client)
        
        # Check for path to existing library
        library_path = self.config.get("libraries", {}).get("preference_library_path")
        append_to_existing = self.config.get("libraries", {}).get("append_to_existing", True)
        
        if self.use_cache and library_path and os.path.exists(library_path):
            try:
                count = preference_library.load_from_file(library_path)
                logger.info(f"Loaded {count} preferences from existing library: {library_path}")
                
                # If not appending, return the loaded library as is
                if not append_to_existing and count > 0:
                    return preference_library
                    
            except Exception as e:
                logger.warning(f"Failed to load preference library from {library_path}: {e}")
        elif not self.use_cache:
            logger.info("Skipping preference library cache as use_cache=False")
        
        logger.debug("Preference library initialized")
        return preference_library
    
    def _prepare_libraries(self) -> None:
        """
        Prepare libraries with sufficient data for generation.
        """
        logger.info("Preparing libraries")
        
        # Prepare user library
        logger.info("Preparing user library...")
        self._prepare_user_library()
        logger.info("User library prepared")
        
        # Prepare event library
        logger.info("Preparing event library...")
        self._prepare_event_library()
        logger.info("Event library prepared")
        
        # Prepare preference library
        logger.info("Preparing preference library...")
        self._prepare_preference_library()
        logger.info("Preference library prepared")
        
        # Save libraries to cache if enabled
        if self.config.get("use_cache", True):
            logger.info("Saving libraries to cache...")
            self._save_libraries()
            logger.info("Libraries saved to cache")
    
    def _prepare_user_library(self) -> None:
        """
        Prepare the user library with sufficient user data.
        """
        # Check if we have enough users
        if self.user_library.get_structure_count() >= self.config["user_count"]:
            logger.debug(f"User library already has sufficient data: {self.user_library.get_structure_count()} users")
            return
        
        # Calculate how many more users we need
        needed_users = self.config["user_count"] - self.user_library.get_structure_count()
        logger.info(f"Generating {needed_users} synthetic users")
        
        # Generate batch of synthetic users
        if self.config["libraries"]["enriched_user_profile"]:
            logger.info(f"Generating {needed_users} synthetic enriched users one by one")
            for i in range(needed_users):
                try:
                    user = self.user_library.generate_synthetic_user(enrich_social_graph=True)
                    user_id = f"user_{self.user_library.get_structure_count() + 1}"
                    self.user_library.add_user(user_id, user)
                    logger.debug(f"Added synthetic user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to generate user {i+1}: {e}")
        else:
            try:
                generated_users = self.user_library.batch_generate_users(
                    count=needed_users,
                    diversity_params={
                        "ensure_ethnic_diversity": True,
                        "age_ranges": [
                            {"min": 18, "max": 30, "weight": 0.3},
                            {"min": 31, "max": 50, "weight": 0.4},
                            {"min": 51, "max": 75, "weight": 0.3}
                        ]
                    }
                )
                
                # Add to library
                for i, user in enumerate(generated_users):
                    user_id = f"user_{self.user_library.get_structure_count() + i + 1}"
                    self.user_library.add_user(user_id, user)
                    
                logger.info(f"Added {len(generated_users)} synthetic users to library")
                
            except Exception as e:
                logger.error(f"Failed to generate synthetic users: {e}")
                
                # Fallback: generate users one by one
                for i in range(needed_users):
                    try:
                        user = self.user_library.generate_synthetic_user()
                        user_id = f"user_{self.user_library.get_structure_count() + 1}"
                        self.user_library.add_user(user_id, user)
                        logger.debug(f"Added synthetic user {user_id}")
                    except Exception as e:
                        logger.error(f"Failed to generate user {i+1}: {e}")
    
    def _prepare_event_library(self) -> None:
        """
        Prepare the event library with sufficient event data.
        """
        # First, check if we should generate or augment categories based on probability
        current_categories = list(self.config.get("category_weights", {}).keys())
        new_category_prob = self.config.get("new_event_category_probability", 0.3)
        
        # If probability is 0, validate we have categories
        if new_category_prob == 0:
            if not current_categories:
                raise ValueError("When new_event_category_probability is 0, category_weights cannot be empty")
            logger.info("Using only provided event categories")
        # Otherwise, randomly decide whether to generate new categories
        # elif random.random() < new_category_prob:
        #     try:
        #         updated_categories = self.event_library.generate_and_augment_categories(current_categories)
        #         self.config["event_categories"] = updated_categories
        #         logger.info(f"Updated event categories: {len(updated_categories)} total categories")
        #     except Exception as e:
        #         logger.warning(f"Failed to generate/augment categories: {e}. Using existing categories.")
        # else:
        #     logger.info("Skipped category generation based on probability")
            
        # Check categories
        existing_categories = set(self.event_library.get_event_categories())
        target_categories = set(self.config.get("category_weights", {}).keys())
        missing_categories = target_categories - existing_categories
        
        if not missing_categories and self.event_library.get_structure_count() >= 50:
            logger.debug(f"Event library already has sufficient data: {self.event_library.get_structure_count()} events")
            return
            
        # Generate events for missing categories
        if missing_categories:
            logger.info(f"Generating synthetic events for categories: {missing_categories}")
            logger.info(f"This will generate {self.config.get('num_initial_events_per_category', 10)} events per category")
            
            try:
                # Generate 10 events per missing category
                logger.info("Starting batch event generation...")
                batch_results = self.event_library.batch_generate_events(
                    categories=list(missing_categories),
                    count_per_category=self.config.get("num_initial_events_per_category", 10)
                )
                logger.info("Batch event generation completed!")
                
                # Add to library
                event_count = 0
                for category, events in batch_results.items():
                    for event in events:
                        event_id = f"{category}_{self.event_library.get_structure_count() + 1}"
                        self.event_library.add_event(event_id, event)
                        event_count += 1
                        
                logger.info(f"Added {event_count} synthetic events to library")
                
            except Exception as e:
                logger.error(f"Failed to generate batch events: {e}")
                
                # Fallback: generate events one by one
                for category in missing_categories:
                    for i in range(10):  # 10 events per category
                        try:
                            event = self.event_library.generate_synthetic_event(category=category)
                            event_id = f"{category}_{self.event_library.get_structure_count() + 1}"
                            self.event_library.add_event(event_id, event)
                            logger.debug(f"Added synthetic event {event_id}")
                        except Exception as e:
                            logger.error(f"Failed to generate event for {category}: {e}")
        
        # Ensure we have enough events in each category
        for category in target_categories:
            category_events = self.event_library.get_events_by_category(category)
            if len(category_events) < 5:
                logger.info(f"Generating additional events for category: {category}")
                
                try:
                    # Generate additional events
                    for i in range(5 - len(category_events)):
                        event = self.event_library.generate_synthetic_event(category=category)
                        event_id = f"{category}_{self.event_library.get_structure_count() + 1}"
                        self.event_library.add_event(event_id, event)
                        logger.debug(f"Added synthetic event {event_id}")
                except Exception as e:
                    logger.error(f"Failed to generate additional events for {category}: {e}")
    
    def _prepare_preference_library(self) -> None:
        """
        Prepare the preference library with sufficient preference data.
        """
        # First, check if we should generate or augment domains based on probability
        current_domains = self.config.get("preference_domains", [])
        new_domain_prob = self.config.get("new_preference_domain_probability", 0.3)
        
        # If probability is 0, validate we have domains
        if new_domain_prob == 0:
            if not current_domains:
                raise ValueError("When new_preference_domain_probability is 0, preference_domains list cannot be empty")
            logger.info("Using only provided preference domains")
        # Otherwise, randomly decide whether to generate new domains
        # elif random.random() < new_domain_prob:
        #     try:
        #         updated_domains = self.preference_library.generate_and_augment_domains(current_domains)
        #         self.config["preference_domains"] = updated_domains
        #         logger.info(f"Updated preference domains: {len(updated_domains)} total domains")
        #     except Exception as e:
        #         logger.warning(f"Failed to generate/augment domains: {e}. Using existing domains.")
        # else:
        #     logger.info("Skipped domain generation based on probability")
            
        # Check domains
        existing_domains = set(self.preference_library.get_preference_domains())
        target_domains = set(self.config["preference_domains"])
        missing_domains = target_domains - existing_domains
        
        if not missing_domains and self.preference_library.get_structure_count() >= 20:
            logger.debug(f"Preference library already has sufficient data: {self.preference_library.get_structure_count()} preferences")
            return
            
        # Generate preferences for missing domains
        if missing_domains:
            logger.info(f"Generating synthetic preferences for domains: {missing_domains}")
            
            try:
                # Generate 5 preferences per missing domain
                batch_results = self.preference_library.batch_generate_preferences(
                    domains=list(missing_domains),
                    count_per_domain=self.config.get("num_initial_preferences_per_domain", 5)
                )
                
                # Add to library
                pref_count = 0
                for domain, preferences in batch_results.items():
                    for pref in preferences:
                        pref_id = f"{domain}_{self.preference_library.get_structure_count() + 1}"
                        self.preference_library.add_preference(pref_id, pref)
                        pref_count += 1
                        
                logger.info(f"Added {pref_count} synthetic preferences to library")
                
            except Exception as e:
                logger.error(f"Failed to generate batch preferences: {e}")
                
                # Fallback: generate preferences one by one
                for domain in missing_domains:
                    for i in range(self.config.get("num_initial_preferences_per_domain", 5)):  # 5 preferences per domain
                        try:
                            pref = self.preference_library.generate_synthetic_preference(domain=domain)
                            pref_id = f"{domain}_{self.preference_library.get_structure_count() + 1}"
                            self.preference_library.add_preference(pref_id, pref)
                            logger.debug(f"Added synthetic preference {pref_id}")
                        except Exception as e:
                            logger.error(f"Failed to generate preference for {domain}: {e}")
        
        # Generate dependency graph
        logger.info("Generating preference dependency graph")
        pref_ids = self.preference_library.get_all_structure_ids()
        
        try:
            # Use LLM to generate dependency graph
            system_prompt = """
            You are tasked with generating a realistic preference dependency graph.
            Consider how different preferences might influence each other.
            For example, a change in music_taste might affect fashion_preference.
            """
            
            # Split into smaller batches if needed
            if len(pref_ids) > 30:
                batches = [pref_ids[i:i+30] for i in range(0, len(pref_ids), 30)]
                
                for batch in batches:
                    prompt = f"""
                    For the following preference structures, create a dependency graph that indicates how changes in
                    one preference are likely to affect others. Assign a dependency strength between 0.0 (no effect)
                    and 1.0 (strong effect) for each directed relationship.
                    
                    Preference structures: {batch}
                    
                    Return a JSON object mapping source preference IDs to objects containing target preference IDs
                    and their dependency strengths. Only include relationships with strength > 0.2.
                    """
                    
                    response = self.llm_client.generate(prompt, response_format="json", system_prompt=system_prompt)
                    dependencies = self.llm_client.parse_json_response(response["response_text"]) or {}
                    
                    # Update dependency graph
                    self.preference_library.update_dependency_graph(dependencies)
            else:
                # Single batch
                prompt = f"""
                For the following preference structures, create a dependency graph that indicates how changes in
                one preference are likely to affect others. Assign a dependency strength between 0.0 (no effect)
                and 1.0 (strong effect) for each directed relationship.
                
                Preference structures: {pref_ids}
                
                Return a JSON object mapping source preference IDs to objects containing target preference IDs
                and their dependency strengths. Only include relationships with strength > 0.2.
                """
                
                response = self.llm_client.generate(prompt, response_format="json", system_prompt=system_prompt)
                dependencies = self.llm_client.parse_json_response(response["response_text"]) or {}
                
                # Update dependency graph
                self.preference_library.update_dependency_graph(dependencies)
                
            logger.info("Preference dependency graph generated")
            
        except Exception as e:
            logger.error(f"Failed to generate preference dependency graph: {e}")
    
    def _save_libraries(self) -> None:
        """
        Save libraries to cache.
        """
        logger.info("Saving libraries to cache")
        
        try:
            # Save user library
            self.user_library.save_to_file(self.config.get("libraries", {}).get("user_library_path"))
            logger.debug(f"Saved user library to {self.config.get('libraries', {}).get('user_library_path')}")
            
            # Save event library
            self.event_library.save_to_file(self.config.get("libraries", {}).get("event_library_path"))
            logger.debug(f"Saved event library to {self.config.get('libraries', {}).get('event_library_path')}")
            
            # Save preference library
            self.preference_library.save_to_file(self.config.get("libraries", {}).get("preference_library_path"))
            logger.debug(f"Saved preference library to {self.config.get('libraries', {}).get('preference_library_path')}")
            
        except Exception as e:
            logger.error(f"Failed to save libraries to cache: {e}")
    
    def _generate_timelines(self) -> List[Dict[str, Any]]:
        """
        Generate user timelines with events, preferences, and conversations.
        
        Returns:
            List of generated timelines
        """
        logger.info(f"Generating {self.config['user_count']} user timelines")
        
        timelines = []
        
        for i in range(self.config["user_count"]):
            logger.info(f"Generating timeline {i+1}/{self.config['user_count']}")
            
            try:
                # Generate timeline
                timeline = self.timeline_generator.generate_user_timeline()
                timelines.append(timeline)
                
                # Timeline will be saved in _save_results() at the end
                    
                logger.info(f"Timeline {i+1} generated with {len(timeline.get('event_record', []))} events and {len(timeline.get('conversation_record', []))} conversations")
                
            except Exception as e:
                logger.error(f"Failed to generate timeline {i+1}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return timelines
    
    def _save_results(self, timelines: List[Dict[str, Any]]) -> None:
        """
        Save all generated data to the output directory.
        
        Args:
            timelines: List of generated timelines
        """
        logger.info("Saving generation results")
        
        try:
            # Create output directory structure
            os.makedirs(self.output_dir, exist_ok=True)
            timelines_dir = os.path.join(self.output_dir, "timelines")
            os.makedirs(timelines_dir, exist_ok=True)
            
            # Load existing timelines if use_cache is True
            existing_timelines = []
            all_timelines_path = os.path.join(self.output_dir, "all_timelines.json")
            if self.use_cache and os.path.exists(all_timelines_path):
                try:
                    with open(all_timelines_path, 'r') as f:
                        existing_timelines = json.load(f)
                    logger.info(f"Loaded {len(existing_timelines)} existing timelines")
                except Exception as e:
                    logger.warning(f"Failed to load existing timelines: {e}")
            
            # Save individual timelines
            logger.info(f"About to save {len(timelines)} timelines")
            for i, timeline in enumerate(timelines):
                try:
                    # Get user info for filename
                    user_info = timeline.get("user", {}).get("attributes", {})
                    user_name = f"{user_info.get('first_name', 'user')}_{user_info.get('last_name', i)}"
                    
                    # Create sanitized filename with timestamp to avoid overwrites
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"timeline_{user_name}_{timestamp}.json"
                    filepath = os.path.join(timelines_dir, filename)
                    
                    # Ensure preference records are properly serialized
                    if "preference_record" in timeline:
                        serialized_prefs = {}
                        for date, prefs in timeline["preference_record"].items():
                            if isinstance(prefs, dict):
                                serialized_prefs[date] = {
                                    pref_id: pref.to_dict() if hasattr(pref, 'to_dict') else pref
                                    for pref_id, pref in prefs.items()
                                }
                            else:
                                serialized_prefs[date] = prefs
                        timeline["preference_record"] = serialized_prefs
                    
                    # Save individual timeline
                    with open(filepath, 'w') as f:
                        json.dump(timeline, f, indent=2)
                        
                    logger.debug(f"Saved timeline to {filepath}")
                    
                except Exception as e:
                    logger.error(f"Failed to save timeline {i}: {e}")
                    continue
            
            # Combine with existing timelines and save all timelines
            all_timelines = existing_timelines + timelines
            with open(all_timelines_path, 'w') as f:
                json.dump(all_timelines, f, indent=2)
                
            logger.info(f"Saved all {len(all_timelines)} timelines to {all_timelines_path}")
            
            # Save metadata
            metadata = {
                "generation_time": datetime.now().isoformat(),
                "config": self.config,
                "stats": self._collect_stats(all_timelines),  # Update stats to include all timelines
                "user_count": len(all_timelines),
                "total_events": sum(len(t.get("event_record", [])) for t in all_timelines),
                "total_conversations": sum(len(t.get("conversation_record", [])) for t in all_timelines)
            }
            
            metadata_path = os.path.join(self.output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
            raise
    
    def _collect_stats(self, timelines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collect statistics about the generated data.
        
        Args:
            timelines: List of generated timelines
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            "user_count": len(timelines),
            "event_count": 0,
            "conversation_count": 0,
            "avg_events_per_user": 0,
            "avg_conversations_per_user": 0,
            "preference_changes": 0,
            "preference_domains": {},
            "event_categories": {},
            "timeline_duration_days": self.config["timeline_length_days"]
        }
        
        total_events = 0
        total_conversations = 0
        preference_changes = 0
        
        for timeline in timelines:
            # Count events
            events = timeline.get("event_record", [])
            total_events += len(events)
            
            # Count conversations
            conversations = timeline.get("conversation_record", [])
            total_conversations += len(conversations)
            
            # Count preference changes
            preference_record = timeline.get("preference_record", {})
            preference_changes += len(preference_record) - 1  # Subtract initial state
            
            # Collect event categories
            for event in events:
                event_data = event.get("event", {})
                category = event_data.get("category", "unknown")
                
                if category in stats["event_categories"]:
                    stats["event_categories"][category] += 1
                else:
                    stats["event_categories"][category] = 1
            
            # Collect preference domains
            for pref_record in preference_record.values():
                for pref_id, pref_data in pref_record.items():
                    if isinstance(pref_data, dict):
                        domain = pref_data.get("domain", "unknown")
                    else:
                        domain = pref_data.get_attribute("domain", "unknown")
                    
                    if domain in stats["preference_domains"]:
                        stats["preference_domains"][domain] += 1
                    else:
                        stats["preference_domains"][domain] = 1
        
        # Calculate averages
        if len(timelines) > 0:
            stats["avg_events_per_user"] = total_events / len(timelines)
            stats["avg_conversations_per_user"] = total_conversations / len(timelines)
        
        stats["event_count"] = total_events
        stats["conversation_count"] = total_conversations
        stats["preference_changes"] = preference_changes

        any_terminated_early = any(
            t.get("timeline_meta", {}).get("terminated_early", False) for t in timelines
        )
        stats["terminated_early"] = any_terminated_early
        if any_terminated_early:
            expected = sum(t.get("timeline_meta", {}).get("expected_event_count", 0) for t in timelines)
            stats["expected_event_count"] = expected
        
        return stats
    
    def generate_single_event_conversation(
        self,
        expression_type: str,
        temporal_preference_recall_days_min: int,
        specified_event_category: Optional[str],
        specified_event_description: Optional[str],
        specified_persona_traits: str,
        specified_assistant_traits: str,
        generator_model_account: str
    ) -> Dict[str, Any]:
        """
        Generate a single event and its conversation with specified parameters.
        
        Args:
            expression_type: Type of preference expression ("explicit", "implicit", or "natural")
            temporal_preference_recall_days_min: Minimum days for temporal preference recall
            specified_event_category: Optional category for the event (if None, samples from config)
            specified_event_description: Optional description for the event
            specified_persona_traits: String description of user persona traits
            specified_assistant_traits: String description of assistant traits
            generator_model_account: Model account name for the generator
            
        Returns:
            Dictionary containing the generated event, conversation, and metadata
        """
        start_time = time.time()
        logger.info("Starting single event and conversation generation")
        
        # Prepare libraries
        logger.info("Step 1/7: Preparing libraries...")
        self._prepare_libraries()
        logger.info("Step 1/7: Libraries prepared successfully")
        
        # Generate user agent from persona traits string
        logger.info("Step 2/7: Generating user agent from persona traits...")
        user_agent = self._generate_user_from_traits(specified_persona_traits)
        logger.info(f"Step 2/7: User agent generated: {user_agent.get_formatted_name()}")
        
        # Generate event (sample category if not specified)
        logger.info("Step 3/7: Generating event...")
        event = self._generate_event(specified_event_category, specified_event_description)
        logger.info(f"Step 3/7: Event generated: {event.category}")
        
        # Generate assistant agent from traits string
        logger.info("Step 4/7: Generating assistant agent...")
        assistant_agent = self._generate_assistant_from_traits(specified_assistant_traits, event, user_agent)
        logger.info(f"Step 4/7: Assistant agent generated: {assistant_agent.get_formatted_name()}")
        
        # Initialize preferences for both agents
        logger.info("Step 5/7: Initializing preferences for both agents...")
        user_preferences = self._initialize_agent_preferences(user_agent)
        assistant_preferences = self._initialize_agent_preferences(assistant_agent)
        logger.info(f"Step 5/7: Preferences initialized (user: {len(user_preferences)}, assistant: {len(assistant_preferences)})")
        
        # Generate conversation
        logger.info("Step 6/7: Generating conversation...")
        conversation_result = self._generate_conversation(
            user_agent, assistant_agent, event,
            user_preferences, assistant_preferences,
            expression_type, temporal_preference_recall_days_min
        )
        logger.info("Step 6/7: Conversation generated successfully")
        
        # Create result
        result = {
            "user_agent": user_agent._attributes,
            "assistant_agent": assistant_agent._attributes,
            "event": {
                "id": event.id,
                "name": event.name,
                "category": event.category,
                "attributes": event._attributes,
                "date": event.date.isoformat()
            },
            "conversation": conversation_result,
            "preferences_user": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in user_preferences.items()},
            "preferences_assistant": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in assistant_preferences.items()},
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "expression_type": expression_type,
                "temporal_preference_recall_days_min": temporal_preference_recall_days_min,
                "generator_model_account": generator_model_account,
                "generation_duration_seconds": time.time() - start_time
            }
        }
        
        # Save result
        self._save_single_event_result(result)
        
        logger.info(f"Single event generation completed in {time.time() - start_time:.2f} seconds")
        return result
    
    def _generate_user_from_traits(self, persona_traits: str) -> AgentModel:
        """Generate user agent from persona traits string using LLM."""
        prompt = f"""Generate a realistic user persona based on these traits: {persona_traits}

Return a JSON object with these required fields:
- first_name (string)
- last_name (string) 
- age (number)
- gender (string)
- occupation (string)
- personality_traits (array of strings)
- interests (array of strings)
- communication_style (string)

Add other relevant attributes based on the traits provided."""

        response = self.llm_client.generate(prompt, response_format="json")
        try:
            user_attributes = json.loads(response["response_text"])
            return AgentModel(attributes=user_attributes, role="user")
        except (json.JSONDecodeError, KeyError, TypeError):
            return AgentModel(attributes={
                "first_name": "User", "last_name": "Person", "age": 30, "gender": "non-binary"
            }, role="user")
    
    def _generate_assistant_from_traits(self, assistant_traits: str, event: EventModel, user_agent: AgentModel) -> AgentModel:
        """Generate assistant agent from traits string using LLM."""
        prompt = f"""Generate an AI assistant persona based on these traits: {assistant_traits}

Context:
- Event category: {event.category}
- Event description: {event.name}
- User: {user_agent._attributes.get('first_name', 'User')} ({user_agent._attributes.get('occupation', 'person')})

Return a JSON object with these required fields:
- first_name (string)
- last_name (string)
- age (number) 
- gender (string)
- role (string)
- expertise (array of strings)
- communication_style (string)
- personality_traits (array of strings)

Make the assistant appropriate for helping with this specific event."""

        response = self.llm_client.generate(prompt, response_format="json")
        try:
            assistant_attributes = json.loads(response["response_text"])
            return AgentModel(attributes=assistant_attributes, role="assistant")
        except (json.JSONDecodeError, KeyError, TypeError):
            return AgentModel(attributes={
                "first_name": "Assistant", "last_name": "AI", "age": 25, "gender": "non-binary", "role": "helper"
            }, role="assistant")
    
    def _generate_event(self, category: Optional[str], description: Optional[str]) -> EventModel:
        """Generate event, sampling category if not specified."""
        # Sample category if not provided
        if not category:
            category = self._sample_event_category()
        
        # Generate event using LLM
        if description:
            prompt = f"""Generate a detailed event structure for this scenario:
Category: {category}
Description: {description}

Return a JSON object with:
- name (string): Short event name
- attributes (object): Event details like scenario_type, context, complexity, etc.
- preference_dependent (object): Which attributes depend on user preferences (true/false for each)

Make it realistic and engaging."""
        else:
            prompt = f"""Generate a detailed event structure for this category:
Category: {category}

Return a JSON object with:
- name (string): Short event name
- description (string): Event description
- attributes (object): Event details like scenario_type, context, complexity, etc.
- preference_dependent (object): Which attributes depend on user preferences (true/false for each)

Make it realistic and engaging."""

        response = self.llm_client.generate(prompt, response_format="json")
        try:
            event_data = json.loads(response["response_text"])
            name = event_data.get("name", f"{category} event")
            attributes = event_data.get("attributes", {})
            if description:
                attributes["description"] = description
            elif "description" in event_data:
                attributes["description"] = event_data["description"]
                
            return EventModel(
                structure_id=f"generated_{category}",
                name=name,
                category=category,
                base_attributes=attributes,
                preference_dependent_attributes=event_data.get("preference_dependent", {}),
                date=datetime.now()
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            fallback_name = description[:50] if description else f"{category} event"
            fallback_attrs = {"description": description} if description else {}
            return EventModel(
                structure_id=f"fallback_{category}",
                name=fallback_name,
                category=category,
                base_attributes=fallback_attrs,
                preference_dependent_attributes={},
                date=datetime.now()
            )
    
    def _sample_event_category(self) -> str:
        """Sample an event category from the configured weights."""
        import random
        
        category_weights = self.config.get("category_weights", {})
        if not category_weights:
            # Default categories if none configured
            return random.choice([
                "emotional_social_support", "collaborative_storytelling", 
                "philosophical_moral_inquiry", "critical_debates_strategic_analysis"
            ])
        
        # Sample based on weights
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        return random.choices(categories, weights=weights)[0]
    
    def _initialize_agent_preferences(self, agent: AgentModel) -> Dict[str, Any]:
        """Initialize preferences for an agent."""
        preferences = {}
        for domain in self.config["preference_domains"]:
            try:
                pref = self.preference_sampler.sample_preference(domain=domain, user=agent, context={})
                if pref:
                    preferences[f"{domain}_pref"] = pref
            except Exception as e:
                logger.warning(f"Failed to sample preference for domain {domain}: {e}")
        return preferences
    
    def _generate_conversation(
        self, user_agent: AgentModel, assistant_agent: AgentModel, event: EventModel,
        user_preferences: Dict[str, Any], assistant_preferences: Dict[str, Any],
        expression_type: str, temporal_preference_recall_days_min: int
    ) -> Dict[str, Any]:
        """Generate conversation between user and assistant."""
        conv_config = self.config.copy()
        conv_config.update({
            "expression_type": expression_type,
            "temporal_preference_recall_days_min": temporal_preference_recall_days_min
        })
        
        conversation_generator = ConversationGenerator(self.llm_client, self.convo_llm_client, conv_config)
        
        # Simple agent intents - create proper IntentModel instances
        from src.causal_framework.models.intent_model import IntentModel
        
        agent_intents = {
            user_agent.id: IntentModel(
                intent_type="seek_assistance", 
                attributes={"goal": f"Get help with {event.category}"}
            ),
            assistant_agent.id: IntentModel(
                intent_type="provide_assistance", 
                attributes={"goal": f"Help user with {event.category}"}
            )
        }
        
        return conversation_generator.generate_conversation(
            user=user_agent, assistant=assistant_agent, event=event,
            preferences_current_user=user_preferences,
            preferences_updated_user=user_preferences,
            preferences_current_assistant=assistant_preferences,
            preferences_updated_assistant=assistant_preferences,
            agent_intents=agent_intents, previous_events=[],
            num_turns=self.config.get("default_conversation_turns", 8)
        )
    
    
    def _save_single_event_result(self, result: Dict[str, Any]) -> None:
        """Save the single event generation result to a file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"single_event_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
            logger.info(f"Saved single event result to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save single event result: {e}")