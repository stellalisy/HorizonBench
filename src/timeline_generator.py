"""
timeline_generator.py

Orchestrates the generation of temporal user timelines with evolving preferences.
This is the core orchestration module of the Causal Preference Evolution Framework.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.libraries.user_library import UserLibrary
from src.libraries.event_library import EventLibrary
from src.libraries.preference_library import PreferenceLibrary
from src.causal_framework.models.agent_model import AgentModel
from src.causal_framework.models.event_model import EventModel
from src.causal_framework.models.preference_model import PreferenceModel
from src.causal_framework.models.intent_model import IntentModel
from src.causal_framework.sampling.event_sampler import EventSampler
from src.causal_framework.sampling.preference_sampler import PreferenceSampler
from src.causal_framework.sampling.intent_sampler import IntentSampler
from src.causal_framework.generation.assistant_persona_generator import AssistantPersonaGenerator
from src.causal_framework.evaluation.intent_fulfillment_assessor import IntentFulfillmentAssessor
from src.causal_framework.evolution.preference_evolve_manager import PreferenceEvolveManager
from src.conversation.conversation_generator import ConversationGenerator
from src.llm.api_client import LLMClient

from src.benchmark_generation import get_alternative_values, get_perturbable_attributes

logger = logging.getLogger(__name__)

class TimelineGenerator:
    """
    Core class responsible for generating synthetic user timelines with 
    evolving preferences and corresponding conversations with agent symmetry.
    """
    
    def __init__(
        self,
        user_library: UserLibrary,
        event_library: EventLibrary,
        preference_library: PreferenceLibrary,
        llm_client: LLMClient,
        convo_llm_client: Optional[LLMClient],
        config: Dict[str, Any] = None,
        validation_llm_clients: Optional[List] = None
    ):
        """
        Initialize the TimelineGenerator with required libraries and components.
        
        Args:
            user_library: Library of user persona structures
            event_library: Library of event structures
            preference_library: Library of preference structures
            llm_client: Client for accessing LLM services
            config: Configuration parameters for timeline generation
        """
        self.user_library = user_library
        self.event_library = event_library
        self.preference_library = preference_library
        self.llm_client = llm_client
        self.convo_llm_client = convo_llm_client
        
        # Set default configuration or update with provided config
        self.config = {
            "timeline_length_days": 180,  # Default 6 months
            "avg_events_per_week": 3,
            "relevant_preferences_per_event": 3,
            "conversation_probability": 0.7,  # Probability of generating a conversation for an event
            "random_seed": 42,
            "max_preference_shifts": 5,  # Maximum number of significant preference shifts allowed
            "output_dir": "./output",  # Added for _generate_conversation
            "stable_agentB": False,  # Whether to maintain stable assistant across events
            "new_preference_domain_probability": 0.3,
            "max_preference_domains": 20,
            "num_initial_preferences_per_domain": 5,
            "new_preference_structure_probability": 0.3,
            "max_preference_structures_per_domain": 20,
        }
        
        if config:
            self.config.update(config)
            
        random.seed(self.config["random_seed"])
        
        # Initialize components
        self.event_sampler = EventSampler(event_library, llm_client, config)
        self.preference_sampler = PreferenceSampler(preference_library, llm_client)
        self.intent_sampler = IntentSampler(llm_client, config)
        self.assistant_persona_generator = AssistantPersonaGenerator(llm_client, config)
        self.intent_fulfillment_assessor = IntentFulfillmentAssessor(llm_client, config)
        self.preference_evolve_manager = PreferenceEvolveManager(llm_client)
        self.conversation_generator = ConversationGenerator(llm_client, convo_llm_client, config)
        self.validation_llm_clients = validation_llm_clients
        
        logger.info("TimelineGenerator initialized with configuration: %s", self.config)
    
    def generate_user_timeline(self, user_id: Optional[str] = None) -> Dict:
        """
        Generate a complete user timeline with evolving preferences and conversations.
        
        Args:
            user_id: Optional user ID to use a specific user from the library.
                    If None, a random user will be sampled.
                    
        Returns:
            A dictionary containing the complete user timeline with events,
            preferences, and conversations.
        """
        # 1. Sample or retrieve a user persona (now AgentModel)
        user_agent = self._instantiate_user_agent(user_id)
        logger.info(f"Generated timeline for user: {user_agent.get_formatted_name()}")
        
        if user_id is None:
            user_id = user_agent.id
            
        # 2. Generate timeline start and end dates
        start_date = datetime.now() - timedelta(days=self.config["timeline_length_days"])
        end_date = datetime.now()
        
        # 3. Initialize empty timeline records
        preference_record = {}
        event_record = []
        conversation_record = []
        intent_record = []
        
        # 4. Initialize empty preferences dictionary - will be populated lazily
        current_preferences_user = {}
        all_preferences_user = {}  # Keep track of all user preferences for temporal recall
        
        # 4b. Initialize assistant tracking for stable_agentB case
        stable_assistant_agent = None
        stable_assistant_preferences = {}
        
        # 5. Generate events across the timeline
        current_date = start_date
        previous_events = []
        
        # Calculate total expected events based on timeline length
        expected_events = int((self.config["timeline_length_days"] / 7) * self.config["avg_events_per_week"])
        
        # Distribute events across the timeline
        event_dates = self._distribute_event_dates(start_date, end_date, expected_events)

        MAX_CONSECUTIVE_FATAL = 5
        consecutive_fatal_errors = 0

        # Generate each event
        for event_index, event_date in enumerate(event_dates):
            try:
                current_date = event_date
            
                # 5a. Sample next event type based on user, preferences, and previous events
                event, temporal_pref_id = self._sample_next_event(user_agent, all_preferences_user, previous_events, current_date)

                # 5b. Generate assistant agent based on event and stable_agentB setting
                if self.config["stable_agentB"] and stable_assistant_agent is not None:
                    # Use the stable assistant
                    assistant_agent = stable_assistant_agent
                    current_preferences_assistant = stable_assistant_preferences.copy()
                else:
                    # Generate new assistant
                    assistant_agent = self._generate_assistant_agent(event, user_agent)
                    current_preferences_assistant = {}
                    
                    # Store as stable assistant if needed
                    if self.config["stable_agentB"]:
                        stable_assistant_agent = assistant_agent
                
                # 5c. Sample intents for both agents
                agent_intents = self._sample_agent_intents(user_agent, assistant_agent, event)

                # 5e. Identify relevant preference structures for this event (conditioned on user intent)
                relevant_pref_ids = self._identify_relevant_preferences(event, user_intent=agent_intents[user_agent.id])

                try:
                    # 5f. Get only the relevant preferences from all historical preferences
                    current_preferences_user = {
                        pref_id: all_preferences_user[pref_id] 
                        for pref_id in relevant_pref_ids.keys() 
                        if pref_id in all_preferences_user
                    }
                    
                    # Update user preferences based on this event (conditioned on event, user persona, user intent)
                    updated_preferences_user = self._evolve_preferences_with_intent(
                        agent=user_agent, 
                        event=event, 
                        current_preferences=current_preferences_user, 
                        relevant_pref_ids=relevant_pref_ids,
                        agent_intent=agent_intents[user_agent.id],
                        previous_events=previous_events
                    )
                    
                    # 5g. Initialize and evolve assistant preferences (conditioned on assistant role, event, and intent)
                    # Note: Use current_preferences_user (before evolution) so assistant plans based on user's starting state
                    updated_preferences_assistant = self._initialize_assistant_action_preferences(
                        assistant_agent=assistant_agent,
                        event=event,
                        user_agent=user_agent,
                        user_preferences=current_preferences_user,
                        assistant_intent=agent_intents[assistant_agent.id],
                        current_assistant_preferences=current_preferences_assistant
                    )
                except Exception as e:
                    print(f"timeline_generator.py:214 error: {e}")
                    # Log the error but continue with fallback
                    logger.warning(f"Failed to evolve preferences: {e}")
                    # Get relevant preferences for fallback
                    current_preferences_user = {
                        pref_id: all_preferences_user[pref_id] 
                        for pref_id in relevant_pref_ids.keys() 
                        if pref_id in all_preferences_user
                    }
                    updated_preferences_user = current_preferences_user.copy()  # Use relevant prefs as fallback
                    updated_preferences_assistant = current_preferences_assistant.copy()  # Empty fallback for assistant

                try:
                    # 5h. Finalize event details based on updated preferences
                    finalized_event = self._finalize_event(event, user_agent, updated_preferences_user)
                except Exception as e:
                    print(f"timeline_generator.py:227 error: {e}")
                    # Log the error but continue with fallback
                    logger.warning(f"Failed to finalize event: {e}")
                    finalized_event = event  # Use original event as fallback
                
                # 5i. Add event to record
                event_record.append({
                    "date": current_date.isoformat(),
                    "event": finalized_event.to_dict() if hasattr(finalized_event, 'to_dict') else finalized_event,
                    "assistant_agent": assistant_agent.to_dict()
                })
                
                # 5j. Add intent record
                intent_record.append({
                    "date": current_date.isoformat(),
                    "event_id": finalized_event.id if hasattr(finalized_event, 'id') else str(finalized_event),
                    "user_intent": agent_intents[user_agent.id].to_dict(),
                    "assistant_intent": agent_intents[assistant_agent.id].to_dict()
                })
                
                random.seed(event_index)
                # 5k. Generate conversation for this event (probabilistic)
                if random.random() < self.config["conversation_probability"]:
                    conversation = self._generate_conversation_with_intents(
                        user_agent=user_agent,
                        assistant_agent=assistant_agent,
                        event=finalized_event, 
                        current_preferences_user=current_preferences_user,
                        updated_preferences_user=updated_preferences_user,
                        current_preferences_assistant=current_preferences_assistant,
                        updated_preferences_assistant=updated_preferences_assistant,
                        agent_intents=agent_intents,
                        previous_events=previous_events,
                        user_id=user_id,
                        event_index=event_index,
                        event_date=event_date,
                        temporal_pref_id=temporal_pref_id,
                        all_preferences_user=all_preferences_user
                    )
                    
                    # Refresh last_expressed_date for every user preference
                    # that was relevant to (i.e. used/expressed in) this conversation.
                    for pref_id in updated_preferences_user:
                        if pref_id in all_preferences_user:
                            all_preferences_user[pref_id].update_expression_tracking(
                                date=current_date,
                                context={
                                    "event_id": finalized_event.id if hasattr(finalized_event, 'id') else None,
                                    "event_name": finalized_event.name if hasattr(finalized_event, 'name') else None,
                                    "event_category": finalized_event.category if hasattr(finalized_event, 'category') else None,
                                    "is_temporal_recall": (pref_id == temporal_pref_id)
                                }
                            )
                    # Also refresh the temporal preference (it's in all_preferences_user
                    # but may have been popped from updated_preferences_user earlier).
                    if temporal_pref_id and temporal_pref_id in all_preferences_user and temporal_pref_id not in updated_preferences_user:
                        all_preferences_user[temporal_pref_id].update_expression_tracking(
                            date=current_date,
                            context={
                                "event_id": finalized_event.id if hasattr(finalized_event, 'id') else None,
                                "event_name": finalized_event.name if hasattr(finalized_event, 'name') else None,
                                "event_category": finalized_event.category if hasattr(finalized_event, 'category') else None,
                                "is_temporal_recall": True
                            }
                        )

                    # Assess intent fulfillment
                    intent_fulfillment = self._assess_intent_fulfillment(
                        agent_intents,
                        conversation,
                        {user_agent.id: user_agent, assistant_agent.id: assistant_agent}
                    )
                    
                    conversation_record.append({
                        "date": current_date.isoformat(),
                        "date_unix": int(current_date.timestamp()),
                        "event_id": finalized_event.id if hasattr(finalized_event, 'id') else str(finalized_event),
                        "conversation": conversation,
                        "intent_fulfillment": intent_fulfillment,
                        "user_agent": user_agent.to_dict() if hasattr(user_agent, 'to_dict') else user_agent,
                        "assistant_agent": assistant_agent.to_dict() if hasattr(assistant_agent, 'to_dict') else assistant_agent
                    })
                
                # 5l. Update all preferences with the evolved ones and set current preferences for this event
                all_preferences_user.update(updated_preferences_user)
                current_preferences_user = updated_preferences_user.copy()  # Only relevant preferences for this event
                
                # Add updated preferences to record if they changed
                if updated_preferences_user:
                    # Ensure updated preferences are properly serialized
                    preference_record[current_date.isoformat()] = {
                        pref_id: pref.to_dict() if hasattr(pref, 'to_dict') else pref
                        for pref_id, pref in updated_preferences_user.items()
                    }
                
                # 5l-evo. Probabilistically trigger a deliberate evolution event on stale preferences
                evolution_prob = self.config.get("preference_evolution_event_probability", 0.0)
                if evolution_prob > 0 and random.random() < evolution_prob and len(all_preferences_user) > 0:
                    try:
                        evolved_pref = self._trigger_evolution_event(
                            user_agent=user_agent,
                            all_preferences=all_preferences_user,
                            current_date=current_date,
                            previous_events=previous_events,
                            event_record=event_record,
                            conversation_record=conversation_record
                        )
                        if evolved_pref:
                            all_preferences_user.update(evolved_pref)
                    except Exception as e:
                        logger.warning(f"Evolution event failed: {e}")

                # 5l2. Update stable assistant preferences if using stable_agentB
                if self.config["stable_agentB"]:
                    stable_assistant_preferences = updated_preferences_assistant.copy()
                
                # 5m. Update previous events list
                previous_events.append(finalized_event)
            except Exception as e:
                error_msg = str(e)
                print(f"timeline_generator.py:304 error, skipping to next event: {e}")
                logger.warning(f"Failed to generate event {event_index}, skipping to next event: {e}")

                FATAL_PATTERNS = ["ExpiredTokenException", "security token", "InvalidClientTokenId",
                                  "AuthFailure", "UnrecognizedClientException", "AccessDeniedException"]
                is_fatal = any(pat in error_msg for pat in FATAL_PATTERNS)
                if is_fatal:
                    consecutive_fatal_errors += 1
                    logger.error(f"Fatal infrastructure error ({consecutive_fatal_errors}/{MAX_CONSECUTIVE_FATAL}): {error_msg[:200]}")
                    if consecutive_fatal_errors >= MAX_CONSECUTIVE_FATAL:
                        logger.error(
                            f"Aborting timeline generation after {consecutive_fatal_errors} consecutive fatal errors. "
                            f"Completed {len(event_record)} / {len(event_dates)} events. "
                            f"Last successful event index: {event_index - consecutive_fatal_errors}"
                        )
                        break
                else:
                    consecutive_fatal_errors = 0
                continue
        
        if consecutive_fatal_errors >= MAX_CONSECUTIVE_FATAL:
            logger.error(f"Timeline generation terminated early due to infrastructure failure.")

        # 6. Assemble and return the complete timeline
        timeline = {
            "user": user_agent.to_dict() if hasattr(user_agent, 'to_dict') else user_agent,
            "assistant": assistant_agent.to_dict() if hasattr(assistant_agent, 'to_dict') else assistant_agent,
            "timeline_meta": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "event_count": len(event_record),
                "expected_event_count": len(event_dates),
                "conversation_count": len(conversation_record),
                "preference_change_count": len(preference_record) - 1,  # Subtract initial state
                "stable_agentB": self.config["stable_agentB"],
                "terminated_early": consecutive_fatal_errors >= MAX_CONSECUTIVE_FATAL,
            },
            "preference_record": preference_record,
            "event_record": event_record,
            "intent_record": intent_record,
            "conversation_record": conversation_record
        }
        
        logger.info(f"Generated timeline with {len(event_record)} events and {len(conversation_record)} conversations")
        return timeline
    
    def _instantiate_user_agent(self, user_id: Optional[str] = None) -> AgentModel:
        """
        Sample a user agent from the library or retrieve a specific user by ID.
        
        Args:
            user_id: Optional user ID to retrieve a specific user
            
        Returns:
            An instantiated AgentModel for the user
        """
        if user_id:
            user_struct = self.user_library.get_user_by_id(user_id)
        else:
            user_struct = self.user_library.sample_user()
            
        # Convert the structure to an AgentModel instance
        user_agent = AgentModel(
            attributes=user_struct,
            role="user"
        )
        
        # Use LLM to enrich user attributes if needed
        missing_attributes = user_agent.get_missing_attributes() if hasattr(user_agent, 'get_missing_attributes') else []
        if missing_attributes:
            enriched_attributes = self._enrich_agent_attributes(user_agent, missing_attributes)
            for attr, value in enriched_attributes.items():
                user_agent.set_attribute(attr, value)
                
        return user_agent
    
    def _enrich_agent_attributes(self, agent: AgentModel, missing_attributes: List[str]) -> Dict:
        """
        Use LLM to generate missing agent attributes for increased realism.
        
        Args:
            agent: The agent model to enrich
            missing_attributes: List of attribute names to generate
            
        Returns:
            Dictionary of generated attributes
        """
        # Create a prompt for the LLM to generate consistent attributes
        prompt = f"""
        Based on the following agent information, generate plausible values for the missing attributes.
        
        Agent information:
        {json.dumps(agent.to_dict(), indent=2)}
        
        Missing attributes: {', '.join(missing_attributes)}
        
        Return a JSON object with only the missing attributes and their generated values:
        """
        
        response = self.llm_client.generate(prompt, response_format="json")

        enriched_attributes = json.loads(response["response_text"])
        return {k: v for k, v in enriched_attributes.items() if k in missing_attributes}

    def _generate_assistant_agent(self, event: EventModel, user_agent: AgentModel) -> AgentModel:
        """
        Generate an assistant agent based on event context and stable_agentB setting.
        
        Args:
            event: Event that triggers the interaction
            user_agent: User agent for context
            
        Returns:
            AgentModel instance for the assistant
        """
        return self.assistant_persona_generator.generate_assistant(
            event=event,
            user_agent=user_agent,
            stable_agentB=self.config["stable_agentB"],
            user_id=user_agent.id if self.config["stable_agentB"] else None
        )
    
    def _sample_agent_intents(
        self,
        user_agent: AgentModel,
        assistant_agent: AgentModel,
        event: EventModel
    ) -> Dict[str, IntentModel]:
        """
        Sample intents for both agents based on event context.
        
        Args:
            user_agent: User agent
            assistant_agent: Assistant agent
            event: Event that triggers the interaction
            
        Returns:
            Dictionary mapping agent IDs to their IntentModel instances
        """
        return self.intent_sampler.sample_intents(
            agent_a=user_agent,
            agent_b=assistant_agent,
            event=event
        )
    
    def _assess_intent_fulfillment(
        self,
        agent_intents: Dict[str, IntentModel],
        conversation: Dict[str, Any],
        agents: Dict[str, AgentModel]
    ) -> Dict[str, Any]:
        """
        Assess how well intents were fulfilled during the conversation.
        
        Args:
            agent_intents: Dictionary mapping agent IDs to their IntentModel instances
            conversation: Generated conversation dictionary
            agents: Dictionary mapping agent IDs to AgentModel instances
            
        Returns:
            Dictionary containing fulfillment scores and metrics
        """
        return self.intent_fulfillment_assessor.assess_intent_fulfillment(
            agent_intents=agent_intents,
            conversation=conversation,
            agents=agents
        )
    
    def _initialize_agent_preferences(
        self, 
        agent: AgentModel, 
        preference_ids: Optional[List[str]] = None
    ) -> Dict[str, PreferenceModel]:
        """
        Sample initial preference states for an agent based on their persona.
        
        Args:
            agent: The agent model
            preference_ids: Optional list of specific preference IDs to initialize. 
                          If None, initializes all available preferences.
            
        Returns:
            Dictionary mapping preference structure IDs to instantiated PreferenceModel objects
        """
        # Get all available preference structures
        all_preference_structures = self.preference_library.get_all_structures()
        
        # Filter to only needed preference structures
        if preference_ids is not None:
            needed_structures = {
                pref_id: all_preference_structures[pref_id]
                for pref_id in preference_ids
                if pref_id in all_preference_structures
            }
        else:
            needed_structures = all_preference_structures
            
        if not needed_structures:
            return {}
        
        # Use batch sampling for consistency and efficiency
        sampled_preferences = self.preference_sampler.batch_sample_preferences(
            user=agent,  # PreferenceSampler expects 'user' parameter
            preference_structures=needed_structures
        )
        
        # Convert sampled values to preference models
        preferences = {}
        for pref_id, pref_values in sampled_preferences.items():
            pref_structure = needed_structures[pref_id]
            
            # Create preference model instance
            pref_model = PreferenceModel(
                structure_id=pref_id,
                attributes=pref_values,
                metadata={
                    "name": pref_structure.get("name", pref_id),
                    "description": pref_structure.get("description", f"Preference {pref_id}"),
                    "domain": pref_structure.get("domain", "unknown")
                }
            )
            
            preferences[pref_id] = pref_model
            
        return preferences
    
    def _initialize_contextual_assistant_preferences(
        self,
        assistant_agent: AgentModel,
        event: EventModel,
        relevant_pref_ids: Dict[str, float],
        user_agent: AgentModel,
        user_preferences: Dict[str, PreferenceModel],
        assistant_intent: IntentModel
    ) -> Dict[str, PreferenceModel]:
        """
        Initialize assistant preferences contextually based on the event, role, user needs, and assistant intent.
        Only creates preferences relevant to the current event/interaction.
        
        Args:
            assistant_agent: Assistant agent model
            event: Current event
            relevant_pref_ids: Preference IDs relevant to this event
            user_agent: User agent for context
            user_preferences: User's current preferences
            assistant_intent: Assistant's intent for this interaction
            
        Returns:
            Dictionary of contextually appropriate assistant preferences
        """
        assistant_preferences = {}
        
        # Get all preference structures to reference
        all_preference_structures = self.preference_library.get_all_structures()
        
        # Only initialize preferences that are relevant to the current event
        for pref_id, relevance_score in relevant_pref_ids.items():
            if relevance_score < 0.2:  # Skip low-relevance preferences
                continue
                
            if pref_id not in all_preference_structures:
                continue
                
            pref_structure = all_preference_structures[pref_id]
            
            # Generate contextual preferences based on assistant role, user needs, and intent
            contextual_prefs = self._generate_contextual_assistant_preference(
                assistant_agent=assistant_agent,
                pref_structure=pref_structure,
                event=event,
                user_agent=user_agent,
                user_preference=user_preferences.get(pref_id),
                assistant_intent=assistant_intent
            )
            
            # Create preference model instance
            pref_model = PreferenceModel(
                structure_id=pref_id,
                attributes=contextual_prefs,
                metadata={
                    "name": pref_structure.get("name", pref_id),
                    "description": pref_structure.get("description", f"Preference {pref_id}"),
                    "domain": pref_structure.get("domain", "unknown"),
                    "contextual": True,  # Mark as contextually generated
                    "assistant_role": assistant_agent.role
                }
            )
            
            assistant_preferences[pref_id] = pref_model
            
        return assistant_preferences
    
    def _initialize_assistant_action_preferences(
        self,
        assistant_agent: AgentModel,
        event: EventModel,
        user_agent: AgentModel,
        user_preferences: Dict[str, PreferenceModel],
        assistant_intent: IntentModel,
        current_assistant_preferences: Dict[str, PreferenceModel]
    ) -> Dict[str, PreferenceModel]:
        """
        Initialize or update assistant preferences based on their role, context, and intent.
        For AI assistants: action strategies. For human assistants: personal preferences.
        
        Args:
            assistant_agent: Assistant agent model
            event: Current event
            user_agent: User agent for context
            user_preferences: User's current preferences
            assistant_intent: Assistant's intent for this interaction
            current_assistant_preferences: Current assistant preferences (for stable assistants)
            
        Returns:
            Dictionary of assistant preferences
        """
        # First, identify which assistant preferences are relevant to this event and context
        relevant_assistant_preferences = self._identify_relevant_assistant_preferences(
            assistant_agent=assistant_agent,
            event=event,
            user_preferences=user_preferences,
            current_assistant_preferences=current_assistant_preferences
        )
        
        # Get assistant type from the agent (determined during generation)
        is_ai_assistant = assistant_agent.get_attribute("assistant_type", "ai") == "ai"
        
        # Create a comprehensive prompt to generate/update all needed preferences
        if is_ai_assistant:
            preference_type = "action strategies and approaches"
            focus_instruction = "Focus on HOW the assistant plans to act to achieve their intent and help the user effectively."
        else:
            preference_type = "personal preferences and interaction style"
            focus_instruction = "Focus on the assistant's personal preferences and natural interaction style as a human."
        
        prompt = f"""
        Generate or update preferences for this assistant based on the current context.
        
        Assistant Context:
        - Role: {assistant_agent.role} ({'AI assistant' if is_ai_assistant else 'Human assistant'})
        - Attributes: {json.dumps(assistant_agent.to_dict(), indent=2)}
        - Intent: {json.dumps(assistant_intent.to_dict(), indent=2)}
        
        Event Context: {json.dumps(event.to_dict(), indent=2)}
        
        User Context:
        - User: {json.dumps(user_agent.to_dict(), indent=2)}
        - User Preferences: {json.dumps({k: v.to_dict() for k, v in user_preferences.items()}, indent=2)}
        
        Current Relevant Assistant Preferences: {json.dumps({k: v.to_dict() for k, v in relevant_assistant_preferences.items()}, indent=2) if relevant_assistant_preferences else "None - no relevant preferences or new assistant"}
        
        TASK: Generate {preference_type} for this assistant.
        {focus_instruction}
        
        REQUIREMENTS:
        1. If current preferences exist, update/evolve them contextually - don't replace everything
        2. Generate 2-4 preference categories that are most relevant to this specific context
        3. Each preference should have 2-4 attributes with specific values
        4. Make preferences realistic and contextually appropriate
        
        PREFERENCE EXAMPLES:
        {"For AI assistants (action strategies):" if is_ai_assistant else "For human assistants (personal preferences):"}
        
        {'''AI Example:
        {
          "communication_approach": {
            "directness": "gentle",
            "formality": "casual", 
            "response_length": "detailed"
          },
          "support_strategy": {
            "type": "emotional_and_practical",
            "timing": "user_paced",
            "intensity": "moderate"
          }
        }''' if is_ai_assistant else '''Human Example:
        {
          "social_interaction": {
            "energy_level": "moderate",
            "humor_usage": "frequent",
            "personal_sharing": "selective"
          },
          "emotional_expression": {
            "openness": "high",
            "empathy_style": "active_listening",
            "support_approach": "encouraging"
          }
        }'''}
        
        Return a JSON object with preference categories as keys and their attributes as values:
        {{
          "preference_category_1": {{
            "attribute1": "value1",
            "attribute2": "value2"
          }},
          "preference_category_2": {{
            "attribute1": "value1", 
            "attribute2": "value2"
          }}
        }}
        """
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            generated_preferences = json.loads(response["response_text"])
            
            # Start with existing relevant preferences
            updated_preferences = relevant_assistant_preferences.copy()
            
            # Convert to PreferenceModel objects and add/update
            for pref_category, attributes in generated_preferences.items():
                if isinstance(attributes, dict):
                    pref_id = f"assistant_{pref_category}"
                    
                    # Create preference model
                    pref_model = PreferenceModel(
                        structure_id=pref_id,
                        attributes=attributes,
                        metadata={
                            "name": pref_category.replace("_", " ").title(),
                            "description": f"Assistant {pref_category} preferences",
                            "domain": pref_category,
                            "assistant_role": assistant_agent.role,
                            "preference_type": "action_strategy" if is_ai_assistant else "personal"
                        }
                    )
                    
                    updated_preferences[pref_id] = pref_model
            
            return updated_preferences
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to generate assistant preferences: {e}")
            # Return existing relevant preferences as fallback
            return relevant_assistant_preferences
    
    def _identify_relevant_assistant_preferences(
        self,
        assistant_agent: AgentModel,
        event: EventModel,
        user_preferences: Dict[str, PreferenceModel],
        current_assistant_preferences: Dict[str, PreferenceModel]
    ) -> Dict[str, PreferenceModel]:
        """
        Identify which assistant preferences are relevant to the current event and context.
        
        Args:
            assistant_agent: Assistant agent model
            event: Current event
            user_preferences: User's current preferences
            current_assistant_preferences: All current assistant preferences
            
        Returns:
            Dictionary of relevant assistant preferences
        """
        if not current_assistant_preferences:
            return {}
        
        # If there are only a few preferences, return all of them
        if len(current_assistant_preferences) <= 3:
            return current_assistant_preferences
        
        # For many preferences, use LLM to identify relevant ones
        prompt = f"""
        Given the current event and user context, identify which assistant preferences are most relevant.
        
        Assistant: {assistant_agent.role}
        Event: {json.dumps(event.to_dict(), indent=2)}
        User Preferences: {json.dumps({k: v.to_dict() for k, v in user_preferences.items()}, indent=2)}
        
        Available Assistant Preferences:
        {json.dumps({k: {"name": v.metadata.get("name", k), "domain": v.metadata.get("domain", "unknown"), "attributes": v.attributes} for k, v in current_assistant_preferences.items()}, indent=2)}
        
        Return a JSON object mapping preference IDs to relevance scores (0.0 to 1.0).
        Only include preferences with relevance > 0.3:
        {{
          "preference_id_1": 0.8,
          "preference_id_2": 0.6
        }}
        """
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            relevance_scores = json.loads(response["response_text"])
            
            # Filter preferences based on relevance
            relevant_preferences = {}
            for pref_id, score in relevance_scores.items():
                if pref_id in current_assistant_preferences and score > 0.3:
                    relevant_preferences[pref_id] = current_assistant_preferences[pref_id]
            
            return relevant_preferences
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to identify relevant assistant preferences: {e}")
            # Fallback: return first 3 preferences
            return dict(list(current_assistant_preferences.items())[:3])
    

    

    
    def _distribute_event_dates(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        event_count: int
    ) -> List[datetime]:
        """
        Distribute events across the timeline with some randomness.
        
        Args:
            start_date: Timeline start date
            end_date: Timeline end date
            event_count: Target number of events to generate
            
        Returns:
            List of event dates in chronological order
        """
        # Calculate timeline duration in days
        timeline_days = (end_date - start_date).days
        
        # Generate random offsets (days from start date)
        offsets = sorted([random.randint(0, timeline_days) for _ in range(event_count)])
        
        # Convert offsets to actual dates
        event_dates = [start_date + timedelta(days=offset) for offset in offsets]
        
        return event_dates
    
    def _sample_next_event(
        self,
        user_agent: AgentModel,
        current_preferences: Dict[str, PreferenceModel],
        previous_events: List[EventModel],
        current_date: datetime
    ) -> Tuple[EventModel, Optional[str]]:
        """
        Sample the next event based on user, preferences, and event history.
        Sometimes uses temporal preference sampling to create opportunities
        for assistants to recall past preferences.
        
        Args:
            user_agent: User agent model
            current_preferences: Current preference states
            previous_events: List of previous events
            current_date: Current date in the timeline
            
        Returns:
            Sampled EventModel instance
        """
        import random
        
        # Check if we should use temporal preference sampling
        temporal_prob = self.config.get("sample_temporal_preference_prob", 0.3)

        selected_pref_id, selected_preference, days_since = None, None, None
        random.seed(int(current_date.timestamp()))
        if random.random() < temporal_prob:
            # Try to sample a temporal preference event
            # temporal_event, selected_pref_id = self._sample_temporal_preference_event(
            #     user_agent, current_preferences, current_date
            # )
            selected_pref_id, selected_preference, days_since = self._sample_temporal_preference_event(
                user_agent, current_preferences, current_date
            )
            if selected_pref_id:
                print(f"Using temporal preference: {selected_preference.structure_id}: {selected_preference}")
            else:
                print("No temporal preference candidates, falling back to regular sampling")
        else:
            print(f"timeline_generator.py:863 Not using temporal preference sampling")
        
        print(f"selected_pref_id: {selected_pref_id}")

        # Use the event sampler to get the next event (regular sampling)
        event_structure = self.event_sampler.sample_event(
            user=user_agent,  # EventSampler expects 'user' parameter
            preferences=current_preferences,
            previous_events=previous_events,
            date=current_date,
            temporal_pref_id=selected_pref_id,
            selected_preference=selected_preference,
        )

        # Initialize event with base attributes (non-preference dependent)

        # Ensure structure has an ID (fallback for synthetic events)
        if "id" not in event_structure:
            event_structure["id"] = f"synthetic_{event_structure.get('category', 'unknown')}_{len(previous_events) + 1}"
        
        event = EventModel(
            structure_id=event_structure["id"],
            name=event_structure["name"],
            category=event_structure["category"],
            base_attributes={
                k: v for k, v in event_structure["attributes"].items() 
                if not event_structure["preference_dependent"].get(k, False)
            },
            preference_dependent_attributes=event_structure["preference_dependent"],
            date=current_date
        )
        
        return event, selected_pref_id  # No temporal preference for regular events
    
    def _sample_temporal_preference_event(
        self,
        user_agent: AgentModel,
        all_current_preferences: Dict[str, PreferenceModel],
        current_date: datetime
    ) -> Tuple[Optional[EventModel], Optional[str]]:
        """
        Sample an event based on previously expressed preferences that haven't been
        revisited in a while. This creates opportunities for the assistant to recall
        past preferences.
        
        Args:
            user_agent: User agent model
            all_current_preferences: Current preference states
            current_date: Current date in the timeline
            
        Returns:
            EventModel instance based on temporal preference, or None if no suitable preference found
        """
        # Get preferences that were expressed in the past but not recently
        temporal_candidates = []
        
        for pref_id, preference in all_current_preferences.items():
            days_since_expressed = preference.get_days_since_last_expressed(current_date)

            # Only consider preferences that were expressed at least 7 days ago
            if days_since_expressed is not None and days_since_expressed >= self.config.get("benchmark_generation", {}).get("temporal_preference_recall_days_min", 7) and days_since_expressed <= self.config.get("benchmark_generation", {}).get("temporal_preference_recall_days_max", 500):
                temporal_candidates.append((pref_id, preference, days_since_expressed))
        
        # If no temporal candidates, return None
        if not temporal_candidates:
            logger.debug("No temporal preference candidates found")
            return None, None, None
        
        # Weight by staleness, with a multiplier for evolved preferences to achieve
        # roughly 50/50 evolved vs static benchmark items.
        evolved_weight_mult = self.config.get("evolved_preference_recall_weight", 3.0)
        weights = []
        evolved_count = 0
        static_count = 0
        for pref_id, preference, days_since in temporal_candidates:
            w = days_since
            has_evo = bool(getattr(preference, 'evolution_history', None))
            if has_evo:
                w *= evolved_weight_mult
                evolved_count += 1
            else:
                static_count += 1
            weights.append(w)
        
        total_weight = sum(weights)
        evolved_weight_sum = sum(w for (_, p, _), w in zip(temporal_candidates, weights) if getattr(p, 'evolution_history', None))
        logger.info(
            f"Temporal pool: {len(temporal_candidates)} candidates "
            f"({evolved_count} evolved, {static_count} static) | "
            f"weight_mult={evolved_weight_mult} | "
            f"evolved_weight_share={evolved_weight_sum/total_weight*100:.1f}% "
            f"static_weight_share={(total_weight-evolved_weight_sum)/total_weight*100:.1f}%"
        )
        
        # Sample one temporal preference
        selected_pref_id, selected_preference, days_since = random.choices(temporal_candidates, weights=weights, k=1)[0]
        has_evo = bool(getattr(selected_preference, 'evolution_history', None))
        
        logger.info(f"Selected temporal preference {selected_pref_id} (expressed {days_since} days ago, evolved={has_evo})")
        
        # # Generate an event relevant to this preference
        # event = self._generate_event_from_preference(
        #     user_agent, selected_preference, current_date, is_temporal_recall=True
        # )
        
        return selected_pref_id, selected_preference, days_since
    
    def _trigger_evolution_event(
        self,
        user_agent: AgentModel,
        all_preferences: Dict[str, PreferenceModel],
        current_date: datetime,
        previous_events: List[EventModel],
        event_record: List[Dict],
        conversation_record: List[Dict]
    ) -> Optional[Dict[str, PreferenceModel]]:
        """
        Generate a single life event that causally changes multiple (2-5) previously
        expressed preferences. After applying the changes, a short conversation is
        generated where the user mentions the event (without stating any preference
        values) so the model under evaluation knows the event occurred.
        
        Returns:
            Dict of {pref_id: evolved PreferenceModel} or None if no evolution occurred.
        """
        min_days = self.config.get("evolution_event_min_days_since_expressed", 14)
        candidates = []
        for pref_id, pref in all_preferences.items():
            days_since = pref.get_days_since_last_expressed(current_date)
            if days_since is not None and days_since >= min_days:
                candidates.append((pref_id, pref, days_since))
        
        if not candidates:
            return None
        
        num_min = self.config.get("evolution_event_num_preferences_min", 2)
        num_max = self.config.get("evolution_event_num_preferences_max", 5)
        k = random.randint(min(num_min, len(candidates)), min(num_max, len(candidates)))
        
        # Sample k preferences weighted by staleness
        weights = [d for _, _, d in candidates]
        selected = []
        remaining = list(range(len(candidates)))
        remaining_weights = list(weights)
        for _ in range(k):
            if not remaining:
                break
            chosen_indices = random.choices(remaining, weights=remaining_weights, k=1)
            idx = chosen_indices[0]
            selected.append(candidates[idx])
            pos = remaining.index(idx)
            remaining.pop(pos)
            remaining_weights.pop(pos)
        
        if len(selected) < 2 and len(candidates) >= 2:
            return None
        if not selected:
            return None
        
        selected_ids = [s[0] for s in selected]
        logger.info(f"Triggering evolution event for {len(selected)} preferences: {selected_ids}")
        
        user_dict = user_agent.to_dict()
        prefs_block = ""
        for pref_id, pref, days in selected:
            prefs_block += f"\n--- {pref_id} (domain: {pref.metadata.get('domain', 'unknown')}) ---\n"
            prefs_block += json.dumps(pref.to_dict(), indent=2, default=str) + "\n"
        
        prompt = f"""Multiple user preferences are about to change due to a SINGLE significant life event.

USER PROFILE:
{json.dumps(user_dict, indent=2, default=str)}

PREFERENCES TO EVOLVE:
{prefs_block}

Generate ONE realistic life event that would MEANINGFULLY change at least one attribute
in EACH of the preferences listed above.

Requirements:
- The event must be causally connected to ALL the preference domains
- It must be plausible for this user's life context
- Changed attributes should be QUALITATIVELY DIFFERENT from current values
- Every listed preference must have at least one attribute change

Return JSON:
{{
  "event_name": "Brief name of the life event",
  "event_category": "Category (e.g., career, health, relationship, personal_growth)",
  "event_description": "1-2 sentence description of what happened",
  "preference_changes": {{
    "<preference_id>": {{
      "<attribute_name>": {{
        "new_value": "the new value after the event",
        "reason": "why this event caused this change"
      }}
    }}
  }}
}}

Every preference ID listed above MUST appear in preference_changes with at least one attribute change.

Please respond with clean JSON only, without explanations or code blocks, do not include "```json"."""
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            result = json.loads(response["response_text"])
        except Exception as e:
            logger.warning(f"Failed to generate evolution event: {e}")
            return None
        
        event_name = result.get("event_name", "Life event")
        event_category = result.get("event_category", "life_event")
        event_description = result.get("event_description", "")
        all_pref_changes = result.get("preference_changes", {})
        
        if not all_pref_changes:
            logger.debug("LLM returned no preference changes, skipping evolution event")
            return None
        
        evolved_prefs = {}
        total_changes = 0
        
        for pref_id, pref, _ in selected:
            pref_changes_spec = all_pref_changes.get(pref_id, {})
            if not pref_changes_spec:
                continue
            
            evolved_pref = pref.clone()
            attr_changes_record = {}
            
            for attr_name, change_info in pref_changes_spec.items():
                new_value = change_info.get("new_value") if isinstance(change_info, dict) else change_info
                if new_value is not None and evolved_pref.has_attribute(attr_name):
                    old_value = evolved_pref.get_attribute(attr_name)
                    if old_value != new_value:
                        evolved_pref.set_attribute(attr_name, new_value)
                        attr_changes_record[attr_name] = {"from": old_value, "to": new_value}
            
            if not attr_changes_record:
                continue
            
            evolved_pref.record_evolution(
                attribute_changes=attr_changes_record,
                event_date=current_date,
                event_name=event_name,
                event_category=event_category,
                trigger="evolution_event"
            )
            
            evolved_prefs[pref_id] = evolved_pref
            total_changes += len(attr_changes_record)
            
            changes_str = ', '.join(
                '{}: {} -> {}'.format(a, c['from'], c['to'])
                for a, c in attr_changes_record.items()
            )
            logger.info(f"  {pref_id}: {changes_str}")
        
        if not evolved_prefs:
            return None
        
        logger.info(f"Evolution event '{event_name}' changed {total_changes} attributes "
                     f"across {len(evolved_prefs)} preferences")
        
        self._generate_evolution_event_conversation(
            user_agent=user_agent,
            event_name=event_name,
            event_category=event_category,
            event_description=event_description,
            current_date=current_date,
            conversation_record=conversation_record,
            event_record=event_record
        )
        
        return evolved_prefs
    
    def _generate_evolution_event_conversation(
        self,
        user_agent: AgentModel,
        event_name: str,
        event_category: str,
        event_description: str,
        current_date: datetime,
        conversation_record: List[Dict],
        event_record: List[Dict]
    ) -> None:
        """
        Generate a short (2-4 turn) conversation where the user casually mentions
        the life event that caused preference evolution. The conversation does NOT
        mention any preferences — it just establishes that the event occurred so
        the model under evaluation has that context.
        """
        user_name = user_agent.get_formatted_name()
        user_dict = user_agent.to_dict()
        
        prompt = f"""Generate a SHORT, realistic conversation (2-4 turns total) between a user and an AI assistant.

USER: {user_name}
USER PROFILE SUMMARY: {user_dict.get('attributes', {}).get('occupation', {}).get('title', '')}, {user_dict.get('attributes', {}).get('age', '')} years old, from {user_dict.get('attributes', {}).get('location', '')}

LIFE EVENT THAT JUST HAPPENED: {event_name}
DESCRIPTION: {event_description}

The user casually mentions this life event and asks for practical help with something
related to the event. The conversation should:
- Be SHORT (2-4 turns, where 1 turn = 1 user message + 1 assistant response)
- Have the user mention the event naturally (e.g., "So I just [event]..." or "Hey, [event] happened and I need help with...")
- Focus on a PRACTICAL request related to the event (NOT about personal preferences, feelings, or how the event changed them)
- The assistant responds helpfully and briefly
- Sound natural and casual (like a real chat)

DO NOT mention any specific preference values, personal style preferences, or how the event
changed the user's tastes/preferences. Keep it purely about the practical situation.

Return JSON:
{{
  "turns": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Please respond with clean JSON only, without explanations or code blocks, do not include "```json"."""
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            convo_data = json.loads(response["response_text"])
        except Exception as e:
            logger.warning(f"Failed to generate evolution event conversation: {e}")
            return
        
        turns = convo_data.get("turns", [])
        if not turns:
            return
        
        conversation = {
            "turns": turns,
            "metadata": {
                "type": "evolution_event",
                "event_name": event_name,
                "event_category": event_category,
                "event_description": event_description,
                "num_turns": len([t for t in turns if t.get("role") == "user"])
            }
        }
        
        user_dir = os.path.join(
            self.config["output_dir"], "conversations", user_name
        )
        os.makedirs(user_dir, exist_ok=True)
        
        event_index = len(event_record)
        date_str = current_date.strftime("%Y%m%d")
        safe_event_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in event_name[:40])
        filename = f"{event_index}_{date_str}_evolution_{safe_event_name}_conversation.json"
        filepath = os.path.join(user_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                "event": {
                    "name": event_name,
                    "category": event_category,
                    "description": event_description,
                    "date": current_date.isoformat(),
                    "type": "evolution_event"
                },
                "conversation": conversation,
                "user": user_agent.to_dict()
            }, f, indent=2)
        
        logger.info(f"Saved evolution event conversation to {filepath}")
        
        conversation_record.append({
            "date": current_date.isoformat(),
            "date_unix": int(current_date.timestamp()),
            "event_id": f"evolution_event_{event_name}",
            "conversation": conversation,
            "intent_fulfillment": None,
            "user_agent": user_agent.to_dict(),
            "assistant_agent": None
        })
    
    def _generate_event_from_preference(
        self,
        user_agent: AgentModel,
        preference: PreferenceModel,
        current_date: datetime,
        is_temporal_recall: bool = False
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
        prompt = f"""
        Generate an event that would be specifically relevant to this user's preference.
        
        User: {json.dumps(user_agent.to_dict(), indent=2)}
        
        Preference: {json.dumps(preference.to_dict(), indent=2)}
        
        {"This is for temporal preference recall - the user expressed this preference a while ago." if is_temporal_recall else ""}
        
        Generate an event that would naturally trigger this preference. The event should:
        - Be realistic and fit the user's life context
        - Directly involve the preference domain
        - Create a natural opportunity for the preference to be expressed or recalled
        
        Return a JSON object with this structure:
        {{
            "id": "generated_event_id",
            "name": "Event name",
            "category": "event_category",
            "description": "Extremely concise description of the event",
            "attributes": {{
                "key": "value"
            }},
            "preference_dependent": {{
                "attribute_name": true/false
            }}
        }}
        """
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            event_data = json.loads(response["response_text"])
            
            # Create event model
            event = EventModel(
                structure_id=event_data["id"],
                name=event_data["name"],
                category=event_data["category"],
                base_attributes={
                    k: v for k, v in event_data["attributes"].items()
                    if not event_data.get("preference_dependent", {}).get(k, False)
                },
                preference_dependent_attributes=event_data.get("preference_dependent", {}),
                date=current_date
            )
            
            # Add metadata to indicate this is a temporal preference event
            event.metadata["generated_for_temporal_preference"] = preference.structure_id
            event.metadata["temporal_recall"] = is_temporal_recall
            
            return event
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to generate temporal preference event: {e}")
            # Fallback to regular event sampling
            return None
    
    def _identify_relevant_preferences(
        self,
        event: EventModel,
        current_preferences: Optional[Dict[str, PreferenceModel]] = None,
        user_intent: Optional[IntentModel] = None
    ) -> Dict[str, float]:
        """
        Identify preference structures most relevant to the given event and user intent.
        Uses a two-step process:
        1. First identifies relevant domains for the event and user intent
        2. Then identifies relevant preferences from those domains
        
        For temporal preference events, ensures the original preference gets high relevance.
        
        Args:
            event: The event model
            current_preferences: Optional current preference states
            user_intent: Optional user intent model for conditioning
            
        Returns:
            Dictionary mapping preference IDs to relevance scores (0-1)
        """
        # Handle temporal preference events specially
        if event.metadata.get("temporal_recall", False):
            temporal_pref_id = event.metadata.get("generated_for_temporal_preference")
            if temporal_pref_id:
                logger.debug(f"Event is temporal recall for preference {temporal_pref_id}")
                # Start with the temporal preference having high relevance
                relevant_preferences = {temporal_pref_id: 1.0}
                
                # Still identify other relevant preferences but give them lower scores
                other_relevant = self._identify_regular_relevant_preferences(
                    event, current_preferences, user_intent, exclude_prefs={temporal_pref_id}
                )
                
                # Add other relevant preferences with reduced scores
                for pref_id, score in other_relevant.items():
                    if pref_id not in relevant_preferences:
                        relevant_preferences[pref_id] = score * 0.7  # Reduce by 30%
                
                return relevant_preferences
        
        # Regular preference identification
        return self._identify_regular_relevant_preferences(event, current_preferences, user_intent)
    
    def _identify_regular_relevant_preferences(
        self,
        event: EventModel,
        current_preferences: Optional[Dict[str, PreferenceModel]] = None,
        user_intent: Optional[IntentModel] = None,
        exclude_prefs: Optional[set] = None
    ) -> Dict[str, float]:
        """
        Regular preference identification logic (extracted from original method).
        
        Args:
            event: The event model
            current_preferences: Optional current preference states
            user_intent: Optional user intent model for conditioning
            exclude_prefs: Optional set of preference IDs to exclude
            
        Returns:
            Dictionary mapping preference IDs to relevance scores (0-1)
        """
        exclude_prefs = exclude_prefs or set()
        
        # Get all preference structures and available domains
        available_domains = self.preference_library.get_preference_domains()
        
        # Step 0: Generate a new preference domain if needed
        generate_new_preference_domain = random.random() < self.config.get("new_preference_domain_probability", 0.3)
        if generate_new_preference_domain and len(available_domains) < self.config["max_preference_domains"]:
            # generate a new preference domain that best fits the event
            new_domain_prompt = (
            "For the following event, generate a new preference domain that is most relevant"
            " and that is NOT already in the available domains.\n\n"
            f"Event: {json.dumps(event.to_dict(), indent=2)}\n\n"
            f"Available domains: {json.dumps(available_domains, indent=2)}\n\n"
            "Return the new domain name as a string and nothing else. Or return 'NONE' if no new domain is relevant."
            )
            new_domain_response = self.llm_client.generate(new_domain_prompt, response_format="text")["response_text"]
            if new_domain_response.lower() == "none":
                new_domain = None
            else:
                new_domain = new_domain_response.strip().strip('"')
                available_domains.append(new_domain)

            # initialize preference structures for the new domain
            self.preference_library.initialize_new_domain(new_domain, count_per_domain=self.config.get("num_initial_preferences_per_domain", 5))

        # Step 1: Identify relevant domains
        domain_prompt = (
            "For the following event and user intent, identify which domains are most relevant.\n\n"
            f"Event: {json.dumps(event.to_dict(), indent=2)}\n\n"
            f"User Intent: {json.dumps(user_intent.to_dict(), indent=2) if user_intent else 'No specific user intent provided'}\n\n"
            f"Available domains: {json.dumps(available_domains, indent=2)}\n\n"
            "Return a JSON object mapping domain names to relevance scores (0.0 to 1.0).\n\n"
            "Consider both the event context and the user's intent when determining relevance.\n"
            "Only include domains that have a relevance score > 0.2."
        )


        domain_response = self.llm_client.generate(domain_prompt, response_format="json")
        relevant_domains = json.loads(domain_response["response_text"])

        valid_domains = {}
        for domain, score in relevant_domains.items():
            if domain not in available_domains:
                logger.warning(f"LLM returned unknown domain: {domain}")
                continue

            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                logger.warning(f"Invalid relevance score for domain {domain}: {score}")
                continue

            valid_domains[domain] = float(score)

        top_domains = [domain for domain, score in valid_domains.items() if score > 0.2]

        if not top_domains:
            logger.warning("No relevant domains identified, using all domains")
            top_domains = available_domains

        # Step 2.0: For each relevant domain, generate a new preference structure if needed
        for domain in top_domains:
            generate_new_preference_struct = random.random() < self.config.get("new_preference_structure_probability", 0.3)
            if generate_new_preference_struct and len(self.preference_library.get_preferences_by_domain(domain)) < self.config.get("max_preference_structures_per_domain", 20):
                pref = self.preference_library.generate_synthetic_preference(domain=domain, event=event)
                pref_id = f"{domain}_{self.preference_library.get_structure_count() + 1}"
                self.preference_library.add_preference(pref_id, pref)
                self.preference_library.save_to_file(self.config.get("libraries", {}).get("preference_library_path"))
            

        # Step 2: Identify relevant preferences from the selected domains
        # Filter preference structures to only those in relevant domains
        all_preference_structures = self.preference_library.get_all_structures()
        domain_preferences = {
            pref_id: struct for pref_id, struct in all_preference_structures.items()
            if struct.get("domain", "unknown") in top_domains
        }
        
        # Sample a subset if there are too many
        max_structures = 200
        if len(domain_preferences) > max_structures:
            sampled_structures = dict(random.sample(list(domain_preferences.items()), max_structures))
        else:
            sampled_structures = domain_preferences
        
        available_preference_structures_str = json.dumps({pref_id: {
            "name": struct.get("name", pref_id), 
            "domain": struct.get("domain", "unknown"), 
            "description": struct.get("description", "")
            } for pref_id, struct in sampled_structures.items()}, indent=2)
        
        preference_prompt = (
        "For the following event and user intent, determine which preference structures are most relevant.\n"
        "Assign a relevance score from 0.0 to 1.0 for each preference structure.\n\n"
        f"Event: {json.dumps(event.to_dict(), indent=2)}\n"
        f"User Intent: {json.dumps(user_intent.to_dict(), indent=2) if user_intent else 'No specific user intent provided'}\n"
        f"Relevant domains: {json.dumps(top_domains, indent=2)}\n"
        f"Available preference structures:\n"
        f'{available_preference_structures_str}\n\n'
        "Consider both the event context and the user's intent when determining relevance.\n"
        "Return a JSON object mapping preference structure IDs to relevance scores (0.0 to 1.0).\n"
        "Only include preferences that have a relevance score > 0.2."
        )
        
        preference_response = self.llm_client.generate(preference_prompt, response_format="json")
        relevance_scores = json.loads(preference_response["response_text"])

        valid_scores = {}
        for pref_id, score in relevance_scores.items():
            if pref_id not in all_preference_structures:
                logger.warning(f"LLM returned unknown preference ID: {pref_id}")
                continue

            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                logger.warning(f"Invalid relevance score for {pref_id}: {score}")
                continue

            valid_scores[pref_id] = float(score)

        valid_scores = {
            pref_id: score for pref_id, score in valid_scores.items()
            if pref_id not in exclude_prefs
        }

        num_prefs = self.config["relevant_preferences_per_event"] + random.randint(0,3) - random.randint(0,3)
        top_preferences = dict(sorted(
            valid_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:num_prefs])

        return top_preferences

    def _evolve_preferences_with_intent(
        self,
        agent: AgentModel,
        event: EventModel,
        current_preferences: Dict[str, PreferenceModel],
        relevant_pref_ids: Dict[str, float],
        agent_intent: IntentModel,
        previous_events: List[EventModel]
    ) -> Dict[str, PreferenceModel]:
        """
        Evolve preferences based on the current event, relevance, and agent intent.
        Will initialize preferences that haven't been initialized yet.
        
        Args:
            agent: Agent model (user or assistant)
            event: Current event
            current_preferences: Current preference states
            relevant_pref_ids: Mapping of preference IDs to relevance scores
            agent_intent: Agent's intent for this interaction
            previous_events: List of previous events
            
        Returns:
            Updated preferences dictionary
        """
        # Create a deep copy of current preferences to modify
        updated_preferences = {
            pref_id: pref_model.clone() for pref_id, pref_model in current_preferences.items()
        }
        
        # Process each relevant preference
        for pref_id, relevance_score in relevant_pref_ids.items():
            # Initialize preference if it doesn't exist yet
            is_newly_initialized = False
            if pref_id not in updated_preferences:
                initialized_prefs = self._initialize_agent_preferences(agent, [pref_id])
                if pref_id in initialized_prefs:
                    updated_preferences[pref_id] = initialized_prefs[pref_id]
                    updated_preferences[pref_id].snapshot_original_attributes()
                    is_newly_initialized = True
            
            # Snapshot if not yet done (for preferences that existed before evolution tracking)
            if updated_preferences.get(pref_id) and updated_preferences[pref_id].original_expressed_attributes is None:
                updated_preferences[pref_id].snapshot_original_attributes()
            
            # Set expression date on first initialization so the preference
            # becomes eligible for temporal recall after the min-days threshold.
            if is_newly_initialized and pref_id in updated_preferences:
                updated_preferences[pref_id].update_expression_tracking(
                    date=event.date,
                    context={
                        "event_id": event.id,
                        "event_name": event.name,
                        "event_category": event.category,
                        "relevance_score": relevance_score,
                        "is_first_expression": True
                    }
                )
            
            # Skip evolution if relevance is too low
            if relevance_score < 0.2:
                continue
            
            # Capture pre-evolution attribute values for change tracking
            pre_evolution_attrs = updated_preferences[pref_id].get_all_attributes() if pref_id in updated_preferences else {}
            
            event_context = f"{event.category}_{agent_intent.type}" if hasattr(agent_intent, 'type') else event.category
            updated_pref = self.preference_evolve_manager.evolve_preference(
                preference=updated_preferences[pref_id],
                user=agent,
                event=event,
                relevance_score=relevance_score,
                preference_history=[
                    event_record.get_preference(pref_id) for event_record in previous_events
                    if event_record.has_preference(pref_id)
                ],
                event_context=event_context
            )

            # Record evolution if attributes actually changed
            post_evolution_attrs = updated_pref.get_all_attributes()
            attr_changes = {}
            for attr_name in set(list(pre_evolution_attrs.keys()) + list(post_evolution_attrs.keys())):
                old_val = pre_evolution_attrs.get(attr_name)
                new_val = post_evolution_attrs.get(attr_name)
                if old_val != new_val:
                    attr_changes[attr_name] = {'from': old_val, 'to': new_val}
            if attr_changes:
                updated_pref.record_evolution(
                    attribute_changes=attr_changes,
                    event_date=event.date if hasattr(event, 'date') else None,
                    event_name=event.name if hasattr(event, 'name') else None,
                    event_category=event.category if hasattr(event, 'category') else None,
                    trigger='contextual_evolution'
                )

            updated_preferences[pref_id] = updated_pref
        
        # Apply interdependence - changes in one preference may affect related preferences
        updated_preferences = self.preference_evolve_manager.apply_interdependence(
            updated_preferences, 
            current_preferences, 
            relevant_pref_ids
        )
        
        return updated_preferences

    def _evolve_preferences(
        self,
        user_agent: AgentModel,
        event: EventModel,
        current_preferences: Dict[str, PreferenceModel],
        relevant_preferences: Dict[str, float],
        previous_events: List[EventModel]
    ) -> Dict[str, PreferenceModel]:
        """
        Evolve preferences based on the current event and relevance.
        Will initialize preferences that haven't been initialized yet.
        
        Args:
            user_agent: User agent model
            event: Current event
            current_preferences: Current preference states
            relevant_preferences: Mapping of preference IDs to relevance scores
            previous_events: List of previous events
            
        Returns:
            Updated preferences dictionary
        """
        # Create a deep copy of current preferences to modify
        updated_preferences = {
            pref_id: pref_model.clone() for pref_id, pref_model in current_preferences.items()
        }
        
        # Process each relevant preference
        for pref_id, relevance_score in relevant_preferences.items():
            # Initialize preference if it doesn't exist yet
            is_newly_initialized = False
            if pref_id not in updated_preferences:
                initialized_prefs = self._initialize_agent_preferences(user_agent, [pref_id])
                if pref_id in initialized_prefs:
                    updated_preferences[pref_id] = initialized_prefs[pref_id]
                    updated_preferences[pref_id].snapshot_original_attributes()
                    is_newly_initialized = True
            
            if updated_preferences.get(pref_id) and updated_preferences[pref_id].original_expressed_attributes is None:
                updated_preferences[pref_id].snapshot_original_attributes()
            
            if is_newly_initialized and pref_id in updated_preferences:
                updated_preferences[pref_id].update_expression_tracking(
                    date=event.date,
                    context={
                        "event_id": event.id,
                        "event_name": event.name,
                        "event_category": event.category,
                        "relevance_score": relevance_score,
                        "is_first_expression": True
                    }
                )
            
            # Skip evolution if relevance is too low
            if relevance_score < 0.2:
                continue
            
            pre_evolution_attrs = updated_preferences[pref_id].get_all_attributes() if pref_id in updated_preferences else {}
            
            updated_pref = self.preference_evolve_manager.evolve_preference(
                preference=updated_preferences[pref_id],
                user=user_agent,
                event=event,
                relevance_score=relevance_score,
                preference_history=[
                    event_record.get_preference(pref_id) for event_record in previous_events
                    if event_record.has_preference(pref_id)
                ]
            )
            
            # Record evolution if attributes actually changed
            post_evolution_attrs = updated_pref.get_all_attributes()
            attr_changes = {}
            for attr_name in set(list(pre_evolution_attrs.keys()) + list(post_evolution_attrs.keys())):
                old_val = pre_evolution_attrs.get(attr_name)
                new_val = post_evolution_attrs.get(attr_name)
                if old_val != new_val:
                    attr_changes[attr_name] = {'from': old_val, 'to': new_val}
            if attr_changes:
                updated_pref.record_evolution(
                    attribute_changes=attr_changes,
                    event_date=event.date if hasattr(event, 'date') else None,
                    event_name=event.name if hasattr(event, 'name') else None,
                    event_category=event.category if hasattr(event, 'category') else None,
                    trigger='contextual_evolution'
                )
            
            updated_preferences[pref_id] = updated_pref
        
        # Apply interdependence - changes in one preference may affect related preferences
        updated_preferences = self.preference_evolve_manager.apply_interdependence(
            updated_preferences, 
            current_preferences, 
            relevant_preferences
        )
        
        return updated_preferences
    
    def _finalize_event(
        self,
        event: EventModel,
        user_agent: AgentModel,
        preferences: Dict[str, PreferenceModel]
    ) -> EventModel:
        """
        Finalize event details based on user preferences.
        
        Args:
            event: Event model with base attributes
            user_agent: User agent model
            preferences: Current preference states
            
        Returns:
            Fully instantiated event with all attributes
        """
        # Get the preference-dependent attributes from event
        pref_dependent_attrs = event.get_preference_dependent_attributes()
        
        if not pref_dependent_attrs:
            return event  # No preference-dependent attributes to fill
        
        # Create a prompt for the LLM to generate preference-dependent event attributes
        prompt = f"""
        Based on the following user, event, and preference information, generate realistic values for the preference-dependent event attributes.
        
        User: {json.dumps(user_agent.to_dict(), indent=2)}
        
        Event (base attributes): {json.dumps(event.to_dict(), indent=2)}
        
        Current preferences: {json.dumps({pid: pref.to_dict() for pid, pref in preferences.items()}, indent=2)}
        
        Preference-dependent attributes to generate: {json.dumps(pref_dependent_attrs, indent=2)}
        
        Return a JSON object containing only the preference-dependent attributes and their values:
        """
        
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            generated_attrs = json.loads(response["response_text"])
            
            # Set the generated attributes on the event
            for attr, value in generated_attrs.items():
                if attr in pref_dependent_attrs:
                    event.set_attribute(attr, value, is_preference_dependent=True)
            
            # Generate a one-sentence description of the event
            description_prompt = f"""
            Write a single, concise sentence describing this event, taking into account:
            - The user's persona and attributes
            - The event's attributes (both base and preference-dependent)
            - The user's relevant preferences
            
            User: {json.dumps(user_agent.to_dict(), indent=2)}
            
            Event: {json.dumps(event.to_dict(), indent=2)}
            
            Relevant preferences: {json.dumps({pid: pref.to_dict() for pid, pref in preferences.items()}, indent=2)}
            
            Return a single sentence that captures the essence of this event in the context of this user's preferences and persona.
            """
            
            description_response = self.llm_client.generate(description_prompt)["response_text"]
            event.set_attribute("description", description_response.strip())
                    
            return event
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            return event
    
    def _generate_conversation_with_intents(
        self,
        user_agent: AgentModel,
        assistant_agent: AgentModel,
        event: EventModel,
        current_preferences_user: Dict[str, PreferenceModel],
        updated_preferences_user: Dict[str, PreferenceModel],
        current_preferences_assistant: Dict[str, PreferenceModel],
        updated_preferences_assistant: Dict[str, PreferenceModel],
        agent_intents: Dict[str, IntentModel],
        previous_events: List[EventModel],
        user_id: Optional[str] = None,
        event_index: Optional[int] = None,
        event_date: Optional[datetime] = None,
        temporal_pref_id: Optional[str] = None,
        all_preferences_user: Optional[Dict[str, PreferenceModel]] = None
    ) -> Dict:
        """
        Generate a conversation related to the event that reflects user preferences.
        
        Args:
            user_agent: User agent model
            assistant_agent: Assistant agent model
            event: Current event
            current_preferences_user: User's current preference states before evolution
            updated_preferences_user: User's updated preference states after evolution
            current_preferences_assistant: Assistant's current preference states before evolution
            updated_preferences_assistant: Assistant's updated preference states after evolution
            agent_intents: Dictionary mapping agent IDs to their IntentModel instances
            previous_events: List of previous events
            temporal_pref_id: ID of temporal preference if this is a temporal recall event
            
        Returns:
            Dictionary containing the conversation
        """
        # Handle temporal preference events specially
        recalled_preference = None
        if temporal_pref_id and all_preferences_user:
            # Get the temporal preference for assistant recall from all preferences
            if temporal_pref_id in all_preferences_user:
                recalled_preference = all_preferences_user[temporal_pref_id]
                logger.debug(f"Setting up temporal preference recall for {temporal_pref_id}")

                # Remove from current conversation context (user shouldn't re-express it)
                if temporal_pref_id in current_preferences_user:
                    current_preferences_user.pop(temporal_pref_id)
                    logger.info(f"Removed temporal preference {temporal_pref_id} from user conversation context")
                if temporal_pref_id in updated_preferences_user:
                    updated_preferences_user.pop(temporal_pref_id)
                    logger.info(f"Removed temporal preference {temporal_pref_id} from updated user conversation context")
                

                    
            else:
                logger.warning("PROBLEM: Temporal preference not found in all user preferences")
            
            
        
        # # Extract the most relevant preference models for the user
        # relevant_pref_models_current_user = {
        #     pref_id: current_preferences_user[pref_id] 
        #     for pref_id in updated_preferences_user
        #     if pref_id in current_preferences_user
        # }
        # relevant_pref_models_updated_user = {
        #     pref_id: updated_preferences_user[pref_id]
        #     for pref_id in updated_preferences_user
        # }
        
        # # For temporal events, remove the temporal preference from user preferences
        # # (since the user already expressed it before and shouldn't express it again)
        # if temporal_pref_id:
        #     # relevant_pref_models_current_user.pop(temporal_pref_id, None)
        #     relevant_pref_models_updated_user.pop(temporal_pref_id, None)
        #     logger.debug(f"Removed temporal preference {temporal_pref_id} from user conversation context")
        
        # Extract the most relevant preference models for the assistant
        relevant_pref_models_current_assistant = {
            pref_id: current_preferences_assistant[pref_id] 
            for pref_id in updated_preferences_assistant
            if pref_id in current_preferences_assistant
        }
        relevant_pref_models_updated_assistant = {
            pref_id: updated_preferences_assistant[pref_id]
            for pref_id in updated_preferences_assistant
        }
        # Use the conversation generator to create a conversation
        conversation = self.conversation_generator.generate_conversation(
            user=user_agent,
            assistant=assistant_agent,
            event=event,
            preferences_current_user=current_preferences_user,
            preferences_updated_user=updated_preferences_user,
            preferences_current_assistant=relevant_pref_models_current_assistant,
            preferences_updated_assistant=relevant_pref_models_updated_assistant,
            agent_intents=agent_intents,
            previous_events=previous_events[-5:] if len(previous_events) > 5 else previous_events,
            recalled_preference=recalled_preference
        )
        
        # Create user-specific directory for conversations
        user_dir = os.path.join(
            self.config["output_dir"],
            "conversations",
            f"{user_agent.get_formatted_name()}"
        )
        os.makedirs(user_dir, exist_ok=True)
        
        # Create a filename using event date and type

        try: event_date = event.get_attribute("date", datetime.now()).strftime("%Y%m%d")
        except Exception: event_date = event_date.strftime("%Y%m%d") if event_date is not None else datetime.now().strftime("%Y%m%d")

        event_type = event.get_attribute("type", event.category).lower().replace(" ", "_")
        filename = f"{event_date}_{event_type}_conversation.json" if event_index is None else f"{event_index}_{event_date}_{event_type}_conversation.json"
        filepath = os.path.join(user_dir, filename)
        
        # Save conversation to file
        conversation_for_saving = conversation.copy()
        if "outline" in conversation_for_saving:
            conversation_for_saving.pop("outline")
            
        with open(filepath, 'w') as f:
            json.dump({
                "event": event.to_dict(),
                "conversation": conversation_for_saving,
                "relevant_preferences_user": {
                    pref_id: pref.to_dict() 
                    for pref_id, pref in updated_preferences_user.items()
                },
                "relevant_preferences_assistant": {
                    pref_id: pref.to_dict() 
                    for pref_id, pref in relevant_pref_models_updated_assistant.items()
                },
                "user": user_agent.to_dict(),
                "assistant": assistant_agent.to_dict()
            }, f, indent=2)
            
        logger.info(f"Saved conversation to {filepath}")

        # Save paired expression mode conversation variants
        paired = conversation.get("paired_conversations", {})
        for mode, paired_conv in paired.items():
            mode_dir = os.path.join(
                self.config["output_dir"],
                f"conversations_{mode}",
                f"{user_agent.get_formatted_name()}"
            )
            os.makedirs(mode_dir, exist_ok=True)
            paired_for_saving = {k: v for k, v in paired_conv.items() if k not in ("outline", "paired_conversations")}
            paired_filepath = os.path.join(mode_dir, filename)
            with open(paired_filepath, 'w') as f:
                json.dump({
                    "event": event.to_dict(),
                    "conversation": paired_for_saving,
                    "expression_mode": mode,
                    "relevant_preferences_user": {
                        pref_id: pref.to_dict()
                        for pref_id, pref in updated_preferences_user.items()
                    },
                    "relevant_preferences_assistant": {
                        pref_id: pref.to_dict()
                        for pref_id, pref in relevant_pref_models_updated_assistant.items()
                    },
                    "user": user_agent.to_dict(),
                    "assistant": assistant_agent.to_dict()
                }, f, indent=2)
            logger.info(f"Saved {mode} conversation variant to {paired_filepath}")

        # Save conversation outline to separate file
        if "outline" in conversation:
            outline_filename = filename.replace(".json", "_outline.json")
            outline_filepath = os.path.join(user_dir, outline_filename)
            
            with open(outline_filepath, 'w') as f:
                json.dump({
                    "event": event.to_dict(),
                    "outline": conversation["outline"],
                    "scenario": conversation.get("scenario", ""),
                    "metadata": {
                        "num_turns": conversation.get("metadata", {}).get("num_turns", 0),
                        "generation_timestamp": event.date.isoformat() if event.date else None,
                        "user_preference_ids": list(updated_preferences_user.keys()),
                        "assistant_preference_ids": list(relevant_pref_models_updated_assistant.keys())
                    }
                }, f, indent=2)
                
            logger.info(f"Saved conversation outline to {outline_filepath}")
        
        # Generate counterfactuals for temporal preference recall AFTER conversation generation
        if temporal_pref_id and recalled_preference:
            self._generate_counterfactual_turns(
                conversation=conversation,
                user_agent=user_agent,
                assistant_agent=assistant_agent,
                event=event,
                temporal_pref_id=temporal_pref_id,
                recalled_preference=recalled_preference,
                user_id=user_id,
                event_index=event_index,
                event_date=event_date
            )

            # Generate counterfactuals for paired expression mode variants
            paired = conversation.get("paired_conversations", {})
            for mode, paired_conv in paired.items():
                mode_suffix = f"_{mode}"
                paired_conv_for_counterfactuals = paired_conv.copy()
                paired_conv_for_counterfactuals.pop("paired_conversations", None)
                self._generate_counterfactual_turns(
                    conversation=paired_conv_for_counterfactuals,
                    user_agent=user_agent,
                    assistant_agent=assistant_agent,
                    event=event,
                    temporal_pref_id=temporal_pref_id,
                    recalled_preference=recalled_preference,
                    user_id=user_id,
                    event_index=event_index,
                    event_date=event_date,
                    output_suffix=mode_suffix
                )
        
        return conversation
    
    def _generate_counterfactual_turns(
        self,
        conversation: Dict[str, Any],
        user_agent: AgentModel,
        assistant_agent: AgentModel,
        event: EventModel,
        temporal_pref_id: str,
        recalled_preference: PreferenceModel,
        user_id: Optional[str] = None,
        event_index: Optional[int] = None,
        event_date: Optional[datetime] = None,
        output_suffix: str = ""
    ) -> None:
        """
        Generate counterfactual turns for temporal preference recall events.
        
        This method:
        1. Identifies which turn(s) in the conversation reference the recalled preference
        2. Generates alternative values for the recalled preference attributes
        3. Creates counterfactual assistant responses using the alternative values
        4. Saves the results
        
        Args:
            conversation: The original conversation that was generated
            user_agent: User agent model
            assistant_agent: Assistant agent model
            event: Current event (temporal preference recall event)
            temporal_pref_id: ID of the temporal preference being recalled
            recalled_preference: The actual preference that should be recalled
            user_id: Optional user ID
            event_index: Optional event index
            event_date: Optional event date
            output_suffix: Suffix appended to output filenames (e.g., "_implicit" for paired mode)
        """
        # Check if counterfactual generation is enabled
        benchmark_config = self.config.get("benchmark_generation", {})
        if not benchmark_config.get("enabled", False):
            return
            
        # Check generation probability
        if random.random() > benchmark_config.get("generation_probability", 1.0):
            return
            
        logger.info(f"Generating counterfactual turns for temporal preference recall: {temporal_pref_id}")
        
        # 1. Find which turn(s) contain references to the recalled preference
        turns_with_preference = []
        turns = conversation.get("turns", [])
        
        for turn_idx, turn in enumerate(turns):
            if turn.get("role") == "assistant":
                # Check if this turn appeals to the temporal preference
                preference_appealed = turn.get("preference_appealed", [])
                if temporal_pref_id in preference_appealed:
                    turns_with_preference.append(turn_idx)
        
        if not turns_with_preference:
            logger.warning(f"No assistant turns found that appeal to temporal preference {temporal_pref_id}")
            return
        
        logger.info(f"Found {len(turns_with_preference)} turns that reference the recalled preference: {turns_with_preference}")
        
        # 2. Generate alternative values for the recalled preference (single API call)
        alternative_values = self._generate_alternative_preference_values(
            recalled_preference=recalled_preference,
            temporal_pref_id=temporal_pref_id,
            num_alternatives=4
        )
        
        if not alternative_values:
            logger.warning(f"No alternative values generated for preference {temporal_pref_id}")
            return
        
        # 3. For each turn that references the preference, generate counterfactual responses
        for turn_idx in turns_with_preference:
            # Generate all counterfactuals for this turn in a single API call
            counterfactual_turns = self._generate_counterfactual_turns_batch(
                conversation=conversation,
                turn_idx=turn_idx,
                user_agent=user_agent,
                assistant_agent=assistant_agent,
                temporal_pref_id=temporal_pref_id,
                alternative_values=alternative_values
            )
            
            # 4. Save counterfactuals for this turn
            if len(counterfactual_turns) > 0:
                self._save_counterfactual_turns(
                    conversation=conversation,
                    turn_idx=turn_idx,
                    counterfactual_turns=counterfactual_turns,
                    user_agent=user_agent,
                    event=event,
                    temporal_pref_id=temporal_pref_id,
                    recalled_preference=recalled_preference,
                    event_index=event_index,
                    event_date=event_date,
                    output_suffix=output_suffix
                )
                
                logger.info(f"Generated {len(counterfactual_turns)} counterfactual turns for turn {turn_idx}")
    
    def _generate_alternative_preference_values(
        self,
        recalled_preference: PreferenceModel,
        temporal_pref_id: str,
        num_alternatives: int = 4
    ) -> List[Tuple[str, Any, Any, bool]]:
        """
        Generate alternative values for the recalled preference attributes.
        Injects pre-evolution value as a hard distractor if the preference has evolved.
        
        Args:
            recalled_preference: The preference model
            temporal_pref_id: ID of the preference
            num_alternatives: Number of alternatives to generate
            
        Returns:
            List of tuples (attribute_name, alternative_value, original_value, is_pre_evolution_value)
        """
        # Get all attributes from the preference
        all_attributes = recalled_preference.get_all_attributes()
        
        if not all_attributes:
            logger.warning(f"No attributes found in recalled preference {temporal_pref_id}")
            return []
        
        # Select one attribute to vary
        selected_attr = random.choice(list(all_attributes.keys()))
        original_value = all_attributes[selected_attr]
        
        logger.debug(f"Selected attribute '{selected_attr}' with original value '{original_value}' for counterfactuals")
        
        # Check if preference has evolved - if so, the pre-evolution value is a hard distractor
        evolution_summary = recalled_preference.get_evolution_summary()
        original_expressed = evolution_summary.get("original_attributes", {}) if evolution_summary["has_evolved"] else {}
        pre_evo_val = original_expressed.get(selected_attr)
        
        # Generate alternative values for this attribute
        try:
            alternative_values = get_alternative_values(
                attr_name=selected_attr,
                orig_value=original_value,
                num_alternatives=num_alternatives,
                pref_id=temporal_pref_id,
                llm_client=self.llm_client
            )
            
            results = []
            
            # Inject pre-evolution value as guaranteed hard distractor if it exists and differs
            if pre_evo_val is not None and str(pre_evo_val) != str(original_value):
                results.append((selected_attr, pre_evo_val, original_value, True))
                # Remove from LLM-generated list if it was already there (dedup)
                alternative_values = [v for v in alternative_values if str(v) != str(pre_evo_val)]
                logger.info(f"Injected pre-evolution distractor for {selected_attr}: {pre_evo_val}")
            
            # Add remaining LLM-generated alternatives
            for alt_val in alternative_values:
                if alt_val != original_value and len(results) < num_alternatives:
                    results.append((selected_attr, alt_val, original_value, False))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate alternative values for {selected_attr}: {e}")
            return []
    
    def _generate_counterfactual_turns_batch(
        self,
        conversation: Dict[str, Any],
        turn_idx: int,
        user_agent: AgentModel,
        assistant_agent: AgentModel,
        temporal_pref_id: str,
        alternative_values: List[Tuple[str, Any, Any, bool]]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple counterfactual turns for different alternative values in a single API call.
        
        Args:
            conversation: The original conversation
            turn_idx: Index of the turn to create counterfactuals for
            user_agent: User agent model
            assistant_agent: Assistant agent model
            temporal_pref_id: ID of the temporal preference
            alternative_values: List of tuples (attribute_name, alternative_value, original_value)
            
        Returns:
            List of counterfactual turn dictionaries
        """
        turns = conversation.get("turns", [])
        if turn_idx >= len(turns):
            logger.error(f"Turn index {turn_idx} out of range for conversation with {len(turns)} turns")
            return []
        
        if not alternative_values:
            logger.warning("No alternative values provided for counterfactual generation")
            return []
        
        original_turn = turns[turn_idx]
        
        # Build context from previous turns (last 5 for efficiency)
        context_turns = turns[:turn_idx][-5:]
        context = "\n".join([
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in context_turns
        ])
        
        # Extract attribute info (assuming all alternatives are for the same attribute)
        attribute_name = alternative_values[0][0]
        original_value = alternative_values[0][2]
        
        # Create prompt for generating multiple counterfactual turns
        prompt = f"""You are generating diverse alternative assistant responses based on different preference values. Each response should reflect a different recalled preference value.

CONTEXT:
{context}

ORIGINAL ASSISTANT RESPONSE:
{original_turn['content']}

TEMPORAL PREFERENCE BEING RECALLED:
- Preference ID: {temporal_pref_id}
- Attribute: {attribute_name}
- Original value: {original_value}

TASK: Generate {len(alternative_values)} DIVERSE alternative responses that each reflect a different preference value. Each response should:
1. Naturally reference the specific alternative preference value
2. Maintain the same conversational tone and helpfulness
3. Be significantly different from the others in how it reflects the preference
4. Keep the same general structure but vary the specific preference content
5. Only change the specific preference value being recalled

ALTERNATIVE PREFERENCE VALUES AND REQUIRED RESPONSES:
"""
        
        # Add each alternative value to the prompt
        option_keys = ["A", "B", "C", "D", "E", "F"]  # Support up to 6 alternatives
        for i, alt_tuple in enumerate(alternative_values[:6]):
            attr_name, alt_value, orig_value = alt_tuple[0], alt_tuple[1], alt_tuple[2]
            option_key = option_keys[i]
            prompt += f"\nOption {option_key} (Preference: {attr_name} = {alt_value}):\n"
            prompt += f"Generate a response that naturally recalls and reflects this preference value: {alt_value}\n"
        
        prompt += f"""

Return your response in the following JSON format:
{{"""
        
        for i, alt_tuple in enumerate(alternative_values[:6]):
            alt_value = alt_tuple[1]
            option_key = option_keys[i]
            preference_appealed_json = json.dumps(original_turn.get('preference_appealed', []))
            prompt += f"""
  "{option_key}": {{
    "role": "assistant",
    "content": "Response that naturally recalls preference value: {alt_value}",
    "preference_appealed": {preference_appealed_json}
  }}"""
            if i < len(alternative_values) - 1 and i < 5:
                prompt += ","
        
        prompt += f"""
}}

Make sure each response:
- Clearly reflects its assigned preference value in a natural way
- Is substantially different from the others
- Maintains the helpful, conversational tone
- References the preference value naturally in context
"""
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            generated_responses = json.loads(response["response_text"])
            
            # Create results for each alternative
            counterfactual_turns = []
            
            for i, alt_tuple in enumerate(alternative_values[:6]):
                attr_name, alt_value, orig_value = alt_tuple[0], alt_tuple[1], alt_tuple[2]
                is_pre_evo = alt_tuple[3] if len(alt_tuple) > 3 else False
                option_key = option_keys[i]
                
                if option_key in generated_responses:
                    cf_turn_data = generated_responses[option_key]
                    
                    # Ensure the response has the correct structure
                    if isinstance(cf_turn_data, dict) and "content" in cf_turn_data:
                        counterfactual_turns.append({
                            "attribute": attr_name,
                            "original_value": orig_value,
                            "alternative_value": alt_value,
                            "is_pre_evolution_distractor": is_pre_evo,
                            "counterfactual_turn": {
                                "role": cf_turn_data.get("role", "assistant"),
                                "content": cf_turn_data["content"],
                                "preference_appealed": cf_turn_data.get("preference_appealed", original_turn.get('preference_appealed', []))
                            }
                        })
                        
                        logger.debug(f"Generated counterfactual turn for {attr_name}: {orig_value} -> {alt_value}")
                    else:
                        logger.warning(f"Invalid response format for option {option_key}")
                else:
                    logger.warning(f"No response generated for option {option_key} (value: {alt_value})")
            
            return counterfactual_turns
            
        except json.JSONDecodeError as e:
            if self.config.get("dev_mode", False):
                raise json.JSONDecodeError(f"Failed to parse JSON response for counterfactual turns: {e}", e.doc, e.pos)
            
            logger.error(f"Failed to parse JSON response for counterfactual turns: {e}")
            # Fallback to individual generation if batch fails
            return self._generate_counterfactual_turns_fallback(
                conversation, turn_idx, user_agent, assistant_agent, temporal_pref_id, alternative_values
            )
        except Exception as e:
            logger.error(f"Failed to generate counterfactual turns batch: {e}")
            # Fallback to individual generation if batch fails
            return self._generate_counterfactual_turns_fallback(
                conversation, turn_idx, user_agent, assistant_agent, temporal_pref_id, alternative_values
            )
    
    def _generate_counterfactual_turns_fallback(
        self,
        conversation: Dict[str, Any],
        turn_idx: int,
        user_agent: AgentModel,
        assistant_agent: AgentModel,
        temporal_pref_id: str,
        alternative_values: List[Tuple[str, Any, Any, bool]]
    ) -> List[Dict[str, Any]]:
        """
        Fallback method to generate counterfactual turns individually if batch generation fails.
        
        Args:
            conversation: The original conversation
            turn_idx: Index of the turn to create counterfactuals for
            user_agent: User agent model
            assistant_agent: Assistant agent model
            temporal_pref_id: ID of the temporal preference
            alternative_values: List of tuples (attribute_name, alternative_value, original_value)
            
        Returns:
            List of counterfactual turn dictionaries
        """
        logger.info("Using fallback individual generation for counterfactual turns")
        
        counterfactual_turns = []
        turns = conversation.get("turns", [])
        original_turn = turns[turn_idx]
        
        # Build context from previous turns
        context_turns = turns[:turn_idx]
        context = "\n".join([
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in context_turns
        ])
        
        for alt_tuple in alternative_values:
            attr_name, alt_value, orig_value = alt_tuple[0], alt_tuple[1], alt_tuple[2]
            is_pre_evo = alt_tuple[3] if len(alt_tuple) > 3 else False
            # Create prompt for generating single counterfactual turn
            prompt = f"""
            Generate a counterfactual assistant response for the following conversation context.
            
            CONTEXT:
            {context}
            
            ORIGINAL ASSISTANT RESPONSE:
            {original_turn['content']}
            
            TEMPORAL PREFERENCE BEING RECALLED:
            - Preference ID: {temporal_pref_id}
            - Attribute: {attr_name}
            - Original value: {orig_value}
            - COUNTERFACTUAL value: {alt_value}
            
            Generate a new assistant response that recalls the COUNTERFACTUAL value instead of the original value.
            The response should:
            1. Naturally reference the counterfactual value ({alt_value}) instead of the original value ({orig_value})
            2. Maintain the same conversational tone and style
            3. Keep the same general structure and helpfulness
            4. Only change the specific preference value being recalled
            
            Return a JSON object with:
            {{
                "role": "assistant",
                "content": "The counterfactual response",
                "preference_appealed": {json.dumps(original_turn.get('preference_appealed', []))}
            }}
            """
            
            try:
                response = self.llm_client.generate(prompt, response_format="json")
                counterfactual_turn = json.loads(response["response_text"])
                
                counterfactual_turns.append({
                    "attribute": attr_name,
                    "original_value": orig_value,
                    "alternative_value": alt_value,
                    "is_pre_evolution_distractor": is_pre_evo,
                    "counterfactual_turn": counterfactual_turn
                })
                
                logger.debug(f"Generated fallback counterfactual turn for {attr_name}: {orig_value} -> {alt_value}")
                
            except Exception as e:
                logger.error(f"Failed to generate fallback counterfactual turn for {attr_name}={alt_value}: {e}")
                continue
        
        return counterfactual_turns
    
    def _save_counterfactual_turns(
        self,
        conversation: Dict[str, Any],
        turn_idx: int,
        counterfactual_turns: List[Dict[str, Any]],
        user_agent: AgentModel,
        event: EventModel,
        temporal_pref_id: str,
        recalled_preference: PreferenceModel,
        event_index: Optional[int] = None,
        event_date: Optional[datetime] = None,
        output_suffix: str = ""
    ) -> None:
        """
        Save counterfactual turns to file.
        
        Args:
            conversation: The original conversation
            turn_idx: Index of the turn that was varied
            counterfactual_turns: List of counterfactual turns
            user_agent: User agent model
            event: Event model
            temporal_pref_id: ID of the temporal preference
            recalled_preference: The recalled preference model
            event_index: Optional event index
            event_date: Optional event date
            output_suffix: Suffix for output directory (e.g., "_implicit" for paired mode)
        """
        # Create benchmark output directory
        subdir = "temporal_preference_recall" + output_suffix
        benchmark_output_dir = os.path.join(
            self.config["output_dir"],
            self.config.get("benchmark_generation", {}).get("output_dir", "benchmark"),
            subdir
        )
        os.makedirs(benchmark_output_dir, exist_ok=True)
        
        # Create filename
        try:
            event_date_attr = event.get_attribute("date", datetime.now())
            if isinstance(event_date_attr, str):
                # If it's a string, try to parse it
                try:
                    event_date_str = datetime.fromisoformat(event_date_attr.replace('Z', '+00:00')).strftime("%Y%m%d")
                except (ValueError, TypeError):
                    event_date_str = datetime.now().strftime("%Y%m%d")
            else:
                event_date_str = event_date_attr.strftime("%Y%m%d")
        except (AttributeError, TypeError):
            if event_date is not None:
                if isinstance(event_date, str):
                    try:
                        event_date_str = datetime.fromisoformat(event_date.replace('Z', '+00:00')).strftime("%Y%m%d")
                    except (ValueError, TypeError):
                        event_date_str = datetime.now().strftime("%Y%m%d")
                else:
                    event_date_str = event_date.strftime("%Y%m%d")
            else:
                event_date_str = datetime.now().strftime("%Y%m%d")
        
        user_name = user_agent.get_formatted_name()
        filename = f"{user_name}_event{event_index}_{event_date_str}_turn{turn_idx}_{temporal_pref_id}_counterfactuals.json"
        filepath = os.path.join(benchmark_output_dir, filename)
        
        # Build evolution metadata for this preference
        evolution_summary = recalled_preference.get_evolution_summary()
        
        # Prepare data for saving
        save_data = {
            "metadata": {
                "user": user_agent.to_dict(),
                "event": event.to_dict(),
                "temporal_preference_id": temporal_pref_id,
                "recalled_preference": recalled_preference.to_dict(),
                "turn_index": turn_idx,
                "event_index": event_index,
                "event_date": event_date_str,
                "generation_timestamp": datetime.now().isoformat(),
                "num_counterfactuals": len(counterfactual_turns)
            },
            "preference_evolution": {
                "has_evolved": evolution_summary["has_evolved"],
                "num_evolution_events": evolution_summary["num_evolution_events"],
                "original_expressed_attributes": evolution_summary["original_attributes"],
                "current_attributes": evolution_summary["current_attributes"],
                "changed_attributes": evolution_summary["changed_attributes"],
                "evolution_history": evolution_summary["history"]
            },
            "original_conversation": conversation,
            "original_turn": conversation.get("turns", [])[turn_idx] if turn_idx < len(conversation.get("turns", [])) else None,
            "counterfactual_turns": counterfactual_turns
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved {len(counterfactual_turns)} counterfactual turns to {filepath}")
    

    def generate_batch(self, num_users: int, output_dir: str) -> List[str]:
        """
        Generate timelines for multiple users and save to files.
        
        Args:
            num_users: Number of user timelines to generate
            output_dir: Directory to save the generated timelines
            
        Returns:
            List of saved file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        for i in range(num_users):
            logger.info(f"Generating timeline for user {i+1}/{num_users}")
            
            # Generate timeline
            timeline = self.generate_user_timeline()
            
            # Create filename using user information
            user_info = timeline["user"]
            filename = f"{user_info['first_name']}_{user_info['last_name']}_{i}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(timeline, f, indent=2)
                
            saved_files.append(filepath)
            logger.info(f"Saved timeline to {filepath}")
            
        return saved_files