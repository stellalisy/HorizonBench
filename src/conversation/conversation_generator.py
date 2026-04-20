"""
conversation_generator.py

Implements conversation generation for the Causal Preference Evolution Framework.
Generates naturalistic conversations between users and AI systems that reflect
preference entities and maintain temporal consistency.
"""
import re
import json
import logging
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple, Union

from src.causal_framework.models.user_model import UserModel
from src.causal_framework.models.agent_model import AgentModel
from src.causal_framework.models.event_model import EventModel
from src.causal_framework.models.preference_model import PreferenceModel
from src.causal_framework.models.intent_model import IntentModel
from src.llm.api_client import LLMClient
from src.conversation.preference_memory_system import PreferenceMemorySystem

from src.conversation.prompts.outline_generation_prompt import outline_generator_prompts
from src.conversation.prompts.conversation_generation_prompt import conversation_generator_prompt

logger = logging.getLogger(__name__)

class ConversationGenerator:
    """
    Generates naturalistic conversations between users and AI systems
    that reflect preference entities and maintain temporal consistency.
    """
    
    def __init__(self, llm_client: LLMClient, convo_llm_client: Optional[LLMClient], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the conversation generator.
        
        Args:
            llm_client: Client for accessing LLM services
            config: Optional configuration parameters
        """
        self.llm_client = llm_client
        self.convo_llm_client = convo_llm_client
        self.preference_memory = PreferenceMemorySystem(llm_client)
        
        # Set default configuration
        self.config = {
            "min_conversation_turns": 3,
            "max_conversation_turns": 10,
            "default_conversation_turns": 5,
            "preference_expression_threshold": 0.7,  # Probability of expressing a preference
            "temporal_context_window": 5,  # Number of previous events to consider for context
            "consistency_verification": True,  # Whether to verify consistency with history
            "random_seed": 42,
            "dev_mode": False,  # Default to production mode
            "expression_type": "natural"  # "natural", "implicit", or "explicit" - controls preference expression style
        }
        
        if config:
            # First, extract conversation_config if it exists (nested structure in YAML)
            conv_config = config.get("conversation_config", {})
            # Update with conversation_config values first
            self.config.update(conv_config)
            # Then update with any top-level overrides (e.g., expression_type passed directly)
            # Only update with specific keys that might be passed at runtime
            runtime_keys = ["expression_type", "paired_expression_modes", "temporal_preference_recall_days_min", "dev_mode", "random_seed"]
            for key in runtime_keys:
                if key in config:
                    self.config[key] = config[key]
            
        random.seed(self.config["random_seed"])
        
        # Creative templates for different conversation scenarios (based on Character.AI usage data)
        self._scenario_templates = {
            "productivity_assistance": [
                "The user is wrestling with {event_topic} and needs the AI assistant's strategic guidance to navigate the complexities and make real progress.",
                "The user is facing {event_type} and wants the AI assistant to help them brainstorm creative solutions and develop an actionable plan.",
                "The user is tackling {event_topic} and seeks the AI assistant's expertise to organize their thoughts and approach this challenge systematically."
            ],
            "entertainment_interaction": [
                "The user is excited about {event_type} and wants the AI assistant to join them in exploring this fun, creative experience with enthusiasm and wit.",
                "The user is looking for engaging interaction around {event_topic} and values the AI assistant's playful, intelligent approach to making this entertaining.",
                "The user wants to dive into {event_type} and seeks the AI assistant as a creative collaborator who can match their energy and imagination."
            ],
            "creative_storytelling": [
                "The user is passionate about developing {event_topic} and wants the AI assistant to be their creative partner in bringing this vision to life.",
                "The user is exploring {event_type} and needs the AI assistant's imaginative input to craft compelling narratives and rich character development.",
                "The user is building {event_topic} and seeks the AI assistant's storytelling expertise to create something truly engaging and memorable."
            ],
            "emotional_social_support": [
                "The user is processing {event_topic} and needs the AI assistant's empathetic guidance to work through these complex emotions and feelings.",
                "The user is struggling with {event_type} and wants the AI assistant's supportive perspective to help them gain clarity and emotional balance.",
                "The user is navigating {event_topic} and seeks the AI assistant's compassionate wisdom to find their way through this challenging situation."
            ],
            "philosophical_inquiry": [
                "The user is fascinated by {event_topic} and wants to explore the deeper philosophical implications with the AI assistant's thoughtful perspective.",
                "The user is questioning {event_type} and seeks the AI assistant's intellectual companionship to examine these profound ideas together.",
                "The user is contemplating {event_topic} and values the AI assistant's ability to engage in meaningful, thought-provoking dialogue about life's big questions."
            ],
            "relationship_guidance": [
                "The user is navigating the complexities of {event_topic} and needs the AI assistant's insightful advice on these delicate interpersonal dynamics.",
                "The user is dealing with {event_type} and wants the AI assistant's perspective on how to approach these relationship challenges with wisdom and care.",
                "The user is working through {event_topic} and seeks the AI assistant's guidance on building healthier, more authentic connections with others."
            ],
            "analytical_discussion": [
                "The user is intrigued by {event_topic} and wants to dissect the underlying patterns and implications with the AI assistant's analytical mind.",
                "The user is examining {event_type} and seeks the AI assistant's logical framework to understand the strategic and critical thinking involved.",
                "The user wants to thoroughly analyze {event_topic} and values the AI assistant's ability to break down complex systems and reveal deeper insights."
            ]
        }
        
        logger.info("ConversationGenerator initialized")
    
    def _get_expression_type_instruction(self, expression_type: str) -> str:
        """
        Generate instruction text based on the configured expression type.
        
        Args:
            expression_type: The configured expression type ("natural", "implicit", or "explicit")
            
        Returns:
            String containing the appropriate instruction for the expression type
        """
        if expression_type == "explicit":
            return """**MAXIMALLY EXPLICIT PREFERENCE EXPRESSION — STRICT RULES:**
The user MUST state every single preference in the most direct, unambiguous, and overt language possible.
Every preference must be expressed as a clear declarative statement using phrases like:
- "I strongly prefer X over Y"
- "I always want Z"
- "My preference is specifically for..."
- "I need you to do X, not Y — that's really important to me"
- "Just so you know, I'm the type of person who..."

REQUIREMENTS:
1. Every preference must be expressed as a first-person declarative statement ("I prefer...", "I want...", "I need...", "I like...", "I always...").
2. Preferences must name the specific attribute and value directly — no hinting, no subtext.
3. The user should volunteer their preferences proactively, even if the assistant doesn't ask.
4. If a preference is relevant to the conversation, the user MUST state it out loud rather than letting it show through behavior.
5. NEVER let a preference be merely implied through choices or reactions — always verbalize it explicitly.
6. If a turn has a 'preference_recap' field, the user MUST explicitly re-state EVERY attribute listed there, naming the exact attribute and value (e.g. "btw, my preferred response format is bullet points", "just a reminder, I really need hyperlinked sources for evidence"). State each attribute-value pair clearly and specifically — these are the exact preferences the user wants to reinforce."""
        
        elif expression_type == "implicit":
            return """**EXTREMELY IMPLICIT & VAGUE PREFERENCE EXPRESSION — THIS IS THE #1 PRIORITY OF THIS GENERATION.**

The user's preferences must be nearly invisible in the conversation. They should surface
only as fleeting, ambiguous, offhand remarks that a very attentive reader *might* connect
to a preference — but that could just as easily be interpreted as idle chatter.

BANNED — the user must NEVER:
- State, declare, or name any preference ("I prefer …", "I want …", "I like …")
- Request a format, style, or approach ("keep it concise", "give me bullets", "be direct")
- Demonstrate or perform the preference (e.g. writing in bullets to show they prefer bullets,
  writing short messages to show they prefer brevity, writing cinematically to show they prefer
  cinematic style)
- React in ways that obviously signal approval/disapproval of content matching/violating the
  preference ("oh nice, that's clear" when bullets are used, "ugh" when prose is used)
- Steer the conversation toward the preference through pointed follow-up questions
  ("what about the numbers?" to signal an analytical preference)
- Use any behavioral pattern that makes the preference easy to infer from this turn alone

WHAT TO DO INSTEAD — the preference should appear ONLY as:
1. **Passing, throwaway remarks** — a brief aside or tangent that has a tenuous link to the
   preference. E.g. for a conciseness preference, "sorry i'm distracted, on my phone between
   meetings" (which vaguely contextualizes why short replies might be valued — but doesn't
   ask for brevity or demonstrate it).
2. **Anecdotes about third parties** — mentioning something someone else did/said that
   tangentially relates. E.g. for a bullet-point preference, "my coworker sent this crazy
   long slack message today lol" (vaguely suggests long text is undesirable — but never
   requests bullets).
3. **Neutral, ambiguous reactions** — "huh", "ok cool", "interesting" that give no clear
   preference signal but keep the conversation moving.
4. **Environmental or contextual details** — mentioning a situation, constraint, or context
   that a very careful reader could connect to the preference, but that doesn't directly
   relate. E.g. for risk-averse: "the market's been wild lately" (establishes uncertainty
   without saying anything about risk preference).

The GOAL: if you showed the conversation to 10 people and asked "what are this user's
preferences?", at most 1-2 would pick up on the vague hint. The rest would see nothing.
The preference should be almost entirely hidden.

KEY RULE: each hint must be deniable — it must have a plausible non-preference interpretation.

If the outline contains turns with empty preference_expressed fields, the user should just
continue the conversation naturally with no hint at all."""
        
        else:  # "natural" - default
            return """**NATURAL PREFERENCE EXPRESSION:**
You can choose the most natural way for preferences to be expressed based on the conversation context.
You can use either:

**EXPLICIT PREFERENCES** - User directly states their preference:
- "I really need someone to be direct with me right now"
- "I prefer when people explain the reasoning behind their advice"

**IMPLICIT PREFERENCES** - User shows preference through behavior/choices:
- User chooses detailed explanation over quick summary
- User asks follow-up questions showing they want depth

Choose whichever feels more natural and authentic for the specific conversation context."""
    
    def generate_conversation(
        self,
        user: "AgentModel",
        assistant: "AgentModel", 
        event: EventModel,
        preferences_current_user: Dict[str, PreferenceModel],
        preferences_updated_user: Dict[str, PreferenceModel],
        preferences_current_assistant: Dict[str, PreferenceModel],
        preferences_updated_assistant: Dict[str, PreferenceModel],
        agent_intents: Dict[str, "IntentModel"],
        previous_events: Optional[List[EventModel]] = None,
        num_turns: Optional[int] = None,
        recalled_preference: Optional[PreferenceModel] = None
    ) -> Dict[str, Any]:
        """
        Generate a conversation between user and assistant based on event, preferences, and intents.
        
        Args:
            user: User agent model
            assistant: Assistant agent model
            event: Event model that triggers the conversation
            preferences_current_user: User's current preference states
            preferences_updated_user: User's updated preference states
            preferences_current_assistant: Assistant's current preference states
            preferences_updated_assistant: Assistant's updated preference states
            agent_intents: Dictionary mapping agent IDs to their IntentModel instances
            previous_events: Optional list of previous events for context
            num_turns: Optional number of conversation turns to generate
            recalled_preference: Optional preference that assistant should recall from previous conversations
            
        Returns:
            Dictionary containing the generated conversation
        """
        if not previous_events:
            previous_events = []
            
        # Determine number of turns
        if num_turns is None:
            num_turns = random.randint(
                self.config["min_conversation_turns"],
                self.config["max_conversation_turns"]
            )

        # 1. Define the conversation scenario
        scenario = self._define_conversation_scenario(event, user)
        
        # 2. Get historical conversation context (not for preference selection)
        historical_context = self.preference_memory.get_conversation_context(
            user_id=user.id,
            lookback_days=90
        )

        # 3. Integrate preferences into conversation prompts
        preference_prompts_user_current, preference_prompts_user_updated = self._create_preference_prompts(
            preferences_current_user, preferences_updated_user, event, historical_context, recalled_preference
        )
        preference_prompts_assistant_current, preference_prompts_assistant_updated = self._create_preference_prompts(
            preferences_current_assistant, preferences_updated_assistant, event, historical_context
        )

        # 4. Consider temporal context from previous events
        temporal_context = self._extract_temporal_context(previous_events, event)
        
        # 5. Generate conversation outline
        outline = self._generate_conversation_outline(
            scenario=scenario,
            user=user,
            assistant=assistant,
            event=event,
            # preferences_current_user=preferences_current_user,
            preferences_updated_user=preferences_updated_user,
            # preferences_current_assistant=preferences_current_assistant,
            # preferences_updated_assistant=preferences_updated_assistant,
            preference_prompts_user_current=preference_prompts_user_current,
            preference_prompts_user_updated=preference_prompts_user_updated,
            preference_prompts_assistant_current=preference_prompts_assistant_current,
            preference_prompts_assistant_updated=preference_prompts_assistant_updated,
            agent_intents=agent_intents,
            temporal_context=temporal_context,
            num_turns=num_turns,
            recalled_preference=recalled_preference
        )
        
        # 6. Generate full conversation
        conversation = self._generate_full_conversation(
            outline=outline,
            user=user,
            assistant=assistant,
            event=event,
            preferences_user=preferences_updated_user,
            preferences_assistant=preferences_updated_assistant,
            agent_intents=agent_intents,
            temporal_context=temporal_context
        )
        
        # 7. Record conversation preferences for future reference
        self._record_conversation_preferences(
            user=user,
            conversation=conversation,
            preferences_updated=preferences_updated_user,
            event_date=event.date or datetime.now()
        )
        
        # 8. Verify consistency if enabled
        if self.config["consistency_verification"] and previous_events:
            is_consistent = self._verify_consistency(conversation, previous_events, preferences_updated_user)
            if not is_consistent:
                # Regenerate with stronger consistency guidance
                logger.warning("Inconsistency detected, regenerating conversation with stronger guidance")
                return self.generate_conversation(
                    user=user,
                    assistant=assistant,
                    event=event,
                    preferences_current_user=preferences_current_user,
                    preferences_updated_user=preferences_updated_user,
                    preferences_current_assistant=preferences_current_assistant,
                    preferences_updated_assistant=preferences_updated_assistant,
                    agent_intents=agent_intents,
                    previous_events=previous_events,
                    num_turns=num_turns
                )
        
        # 9. Format the final conversation
        formatted_conversation = {
            "scenario": scenario,
            "turns": conversation["turns"],
            "event_id": event.id,
            "preferences_expressed": conversation.get("preferences_expressed", []),
            "agent_intents": {
                "user_intent": agent_intents[user.id].to_dict(),
                "assistant_intent": agent_intents[assistant.id].to_dict()
            },
            "metadata": {
                "num_turns": len(conversation["turns"]),
                "generation_timestamp": event.date.isoformat() if event.date else None,
                "user_preference_ids": list(preferences_updated_user.keys()),
                "assistant_preference_ids": list(preferences_updated_assistant.keys()),
                "assistant_type": assistant.get_attribute("assistant_type", "unknown")
            }
        }
        
        # Add outline to the returned data for saving
        formatted_conversation["outline"] = outline

        # Paired expression mode: generate a second conversation variant from the same outline
        paired_modes = self.config.get("paired_expression_modes")
        if paired_modes:
            paired_conversations = {}
            original_expression_type = self.config.get("expression_type", "natural")
            for mode in paired_modes:
                if mode == original_expression_type:
                    paired_conversations[mode] = formatted_conversation
                    continue
                self.config["expression_type"] = mode
                paired_conv = self._generate_full_conversation(
                    outline=outline,
                    user=user,
                    assistant=assistant,
                    event=event,
                    preferences_user=preferences_updated_user,
                    preferences_assistant=preferences_updated_assistant,
                    agent_intents=agent_intents,
                    temporal_context=temporal_context
                )
                paired_formatted = {
                    "scenario": scenario,
                    "turns": paired_conv["turns"],
                    "event_id": event.id,
                    "preferences_expressed": paired_conv.get("preferences_expressed", []),
                    "agent_intents": formatted_conversation["agent_intents"],
                    "metadata": {
                        **formatted_conversation["metadata"],
                        "expression_type": mode
                    },
                    "outline": outline
                }
                paired_conversations[mode] = paired_formatted
            self.config["expression_type"] = original_expression_type
            formatted_conversation["paired_conversations"] = paired_conversations
        
        logger.info(f"Generated conversation with {len(conversation['turns'])} turns for event {event.id}")
        return formatted_conversation
    
    def _define_conversation_scenario(self, event: EventModel, user: AgentModel) -> str:
        """
        Define the conversation scenario based on the event.
        
        Args:
            event: Event model
            user: User model
            
        Returns:
            String describing the conversation scenario
        """
        # Extract event category
        event_metadata = event.metadata or {}
        event_category = event.category
        
        if not event_category:
            # Try to determine category from tags or attributes
            if event.has_attribute("category"):
                event_category = event.get_attribute("category")
            elif "tags" in event_metadata and event_metadata["tags"]:
                # Use first tag as category
                event_category = event_metadata["tags"][0]
            else:
                # Default to generic category
                event_category = "social"
        
        # Map to scenario template category
        template_category = event_category
        if template_category not in self._scenario_templates:
            # Create a generic template for novel categories
            template = f"The user is seeking help with {{event_topic}} and values the AI assistant's expertise in {template_category.replace('_', ' ')}."
        else:
            # Select random template from predefined category
            templates = self._scenario_templates[template_category]
            template = random.choice(templates)
        
        # Extract event attributes for template
        event_type = event.name
        event_topic = event.get_attribute("topic", event_type)
        
        # Format the template
        scenario = template.format(
            event_type=event_type,
            event_topic=event_topic,
            user_name=user.get_formatted_name()
        )
        
        logger.debug(f"Defined conversation scenario: {scenario}")
        return scenario
    
    def _create_preference_prompts(
        self,
        preferences_current: Dict[str, PreferenceModel],
        preferences_updated: Dict[str, PreferenceModel],
        event: EventModel,
        historical_context: Dict[str, Any],
        recalled_preference: Optional[PreferenceModel] = None
    ) -> Tuple[str, str]:
        """
        Create prompts for integrating preferences into the conversation.
        
        Args:
            preferences_current: Current preference models (what assistant knows)
            preferences_updated: Updated preference models (user's actual preferences)
            event: Event model
            historical_context: Historical conversation context
            recalled_preference: Optional preference that assistant should recall from previous conversations
        Returns:
            Tuple of (current_preferences_prompt, updated_preferences_prompt)
        """
        # Create current preferences prompt (what assistant knows)
        current_preferences_prompt = f"""
        ASSISTANT'S KNOWN PREFERENCES (can appeal to these):
        {json.dumps({pid: pref.to_dict() for pid, pref in preferences_current.items()}, indent=2)}
        
        HISTORICAL CONTEXT:
        {historical_context.get("context_summary", "No previous conversation history")}
        """

        if recalled_preference:
            current_preferences_prompt += (
            "IMPORTANT - TEMPORAL PREFERENCE RECALL:\n"
            "The assistant should recall and reference this preference that the user previously expressed:\n"
            f"{json.dumps(recalled_preference.to_dict(), indent=2)}\n"
            )

        # Create updated preferences prompt (user's actual preferences)
        updated_preferences_prompt = f"""
        USER'S ACTUAL PREFERENCES (for natural expression):
        {json.dumps({pid: pref.to_dict() for pid, pref in preferences_updated.items()}, indent=2)}
        """
        
        return current_preferences_prompt, updated_preferences_prompt
    
    def _extract_temporal_context(
        self,
        previous_events: List[EventModel],
        current_event: EventModel
    ) -> str:
        """
        Extract temporal context from previous events.
        
        Args:
            previous_events: List of previous events
            current_event: Current event
            
        Returns:
            String describing temporal context
        """
        if not previous_events:
            return "This is the first recorded interaction with this user."
            
        # Limit to recent events within the temporal context window
        recent_events = previous_events[-self.config["temporal_context_window"]:]
        
        # Create context description
        context_parts = []
        
        for event in recent_events:
            # Skip if no date
            if not event.date:
                continue
                
            # Calculate time difference
            if current_event.date:
                time_diff = current_event.date - event.date
                days_ago = time_diff.days
                
                if days_ago == 0:
                    time_str = "earlier today"
                elif days_ago == 1:
                    time_str = "yesterday"
                else:
                    time_str = f"{days_ago} days ago"
            else:
                time_str = "previously"
                
            # Create event description
            event_name = event.name
            context_parts.append(f"The user experienced a {event_name} {time_str}.")
            
            # Add associated preferences if available
            associated_prefs = event.get_all_preferences()
            if associated_prefs:
                for pref_id, pref_data in associated_prefs.items():
                    if isinstance(pref_data, dict) and "attributes" in pref_data:
                        pref_str = f"During this event, the user expressed preferences related to {pref_id.replace('_', ' ')}."
                        context_parts.append(pref_str)
        
        if not context_parts:
            return "There is no significant recent history with this user."
            
        return " ".join(context_parts)
    
    def _generate_conversation_outline(
        self,
        scenario: str,
        user: AgentModel,
        assistant: AgentModel,
        event: EventModel,
        # preferences_current_user: Dict[str, PreferenceModel],
        preferences_updated_user: Dict[str, PreferenceModel],
        # preferences_current_assistant: Dict[str, PreferenceModel],
        # preferences_updated_assistant: Dict[str, PreferenceModel],
        preference_prompts_user_current: str,
        preference_prompts_user_updated: str,
        preference_prompts_assistant_current: str,
        preference_prompts_assistant_updated: str,
        agent_intents: Dict[str, IntentModel],
        temporal_context: str,
        num_turns: int,
        recalled_preference: Optional[PreferenceModel] = None
    ) -> Dict[str, Any]:
        """
        Generate a conversation outline using LLM.
        
        Args:
            scenario: Conversation scenario
            user: User agent model
            assistant: Assistant agent model
            event: Event model
            # preferences_current_user: Dictionary of current user preference models
            preferences_updated_user: Dictionary of updated user preference models
            # preferences_current_assistant: Dictionary of current assistant preference models
            # preferences_updated_assistant: Dictionary of updated assistant preference models
            preference_prompts_user_current: Current user preference integration prompts
            preference_prompts_user_updated: Updated user preference integration prompts
            preference_prompts_assistant_current: Current assistant preference integration prompts
            preference_prompts_assistant_updated: Updated assistant preference integration prompts
            agent_intents: Dictionary mapping agent IDs to their IntentModel instances
            temporal_context: String describing temporal context
            num_turns: Number of conversation turns
            
        Returns:
            Dictionary containing the conversation outline
        """
        # Create a prompt for the LLM
        user_information = json.dumps(user.to_dict(), indent=2)
        assistant_information = json.dumps(assistant.to_dict(), indent=2)
        event_context = json.dumps(event.to_dict(), indent=2)
        user_intent = json.dumps(agent_intents[user.id].to_dict(), indent=2)
        assistant_intent = json.dumps(agent_intents[assistant.id].to_dict(), indent=2)    
        
        # Add recalled preference information if available
        recalled_preference_context = ""
        if recalled_preference:
            recalled_preference_context = (
            "\nIMPORTANT - TEMPORAL PREFERENCE RECALL:\n"
            "The assistant should recall and reference this preference that the user previously expressed:\n"
            f"{json.dumps(recalled_preference.to_dict(), indent=2)}\n"
            
            "The assistant should naturally recall this preference and appeal to it during the conversation.\n"
            "The USER should NOT re-express this preference since they already expressed it before.\n"
            )
        
        # Get expression type instruction
        expression_type = self.config.get("expression_type", "natural")
        expression_type_instruction = self._get_expression_type_instruction(expression_type)
        
        prompt = outline_generator_prompts.format(
            scenario=scenario,
            user_information=user_information,
            assistant_information=assistant_information,
            event_context=event_context,
            temporal_context=temporal_context,
            preferences_updated_user=preference_prompts_user_updated,
            preferences_current_user=preference_prompts_user_current,
            preferences_updated_assistant=preference_prompts_assistant_updated,
            user_intent=user_intent,
            assistant_intent=assistant_intent,
            num_turns=num_turns,
            num_turns_minus_one=num_turns - 1,
            num_turns_plus_one=num_turns + 1,
            recalled_preference_context=recalled_preference_context,
            expression_type=expression_type,
            expression_type_instruction=expression_type_instruction
        )
        
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            outline = json.loads(response["response_text"])
            logger.debug(f"Generated conversation outline with {len(outline.get('outline', []))} turns")
            return outline
        except json.JSONDecodeError as e:
            if self.config.get("dev_mode", False):
                raise json.JSONDecodeError(f"LLM response for conversation outline could not be parsed as JSON: {response}", e.doc, e.pos)
            
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            # Fallback: create a simple outline
            return {
                "outline": [
                    {
                        "turn_number": i + 1,
                        "user_intent": "Discuss the event",
                        "assistant_response_goal": "Respond helpfully",
                        "preference_expressions": list(preferences_updated_user.keys()) if i == 0 else []
                    }
                    for i in range(num_turns)
                ]
            }
    
    def _generate_full_conversation(
        self,
        outline: Dict[str, Any],
        user: AgentModel,
        assistant: AgentModel,
        event: EventModel,
        preferences_user: Dict[str, PreferenceModel],
        preferences_assistant: Dict[str, PreferenceModel],
        agent_intents: Dict[str, IntentModel],
        temporal_context: str,
        num_retries: int = 10
    ) -> Dict[str, Any]:
        """
        Generate the full conversation based on the outline, with retry logic.

        Args:
            outline: Conversation outline
            user: User agent model
            assistant: Assistant agent model
            event: Event model
            preferences_user: Dictionary of user preference models
            preferences_assistant: Dictionary of assistant preference models
            agent_intents: Dictionary mapping agent IDs to their IntentModel instances
            temporal_context: String describing temporal context
            num_retries: Number of times to retry generation if anything fails

        Returns:
            Dictionary containing the full conversation
        """
        outline_json = json.dumps(outline, indent=2)
        
        # Build rich user persona description
        user_attrs = user.get_all_attributes()
        user_persona_parts = []
        user_persona_parts.append(f"Name: {user.get_formatted_name()}")
        if user_attrs.get("age"):
            user_persona_parts.append(f"Age: {user_attrs['age']}")
        if user_attrs.get("gender"):
            user_persona_parts.append(f"Gender: {user_attrs['gender']}")
        if user_attrs.get("ethnicity"):
            user_persona_parts.append(f"Background: {user_attrs['ethnicity']}")
        if user_attrs.get("location"):
            user_persona_parts.append(f"Location: {user_attrs['location']}")
        if user_attrs.get("occupation"):
            occ = user_attrs["occupation"]
            if isinstance(occ, dict):
                user_persona_parts.append(f"Occupation: {occ.get('title', '')} at {occ.get('organization', '')}")
            else:
                user_persona_parts.append(f"Occupation: {occ}")
        if user_attrs.get("education"):
            edu = user_attrs["education"]
            if isinstance(edu, dict):
                user_persona_parts.append(f"Education: {edu.get('degree', '')} in {edu.get('major', '')} from {edu.get('institution', '')}")
            else:
                user_persona_parts.append(f"Education: {edu}")
        if user_attrs.get("interests"):
            interests = user_attrs["interests"]
            if isinstance(interests, list):
                user_persona_parts.append(f"Interests: {', '.join(interests[:5])}")  # Limit to 5
            else:
                user_persona_parts.append(f"Interests: {interests}")
        if user_attrs.get("traits"):
            traits = user_attrs["traits"]
            if isinstance(traits, list):
                # Extract key personality traits (limit to most distinctive ones)
                key_traits = [t.split(" - ")[0] if " - " in t else t for t in traits[:5]]
                user_persona_parts.append(f"Personality: {', '.join(key_traits)}")
        
        user_persona = "\n    ".join(user_persona_parts)
        
        assistant_name = assistant.get_formatted_name()
        assistant_role = assistant.role
        event_context = json.dumps(event.to_dict(), indent=2)
        user_preferences_json = json.dumps([p.to_dict() for p in preferences_user.values()])
        assistant_preferences_json = json.dumps([p.to_dict() for p in preferences_assistant.values()])
        user_intent_json = json.dumps(agent_intents[user.id].to_dict(), indent=2)
        assistant_intent_json = json.dumps(agent_intents[assistant.id].to_dict(), indent=2)

        # Get expression type instruction
        expression_type = self.config.get("expression_type", "natural")
        expression_type_instruction = self._get_expression_type_instruction(expression_type)
        
        prompt = conversation_generator_prompt.format(
            outline=outline_json,
            user_persona=user_persona,
            assistant_name=assistant_name,
            assistant_role=assistant_role,
            event_context=event_context,
            temporal_context=temporal_context,
            user_preferences=user_preferences_json,
            assistant_preferences=assistant_preferences_json,
            user_intent=user_intent_json,
            assistant_intent=assistant_intent_json,
            expression_type=expression_type,
            expression_type_instruction=expression_type_instruction
        )

        last_exception = None
        for attempt in range(num_retries + 1):
            try:
                response = self.convo_llm_client.generate(prompt, response_format="json")['response_text']
                logger.debug(f"Raw LLM response for conversation generation: {response[:500]}...")
                conversation = json.loads(response)

                # Validate and clean up turns
                if "turns" in conversation:
                    valid_turns = []
                    for i, turn in enumerate(conversation["turns"]):
                        if turn is None:
                            logger.warning(f"Skipping None turn at index {i}")
                            continue
                        if not isinstance(turn, dict):
                            logger.warning(f"Skipping invalid turn at index {i}: {turn}")
                            continue
                        if "role" not in turn or "content" not in turn:
                            logger.warning(f"Skipping incomplete turn at index {i}: {turn}")
                            continue
                        if "preference_appealed" in turn and turn["preference_appealed"] is None:
                            turn["preference_appealed"] = {}
                        if "preference_expressed" in turn and turn["preference_expressed"] is None:
                            turn["preference_expressed"] = {}
                        valid_turns.append(turn)

                    conversation["turns"] = valid_turns

                    # If no valid turns, create fallback or raise error in dev mode
                    if not valid_turns:
                        if self.config.get("dev_mode", False):
                            raise ValueError("No valid turns found in LLM response - all turns were None or malformed")
                        logger.warning("No valid turns found, creating fallback conversation")
                        conversation = self._create_fallback_conversation(event)
                else:
                    if self.config.get("dev_mode", False):
                        raise ValueError("No 'turns' field found in LLM response")

                # Ensure preferences_expressed field exists
                if "preferences_expressed" not in conversation:
                    conversation["preferences_expressed"] = list(preferences_user.keys())

                logger.debug(f"Generated full conversation with {len(conversation.get('turns', []))} turns")
                return conversation

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Attempt {attempt+1}/{num_retries}: LLM response could not be parsed as JSON or was invalid: {e}")
                if attempt < num_retries:
                    logger.info("Retrying conversation generation...")
                    continue
                else:
                    if self.config.get("dev_mode", False):
                        # Raise the last exception with more context
                        if isinstance(e, json.JSONDecodeError):
                            raise json.JSONDecodeError(
                                f"LLM response could not be parsed as JSON after {num_retries} attempts: {response}",
                                getattr(e, "doc", response),
                                getattr(e, "pos", 0)
                            )
                        else:
                            raise
                    logger.error(f"LLM response could not be parsed as JSON after {num_retries} attempts: {response if 'response' in locals() else 'No response'}")
                    # Final fallback: create a simple conversation
                    return self._create_fallback_conversation(event)
        # Should never reach here, but fallback just in case
        return self._create_fallback_conversation(event)
    
    def _create_fallback_conversation(self, event: EventModel) -> Dict[str, Any]:
        """
        Create a simple fallback conversation when LLM generation fails.
        
        Args:
            event: Event model
            
        Returns:
            Dictionary containing a simple fallback conversation
        """
        return {
            "turns": [
                {
                    "role": "user",
                    "content": f"Hi, I wanted to talk about my {event.get_attribute('name', 'event')}.",
                    "preference_expressed": []
                },
                {
                    "role": "assistant",
                    "content": "I'd be happy to discuss that with you. What would you like to share?",
                    "preference_appealed": []
                },
                {
                    "role": "user",
                    "content": "I've been thinking about it and wanted your perspective.",
                    "preference_expressed": []
                },
                {
                    "role": "assistant", 
                    "content": "I'm here to help. What specific aspects would you like my perspective on?",
                    "preference_appealed": []
                }
            ],
            "preferences_expressed": []
        }
    
    def _verify_consistency(
        self,
        conversation: Dict[str, Any],
        previous_events: List[EventModel],
        current_preferences: Dict[str, PreferenceModel]
    ) -> bool:
        """
        Verify that the conversation is consistent with preference history.
        
        Args:
            conversation: Generated conversation
            previous_events: List of previous events
            current_preferences: Dictionary of current preference models
            
        Returns:
            True if consistent, False otherwise
        """
        # Extract historical preferences from previous events
        historical_preferences = {}
        
        for event in previous_events:
            event_prefs = event.get_all_preferences()
            for pref_id, pref_data in event_prefs.items():
                if isinstance(pref_data, dict) and "attributes" in pref_data:
                    if pref_id not in historical_preferences:
                        historical_preferences[pref_id] = []
                    historical_preferences[pref_id].append(pref_data)
        
        # If no historical preferences, assume consistent
        if not historical_preferences:
            return True
            
        # Create a prompt for the LLM to check consistency
        prompt = """
        Analyze the following conversation to determine if it's consistent with the user's preference history.
        
        Conversation:
        """
        
        # Add conversation turns
        for i, turn in enumerate(conversation.get("turns", [])):
            if turn.get("role") == "user":
                prompt += f"\nTurn {i+1}:\nUser: {turn.get('content', '')}\n"
            elif turn.get("role") == "assistant":
                prompt += f"Assistant: {turn.get('content', '')}\n"
            
        prompt += "\nCurrent preferences:\n"
        
        # Add current preferences
        for pref_id, pref_model in current_preferences.items():
            prompt += f"- {pref_id}:\n"
            for attr, value in pref_model.get_all_attributes().items():
                prompt += f"  - {attr}: {value}\n"
                
        prompt += "\nHistorical preferences:\n"
        
        # Add historical preferences
        for pref_id, pref_history in historical_preferences.items():
            if pref_id in current_preferences:
                prompt += f"- {pref_id}:\n"
                for i, pref_data in enumerate(pref_history):
                    prompt += f"  Instance {i+1}:\n"
                    for attr, value in pref_data.get("attributes", {}).items():
                        prompt += f"    - {attr}: {value}\n"
        
        prompt += """
        Analyze the conversation and determine if there are any inconsistencies between the user's statements in the conversation and their preference history.
        
        Return a JSON object with the following structure:
        {
          "is_consistent": true/false,
          "inconsistencies": [
            {
              "turn_number": N,
              "preference_id": "preference_id",
              "attribute": "attribute_name",
              "conversation_value": "value expressed in conversation",
              "historical_value": "value in history"
            },
            ...
          ]
        }
        """
        
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            consistency_check = json.loads(response["response_text"])
            is_consistent = consistency_check.get("is_consistent", True)
            
            if not is_consistent:
                inconsistencies = consistency_check.get("inconsistencies", [])
                for inconsistency in inconsistencies:
                    logger.warning(f"Detected inconsistency: {json.dumps(inconsistency)}")
                    
            return is_consistent
        except json.JSONDecodeError as e:
            if self.config.get("dev_mode", False):
                raise json.JSONDecodeError(f"LLM response for consistency check could not be parsed as JSON: {response}", e.doc, e.pos)
            
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            # Default to consistent if parsing fails
            return True
    
    def regenerate_conversation_turn(
        self,
        conversation: Dict[str, Any],
        turn_index: int,
        user: AgentModel,
        event: EventModel,
        preferences: Dict[str, PreferenceModel],
        original_preferences: Optional[Dict[str, PreferenceModel]] = None
    ) -> Dict[str, Any]:
        """
        Regenerate a specific turn in the conversation.
        
        Args:
            conversation: Existing conversation
            turn_index: Index of the turn to regenerate
            user: User model
            event: Event model
            preferences: Dictionary of preference models
            
        Returns:
            Updated conversation with regenerated turn
            
        Raises:
            ValueError: If turn_index is out of range
        """
        turns = conversation.get("turns", [])
        
        if turn_index < 0 or turn_index >= len(turns):
            raise ValueError(f"Turn index {turn_index} out of range (0-{len(turns)-1})")
            
        # Create context from previous turns
        context = ""
        for i, turn in enumerate(turns[:turn_index]):
            if turn.get("role") == "user":
                context += f"Turn {i+1}:\nUser: {turn.get('content', '')}\n"
            elif turn.get("role") == "assistant":
                context += f"Assistant: {turn.get('content', '')}\n\n"
        
        # Determine if this turn should be user or assistant
        current_turn = turns[turn_index]
        is_user_turn = current_turn.get("role") == "user"
        
        # Get the preferences that were originally appealed to/expressed in this turn
        if is_user_turn:
            original_turn_prefs = current_turn.get("preference_expressed", [])
        else:
            original_turn_prefs = current_turn.get("preference_appealed", [])
        
        # Filter preferences to only include those originally used in this turn
        relevant_preferences = {pref_id: preferences[pref_id] for pref_id in original_turn_prefs if pref_id in preferences}
        
        # Detect which preferences have been changed
        changed_preferences = {}
        if original_preferences:
            for pref_id in relevant_preferences:
                if pref_id in original_preferences:
                    orig_attrs = original_preferences[pref_id].get_all_attributes()
                    new_attrs = preferences[pref_id].get_all_attributes()
                    for attr, new_value in new_attrs.items():
                        orig_value = orig_attrs.get(attr)
                        if orig_value != new_value:
                            if pref_id not in changed_preferences:
                                changed_preferences[pref_id] = {}
                            changed_preferences[pref_id][attr] = {
                                "original": orig_value,
                                "new": new_value
                            }
        
        # Get expression type instruction
        expression_type = self.config.get("expression_type", "natural")
        expression_type_instruction = self._get_expression_type_instruction(expression_type)
        
        # Create a prompt for the LLM
        if is_user_turn:
            prompt = f"""
            Regenerate the user message for turn {turn_index + 1} in the following conversation between a user and an AI assistant.
            
            Previous conversation:
            {context}
            
            User information:
            Name: {user.get_formatted_name()}
            
            Event context:
            {event.get_attribute("name", "event")}
            
            PREFERENCE EXPRESSION STYLE: {expression_type}
            {expression_type_instruction}
            
            Relevant preferences:
            """
            
            # Focus only on the changed attributes if any exist, otherwise show minimal context
            if changed_preferences:
                prompt += "\nCRITICAL - Your preferences have been modified. Focus ONLY on these changes:\n"
                for pref_id, changes in changed_preferences.items():
                    for attr, change_info in changes.items():
                        prompt += f"- {pref_id}.{attr}: NOW = '{change_info['new']}' (was '{change_info['original']}')\n"
                        prompt += f"  Express this NEW preference value in your message.\n"
                prompt += "\nDo NOT express any other preference attributes unless absolutely necessary for context.\n"
            else:
                # If no changes, show minimal preference context
                prompt += "\nRelevant preferences (only those originally expressed):\n"
                for pref_id, pref_model in relevant_preferences.items():
                    prompt += f"- {pref_id}: {pref_model.metadata.get('description', 'User preference')}\n"
                
            prompt += f"""
            
            Generate a new user message for turn {turn_index + 1} that:
            1. Maintains consistency with the previous conversation
            2. Naturally expresses the user's preferences (using the updated values if any preferences were modified)
            3. Flows naturally from the previous conversation
            
            Return a JSON object with the following structure:
            {{
              "role": "user",
              "content": "User's message",
              "preference_expressed": {original_turn_prefs}
            }}
            
            IMPORTANT: Only use the preference IDs that were originally expressed in this turn: {original_turn_prefs}
            """
        else:
            prompt = f"""
            Regenerate the assistant response for turn {turn_index + 1} in the following conversation between a user and an AI assistant.
            
            Previous conversation:
            {context}
            
            User information:
            Name: {user.get_formatted_name()}
            
            Event context:
            {event.get_attribute("name", "event")}
            
            Relevant preferences:
            """
            
            # Focus only on the changed attributes if any exist, otherwise show minimal context
            if changed_preferences:
                prompt += "\nCRITICAL - The user's preferences have been modified. Focus ONLY on these changes:\n"
                for pref_id, changes in changed_preferences.items():
                    for attr, change_info in changes.items():
                        prompt += f"- {pref_id}.{attr}: NOW = '{change_info['new']}' (was '{change_info['original']}')\n"
                        prompt += f"  Your response must reflect this NEW preference value.\n"
                prompt += "\nDo NOT mention or appeal to any other preference attributes unless absolutely necessary for context.\n"
            else:
                # If no changes, show minimal preference context
                prompt += "\nRelevant preferences (only those originally appealed to):\n"
                for pref_id, pref_model in relevant_preferences.items():
                    prompt += f"- {pref_id}: {pref_model.metadata.get('description', 'User preference')}\n"
                
            prompt += f"""
            
            Generate a new assistant response for turn {turn_index + 1} that:
            1. Maintains consistency with the previous conversation
            2. If preferences were modified, ONLY appeal to the specific changed attributes with their NEW values
            3. Do NOT mention unchanged preference attributes (like instructor experience, group size, etc.)
            4. Keep the response focused and concise, only addressing the modified preferences
            5. Provides a helpful and engaging response
            
            Return a JSON object with the following structure:
            {{
              "role": "assistant",
              "content": "Assistant's response",
              "preference_appealed": {original_turn_prefs}
            }}
            
            IMPORTANT: Only use the preference IDs that were originally appealed to in this turn: {original_turn_prefs}
            Do NOT add additional preferences beyond those originally appealed to.
            """
        
        # Define the JSON schema for turn regeneration
        turn_schema = {
            "type": "object",
            "properties": {
                "role": {"type": "string", "enum": ["user", "assistant"]},
                "content": {"type": "string"},
                "preference_expressed": {
                    "type": "object",
                    "patternProperties": {
                        ".*": {"type": "string"}
                    },
                    "additionalProperties": False
                },
                "preference_appealed": {
                    "type": "object",
                    "patternProperties": {
                        ".*": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            },
            "required": ["role", "content", "preference_expressed", "preference_appealed"],
            "additionalProperties": False
        }
        
        response = self.convo_llm_client.generate(prompt, response_format="json")
        if "response_time" in response: logger.debug(f"{self.convo_llm_client.model_name} generation response time: {response['response_time']}")

        try:
            new_turn = json.loads(response["response_text"])
            
            # Update the conversation
            updated_conversation = conversation.copy()
            updated_turns = turns.copy()
            updated_turns[turn_index] = new_turn
            updated_conversation["turns"] = updated_turns
            
            logger.debug(f"Regenerated turn {turn_index + 1} in conversation")
            return updated_conversation
        except json.JSONDecodeError as e:
            if self.config.get("dev_mode", False):
                raise json.JSONDecodeError(f"LLM response for conversation turn regeneration could not be parsed as JSON: {response}", e.doc, e.pos)
            
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            # Return original conversation if parsing fails
            return conversation
    
    def _record_conversation_preferences(
        self,
        user: AgentModel,
        conversation: Dict[str, Any],
        preferences_updated: Dict[str, PreferenceModel],
        event_date: datetime
    ) -> None:
        """
        Record preferences from this conversation for future reference.
        
        Args:
            user: User model
            conversation: Generated conversation
            preferences_updated: User's actual preferences
            event_date: Date of the conversation
        """
        # Extract expressed preferences from conversation
        expressed_preferences = {}
        appealed_preferences = {}
        
        # Debug: Log conversation structure
        turns = conversation.get("turns", [])
        logger.debug(f"Recording preferences from conversation with {len(turns)} turns")
        for i, turn in enumerate(turns):
            if turn is None:
                logger.error(f"Found None turn at index {i}")
        
        for turn in turns:
            # Skip None turns (failed generation)
            if turn is None:
                logger.warning("Skipping None turn in preference recording")
                continue
                
            try:
                if turn["role"] == "user" and "preference_expressed" in turn:
                    if not turn.get("preference_expressed", {}):
                        turn["preference_expressed"] = {}
                    for pref_id in turn.get("preference_expressed", {}):
                        if pref_id in preferences_updated:
                            expressed_preferences[pref_id] = preferences_updated[pref_id]
                
                elif turn["role"] == "assistant" and "preference_appealed" in turn:
                    if not turn.get("preference_appealed", {}):
                        turn["preference_appealed"] = {}
                    for pref_id in turn.get("preference_appealed", {}):
                        if pref_id in preferences_updated:
                            appealed_preferences[pref_id] = preferences_updated[pref_id]
            except Exception as e:
                logger.error(f"Error recording preferences from turn {i}: {e}")
                if self.config.get("dev_mode", False):
                    raise e
        
        # Record in preference memory system
        if expressed_preferences or appealed_preferences:
            self.preference_memory.record_conversation_preferences(
                user_id=user.id,
                conversation=conversation,
                expressed_preferences=expressed_preferences,
                appealed_preferences=appealed_preferences,
                event_date=event_date
            )