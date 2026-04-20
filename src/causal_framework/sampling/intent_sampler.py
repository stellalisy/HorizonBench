"""
intent_sampler.py

Implements intent sampling for the Causal Preference Evolution Framework.
Samples intents based on event context and agent persona/role.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from src.causal_framework.models.agent_model import AgentModel
from src.causal_framework.models.event_model import EventModel
from src.causal_framework.models.intent_model import IntentModel
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class IntentSampler:
    """
    Samples intents based on event context and agent persona/role.
    Uses LLM to generate appropriate intents for each agent.
    """
    
    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the intent sampler.
        
        Args:
            llm_client: Client for accessing LLM services
            config: Optional configuration parameters
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        logger.info("IntentSampler initialized")
    
    def sample_intents(
        self,
        agent_a: AgentModel,
        agent_b: AgentModel,
        event: EventModel
    ) -> Dict[str, IntentModel]:
        """
        Sample intents for both agents based on event context.
        
        Args:
            agent_a: First agent (typically user)
            agent_b: Second agent (typically assistant)
            event: Event that triggers the interaction
            
        Returns:
            Dictionary mapping agent IDs to their IntentModel instances
        """
        # Sample intents for both agents
        intent_a = self._sample_agent_intent(agent_a, agent_b, event)
        intent_b = self._sample_agent_intent(agent_b, agent_a, event)
        
        return {
            agent_a.id: intent_a,
            agent_b.id: intent_b
        }
    
    def _sample_agent_intent(
        self,
        agent: AgentModel,
        other_agent: AgentModel,
        event: EventModel
    ) -> IntentModel:
        """
        Sample intent for a specific agent based on event and context.
        
        Args:
            agent: Agent to sample intent for
            other_agent: The other agent in the conversation
            event: Event that triggers the interaction
            
        Returns:
            IntentModel instance for the agent
        """
        # Create prompt for LLM
        prompt = self._create_intent_sampling_prompt(agent, other_agent, event)
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            intent_data = json.loads(response["response_text"])
            
            # Create IntentModel from response
            intent_model = IntentModel(
                intent_type=intent_data["intent_type"],
                attributes=intent_data["attributes"],
                metadata={
                    "event_id": event.id,
                    "agent_id": agent.id,
                    "generation_method": "llm",
                    "description": intent_data.get("description", "")
                }
            )
            
            logger.debug(f"Sampled intent {intent_model.intent_type} for agent {agent.id}")
            return intent_model
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse intent response: {e}")
            # Fallback to basic intent
            return self._create_fallback_intent(agent, event)
    
    def _create_intent_sampling_prompt(
        self,
        agent: AgentModel,
        other_agent: AgentModel,
        event: EventModel
    ) -> str:
        """
        Create a prompt for LLM intent sampling.
        
        Args:
            agent: Agent to sample intent for
            other_agent: The other agent in the conversation
            event: Event that triggers the interaction
            
        Returns:
            Formatted prompt string
        """
        # Determine agent perspective
        if agent.is_user():
            perspective = "user"
            role_context = "seeking assistance or interaction"
        else:
            perspective = "assistant"
            role_context = f"providing {agent.role} services"
            
        # Create event context
        event_context = {
            "name": event.name,
            "category": event.category,
            "key_attributes": {
                k: v for k, v in event.get_all_attributes().items()
                if k in ["topic", "context", "urgency", "complexity"]
            }
        }
        
        # Create agent context
        agent_context = {
            "name": agent.get_formatted_name(),
            "role": agent.role,
            "age": agent.get_attribute("age"),
            "interests": agent.get_interests()[:3],  # Limit to top 3
            "personality_traits": agent.get_personality_traits()[:3]  # Limit to top 3
        }
        
        # Create other agent context
        other_context = {
            "name": other_agent.get_formatted_name(),
            "role": other_agent.role
        }
        
        prompt = f"""
        Generate an appropriate intent for an agent in a conversation scenario.
        
        AGENT CONTEXT (the agent whose intent you're generating):
        {json.dumps(agent_context, indent=2)}
        
        OTHER AGENT CONTEXT (the other participant):
        {json.dumps(other_context, indent=2)}
        
        EVENT CONTEXT (what triggered this interaction):
        {json.dumps(event_context, indent=2)}
        
        INSTRUCTIONS:
        - Generate a realistic intent for the agent from the {perspective} perspective
        - Consider the agent's role: {role_context}
        - The intent should be specific to this event and context
        - Include concrete attributes that define the intent's characteristics
        - Make the intent authentic and goal-oriented
        
        INTENT TYPES (examples):
        - For users: "seek_emotional_support", "get_advice", "learn_skill", "creative_collaboration", "vent_frustration"
        - For assistants: "provide_guidance", "offer_support", "teach_knowledge", "facilitate_creativity", "listen_actively"
        
        Return a JSON object with the following structure:
        {{
            "intent_type": "specific_intent_type",
            "description": "Brief description of what the agent wants to achieve",
            "attributes": {{
                "primary_goal": "main objective",
                "interaction_style": "how they want to interact",
                "desired_outcome": "what they hope to accomplish",
                "urgency_level": "low/medium/high",
                "emotional_tone": "their emotional approach"
            }}
        }}
        """
        
        return prompt
    
    def _create_fallback_intent(self, agent: AgentModel, event: EventModel) -> IntentModel:
        """
        Create a fallback intent when LLM sampling fails.
        
        Args:
            agent: Agent to create intent for
            event: Event context
            
        Returns:
            Basic IntentModel instance
        """
        if agent.is_user():
            intent_type = "seek_assistance"
            attributes = {
                "primary_goal": "get help",
                "interaction_style": "collaborative",
                "desired_outcome": "problem resolution",
                "urgency_level": "medium",
                "emotional_tone": "open"
            }
        else:
            intent_type = "provide_assistance"
            attributes = {
                "primary_goal": "help user",
                "interaction_style": "supportive",
                "desired_outcome": "user satisfaction",
                "urgency_level": "medium",
                "emotional_tone": "professional"
            }
        
        return IntentModel(
            intent_type=intent_type,
            attributes=attributes,
            metadata={
                "event_id": event.id,
                "agent_id": agent.id,
                "generation_method": "fallback"
            }
        ) 