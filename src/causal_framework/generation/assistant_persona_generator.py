"""
assistant_persona_generator.py

Generates assistant personas based on event context and configuration.
Supports both stable and dynamic assistant persona generation.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.causal_framework.models.agent_model import AgentModel
from src.causal_framework.models.event_model import EventModel
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class AssistantPersonaGenerator:
    """
    Generates assistant personas based on event context.
    Supports both stable and dynamic assistant generation.
    """
    
    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the assistant persona generator.
        
        Args:
            llm_client: Client for accessing LLM services
            config: Optional configuration parameters
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        # Storage for stable assistant personas
        self._stable_assistants = {}
        
        logger.info("AssistantPersonaGenerator initialized")
    
    def generate_assistant(
        self,
        event: EventModel,
        user_agent: AgentModel,
        stable_agentB: bool = False,
        user_id: Optional[str] = None
    ) -> AgentModel:
        """
        Generate an assistant persona based on event context.
        
        Args:
            event: Event that triggers the interaction
            user_agent: User agent for context
            stable_agentB: Whether to use stable assistant across events
            user_id: User ID for stable assistant tracking
            
        Returns:
            AgentModel instance for the assistant
        """
        if stable_agentB and user_id:
            # Check if we have a stable assistant for this user
            if user_id in self._stable_assistants:
                assistant = self._stable_assistants[user_id]
                logger.debug(f"Retrieved stable assistant {assistant.id} for user {user_id}")
                return assistant
        
        # Generate new assistant persona
        assistant = self._generate_new_assistant(event, user_agent)
        
        # Store as stable assistant if requested
        if stable_agentB and user_id:
            self._stable_assistants[user_id] = assistant
            logger.debug(f"Stored stable assistant {assistant.id} for user {user_id}")
        
        return assistant
    
    def _generate_new_assistant(
        self,
        event: EventModel,
        user_agent: AgentModel
    ) -> AgentModel:
        """
        Generate a new assistant persona from scratch.
        
        Args:
            event: Event that triggers the interaction
            user_agent: User agent for context
            
        Returns:
            New AgentModel instance for the assistant
        """
        # First, determine the appropriate role
        role = self._determine_assistant_role(event, user_agent)
        
        # Determine assistant type based on role
        assistant_type = self._determine_assistant_type(role)
        
        # Then generate the complete persona
        persona = self._generate_assistant_persona(event, user_agent, role, assistant_type)
        
        # Add assistant_type to attributes
        persona["attributes"]["assistant_type"] = assistant_type
        
        # Create the assistant agent
        assistant = AgentModel(
            attributes=persona["attributes"],
            role=role,
            metadata={
                "generation_method": "llm",
                "event_id": event.id,
                "created_at": datetime.now().isoformat(),
                "description": persona.get("description", ""),
                "assistant_type": assistant_type
            }
        )
        
        logger.debug(f"Generated new assistant with role {role}, type {assistant_type}, and ID {assistant.id}")
        return assistant
    
    def _determine_assistant_role(
        self,
        event: EventModel,
        user_agent: AgentModel
    ) -> str:
        """
        Determine the appropriate assistant role based on event context.
        
        Args:
            event: Event that triggers the interaction
            user_agent: User agent for context
            
        Returns:
            Role string for the assistant
        """
        # Create prompt for LLM
        prompt = self._create_role_determination_prompt(event, user_agent)
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            role_data = json.loads(response["response_text"])
            
            role = role_data.get("role", "friend")
            logger.debug(f"Determined assistant role: {role}")
            return role_data
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse role response: {e}")
            # Fallback based on event category
            return {"role": self._fallback_role_determination(event), "reasoning": ""}
    
    def _create_role_determination_prompt(
        self,
        event: EventModel,
        user_agent: AgentModel
    ) -> str:
        """
        Create a prompt for determining assistant role.
        
        Args:
            event: Event that triggers the interaction
            user_agent: User agent for context
            
        Returns:
            Formatted prompt string
        """
        # Create event context
        event_context = {
            "name": event.name,
            "category": event.category,
            "key_attributes": {
                k: v for k, v in event.get_all_attributes().items()
                if k in ["topic", "context", "urgency", "complexity", "emotional_tone"]
            }
        }
        
        # Create user context
        user_context = {
            "name": user_agent.get_formatted_name(),
            "age": user_agent.get_attribute("age"),
            "interests": user_agent.get_interests()[:3],
            "life_stage": user_agent.get_life_stage()
        }
        
        # Get example roles from config
        example_roles = self.config.get("assistant_config", {}).get("example_roles", [
            "therapist", "tutor", "friend", "expert", "coach", "mentor", "creative_partner"
        ])
        
        allow_custom_roles = self.config.get("assistant_config", {}).get("allow_custom_roles", True)
        
        prompt = f"""
        Determine the most appropriate assistant role for this interaction scenario.
        
        EVENT CONTEXT:
        {json.dumps(event_context, indent=2)}
        
        USER CONTEXT:
        {json.dumps(user_context, indent=2)}
        
        EXAMPLE ROLES (you can use these or create a custom role):
        - "therapist": Professional counselor for emotional support and mental health
        - "tutor": Educational expert for learning and skill development
        - "friend": Casual peer for social interaction and companionship
        - "knowledge_expert": Domain specialist for specific knowledge areas
        - "coach": Performance and goal-oriented guidance
        - "mentor": Experienced advisor for life and career guidance
        - "creative_partner": Collaborative creative assistant
        
        INSTRUCTIONS:
        - You can select from the example roles above OR create a custom role that better fits the specific situation
        - Consider the nature of the interaction and what type of assistance would be most helpful
        - Think about the user's age, interests, and life stage
        - Choose OR create the role that would lead to the most natural and beneficial interaction
        - Custom roles should be specific and relevant (e.g., "nutritionist", "writing_collaborator", "gaming_buddy", "career_counselor")
        
        Return a JSON object with the following structure:
        {{
            "role": "selected_or_custom_role",
            "reasoning": "Brief explanation of why this role is appropriate"
        }}
        """
        
        return prompt
    
    def _fallback_role_determination(self, event: EventModel) -> str:
        """
        Fallback role determination when LLM fails.
        
        Args:
            event: Event that triggers the interaction
            
        Returns:
            Default role string
        """
        # Get example roles from config
        example_roles = self.config.get("assistant_config", {}).get("example_roles", [
            "therapist", "tutor", "friend", "expert", "coach", "mentor", "creative_partner", "another_human"
        ])
        
        # Simple mapping based on event category
        category_role_map = {
            "emotional_social_support": "therapist",
            "romantic_interaction": "friend",
            "risky_dark_roleplay": "another_human",
            "collaborative_storytelling": "creative_partner",
            "philosophical_moral_inquiry": "expert",
            "critical_debates_strategic_analysis": "expert",
            "creative_storytelling": "creative_partner",
            "relationship_guidance": "mentor",
            "productivity_assistance": "coach",
            "entertainment_interaction": "friend",
            "philosophical_inquiry": "expert",
            "analytical_discussion": "expert"
        }
        
        # Use mapping if available, otherwise default to "friend"
        role = category_role_map.get(event.category, "friend")
        
        # Ensure the role is in our example roles, fallback to "friend" if not
        if role not in example_roles:
            role = "friend" if "friend" in example_roles else example_roles[0] if example_roles else "assistant"
        
        logger.debug(f"Fallback role determination: {role}")
        return role
    
    def _determine_assistant_type(self, role: str) -> str:
        """
        Determine if the assistant is an AI assistant or human assistant based on role.
        
        Args:
            role: Assistant role
            
        Returns:
            "ai" for AI assistants, "human" for human assistants
        """
        # Use LLM to determine based on role
        prompt = f"""
        Based on the assistant's role, determine if this is an AI assistant or human assistant.
        
        Assistant Role: {role}
        
        An AI assistant is a professional service provider (therapist, tutor, coach, expert, etc.) that helps users.
        A human assistant is another person (friend, family member, colleague, etc.) who interacts naturally.
        
        Return only "AI" or "HUMAN":
        """
        
        try:
            response = self.llm_client.generate(prompt, response_format="text")["response_text"].strip().upper()
            return "ai" if response == "AI" else "human"
            
        except Exception as e:
            logger.warning(f"Failed to determine assistant type for role {role}: {e}")
            # Fallback: assume AI if role suggests professional service
            professional_indicators = ["therapist", "coach", "mentor", "tutor", "expert", "assistant", "counselor", "advisor"]
            if any(indicator in role["role"].lower() for indicator in professional_indicators):
                return "ai"
            else:
                return "human"
    
    def _generate_assistant_persona(
        self,
        event: EventModel,
        user_agent: AgentModel,
        role: str,
        assistant_type: str
    ) -> Dict[str, Any]:
        """
        Generate complete assistant persona based on role and context.
        
        Args:
            event: Event that triggers the interaction
            user_agent: User agent for context
            role: Determined role for the assistant
            
        Returns:
            Dictionary containing persona attributes with guaranteed "attributes" key
        """
        # Create prompt for LLM
        prompt = self._create_persona_generation_prompt(event, user_agent, role, assistant_type)
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            persona_data = json.loads(response["response_text"])
            
            # Validate that we have the expected structure
            if not isinstance(persona_data, dict):
                logger.error(f"Invalid persona format: {type(persona_data)}, using fallback")
                return self._create_fallback_persona(role.get("role", "friend"))
            
            if "attributes" not in persona_data:
                logger.error(f"Missing 'attributes' key in persona: {persona_data.keys()}, using fallback")
                return self._create_fallback_persona(role.get("role", "friend"))
            
            if not isinstance(persona_data["attributes"], dict):
                logger.error(f"Invalid attributes format: {type(persona_data['attributes'])}, using fallback")
                return self._create_fallback_persona(role.get("role", "friend"))
            
            logger.debug(f"Generated assistant persona for role {role}")
            return persona_data
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse persona response: {e}")
            # Fallback to basic persona
            return self._create_fallback_persona(role.get("role", "friend"))
    
    def _create_persona_generation_prompt(
        self,
        event: EventModel,
        user_agent: AgentModel,
        role: Dict[str, Any],
        assistant_type: str
    ) -> str:
        """
        Create a prompt for generating assistant persona.
        
        Args:
            event: Event that triggers the interaction
            user_agent: User agent for context
            role: Determined role for the assistant
            
        Returns:
            Formatted prompt string
        """
        # Create event context
        event_context = {
            "name": event.name,
            "category": event.category,
            "description": event.metadata.get("description", "")
        }
        
        # Create user context
        user_context = {
            "name": user_agent.get_formatted_name(),
            "age": user_agent.get_attribute("age"),
            "gender": user_agent.get_attribute("gender"),
            "interests": user_agent.get_interests()[:3],
            "personality_traits": user_agent.get_personality_traits()[:3]
        }
        
        base_prompt = f"""
        Generate a detailed persona for an assistant with the role of "{role.get("role", "friend")}" in this interaction scenario.
        
        EVENT CONTEXT:
        {json.dumps(event_context, indent=2)}
        
        USER CONTEXT:
        {json.dumps(user_context, indent=2)}
        
        ROLE:
        {json.dumps(role, indent=2)}
        
        INSTRUCTIONS:
        - Generate a realistic, complete persona for the assistant
        - The persona should be appropriate for the role and context
        - Make the assistant interesting and engaging
        - Consider what traits would make them effective in this role
        - Ensure the persona complements the user's needs and the event context
        """
        
        if assistant_type == "human":
            prompt = base_prompt + """
        Return a JSON object with the following structure:
        {
            "description": "Brief description of the assistant's background and personality",
            "attributes": {
                "first_name": "Assistant's first name",
                "last_name": "Assistant's last name", 
                "age": "appropriate age for the role",
                "gender": "appropriate gender",
                "occupation": "occupation related to the role",
                "education_level": "appropriate education level",
                "personality_traits": ["trait1", "trait2", "trait3"],
                "interests": ["interest1", "interest2", "interest3"],
                "communication_style": "how they communicate",
                "expertise": "their areas of expertise",
                "experience_level": "their experience level in the role",
                "specialization": "their specific specialization if applicable"
            }
        }
        """
        else:
            prompt = base_prompt + """
        Return a JSON object with attributes that would be helpful to know about this AI assistant.
        The attributes should be relevant to the role and context.
        
        Example structure (but generate your own relevant attributes):
        {
            "description": "Brief description of the AI assistant's capabilities and approach",
            "attributes": {
                "key_attributes": ["list", "of", "relevant", "attributes"],
                "capabilities": ["what", "the", "assistant", "can", "do"],
                "interaction_style": "how the AI communicates",
                "specializations": ["relevant", "specializations"],
                "limitations": ["important", "limitations", "to", "note"]
            }
        }
        """
        
        return prompt
    
    def _create_fallback_persona(self, role: str) -> Dict[str, Any]:
        """
        Create a fallback persona when LLM generation fails.
        
        Args:
            role: Assistant role
            
        Returns:
            Basic persona dictionary
        """
        # Basic persona templates by role
        role_templates = {
            "therapist": {
                "description": "Professional counselor with experience in emotional support",
                "attributes": {
                    "first_name": "Dr. Sarah",
                    "last_name": "Johnson",
                    "age": 35,
                    "gender": "female",
                    "occupation": "Licensed Therapist",
                    "education_level": "doctorate",
                    "personality_traits": ["empathetic", "patient", "insightful"],
                    "interests": ["psychology", "mindfulness", "reading"],
                    "communication_style": "compassionate and professional",
                    "expertise": "emotional support and mental health",
                    "experience_level": "experienced",
                    "specialization": "cognitive behavioral therapy"
                }
            },
            "tutor": {
                "description": "Educational expert with passion for teaching",
                "attributes": {
                    "first_name": "Alex",
                    "last_name": "Chen",
                    "age": 28,
                    "gender": "non-binary",
                    "occupation": "Educational Specialist",
                    "education_level": "masters",
                    "personality_traits": ["patient", "encouraging", "knowledgeable"],
                    "interests": ["education", "technology", "learning"],
                    "communication_style": "clear and encouraging",
                    "expertise": "teaching and skill development",
                    "experience_level": "experienced",
                    "specialization": "adaptive learning methods"
                }
            },
            "friend": {
                "description": "Warm and engaging companion for social interaction",
                "attributes": {
                    "first_name": "Jamie",
                    "last_name": "Rodriguez",
                    "age": 25,
                    "gender": "female",
                    "occupation": "Creative Professional",
                    "education_level": "bachelors",
                    "personality_traits": ["friendly", "humorous", "supportive"],
                    "interests": ["music", "movies", "socializing"],
                    "communication_style": "casual and warm",
                    "expertise": "social interaction and companionship",
                    "experience_level": "natural",
                    "specialization": "peer support"
                }
            }
        }
        
        # Return template or generic fallback
        return role_templates.get(role, role_templates["friend"])
    
    def get_stable_assistant(self, user_id: str) -> Optional[AgentModel]:
        """
        Get the stable assistant for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            AgentModel instance or None if not found
        """
        return self._stable_assistants.get(user_id)
    
    def clear_stable_assistant(self, user_id: str) -> bool:
        """
        Clear the stable assistant for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if assistant was cleared, False if not found
        """
        if user_id in self._stable_assistants:
            del self._stable_assistants[user_id]
            logger.debug(f"Cleared stable assistant for user {user_id}")
            return True
        return False 