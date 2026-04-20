"""
preference_memory_system.py

Sophisticated preference memory and retrieval system for long-term conversation personalization.
Enables assistants to remember and reference user preferences from previous conversations.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from src.causal_framework.models.preference_model import PreferenceModel
from src.causal_framework.models.event_model import EventModel
from src.causal_framework.models.user_model import UserModel
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class PreferenceMemorySystem:
    """
    Sophisticated system for managing long-term preference memory across conversations.
    Enables natural, personalized AI assistance by remembering user communication styles and preferences.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the preference memory system.
        
        Args:
            llm_client: LLM client for intelligent preference analysis
        """
        self.llm_client = llm_client
        self.preference_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.style_profiles: Dict[str, Dict[str, Any]] = {}
        
    def record_conversation_preferences(
        self,
        user_id: str,
        conversation: Dict[str, Any],
        expressed_preferences: Dict[str, PreferenceModel],
        appealed_preferences: Dict[str, PreferenceModel],
        event_date: datetime
    ) -> None:
        """
        Record preferences expressed or appealed to in a conversation for future retrieval.
        
        Args:
            user_id: User identifier
            conversation: The conversation data
            expressed_preferences: Preferences expressed by user in this conversation
            appealed_preferences: Preferences appealed to by assistant
            event_date: Date of the conversation
        """
        # Extract conversation context for better preference understanding
        conversation_context = self._extract_conversation_context(conversation)
        
        # Record expressed preferences with rich context
        for pref_id, pref_model in expressed_preferences.items():
            preference_record = {
                "preference_id": pref_id,
                "preference_data": pref_model.to_dict(),
                "expression_context": conversation_context,
                "expressed_date": event_date.isoformat(),
                "conversation_type": conversation.get("scenario", "unknown"),
                "confidence": self._calculate_expression_confidence(conversation, pref_id),
                "style_elements": self._extract_style_elements(pref_model)
            }
            
            self.preference_history[user_id].append(preference_record)
            
        # Update the user's style profile
        self._update_style_profile(user_id, expressed_preferences, conversation_context)
        
        logger.debug(f"Recorded {len(expressed_preferences)} preferences for user {user_id}")
    
    def get_conversation_context(
        self,
        user_id: str,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Get simple conversation context without doing preference selection.
        This is used by the conversation generator to get historical context.
        
        Args:
            user_id: User identifier
            lookback_days: How far back to look for context
            
        Returns:
            Dictionary containing conversation context summary
        """
        if user_id not in self.preference_history:
            return {
                "context_summary": "No previous conversation history with this user.",
                "conversations": [],
                "interaction_count": 0
            }
        
        # Get cutoff date for lookback
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        
        # Filter records within lookback window
        relevant_history = [
            record for record in self.preference_history[user_id]
            if record["expressed_date"] >= cutoff_date
        ]
        
        if not relevant_history:
            return {
                "context_summary": "No recent conversation history.",
                "conversations": [],
                "interaction_count": 0
            }
        
        # Create simple context summary
        conversation_types = [record.get("conversation_type", "unknown") for record in relevant_history]
        unique_types = list(set(conversation_types))
        
        context_summary = f"Previous conversations ({len(relevant_history)} interactions): "
        if len(unique_types) <= 3:
            context_summary += f"Focused on {', '.join(unique_types)}."
        else:
            context_summary += f"Covered diverse topics including {', '.join(unique_types[:3])} and others."
        
        return {
            "context_summary": context_summary,
            "conversations": relevant_history[-5:],  # Last 5 conversations for context
            "interaction_count": len(relevant_history)
        }
    
    def retrieve_historical_preferences(
        self,
        user_id: str,
        current_event: EventModel,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Retrieve relevant historical preferences for the current conversation context.
        
        Args:
            user_id: User identifier
            current_event: Current event context
            lookback_days: How far back to look for preferences
            
        Returns:
            Dictionary containing relevant historical preferences and style information
        """
        if user_id not in self.preference_history:
            return {"preferences": {}, "style_profile": {}, "memory_context": "No previous conversations"}
        
        # Get cutoff date for lookback
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        
        # Filter preferences within lookback window
        relevant_history = [
            record for record in self.preference_history[user_id]
            if record["expressed_date"] >= cutoff_date
        ]
        
        if not relevant_history:
            return {"preferences": {}, "style_profile": {}, "memory_context": "No recent conversation history"}
        
        # Use LLM to intelligently select relevant preferences for current context
        relevant_preferences = self._llm_select_relevant_preferences(
            relevant_history, current_event
        )
        
        # Get user's style profile
        style_profile = self.style_profiles.get(user_id, {})
        
        # Create memory context summary
        memory_context = self._create_memory_context_summary(relevant_preferences, style_profile)
        
        return {
            "preferences": relevant_preferences,
            "style_profile": style_profile,
            "memory_context": memory_context,
            "conversation_count": len(relevant_history),
            "oldest_memory": min(record["expressed_date"] for record in relevant_history) if relevant_history else None
        }
    
    def _extract_conversation_context(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meaningful context from a conversation for better preference understanding."""
        context = {
            "scenario": conversation.get("scenario", ""),
            "turn_count": len(conversation.get("turns", [])),
            "user_goals": [],
            "assistance_type": "",
            "emotional_tone": "",
            "interaction_style": ""
        }
        
        # Analyze conversation content for richer context
        turns = conversation.get("turns", [])
        if turns:
            # Extract user goals and assistant response patterns
            user_messages = [turn["content"] for turn in turns if turn["role"] == "user"]
            assistant_messages = [turn["content"] for turn in turns if turn["role"] == "assistant"]
            
            if user_messages and assistant_messages:
                # Use simple heuristics or could enhance with LLM analysis
                context["assistance_type"] = self._infer_assistance_type(user_messages[0])
                context["emotional_tone"] = self._infer_emotional_tone(user_messages)
                context["interaction_style"] = self._infer_interaction_style(assistant_messages)
        
        return context
    
    def _calculate_expression_confidence(self, conversation: Dict[str, Any], pref_id: str) -> float:
        """Calculate confidence that a preference was genuinely expressed vs. inferred."""
        # Simple confidence scoring - could be enhanced
        turns = conversation.get("turns", [])
        explicit_mentions = 0
        
        for turn in turns:
            if turn["role"] == "user" and pref_id in turn.get("preference_expressed", {}):
                explicit_mentions += 1
        
        # Higher confidence for explicitly mentioned preferences
        if explicit_mentions > 0:
            return min(0.8 + (explicit_mentions * 0.1), 1.0)
        else:
            return 0.5  # Medium confidence for inferred preferences
    
    def _extract_style_elements(self, pref_model: PreferenceModel) -> Dict[str, Any]:
        """Extract communication style elements from a preference model."""
        attributes = pref_model.get_all_attributes()
        style_elements = {}
        
        # Map preference attributes to style categories
        style_mappings = {
            "communication_style": ["directness", "formality", "warmth", "humor"],
            "support_style": ["encouragement_level", "challenge_level", "empathy_focus"],
            "learning_style": ["detail_level", "example_preference", "explanation_style"],
            "interaction_style": ["collaboration_level", "guidance_style", "autonomy_preference"]
        }
        
        for style_category, attr_patterns in style_mappings.items():
            category_attrs = {}
            for attr_name, attr_value in attributes.items():
                if any(pattern in attr_name.lower() for pattern in attr_patterns):
                    category_attrs[attr_name] = attr_value
            
            if category_attrs:
                style_elements[style_category] = category_attrs
        
        return style_elements
    
    def _update_style_profile(
        self,
        user_id: str,
        preferences: Dict[str, PreferenceModel],
        conversation_context: Dict[str, Any]
    ) -> None:
        """Update the user's overall communication style profile."""
        if user_id not in self.style_profiles:
            self.style_profiles[user_id] = {
                "communication_patterns": {},
                "preferred_interaction_styles": {},
                "emotional_preferences": {},
                "learning_preferences": {},
                "last_updated": datetime.now().isoformat(),
                "conversation_count": 0
            }
        
        profile = self.style_profiles[user_id]
        profile["conversation_count"] += 1
        profile["last_updated"] = datetime.now().isoformat()
        
        # Aggregate style information from all preferences
        for pref_model in preferences.values():
            style_elements = self._extract_style_elements(pref_model)
            
            for style_category, elements in style_elements.items():
                if style_category not in profile:
                    profile[style_category] = {}
                
                # Simple aggregation - could use more sophisticated merging
                for attr, value in elements.items():
                    if attr in profile[style_category]:
                        # Average numeric values, keep most recent for others
                        if isinstance(value, (int, float)) and isinstance(profile[style_category][attr], (int, float)):
                            profile[style_category][attr] = (profile[style_category][attr] + value) / 2
                        else:
                            profile[style_category][attr] = value
                    else:
                        profile[style_category][attr] = value
    
    def _llm_select_relevant_preferences(
        self,
        preference_history: List[Dict[str, Any]],
        current_event: EventModel
    ) -> Dict[str, Any]:
        """Use LLM to intelligently select relevant historical preferences for current context."""
        if not preference_history:
            return {}
        
        # Create prompt for LLM to select relevant preferences
        prompt = f"""
        Given the current AI assistance context and a user's historical preference expressions, 
        select the most relevant preferences that should influence how the AI assistant interacts with them.
        
        CURRENT CONTEXT:
        {json.dumps(current_event.to_dict(), indent=2)}
        
        HISTORICAL PREFERENCES (from previous conversations):
        {json.dumps([{
            "preference_id": record["preference_id"],
            "preference_data": record["preference_data"],
            "expressed_date": record["expressed_date"],
            "conversation_type": record["conversation_type"],
            "confidence": record["confidence"],
            "style_elements": record.get("style_elements", {})
        } for record in preference_history[-10:]], indent=2)}  # Last 10 for context
        
        Select preferences that would be NATURALLY relevant for the assistant to remember and use 
        in the current context. Focus on communication style, interaction preferences, and 
        approaches that would enhance user engagement.
        
        Return a JSON object with:
        {{
            "selected_preferences": {{
                "preference_id": {{
                    "preference_data": "full preference data",
                    "relevance_reason": "why this preference is relevant to current context",
                    "memory_age": "how long ago this was expressed",
                    "usage_suggestion": "how the assistant should naturally use this"
                }}
            }},
            "overall_memory_summary": "Brief summary of the user's communication style based on history"
        }}
        
        Prioritize:
        1. Communication and interaction style preferences
        2. Preferences that enhance engagement and helpfulness
        3. Recently expressed preferences (higher relevance)
        4. Preferences with high confidence scores
        """
        
        try:
            response = self.llm_client.generate(prompt, response_format="json")
            result = json.loads(response["response_text"])
            return result.get("selected_preferences", {})
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response for preference selection: {response}")
            # Fallback: return most recent high-confidence preferences
            return self._fallback_preference_selection(preference_history)
    
    def _fallback_preference_selection(self, preference_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback preference selection when LLM parsing fails."""
        # Select most recent high-confidence preferences
        sorted_prefs = sorted(preference_history, key=lambda x: (x["confidence"], x["expressed_date"]), reverse=True)
        
        selected = {}
        for record in sorted_prefs[:5]:  # Top 5 preferences
            selected[record["preference_id"]] = {
                "preference_data": record["preference_data"],
                "relevance_reason": "Recent high-confidence preference",
                "memory_age": record["expressed_date"],
                "usage_suggestion": "Use to personalize communication style"
            }
        
        return selected
    
    def _create_memory_context_summary(
        self,
        relevant_preferences: Dict[str, Any],
        style_profile: Dict[str, Any]
    ) -> str:
        """Create a natural language summary of the user's preference memory for the assistant."""
        if not relevant_preferences and not style_profile:
            return "No previous conversations to reference."
        
        summary_parts = []
        
        if relevant_preferences:
            pref_count = len(relevant_preferences)
            summary_parts.append(f"I remember {pref_count} relevant preference{'s' if pref_count != 1 else ''} from our previous conversations.")
        
        if style_profile and style_profile.get("conversation_count", 0) > 0:
            conv_count = style_profile["conversation_count"]
            summary_parts.append(f"We've talked {conv_count} time{'s' if conv_count != 1 else ''} before.")
        
        # Add key style elements if available
        if style_profile:
            style_elements = []
            for category, attrs in style_profile.items():
                if isinstance(attrs, dict) and category not in ["last_updated", "conversation_count"]:
                    if attrs:  # Non-empty
                        style_elements.append(category.replace("_", " "))
            
            if style_elements:
                summary_parts.append(f"I have notes about your {', '.join(style_elements[:3])}.")
        
        return " ".join(summary_parts) if summary_parts else "Building our conversation history."
    
    def _infer_assistance_type(self, first_user_message: str) -> str:
        """Infer the type of assistance from the user's first message."""
        message_lower = first_user_message.lower()
        
        if any(word in message_lower for word in ["write", "draft", "compose", "create"]):
            return "writing_assistance"
        elif any(word in message_lower for word in ["advice", "suggest", "recommend", "help me decide"]):
            return "decision_support"
        elif any(word in message_lower for word in ["explain", "understand", "learn", "teach"]):
            return "educational_support"
        elif any(word in message_lower for word in ["feel", "stressed", "worried", "upset", "excited"]):
            return "emotional_support"
        else:
            return "general_assistance"
    
    def _infer_emotional_tone(self, user_messages: List[str]) -> str:
        """Infer emotional tone from user messages."""
        combined_text = " ".join(user_messages).lower()
        
        if any(word in combined_text for word in ["excited", "happy", "great", "awesome", "love"]):
            return "positive"
        elif any(word in combined_text for word in ["worried", "stressed", "anxious", "frustrated", "upset"]):
            return "concerned"
        elif any(word in combined_text for word in ["urgent", "quickly", "asap", "deadline", "hurry"]):
            return "urgent"
        else:
            return "neutral"
    
    def _infer_interaction_style(self, assistant_messages: List[str]) -> str:
        """Infer interaction style from assistant messages."""
        combined_text = " ".join(assistant_messages).lower()
        
        if any(phrase in combined_text for phrase in ["let's", "we can", "together", "collaborate"]):
            return "collaborative"
        elif any(word in combined_text for word in ["step", "first", "next", "then", "process"]):
            return "structured"
        elif any(word in combined_text for word in ["understand", "feel", "sounds like", "empathy"]):
            return "empathetic"
        else:
            return "informational" 