"""
intent_fulfillment_assessor.py

Evaluates how well agent intents were fulfilled during conversations.
Used for metrics and evaluation purposes only.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from src.causal_framework.models.agent_model import AgentModel
from src.causal_framework.models.intent_model import IntentModel
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class IntentFulfillmentAssessor:
    """
    Assesses how well agent intents were fulfilled during conversations.
    Provides metrics for intent satisfaction and conversation success.
    """
    
    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the intent fulfillment assessor.
        
        Args:
            llm_client: Client for accessing LLM services
            config: Optional configuration parameters
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        logger.info("IntentFulfillmentAssessor initialized")
    
    def assess_intent_fulfillment(
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
        fulfillment_results = {}
        
        # Assess fulfillment for each agent
        for agent_id, intent_model in agent_intents.items():
            if agent_id in agents:
                agent = agents[agent_id]
                fulfillment_score = self._assess_single_intent_fulfillment(
                    intent_model, conversation, agent
                )
                fulfillment_results[agent_id] = fulfillment_score
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(fulfillment_results)
        
        return {
            "individual_fulfillment": fulfillment_results,
            "overall_metrics": overall_metrics,
            "conversation_success": overall_metrics["average_fulfillment"] > 0.7
        }
    
    def _assess_single_intent_fulfillment(
        self,
        intent_model: IntentModel,
        conversation: Dict[str, Any],
        agent: AgentModel
    ) -> Dict[str, Any]:
        """
        Assess fulfillment for a single agent's intent.
        
        Args:
            intent_model: Intent model to assess
            conversation: Generated conversation dictionary
            agent: Agent model for context
            
        Returns:
            Dictionary containing fulfillment scores and details
        """
        # Use LLM to assess intent fulfillment
        fulfillment_score = self._llm_assess_intent_fulfillment(
            intent_model, conversation, agent
        )
        
        if fulfillment_score is None:
            # Fallback to heuristic assessment
            fulfillment_score = self._heuristic_assess_intent_fulfillment(
                intent_model, conversation, agent
            )
        
        return fulfillment_score
    
    def _llm_assess_intent_fulfillment(
        self,
        intent_model: IntentModel,
        conversation: Dict[str, Any],
        agent: AgentModel
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to assess intent fulfillment.
        
        Args:
            intent_model: Intent model to assess
            conversation: Generated conversation dictionary
            agent: Agent model for context
            
        Returns:
            Dictionary containing fulfillment assessment or None if failed
        """
        try:
            prompt = self._create_intent_assessment_prompt(
                intent_model, conversation, agent
            )
            
            response = self.llm_client.generate(prompt, response_format="json")
            assessment_data = json.loads(response["response_text"])
            
            return {
                "fulfillment_score": assessment_data.get("fulfillment_score", 0.5),
                "fulfillment_reasons": assessment_data.get("fulfillment_reasons", []),
                "unfulfilled_aspects": assessment_data.get("unfulfilled_aspects", []),
                "assessment_confidence": assessment_data.get("assessment_confidence", 0.5),
                "assessment_method": "llm"
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse intent assessment response: {e}")
            return None
    
    def _create_intent_assessment_prompt(
        self,
        intent_model: IntentModel,
        conversation: Dict[str, Any],
        agent: AgentModel
    ) -> str:
        """
        Create a prompt for LLM intent assessment.
        
        Args:
            intent_model: Intent model to assess
            conversation: Generated conversation dictionary
            agent: Agent model for context
            
        Returns:
            Formatted prompt string
        """
        # Extract conversation turns
        turns = conversation.get("turns", [])
        
        # Create conversation text
        conversation_text = ""
        for i, turn in enumerate(turns):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            conversation_text += f"{role.capitalize()}: {content}\n"
        
        # Create intent context
        intent_context = {
            "intent_type": intent_model.intent_type,
            "description": intent_model.metadata.get("description", ""),
            "attributes": intent_model.get_all_attributes()
        }
        
        # Create agent context
        agent_context = {
            "name": agent.get_formatted_name(),
            "role": agent.role,
            "is_user": agent.is_user()
        }
        
        prompt = f"""
        Assess how well the agent's intent was fulfilled during this conversation.
        
        AGENT CONTEXT:
        {json.dumps(agent_context, indent=2)}
        
        AGENT'S INTENT:
        {json.dumps(intent_context, indent=2)}
        
        CONVERSATION:
        {conversation_text}
        
        ASSESSMENT CRITERIA:
        - Did the agent achieve their primary goal?
        - Was the desired interaction style maintained?
        - Did the conversation produce the desired outcome?
        - How well were the intent's specific attributes addressed?
        - Consider both explicit achievements and implicit satisfaction
        
        SCORING:
        - 0.0-0.2: Intent completely unfulfilled
        - 0.2-0.4: Intent mostly unfulfilled
        - 0.4-0.6: Intent partially fulfilled
        - 0.6-0.8: Intent mostly fulfilled
        - 0.8-1.0: Intent completely fulfilled
        
        Return a JSON object with the following structure:
        {{
            "fulfillment_score": 0.75,
            "fulfillment_reasons": ["reason1", "reason2"],
            "unfulfilled_aspects": ["aspect1", "aspect2"],
            "assessment_confidence": 0.8
        }}
        """
        
        return prompt
    
    def _heuristic_assess_intent_fulfillment(
        self,
        intent_model: IntentModel,
        conversation: Dict[str, Any],
        agent: AgentModel
    ) -> Dict[str, Any]:
        """
        Heuristic assessment when LLM fails.
        
        Args:
            intent_model: Intent model to assess
            conversation: Generated conversation dictionary
            agent: Agent model for context
            
        Returns:
            Dictionary containing basic fulfillment assessment
        """
        # Simple heuristics based on conversation length and turn count
        turns = conversation.get("turns", [])
        
        if not turns:
            return {
                "fulfillment_score": 0.0,
                "fulfillment_reasons": [],
                "unfulfilled_aspects": ["no conversation occurred"],
                "assessment_confidence": 0.9,
                "assessment_method": "heuristic"
            }
        
        # Basic scoring based on conversation characteristics
        turn_count = len(turns)
        agent_turns = [t for t in turns if t.get("role") == ("user" if agent.is_user() else "assistant")]
        
        # More turns generally means better engagement
        engagement_score = min(1.0, turn_count / 6.0)  # 6 turns = full engagement
        
        # Agent participation
        participation_score = len(agent_turns) / max(1, turn_count / 2)
        participation_score = min(1.0, participation_score)
        
        # Combine scores
        fulfillment_score = (engagement_score + participation_score) / 2
        
        return {
            "fulfillment_score": fulfillment_score,
            "fulfillment_reasons": [
                f"Conversation had {turn_count} turns",
                f"Agent participated in {len(agent_turns)} turns"
            ],
            "unfulfilled_aspects": [] if fulfillment_score > 0.6 else ["limited engagement"],
            "assessment_confidence": 0.5,
            "assessment_method": "heuristic"
        }
    
    def _calculate_overall_metrics(
        self,
        fulfillment_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall metrics from individual fulfillment results.
        
        Args:
            fulfillment_results: Dictionary of individual fulfillment results
            
        Returns:
            Dictionary containing overall metrics
        """
        if not fulfillment_results:
            return {
                "average_fulfillment": 0.0,
                "min_fulfillment": 0.0,
                "max_fulfillment": 0.0,
                "fulfillment_variance": 0.0,
                "agents_assessed": 0
            }
        
        # Extract scores
        scores = [
            result.get("fulfillment_score", 0.0)
            for result in fulfillment_results.values()
        ]
        
        # Calculate metrics
        average_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Calculate variance
        variance = sum((score - average_score) ** 2 for score in scores) / len(scores)
        
        return {
            "average_fulfillment": average_score,
            "min_fulfillment": min_score,
            "max_fulfillment": max_score,
            "fulfillment_variance": variance,
            "agents_assessed": len(scores)
        } 