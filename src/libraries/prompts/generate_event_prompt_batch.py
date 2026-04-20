generate_batch_event_prompt_template = """
Generate {count_per_category} diverse AI assistant interaction scenarios for the category: {category}

Each scenario should represent a realistic situation where someone would seek help from an AI assistant.
{social_graph_info}

Each scenario structure should include:
1. A descriptive "name" for the scenario (e.g., "Drafting apology email to colleague", "Brainstorming blog post ideas")
2. A set of "attributes" representing the scenario's context and requirements
3. A dictionary of "preference_dependent" flags indicating which attributes depend on user's AI interaction preferences

Make the scenarios diverse within the category, covering different contexts, complexity levels, and interaction styles.

Focus on realistic scenarios based on actual AI chatbot usage patterns. You can create scenarios in the target category: {category}

OR if you want to be more diverse, explore other realistic AI assistance categories such as:
- educational_tutoring: homework help, concept explanation, skill development
- technical_support: coding help, troubleshooting, software guidance
- health_wellness: fitness advice, nutrition guidance, wellness planning
- financial_planning: budgeting, investment advice, expense tracking
- travel_planning: itinerary creation, destination research, travel tips
- legal_advice: basic legal questions, document understanding, rights explanation
- parenting_guidance: child development, discipline strategies, family dynamics
- career_coaching: job search, interview prep, professional development
- hobby_exploration: learning new skills, project guidance, community finding
- language_learning: translation, grammar help, conversation practice
- cooking_assistance: recipe help, cooking techniques, meal planning
- mental_health_support: stress management, coping strategies, emotional regulation
- academic_research: paper writing, research methodology, citation help
- personal_development: goal setting, habit formation, self-improvement
- cultural_exploration: traditions, customs, cultural understanding
- news_analysis: current events discussion, fact-checking, perspective analysis

Be creative and think about the full spectrum of ways people use AI assistants!

Return a JSON object with the following format:
{{
  "events": [
    {{
      "name": "AI assistance scenario name",
      "category": "{category}", 
      "description": "Extremely concise description of the event",
      "attributes": {{
        "attribute1": "value1",
        "attribute2": "value2",
        ...
      }},
      "preference_dependent": {{
        "attribute1": true/false,
        "attribute2": true/false,
        ...
      }}
    }},
    ... ({count_per_category} scenarios total)
  ]
}}
"""