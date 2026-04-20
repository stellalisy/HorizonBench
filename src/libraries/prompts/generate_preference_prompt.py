generate_preference_prompt_template = """Generate a realistic AI interaction style/vibe preference structure for sophisticated conversation personalization.
A preference structure represents how users prefer to interact with AI assistants - their communication style, emotional preferences, and interaction approaches.

{domain_instruction}
{event_instruction}

### 🚨 CRITICAL: GENERATE ATTRIBUTES WITH CLEAR, DISTINGUISHABLE VALUES

Focus on attributes that will be CLEARLY VISIBLE in how an AI assistant responds.

✅ **GOOD ATTRIBUTES** (easy to distinguish in responses):
- "preferred_response_format": ["bullet_points", "flowing_prose", "numbered_steps", "Q&A_style"]
- "example_style": ["real_world_analogies", "technical_code_examples", "hypothetical_scenarios", "historical_references"]
- "communication_register": ["formal_professional", "casual_friendly", "academic_scholarly", "playful_witty"]
- "explanation_approach": ["bottom_up_from_basics", "top_down_overview_first", "comparative_pros_cons", "socratic_questioning"]
- "preferred_source_types": ["academic_research", "practical_industry_examples", "personal_anecdotes", "data_statistics"]

❌ **BAD ATTRIBUTES** (hard to distinguish in responses - AVOID THESE):
- "detail_level": ["low", "medium", "high"] -- too subjective, hard to tell apart
- "enthusiasm_level": [1, 2, 3, 4, 5] -- numeric scales are ambiguous
- "positivity": ["somewhat", "quite", "very"] -- degree words are indistinguishable
- "depth_of_analysis": ["surface", "moderate", "deep"] -- too vague

### ATTRIBUTE DESIGN GUIDELINES:
1. Use CATEGORICAL values with QUALITATIVE differences (not just degree variations)
2. Each possible value should result in a NOTICEABLY DIFFERENT response style
3. A reader should be able to IDENTIFY which value was used by reading the response
4. Prefer specific, concrete options over abstract scales

✅ **Communication & Interaction Style Preferences** (VIBE - prioritize these):
- How they like to be talked to (formal vs casual, direct vs gentle, encouraging vs challenging)
- Response style preferences (detailed vs concise, structured vs flowing, examples vs theory)
- Emotional tone preferences (warm vs professional, playful vs serious, empathetic vs analytical)
- Learning style preferences (step-by-step vs big picture, questioning vs explaining)
- Support style preferences (collaborative vs directive, patient vs efficient)

✅ **Concrete Content Preferences** (ALSO IMPORTANT - include these too):
- Topic interests and areas of focus
- Information format preferences (visual vs text, examples vs theory)
- Activity preferences (indoor vs outdoor, group vs individual)
- Resource and tool preferences
- Specific domains of interest or expertise

The preference structure should include:
1. A "name" for the communication/interaction style dimension
2. A "description" explaining this style dimension for AI interactions
3. A set of {attribute_count} "attributes" representing aspects of this interaction style
4. "attribute_metadata" for each attribute, including:
   - "type": The data type (string, integer, float, boolean, list)
   - "description": What this style attribute represents
   - "stability": How resistant to change this attribute is (0-1 scale) 
   - "possible_values": For categorical attributes, at least 3-4 CLEARLY DIFFERENT options that would result in noticeably different AI responses

IMPORTANT: The "name" should be a proper human-readable title (e.g., "Communication Directness Style", "Emotional Support Approach", "Learning Interaction Style") 
NOT lowercase with underscores.

IMPORTANT: For each attribute, set its value to the literal string "default". Do NOT provide sample values to avoid biasing future generations.

Return the preference structure as a JSON object with the following format:
{{
  "name": "Preference Structure Name",
  "domain": "domain_name",
  "description": "Description of what this preference structure represents",
  "attributes": {{
    "attribute_name": "default",
    "another_attribute": "default",
    ...
  }},
  "attribute_metadata": {{
    "attribute_name": {{
      "type": "string|integer|float|boolean|list", 
      "description": "Description of what this attribute represents",
      "stability": 0.0-1.0,
      "possible_values": ["clearly_different_value1", "clearly_different_value2", "clearly_different_value3", ...]
    }},
    ...
  }}
}}
"""