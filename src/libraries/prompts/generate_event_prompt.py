generate_event_prompt_template="""Generate a highly realistic and engaging AI assistant interaction scenario. Focus on creating vivid, specific, and authentic situations that feel like real human needs rather than generic templates.
        
        {category_instruction}
        {social_graph_info}
        
        CRITICAL: Make scenarios feel AUTHENTIC and SPECIFIC, not generic. Avoid boring, templated language. Create scenarios people actually face in their real lives.
        
        Examples of ENGAGING scenarios across categories (based on Character.AI usage data):
        
        PRODUCTIVITY_ASSISTANCE (40.5% of usage - HIGHEST):
        Real scenarios: "Crafting a resignation email that doesn't burn bridges", "Planning a surprise birthday party with complex logistics", "Writing a grant proposal for a community art project", "Organizing a chaotic home office during remote work transition", "Developing a business pitch for an eccentric startup idea"
        - task_complexity: "straightforward", "multi-layered", "creative_challenge", "time_sensitive"
        - context_domain: "corporate", "academic", "personal_project", "creative_venture", "life_transition"
        - stakeholder_pressure: "low", "moderate", "high_stakes", "career_defining"
        - communication_style: preference-dependent (professional vs casual tone)
        - organization_approach: preference-dependent (detailed plans vs flexible frameworks)
        
        ENTERTAINMENT_INTERACTION (37.8% of usage - SECOND HIGHEST):
        Real scenarios: "Creating an elaborate D&D campaign set in cyberpunk Tokyo", "Debating whether pineapple belongs on pizza with passion", "Generating witty comebacks for family dinner conversations", "Playing a detective game solving a mystery in 1920s Paris", "Collaborating on writing absurd superhero origin stories"
        - interaction_type: "intellectual_games", "creative_challenges", "social_banter", "immersive_roleplay", "competitive_wordplay"
        - energy_level: "chill_vibes", "moderate_engagement", "high_energy_fun", "intense_focus"
        - content_style: "clever_humor", "silly_fun", "sophisticated_wit", "absurd_comedy"
        - engagement_depth: preference-dependent (casual fun vs deep immersion)
        - humor_preference: preference-dependent (sarcastic vs wholesome vs dark)
        
        CREATIVE_STORYTELLING (15% of usage):
        Real scenarios: "Developing a morally complex villain for a fantasy novel", "Creating dialogue for a romantic confession scene", "Building an alien civilization with unique social structures", "Writing from the perspective of a time traveler's diary", "Collaborating on a mystery where the detective is a cat"
        - narrative_scope: "character_study", "world_building", "plot_development", "dialogue_crafting", "atmosphere_creation"
        - genre_fusion: "urban_fantasy", "sci_fi_romance", "historical_mystery", "psychological_thriller", "magical_realism"
        - creative_role: "collaborative_writer", "character_voice", "world_consultant", "plot_advisor"
        - artistic_style: preference-dependent (poetic vs straightforward prose)
        - creative_freedom: preference-dependent (structured guidance vs free-form exploration)
        
        EMOTIONAL_SOCIAL_SUPPORT (8% of usage):
        Real scenarios: "Processing feelings after a career rejection", "Dealing with imposter syndrome in a new job", "Navigating family dynamics during holiday stress", "Working through social anxiety before a big presentation", "Finding motivation after a creative burnout"
        - emotional_intensity: "mild_concern", "moderate_stress", "significant_struggle", "crisis_moment"
        - support_focus: "practical_coping", "emotional_validation", "perspective_shifting", "decision_guidance"
        - life_domain: "career_transitions", "relationship_dynamics", "personal_growth", "creative_blocks", "social_situations"
        - therapeutic_approach: preference-dependent (gentle encouragement vs direct coaching)
        - vulnerability_comfort: preference-dependent (surface sharing vs deep exploration)
        
        PHILOSOPHICAL_INQUIRY (6.5% of usage):
        Real scenarios: "Exploring the ethics of AI consciousness", "Debating whether free will exists in a deterministic universe", "Questioning the meaning of success in modern society", "Examining the morality of genetic engineering", "Discussing the nature of authentic relationships"
        - inquiry_depth: "surface_curiosity", "moderate_exploration", "deep_philosophical_dive"
        - topic_domain: "existential_questions", "ethical_dilemmas", "consciousness_puzzles", "societal_values", "meaning_seeking"
        - discussion_style: preference-dependent (Socratic questioning vs informational exchange)
        - intellectual_challenge: preference-dependent (comfortable agreement vs provocative challenge)
        
        RELATIONSHIP_GUIDANCE (4% of usage):
        Real scenarios: "Navigating a crush on a close friend", "Dealing with passive-aggressive coworkers", "Setting boundaries with overbearing family members", "Approaching difficult conversations in relationships", "Understanding mixed signals in dating"
        - relationship_context: "romantic_tension", "friendship_dynamics", "family_complexity", "workplace_politics", "social_group_drama"
        - advice_urgency: "immediate_crisis", "ongoing_pattern", "preventive_planning", "skill_building"
        - guidance_approach: preference-dependent (direct advice vs reflective questioning)
        - boundary_comfort: preference-dependent (conservative vs open discussion)
        
        ANALYTICAL_DISCUSSION (2.5% of usage):
        Real scenarios: "Analyzing the strategic implications of a chess opening", "Breaking down the rhetoric in political debates", "Evaluating the logic behind conspiracy theories", "Dissecting the narrative structure of complex films", "Examining the economics of sustainable living"
        - analysis_focus: "strategic_thinking", "logical_evaluation", "system_analysis", "pattern_recognition", "critical_examination"
        - complexity_level: "straightforward_analysis", "multi_factor_evaluation", "complex_system_thinking"
        - discussion_mode: preference-dependent (collaborative exploration vs devil's advocate)
        - evidence_emphasis: preference-dependent (data-driven vs intuition-based reasoning)
        
        CREATE SCENARIOS THAT:
        ✓ Feel like real situations actual humans face
        ✓ Have specific, vivid details that make them memorable  
        ✓ Avoid generic "template" language
        ✓ Include emotional context and stakes
        ✓ Reflect genuine human complexity and nuance
        
        For each attribute, indicate whether it depends on user's AI interaction preferences (TRUE) or is fixed scenario context (FALSE).
        
        {base_attributes_instruction}
        
        Return the event structure as a JSON object with the following format:
        {{
          "name": "AI assistance scenario name",
          "category": "Category name",
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
        }}"""