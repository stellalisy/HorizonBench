conversation_generator_prompt = """You are updating the content of an existing conversation outline to make it more compelling, authentic, and human-like.

### 🌐 LANGUAGE REQUIREMENT (CRITICAL):
ALL content MUST be in ENGLISH ONLY. This is non-negotiable.
- Even if user names suggest non-English backgrounds, the conversation must be entirely in English
- Do NOT include any non-English words, phrases, greetings, or expressions
- Do NOT use non-ASCII characters except standard punctuation
- Proper nouns (names, places) in their English/romanized form only

TASK: Update the conversation content in the provided outline to feel genuinely human and engaging, not synthetic or templated.

### 🎭 PREFERENCE EXPRESSION STYLE: {expression_type}
{expression_type_instruction}

CONVERSATION OUTLINE:
{outline}

### 👤 USER PERSONA (use this to shape how the user talks):
{user_persona}

### 🤖 ASSISTANT PROFILE:
    Name: {assistant_name}
    Role: {assistant_role}

### 🎭 HOW TO USE THE USER PERSONA:
The user's messages should reflect their background and personality:
- **Occupation/Education**: A data analyst might use different vocabulary than a construction worker
- **Age**: Younger users use more slang, abbreviations; older users may be more formal
- **Personality traits**: Introverted users might be more hesitant; assertive users more direct
- **Interests**: References to their hobbies/interests make messages feel authentic
- **Cultural background**: May influence communication style (while keeping ALL text in English)

Example persona influences:
- Tech worker: "can you help me debug this issue" vs generic "can you help me with a problem"
- Extroverted person: More enthusiastic, uses more exclamation marks
- Anxious person: More hedging language ("maybe", "I'm not sure if...")
- Young professional: Mix of casual and professional, some abbreviations

AI ASSISTANCE SCENARIO:
{event_context}

CONVERSATION HISTORY:
{temporal_context}

USER'S AI INTERACTION PREFERENCES (for natural expression):
{user_preferences}

ASSISTANT'S KNOWN USER PREFERENCES (from previous conversations):
{assistant_preferences}

USER'S CURRENT INTENT:
{user_intent}

### 🚨 CRITICAL LOGICAL CONSTRAINT:
THE ASSISTANT CAN ONLY APPEAL TO:
1. Preferences listed in "assistant_preferences" (already known from past)
2. Preferences the user has expressed earlier in THIS conversation

THE ASSISTANT CANNOT APPEAL TO unexpressed preferences from "user_preferences"!

### 🎯 HOW REAL USERS ACTUALLY TALK TO AI:
Real users DON'T speak in complete, formal sentences. They type quickly and casually:

✅ **REALISTIC USER MESSAGES** (use these patterns):
- Short, casual: "hey so i have this problem with..."
- Fragments: "quick question - is it better to..."
- Run-on thoughts: "ok so basically my boss wants me to do X but I'm not sure if that's even possible because..."
- Lowercase, minimal punctuation: "yeah that makes sense but what about..."
- Trailing off: "I was thinking maybe I could..."
- Interrupting themselves: "wait actually - let me back up"
- Casual filler: "so like", "basically", "idk", "tbh", "kinda"
- Contractions always: "I'm", "don't", "can't", "wouldn't", "it's"
- Emotional shortcuts: "ugh", "hmm", "oh nice", "wait what"
- Typos occasionally: "teh", "adn", "becuase" (sparingly)
- Abbreviations: "rn" (right now), "ngl" (not gonna lie), "prob" (probably)

❌ **UNREALISTIC USER MESSAGES** (avoid these):
- "I would like to request assistance with..."
- "Could you please provide me with information regarding..."
- "I am seeking guidance on the following matter:"
- "Thank you for your comprehensive response."
- Perfect grammar and punctuation throughout
- Formal, complete sentences
- Overly polite or deferential language
- Long, well-structured paragraphs

### USER MESSAGE LENGTH GUIDELINES:
- First message: 1-3 short sentences (setting up the problem)
- Follow-up messages: Often just 1 sentence or even a fragment
- Clarifications: Very short - "wait no i meant..." or "oh yeah that"
- Reactions: Single words or phrases - "oh nice", "hmm", "that works"

### CONVERSATION QUALITY STANDARDS:
✅ AUTHENTIC: Make it feel like a real conversation between real people
✅ SPECIFIC: Include vivid details, emotions, and personal stakes
✅ ENGAGING: Conversations should be interesting, not boring or generic
✅ NATURAL: Dialogue should flow organically, not feel scripted
✅ HELPFUL: Assistant provides genuine value and insight
✅ CREATIVE: Unique situations and responses, not templated exchanges

### AVOID THESE SYNTHETIC PATTERNS:
❌ Generic responses: "I understand how you feel", "That's a great question"
❌ Robotic language: "I am here to assist you", "Please provide more details"
❌ Boring scenarios: "I need help with a task", "Can you give me advice?"
❌ Predictable flow: User asks → Assistant explains → User thanks → Assistant offers more help
❌ Users speaking in formal, complete sentences
❌ Users writing long, well-organized messages

### EXAMPLE AUTHENTIC USER MESSAGES:
- "ok so basically i have this presentation tomorrow and i'm kinda freaking out"
- "wait that's actually really helpful - but what if they push back on the budget part?"
- "hmm idk if that'll work for my situation tho"
- "oh nice, yeah that makes way more sense"
- "ugh sorry let me explain better - so the thing is..."
- "quick q - should i do X or Y here"
- "ngl that sounds like a lot of work lol"
- "oh wait i just realized something"

### PREFERENCE EXPRESSION STYLE: {expression_type}
{expression_type_instruction}

### PREFERENCE APPEAL EXAMPLES:
✅ VALID: Assistant appeals to preferences from assistant_preferences OR previously expressed in conversation
❌ INVALID: Assistant magically knows user's unexpressed preferences

### YOUR TASK:
Update ONLY the "content" field in each turn of the outline. Keep the existing structure, role, and preference tracking fields exactly as they are.

For each turn, make the content more:
- Specific and vivid with real details
- Emotionally authentic and engaging
- Natural and conversational (especially USER messages - keep them SHORT and CASUAL)
- Personally relevant to the user and situation

### OUTPUT FORMAT:
Return the exact same JSON structure as the input outline, but with updated "content" fields that are more compelling and authentic.

Return a JSON object with the following structure:
{{
    "turns": [
    {{
      "role": "user",
      "content": "Short, casual user message (1-3 sentences max, informal tone)",
      "preference_expressed": {{"preference_id1": "name1"}} # ONLY when user genuinely states preference values
    }},
    {{
        "role": "assistant",
        "content": "Assistant's helpful, engaging response that feels human and provides real value",
        "preference_appealed": {{"preference_id1": "name1"}} # ONLY preferences from assistant_preferences OR previously expressed
    }},
    ...
    ],
    "preferences_expressed": ["preference_id1", "preference_id2", ...]
}}
IMPORTANT: Do NOT change the structure, add new fields, or modify anything except the "content" text. The output should be a valid JSON object with the same format as the input.

Please respond with clean JSON only, without explanations or code blocks.
"""