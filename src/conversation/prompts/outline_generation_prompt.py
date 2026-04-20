outline_generator_prompts = """Generate a conversation outline for an AI assistant interaction that feels genuinely human and engaging.

### 🚨 CONVERSATION LENGTH (STRICT REQUIREMENT):
Generate EXACTLY {num_turns} turns. Not {num_turns_minus_one}. Not {num_turns_plus_one}. EXACTLY {num_turns} turns.
- Count carefully: your output must have EXACTLY {num_turns} conversation turns
- A "turn" = one user message + one assistant response (2 messages total per turn)
- Do NOT exceed this limit under any circumstances

### 🌐 LANGUAGE REQUIREMENT (CRITICAL):
ALL content MUST be in ENGLISH ONLY. This is non-negotiable.
- Even if user names suggest non-English backgrounds, the conversation must be entirely in English
- Do NOT include any non-English words, phrases, greetings, or expressions
- Do NOT use non-ASCII characters except standard punctuation
- Proper nouns (names, places) in their English/romanized form only

TASK: Create a structured outline for a conversation between a user and AI assistant based on the given scenario and context.

SCENARIO: {scenario}

USER PROFILE:
{user_information}

ASSISTANT PROFILE:
{assistant_information}

AI ASSISTANCE SCENARIO:
{event_context}

CONVERSATION HISTORY:
{temporal_context}

USER'S AI INTERACTION PREFERENCES (unknown to assistant until expressed):
{preferences_updated_user}

ASSISTANT'S KNOWN USER PREFERENCES (from previous conversations):
{preferences_current_user}

ASSISTANT'S OWN PREFERENCES:
{preferences_updated_assistant}

USER'S CURRENT INTENT:
{user_intent}

ASSISTANT'S CURRENT INTENT:
{assistant_intent}

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

❌ **UNREALISTIC USER MESSAGES** (avoid these):
- "I would like to request assistance with..."
- "Could you please provide me with information regarding..."
- "I am seeking guidance on the following matter:"
- "Thank you for your comprehensive response."
- Perfect grammar and punctuation throughout
- Formal, complete sentences
- Overly polite or deferential language

### CONVERSATION QUALITY STANDARDS:
✅ AUTHENTIC: Create scenarios that feel like real human interactions
✅ SPECIFIC: Include vivid details, emotions, and personal stakes
✅ ENGAGING: Make conversations interesting and memorable, not boring
✅ CREATIVE: Unique situations and responses, avoid templated exchanges
✅ NATURAL: Dialogue should flow organically and feel unscripted

### AVOID THESE SYNTHETIC PATTERNS:
❌ Generic requests: "I need help with a task", "Can you give me advice?"
❌ Robotic assistant responses: "I am here to assist you", "Please provide more details"
❌ Predictable flow: User asks → Assistant explains → User thanks → Assistant offers more help
❌ Boring scenarios: Only having basic information seeking, simple question-answer exchanges
❌ Users speaking in formal, complete sentences

### OUTLINE REQUIREMENTS:
1. The conversation should have {num_turns} turns (user message + assistant response = 1 turn)
2. Each turn should have vivid, specific content that advances the scenario
3. Track preference flow carefully:
   - Turn 1: Assistant typically knows NO preferences (unless assistant_preferences is not empty)
   - User must EXPRESS preferences before assistant can appeal to them
   - Once expressed, assistant can appeal to those preferences in future turns
4. Include emotional context, stakes, and genuine human complexity
5. Focus on authentic dialogue that reveals character and personality
6. USER MESSAGES should be SHORT (1-3 sentences typically), CASUAL, and NATURAL

### PREFERENCE EXPRESSION STYLE: {expression_type}
{expression_type_instruction}

### PREFERENCE LOGIC:
- **preference_expressed**: ONLY when user genuinely states or demonstrates preference values
- **preference_appealed**: ONLY preferences previously expressed OR in assistant_preferences

### WHAT COUNTS AS "GENUINE PREFERENCE EXPRESSION":
✅ **DOES COUNT** (explicit, but still casual):
- "btw i really prefer when things are explained step by step"
- "just be direct with me, i don't need the fluff"
- "i learn better with examples if that helps"

✅ **DOES COUNT** (implicit):
- User chooses detailed option over simple one
- User asks for more depth/specificity
- User shows frustration with surface-level responses

❌ **DOES NOT COUNT**:
- "How should I approach this?" (question, not preference)
- "What do you think?" (seeking opinion, not stating preference)
- "Can you help me?" (request for help, not preference)

### EXAMPLE AUTHENTIC USER MESSAGES:
- "ok so basically i have this presentation tomorrow and i'm kinda freaking out"
- "wait that's actually really helpful - but what if they push back on the budget part?"
- "hmm idk if that'll work for my situation tho"
- "oh nice, yeah that makes way more sense"
- "ugh sorry let me explain better - so the thing is..."

{recalled_preference_context}

### OUTPUT FORMAT:
Return a JSON object with this structure:

{{
  "turns": [
    {{
        "role": "user",
        "content": "Brief, casual description of user's message (short, natural, conversational)",
        "preference_expressed": {{"preference_id1": "name1"}} # ONLY if user genuinely states preference values
    }},
    {{
        "role": "assistant", 
        "content": "Specific, engaging description of how assistant provides thoughtful, personalized help",
        "preference_appealed": {{"preference_id1": "name1"}} # ONLY preferences from assistant_preferences OR previously expressed in this conversation
    }},
    ...
  ]
}}

Please respond with clean JSON only, without explanations or code blocks.
"""