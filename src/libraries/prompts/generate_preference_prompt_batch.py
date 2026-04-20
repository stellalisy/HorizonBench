generate_batch_preference_prompt_template = """Generate {count_per_domain} diverse preference structures for the domain: {domain}

Each preference structure should represent a different dimension of user preference within this domain.
Make the preference dimensions diverse, covering different aspects of user preference in the {domain} domain.

Each preference structure should include:
1. A "name" for the preference dimension
2. A "description" explaining what this preference dimension represents
3. A set of 3-7 "attributes" representing aspects of this preference
4. "attribute_metadata" for each attribute, including:
   - "type": The data type (string, integer, float, boolean, list)
   - "description": What this attribute represents
   - "stability": How resistant to change this attribute is (0-1 scale)
   - "possible_values": For categorical attributes, possible values

IMPORTANT: For each attribute, set its value to the literal string "default". Do NOT provide concrete values.

Return a JSON object with the following format:
{{
  "preferences": [
    {{
      "name": "Proper Human Readable Name",
      "domain": "{domain}",
      "description": "Detailed description", 
      "attributes": {{
        "attribute1": "default",
        "attribute2": "default",
        ...
      }},
      "attribute_metadata": {{
        "attribute1": {{
          "type": "string",
          "description": "Description of attribute1",
          "stability": 0.7,
          "possible_values": ["value1", "value2", ...]
        }},
        ...
      }}
    }},
    ... ({count_per_domain} preference structures)
  ]
}}
"""