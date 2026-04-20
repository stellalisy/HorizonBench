"""
user_library.py

Implements the Universal User Library for the Causal Preference Evolution Framework.
Stores and provides access to user persona structures.
"""

import json
import logging
import random
import pandas as pd
import os
import datasets
from typing import Dict, Any, List, Optional, Callable, Tuple, Set, Union

from src.libraries.base_library import BaseLibrary
from src.causal_framework.models.user_model import UserModel
from src.llm.api_client import LLMClient

logger = logging.getLogger(__name__)

class UserLibrary(BaseLibrary[Dict[str, Any]]):
    """
    Universal User Library for storing and accessing user persona structures.
    User structures contain attributes like demographics, personal identifiers,
    and other traits that define a user's core identity.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the user library.
        
        Args:
            llm_client: Optional LLM client for augmenting user structures
        """
        super().__init__("Universal User Library")
        self.llm_client = llm_client
        self.config = config
        self.personality_data, self.personality_legend, self.personas = self.load_personas_data()        
        self._demographic_distributions = {}
        logger.info("Initialized Universal User Library")


    def load_personas_data(self):
        data_loc = self.config.get("libraries", {}).get("enriched_data_dir", "data/PAPI")
        file_name = "IPIP_NEO_300.csv"
        key_file_name = "IPIP-NEO-ItemKey.csv"
        data_path = os.path.join(data_loc, file_name)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Personality data not found at {data_path}."
            )
        personality_data = pd.read_csv(data_path)
        personality_data = personality_data.iloc[:, 10:]
        legend = pd.read_csv(os.path.join(data_loc, key_file_name))
        # cat facet and item, remove period in item
        legend['trait'] = legend['Facet'] + ' - ' + legend['Item'].astype(str).str.replace('.', '', regex=False)
        legend['Full_key'] = "i" + legend['Full#'].astype(str)
        legend_dict = legend.set_index('Full_key')['trait'].to_dict()
        personas = [p for p in datasets.load_dataset("proj-persona/PersonaHub", "persona")['train']]
        return personality_data, legend_dict, personas
    

    def add_user(self, user_id: str, user_structure: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a user structure to the library.
        
        Args:
            user_id: Unique identifier for the user
            user_structure: Dictionary of user attributes
            metadata: Optional metadata about the user
        """
        # Validate required attributes
        required_attrs = {"first_name", "last_name", "age", "gender"}
        missing_attrs = required_attrs - set(user_structure.keys())
        
        if missing_attrs:
            if self.llm_client:
                # Try to fill in missing attributes using LLM
                augmented_structure = self._augment_user_structure(user_structure, list(missing_attrs))
                user_structure.update(augmented_structure)
                logger.debug(f"Augmented user {user_id} with missing attributes: {missing_attrs}")
            else:
                logger.warning(f"User {user_id} is missing required attributes: {missing_attrs}")
        
        # Add to library
        self.add_structure(user_id, user_structure, metadata)
        
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a user structure by ID.
        
        Args:
            user_id: ID of the user to retrieve
            
        Returns:
            User structure if found, None otherwise
        """
        return self.get_structure(user_id)
    
    def sample_user(self, filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Dict[str, Any]:
        """
        Sample a random user from the library.
        
        Args:
            filter_func: Optional function to filter users
            
        Returns:
            Sampled user structure, or empty dict if no users match
        """
        result = self.sample_structure(filter_func)
        
        if result is None:
            logger.warning("No matching users found for sampling")
            return {}
            
        user_id, user_structure = result
        logger.debug(f"Sampled user {user_id}")
        return user_structure
    
    def sample_user_by_demographics(self, demographics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample a user matching specific demographic criteria.
        
        Args:
            demographics: Dictionary of demographic attributes to match
            
        Returns:
            Matching user structure, or empty dict if no users match
        """
        def demographic_filter(user: Dict[str, Any]) -> bool:
            return all(user.get(attr) == value for attr, value in demographics.items())
            
        return self.sample_user(demographic_filter)
    
    def sample_diverse_users(self, count: int, diversity_attributes: List[str] = None) -> List[Dict[str, Any]]:
        """
        Sample a diverse set of users by ensuring variety in key attributes.
        
        Args:
            count: Number of users to sample
            diversity_attributes: Attributes to maximize diversity for
            
        Returns:
            List of sampled user structures
        """
        if diversity_attributes is None:
            diversity_attributes = ["age_group", "gender", "ethnicity"]
            
        # Get all users
        all_users = list(self.get_all_structures().items())
        
        if len(all_users) <= count:
            return [user for _, user in all_users]
            
        # Initialize with a random user
        selected_user_ids = set()
        selected_users = []
        
        # If we have users, start with a random one
        if all_users:
            first_id, first_user = random.choice(all_users)
            selected_user_ids.add(first_id)
            selected_users.append(first_user)
        
        # Track value frequencies for diversity attributes
        attribute_values = {attr: {} for attr in diversity_attributes}
        
        # Add the first user's values
        if selected_users:
            for attr in diversity_attributes:
                value = selected_users[0].get(attr)
                if value is not None:
                    attribute_values[attr][value] = 1
        
        # Select remaining users to maximize diversity
        while len(selected_users) < count and len(selected_users) < len(all_users):
            best_user_id = None
            best_user = None
            best_diversity_score = -1
            
            # Shuffle to avoid deterministic selection
            random.shuffle(all_users)
            
            for user_id, user in all_users:
                if user_id in selected_user_ids:
                    continue
                    
                # Calculate diversity score
                diversity_score = 0
                for attr in diversity_attributes:
                    value = user.get(attr)
                    if value is None:
                        continue
                        
                    # Lower score for values we've seen more frequently
                    current_frequency = attribute_values[attr].get(value, 0)
                    diversity_score -= current_frequency
                
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_user_id = user_id
                    best_user = user
            
            if best_user_id is None:
                break
                
            # Add the best user
            selected_user_ids.add(best_user_id)
            selected_users.append(best_user)
            
            # Update attribute value frequencies
            for attr in diversity_attributes:
                value = best_user.get(attr)
                if value is not None:
                    attribute_values[attr][value] = attribute_values[attr].get(value, 0) + 1
        
        logger.debug(f"Sampled {len(selected_users)} diverse users")
        return selected_users
    
    def sample_enriched_persona(self):
        persona = random.choice(self.personas)
        # sample 20 random trait indices from 0 to 299
        sampled_traits = random.sample(range(300), 20)
        # sample random row from personality_data
        sampled_row = self.personality_data.iloc[random.randint(0, len(self.personality_data) - 1)]
        sampled_trait_values = sampled_row[sampled_traits].to_dict()
        sampled_trait_pretty = {self.personality_legend[k]: v for k, v in sampled_trait_values.items()}
        persona_string = persona['persona'] + "\n"
        persona_string += "Personality Traits (On a scale of 1-5):\n"
        for trait, value in sampled_trait_pretty.items():
            persona_string += f"{trait}: {value}\n"
        return persona_string.strip()


    def generate_enriched_synthetic_user(self, base_attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a synthetic user structure using LLM, enriched with realistic personality traits, persona and related social graph.
        
        Args:
            base_attributes: Optional base attributes to include
            
        Returns:
            Generated user structure
            
        Raises:
            ValueError: If LLM client is not available
        """
        # Start with base attributes or empty dict
        user_structure = base_attributes.copy() if base_attributes else {}

        # Sample a persona
        enriched_persona = self.sample_enriched_persona()
        persona_desc, traits = enriched_persona.split("\nPersonality Traits (On a scale of 1-5):")

        # Fill out basic persona card (with required base attributes)
        fill_card_prompt = """Fill out this information card in a manner that aligns with the persona mentioned below. The persona also mentioned personality traits. The profile should be realistic and plausible given the persona and their personality traits. 
Return the user persona as a JSON object with the following recommended format.
{{
  "name": <insert text>,
  "first name" : <insert text>,
  "last name" : <last name>,
  "age": <insert text>,
  "gender": <insert text>,
  "ethnicity": <insert text>,
  "location": <insert text>,

  "occupation": {{
    "organization": <insert text>,
    "title": <insert text>
  }},

  "education": {{
    "institution": <insert text>,
    "degree": <insert text>,
    "major": <insert text>
  }}

  "interests": [<insert text>]
}}

Persona and their personality traits: {persona}
"""
        prompt = fill_card_prompt.format(persona=enriched_persona)

        # Generate user structure using LLM
        completed_persona = self.llm_client.generate(prompt, response_format="json")

        try:
            completed_persona = json.loads(completed_persona["response_text"])
            # Merge with base attributes
            completed_persona["persona"] = persona_desc
            completed_persona["traits"] = [item.strip() for item in traits.split("\n") if item.strip() != ""]
            for key, value in user_structure.items():
                completed_persona[key] = value
            logger.debug("Generated enriched persona traits")
        #     return generated_user    
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {completed_persona}")
            # Return base attributes if parsing fails
            return user_structure
        

        sg_prompt = """You are an expert data generator tasked with creating realistic, synthetic social graphs with plausible social structures and connections. You are provided detailed information about a person, including their personality traits, background information, etc.
Your task is to generate the names and personas of two other people that form a social or professional connection with this person. After this, you will also generate the edges that connect these people in a social graph. The output should be formatted as a JSON object with nodes and edges.

Persona:
{completed_persona}

Recommended output for social graph:
```
{{
"nodes": {{
  "Name of given person": "Persona of the given person",
  "Name 1": "Persona 1",
  "Name 2": "Persona 2"
  }},
"edges": {{
  "Edge 1": {{
    "Nodes": ["Given Persona", "Name 1"],
    "Relation": "Relation between Given Persona and Name 1"
    }},
  "Edge 2": {{
    "Nodes": ["Given Persona", "Name 2"],
    "Relation": "Relation between Given Persona and Name 2"
    }},
  }}
}}
```
"""
        prompt = sg_prompt.format(completed_persona=json.dumps(completed_persona, indent=2))

        # Generate user structure using LLM
        small_sg = self.llm_client.generate(prompt, response_format="json")["response_text"]

        prompt_expand_sg = """You are an expert data generator tasked with creating realistic, synthetic social graphs with plausible social structures and connections. Each graph consists of nodes representing individuals and edges representing relationships. Follow these guidelines to ensure a coherent and realistic output:
The expanded social graph should be formatted as:
```
{{
"nodes": {{
    "Name 1": "Persona 1"
    "Name 2": "Persona 2"
    ...
  }},
"edges": {{
    "Edge 1" : {{
      "Nodes":  ["Name 1", "Name 2"],
      "Relation": "Relation between Name 1 and Name 2"
    }},
    "Edge 2" : {{
      "Nodes":  ["Name 1", "Name 3"],
      "Relation": "Relation between Name 1 and Name 3"
    }},
    ...
  }}
}}

Instruction: Given the following existing nodes: {{"Name 1": "Persona 1", "Name 2": "Persona 2", ...}}, please expand the social graph that connects these people, to include more diverse people with distinct personas.
1. Add atleast 5 more nodes to the existing graph and atleast 4 edges.
2. Ensure that the graph is fully connected.

Current graph:
```
{small_sg}
```

Expanded graph:
"""
        prompt = prompt_expand_sg.format(small_sg=small_sg)

        # Generate user structure using LLM
        expanded_social_graph = self.llm_client.generate(prompt, response_format="json")

        try:
            completed_persona["social_graph"] = json.loads(expanded_social_graph["response_text"])
        except Exception as e:
            logger.debug(f"Generated expanded social graph is not well formatted: {e}")
        
        return completed_persona

    def generate_synthetic_user(self, base_attributes: Optional[Dict[str, Any]] = None, enrich_social_graph: Optional[bool] = False) -> Dict[str, Any]:
        """
        Generate a synthetic user structure using LLM.
        
        Args:
            base_attributes: Optional base attributes to include
            
        Returns:
            Generated user structure
            
        Raises:
            ValueError: If LLM client is not available
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for synthetic user generation")
        
        if enrich_social_graph:
            return self.generate_enriched_synthetic_user(base_attributes)
            
        # Start with base attributes or empty dict
        user_structure = base_attributes.copy() if base_attributes else {}
        
        # Create a prompt for the LLM
        prompt = """
        Generate a realistic and diverse user persona with the following attributes.
        Create someone interesting and unique - avoid generic names like "John Doe" or boring occupations.
        
        Required attributes:
        - first_name: Diverse first name (not generic like John, Jane, etc.)
        - last_name: Diverse last name (not generic like Doe, Smith, etc.)
        - age: Age (integer)
        - gender: Gender (male, female, non-binary, etc.)
        - ethnicity: Ethnicity (ensure cultural diversity)
        - occupation: Interesting and diverse occupation (artist, marine biologist, chef, software engineer, teacher, farmer, etc.)
        - education_level: Education level
        - location: Specific location (city, state/province, country)
        - relationship_status: Relationship status
        - interests: List of 3-5 specific and diverse interests/hobbies
        - personality_traits: List of 3-5 distinctive personality traits
        
        IMPORTANT GUIDELINES:
        1. Create diverse, realistic personas from different backgrounds
        2. Avoid stereotypical or generic combinations
        3. Make occupations interesting and varied (not just "office worker" or "manager")
        4. Include people from different cultures, ethnicities, and regions
        5. Make interests specific and engaging
        6. DO NOT include an "id" field - this will be handled separately
        
        If any attributes are provided below, maintain those values in your response:
        {base_attributes}
        
        Ensure the attributes are consistent with each other (e.g., age-appropriate occupation and education).
        Return the user persona as a JSON object:
        """
        
        # Add base attributes to the prompt if provided
        if base_attributes:
            base_attr_str = json.dumps(base_attributes, indent=2)
        else:
            base_attr_str = "{}"
            
        prompt = prompt.format(base_attributes=base_attr_str)
        
        # Generate user structure using LLM
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            generated_user = json.loads(response["response_text"])
            
            # Remove any "id" field that might have been generated - we handle IDs separately
            if "id" in generated_user:
                del generated_user["id"]
            
            # Merge with base attributes
            for key, value in user_structure.items():
                generated_user[key] = value

            logger.debug("Generated synthetic user structure using LLM")
            return generated_user
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            # Return base attributes if parsing fails
            return user_structure
    
    def batch_generate_users(self, count: int, diversity_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate a batch of synthetic users with diversity constraints.
        
        Args:
            count: Number of users to generate
            diversity_params: Optional parameters for ensuring diversity
            
        Returns:
            List of generated user structures
            
        Raises:
            ValueError: If LLM client is not available
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for batch user generation")
            
        # Default diversity parameters
        params = {
            "gender_distribution": {"male": 0.49, "female": 0.49, "non-binary": 0.02},
            "age_ranges": [
                {"min": 18, "max": 30, "weight": 0.3},
                {"min": 31, "max": 50, "weight": 0.4},
                {"min": 51, "max": 75, "weight": 0.3}
            ],
            "ensure_ethnic_diversity": True
        }
        
        # Update with provided parameters
        if diversity_params:
            params.update(diversity_params)
            
        generated_users = []
        
        # Generate diverse base attributes
        base_attribute_sets = []
        for _ in range(count):
            # Sample gender based on distribution
            gender = random.choices(
                list(params["gender_distribution"].keys()),
                weights=list(params["gender_distribution"].values())
            )[0]
            
            # Sample age range
            age_range = random.choices(
                params["age_ranges"],
                weights=[r["weight"] for r in params["age_ranges"]]
            )[0]
            
            # Sample specific age within range
            age = random.randint(age_range["min"], age_range["max"])
            
            base_attrs = {
                "gender": gender,
                "age": age
            }
            
            base_attribute_sets.append(base_attrs)
        
        # Create a batch generation prompt for the LLM
        batch_prompt = f"""
        Generate {count} diverse and interesting user personas based on the provided base attributes.
        Create unique, realistic people from different backgrounds - avoid generic names and boring occupations.
        
        Each user should have the following attributes:
        - first_name: Diverse first name appropriate for the gender and ethnicity (avoid generic names)
        - last_name: Diverse last name appropriate for the ethnicity (avoid generic surnames)
        - ethnicity: Ethnicity (ensure diversity across users - include various backgrounds)
        - occupation: Interesting and diverse occupation appropriate for the age (artist, marine biologist, chef, software engineer, teacher, farmer, entrepreneur, therapist, etc.)
        - education_level: Education level appropriate for the age and occupation
        - location: Specific location (city, state/province, country) - include various regions and countries
        - relationship_status: Relationship status
        - interests: List of 3-5 specific and engaging interests/hobbies per person
        - personality_traits: List of 3-5 distinctive personality traits per person
        
        IMPORTANT GUIDELINES:
        1. Make each person unique and interesting
        2. Ensure cultural and ethnic diversity across the batch
        3. Vary occupations significantly (include creative, technical, service, academic, outdoor, etc.)
        4. Include people from different countries and regions
        5. Make interests specific and varied
        6. DO NOT include "id" fields - these are handled separately
        7. Avoid stereotypical combinations of ethnicity, name, and occupation
        
        Base attributes for each user:
        {json.dumps(base_attribute_sets, indent=2)}
        
        Return a JSON array with {count} user objects:
        """
        
        response = self.llm_client.generate(batch_prompt, response_format="json")
        
        try:
            batch_response = json.loads(response["response_text"])
            
            # Handle different response formats
            if isinstance(batch_response, list):
                batch_users = batch_response
            elif isinstance(batch_response, dict) and "users" in batch_response:
                batch_users = batch_response["users"]
            elif isinstance(batch_response, dict) and len(batch_response) == 1:
                # Try to extract the list from the single key
                batch_users = list(batch_response.values())[0]
            else:
                logger.error(f"Unexpected response format: {type(batch_response)} - {batch_response}")
                raise ValueError("LLM response format not recognized")
            
            # Validate that we got a list
            if not isinstance(batch_users, list):
                logger.error(f"Expected list but got {type(batch_users)}: {batch_users}")
                raise ValueError("LLM did not return a list of users")
            
            # Merge with base attributes and validate
            for i, user in enumerate(batch_users):
                # Validate that each user is a dictionary
                if not isinstance(user, dict):
                    logger.error(f"User {i+1} is not a dictionary: {type(user)} - {user}")
                    continue
                
                if i < len(base_attribute_sets):
                    # Remove any "id" field that might have been generated
                    if "id" in user:
                        del user["id"]
                    
                    # Ensure base attributes are preserved
                    for key, value in base_attribute_sets[i].items():
                        user[key] = value
                        
                generated_users.append(user)
                
            logger.info(f"Generated batch of {len(generated_users)} synthetic users")
            return generated_users
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            
            # Fallback: generate users one by one
            for base_attrs in base_attribute_sets:
                user = self.generate_synthetic_user(base_attrs)
                generated_users.append(user)
                
            return generated_users
    
    def update_demographic_distributions(self, distributions: Dict[str, Dict[str, float]]) -> None:
        """
        Update the demographic distributions used for sampling.
        
        Args:
            distributions: Dictionary mapping demographic attributes to value distribution dicts
        """
        self._demographic_distributions.update(distributions)
        logger.debug(f"Updated demographic distributions: {list(distributions.keys())}")
    
    def get_demographic_distributions(self) -> Dict[str, Dict[str, float]]:
        """
        Get the current demographic distributions.
        
        Returns:
            Dictionary of demographic distributions
        """
        return self._demographic_distributions.copy()
    
    def _augment_user_structure(self, user_structure: Dict[str, Any], missing_attrs: List[str]) -> Dict[str, Any]:
        """
        Use LLM to augment a user structure with missing attributes.
        
        Args:
            user_structure: Existing user structure
            missing_attrs: List of missing attribute names
            
        Returns:
            Dictionary of generated attributes
        """
        if not self.llm_client:
            return {}
            
        prompt = f"""
        Based on the following partial user information, generate plausible values for the missing attributes.
        
        Existing user attributes:
        {json.dumps(user_structure, indent=2)}
        
        Missing attributes to generate: {', '.join(missing_attrs)}
        
        Return a JSON object with only the missing attributes and their generated values:
        """
        
        response = self.llm_client.generate(prompt, response_format="json")
        
        try:
            augmented_attributes = json.loads(response["response_text"])
            return {k: v for k, v in augmented_attributes.items() if k in missing_attrs}
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            return {}
    
    def _parse_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a user structure from its serialized form.
        
        Args:
            data: Serialized user data
            
        Returns:
            Parsed user structure
        """
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data
    
    def _serialize_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize a user structure to a dictionary.
        
        Args:
            structure: User structure to serialize
            
        Returns:
            Serialized user data
        """
        return structure
