"""
preference_library.py

Implements the Universal Preference Library for the Causal Preference Evolution Framework.
Stores and provides access to preference structures that model distinct preference dimensions.
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional, Callable, Tuple, Set, Union

from src.libraries.base_library import BaseLibrary
from src.llm.api_client import LLMClient
from src.causal_framework.models.event_model import EventModel
from src.libraries.prompts.generate_preference_prompt import generate_preference_prompt_template
from src.libraries.prompts.generate_preference_prompt_batch import generate_batch_preference_prompt_template

logger = logging.getLogger(__name__)

class PreferenceLibrary(BaseLibrary[Dict[str, Any]]):
    """
    Universal Preference Library for storing and accessing preference structures.
    Preference structures model distinct preference dimensions relevant to human-AI
    interactions, each with multiple attributes that characterize the preference.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the preference library.
        
        Args:
            llm_client: Optional LLM client for augmenting preference structures
        """
        super().__init__("Universal Preference Library")
        self.llm_client = llm_client
        self._preference_domains = {}
        self._preference_dependencies = {}
        logger.info("Initialized Universal Preference Library")
    
    def add_preference(
        self, 
        preference_id: str, 
        preference_structure: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a preference structure to the library.
        
        Args:
            preference_id: Unique identifier for the preference
            preference_structure: Dictionary describing the preference
            metadata: Optional metadata about the preference
        """
        # Ensure preference structure has required fields
        if "attributes" not in preference_structure:
            preference_structure["attributes"] = {}
            
        if "description" not in preference_structure:
            preference_structure["description"] = f"Preference structure: {preference_id}"
            
        # Add domain to metadata if present
        if "domain" in preference_structure and metadata is not None:
            metadata["domain"] = preference_structure["domain"]
            
            # Track domains for efficient filtering
            domain = preference_structure["domain"]
            if domain not in self._preference_domains:
                self._preference_domains[domain] = set()
            self._preference_domains[domain].add(preference_id)
            
        # Add to library
        self.add_structure(preference_id, preference_structure, metadata)
        
    def get_preference_by_id(self, preference_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a preference structure by ID.
        
        Args:
            preference_id: ID of the preference to retrieve
            
        Returns:
            Preference structure if found, None otherwise
        """
        return self.get_structure(preference_id)

    # def get_preference_structures(self, preference_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    #     """
    #     Get multiple preference structures by their IDs.
        
    #     Args:
    #         preference_ids: List of preference IDs to retrieve
            
    #     Returns:
    #         Dictionary mapping preference IDs to preference structures
    #     """
    #     if not preference_ids:
    #         return self._structures
    #     return {pref_id: self.get_structure(pref_id) for pref_id in preference_ids}
    
    def sample_preference(
        self, 
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sample a random preference from the library.
        
        Args:
            filter_func: Optional function to filter preferences
            domain: Optional domain to filter by
            
        Returns:
            Sampled preference structure, or empty dict if no preferences match
        """
        # If domain is specified, restrict to preferences in that domain
        if domain is not None:
            domain_prefs = self._preference_domains.get(domain, set())
            if not domain_prefs:
                logger.warning(f"No preferences found in domain: {domain}")
                return {}
                
            # Create a filter function that checks both domain and user-provided filter
            def combined_filter(pref: Dict[str, Any]) -> bool:
                domain_match = pref.get("domain") == domain
                filter_match = filter_func(pref) if filter_func else True
                return domain_match and filter_match
                
            filter_to_use = combined_filter
        else:
            filter_to_use = filter_func
        
        # Sample a preference
        result = self.sample_structure(filter_to_use)
        
        if result is None:
            logger.warning("No matching preferences found for sampling")
            return {}
            
        pref_id, pref_structure = result
        logger.debug(f"Sampled preference {pref_id}")
        return pref_structure
    
    def get_preference_domains(self) -> List[str]:
        """
        Get all preference domains in the library.
        
        Returns:
            List of domain names
        """
        return list(self._preference_domains.keys())
    
    def get_preferences_by_domain(self, domain: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all preferences in a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary mapping preference IDs to preference structures
        """
        if domain not in self._preference_domains:
            return {}
            
        return {
            pref_id: self.get_structure(pref_id)
            for pref_id in self._preference_domains[domain]
            if self.get_structure(pref_id) is not None
        }
    
    def get_preference_attribute_metadata(
        self,
        preference_id: str,
        attribute_name: str
    ) -> Dict[str, Any]:
        """
        Get metadata for a specific preference attribute.
        
        Args:
            preference_id: ID of the preference
            attribute_name: Name of the attribute
            
        Returns:
            Attribute metadata, empty dict if not found
        """
        preference = self.get_structure(preference_id)
        if not preference or "attribute_metadata" not in preference:
            return {}
            
        attr_metadata = preference.get("attribute_metadata", {})
        return attr_metadata.get(attribute_name, {})
    
    def get_related_preferences(
        self,
        preference_id: str,
        min_strength: float = 0.5
    ) -> Dict[str, float]:
        """
        Get preferences related to the given preference.
        
        Args:
            preference_id: ID of the source preference
            min_strength: Minimum relationship strength (0-1)
            
        Returns:
            Dictionary mapping related preference IDs to relationship strengths
        """
        # Check the dependency graph
        if preference_id not in self._preference_dependencies:
            if self.llm_client:
                # Generate dependencies if not cached
                self._generate_preference_dependencies(preference_id)
            else:
                return {}
                
        # Filter by minimum strength
        return {
            related_id: strength 
            for related_id, strength in self._preference_dependencies.get(preference_id, {}).items()
            if strength >= min_strength
        }
    
    def generate_synthetic_preference(
        self, 
        domain: Optional[str] = None,
        attribute_count: int = 5,
        event: Optional[EventModel] = None
    ) -> Dict[str, Any]:
        """
        Generate a synthetic preference structure using LLM.
        
        Args:
            domain: Optional preference domain
            attribute_count: Number of attributes to generate
            
        Returns:
            Generated preference structure
            
        Raises:
            ValueError: If LLM client is not available
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for synthetic preference generation")
            
        # Base preference structure
        preference_structure = {
            "attributes": {},
            "attribute_metadata": {},
            "description": ""
        }
            
        if domain:
            preference_structure["domain"] = domain
            
        # Create a prompt for the LLM
        
        # Add domain instruction if specified
        if domain:
            domain_instruction = f"\nThe preference should be in the domain: {domain}.\n"
        else:
            domain_instruction = "\nChoose an appropriate domain for the preference (e.g., entertainment, food, communication, productivity, lifestyle, etc.).\n"
        
        if event:
            event_instruction = f"\nThe preference should be relevant to the following event: {json.dumps(event.to_dict(), indent=2)}.\n"
        else:
            event_instruction = ""
        
        prompt = generate_preference_prompt_template.format(
            domain_instruction=domain_instruction,
            event_instruction=event_instruction,
            attribute_count=attribute_count
        )
        
        # Generate preference structure using LLM
        response = self.llm_client.generate(prompt, response_format="json")


        # # Generate preference structure using LLM with randomness
        # # Add timestamp and random element to ensure diverse responses
        # import time
        # random_element = random.randint(1000, 9999)
        # time_element = int(time.time() * 1000) % 10000
        
        # prompt_with_diversity = prompt + f"\n\nGeneration seed: {random_element}-{time_element} (use this for randomness in your response)"
        
        # response = self.llm_client.generate(prompt_with_diversity, response_format="json")
        
        try:
            generated_preference = json.loads(response["response_text"])
            
            # Ensure required fields exist
            if "attributes" not in generated_preference:
                generated_preference["attributes"] = {}
                
            if "attribute_metadata" not in generated_preference:
                generated_preference["attribute_metadata"] = {}
                
            # Ensure domain is set
            if "domain" not in generated_preference and domain:
                generated_preference["domain"] = domain
                
            # Ensure description is set
            if "description" not in generated_preference:
                generated_preference["description"] = f"Preference structure for {generated_preference.get('name', 'unknown')}"
            
            logger.debug(f"Generated synthetic preference structure using LLM: {generated_preference.get('name', 'unknown')}")
            return generated_preference
            
        except json.JSONDecodeError:
            logger.error(f"LLM response could not be parsed as JSON: {response}")
            # Return basic structure if parsing fails
            # return preference_structure

            generated_preference = json.loads(response)
            
            # Ensure required fields exist
            if "attributes" not in generated_preference:
                generated_preference["attributes"] = {}
                
            if "attribute_metadata" not in generated_preference:
                generated_preference["attribute_metadata"] = {}
                
            # Ensure domain is set
            if "domain" not in generated_preference and domain:
                generated_preference["domain"] = domain
                
            # Ensure description is set
            if "description" not in generated_preference:
                generated_preference["description"] = f"Preference structure for {generated_preference.get('name', 'unknown')}"
                
            logger.debug("Generated synthetic preference structure using LLM")
            return generated_preference


    def initialize_new_domain(self, new_domain: Optional[str] = None, count_per_domain: int = 5) -> None:
        """
        Initialize a new preference domain.
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for domain initialization")
        
        if new_domain is None:
            # Generate a new domain using LLM
            domain_prompt = f"""
            Generate a new preference domain that is NOT already in the following list:
            {json.dumps(self.get_preference_domains(), indent=2)}

            Return the new domain name as a string and nothing else.
            """
            response = self.llm_client.generate(domain_prompt, response_format="text")["response_text"]
            if response.lower() == "none":
                return None
            else:
                new_domain = response.strip()
        # Generate a batch of preferences for the new domain
        batch_results = self.batch_generate_preferences(
            domains=[new_domain],
            count_per_domain=count_per_domain
        )
        pref_count = 0
        for domain, preferences in batch_results.items():
            for pref in preferences:
                pref_id = f"{domain}_{self.get_structure_count() + 1}"
                self.add_preference(pref_id, pref)
                pref_count += 1
        logger.info(f"Added {pref_count} synthetic preferences to library")
    
    def batch_generate_preferences(self, domains: List[str], count_per_domain: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a batch of synthetic preferences across multiple domains.
        
        Args:
            domains: List of preference domains to generate
            count_per_domain: Number of preferences to generate per domain
            
        Returns:
            Dictionary mapping domains to lists of generated preference structures
            
        Raises:
            ValueError: If LLM client is not available
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for batch preference generation")
            
        generated_preferences = {}
        
        for domain in domains:
            domain_preferences = []
            
            # Generate a batch prompt for this domain
            batch_prompt = generate_batch_preference_prompt_template.format(
                count_per_domain=count_per_domain,
                domain=domain
            )
            
            response = self.llm_client.generate(batch_prompt, response_format="json")

            batch_preferences = json.loads(response["response_text"])["preferences"]

            for pref in batch_preferences:
                if "attributes" not in pref:
                    pref["attributes"] = {}

                if "attribute_metadata" not in pref:
                    pref["attribute_metadata"] = {}

                if "domain" not in pref:
                    pref["domain"] = domain

                if "description" not in pref:
                    pref["description"] = f"Preference structure for {pref.get('name', 'unknown')}"

                domain_preferences.append(pref)

            generated_preferences[domain] = domain_preferences
            logger.info(f"Generated {len(domain_preferences)} preferences for domain {domain}")
            
        return generated_preferences
    
    def update_dependency_graph(self, dependencies: Dict[str, Dict[str, float]]) -> None:
        """
        Update the preference dependency graph.
        
        Args:
            dependencies: Nested dictionary mapping source preference IDs to target
                        preference IDs with dependency strengths
        """
        for source_id, targets in dependencies.items():
            if source_id not in self._preference_dependencies:
                self._preference_dependencies[source_id] = {}
                
            self._preference_dependencies[source_id].update(targets)
            
        logger.debug(f"Updated preference dependency graph with {len(dependencies)} source preferences")
    
    def get_dependency_graph(self) -> Dict[str, Dict[str, float]]:
        """
        Get the current preference dependency graph.
        
        Returns:
            Nested dictionary of preference dependencies
        """
        return self._preference_dependencies.copy()
    
    def _generate_preference_dependencies(self, preference_id: str) -> None:
        """
        Use LLM to generate dependencies for a preference.
        
        Args:
            preference_id: ID of the preference to generate dependencies for
        """
        if not self.llm_client:
            return
            
        preference = self.get_structure(preference_id)
        if not preference:
            return
            
        # Get all other preferences
        all_preferences = self.get_all_structures()
        other_prefs = {
            pid: pref for pid, pref in all_preferences.items()
            if pid != preference_id
        }
        
        if not other_prefs:
            return
            
        # Create a summary of other preferences
        other_prefs_summary = "\n".join([
            f"- {pid}: {pref.get('name', pid)} ({pref.get('domain', 'unknown domain')})"
            for pid, pref in other_prefs.items()
        ])
        
        # Create a prompt for the LLM
        prompt = (
        "Analyze the potential dependencies between the following preference and other preferences.\n"
        "Assign a dependency strength between 0.0 (no effect) and 1.0 (strong effect) for each relationship.\n"
        "Only include relationships with strength > 0.2.\n"
        
        "Source preference:\n"
        f"ID: {preference_id}\n"
        f"Name: {preference.get('name', preference_id)}\n"
        f"Domain: {preference.get('domain', 'unknown')}\n"
        f"Description: {preference.get('description', '')}\n"
        f"Attributes: {json.dumps(preference.get('attributes', {}), indent=2)}\n"
        
        "Other preferences:\n"
        f"{other_prefs_summary}\n"
        
        "Return a JSON object mapping target preference IDs to dependency strengths:\n"
        "{{\n"
          '  "target_preference_id1": 0.7,\n'
          '  "target_preference_id2": 0.4,\n'
          '  ...\n'
        '}}\n'
        )
        
        response = self.llm_client.generate(prompt, response_format="json")

        dependencies = json.loads(response["response_text"])

        validated_deps = {}
        for target_id, strength in dependencies.items():
            if target_id in other_prefs and isinstance(strength, (int, float)) and 0 <= strength <= 1:
                validated_deps[target_id] = strength

        self._preference_dependencies[preference_id] = validated_deps
        logger.debug(f"Generated {len(validated_deps)} dependencies for preference {preference_id}")
    
    def _parse_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a preference structure from its serialized form.
        
        Args:
            data: Serialized preference data
            
        Returns:
            Parsed preference structure
        """
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data
    
    def _serialize_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize a preference structure to a dictionary.
        
        Args:
            structure: Preference structure to serialize
            
        Returns:
            Serialized preference data
        """
        return structure

    def add_structure(self, structure_id: str, structure: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a structure to the library.
        
        Args:
            structure_id: Unique identifier for the structure
            structure: The structure to add
            metadata: Optional metadata about the structure
        """
        self._structures[structure_id] = structure
        self._metadata[structure_id] = metadata or {}
        if "domain" in structure:
            domain = structure["domain"]
            if domain not in self._preference_domains:
                self._preference_domains[domain] = set()
            self._preference_domains[domain].add(structure_id)
        logger.debug(f"Added structure {structure_id} to {self.library_name}")

    def generate_and_augment_domains(self, existing_domains: Optional[List[str]] = None) -> List[str]:
        """
        Generate new preference domains or augment existing ones using LLM.
        Ensures semantic uniqueness from existing domains.
        
        Args:
            existing_domains: Optional list of existing domains to augment
            
        Returns:
            List of all domains (existing + new)
            
        Raises:
            ValueError: If LLM client is not available
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for domain generation")

        # Base prompt for domain generation
        prompt = """
        You are a domain expert designing a comprehensive taxonomy of human preference domains.
        
        Output a valid parsable JSON object containing only a list of major preference domains.
        Each domain should:
        - Have a distinguishing and easily understandable name
        - Be broad but distinct from others
        - Focus on a major area of human interest, lifestyle, or behavior
        - Be semantically unique (not just different wording for the same concept)
        
        Constraints:
        1. Generate at least 30 distinct domain names
        2. The domains should be on the same abstract level, cannot be a subdomain of another
        3. Make sure the domains are semantically different, not just different phrasings
        4. Try NOT to use the format "A and B" unless absolutely necessary
        
        {existing_domains_instruction}
        
        Return ONLY a valid JSON object with the following structure:
        {{
          "domains": ["Domain1", "Domain2", "Domain3", ...]
        }}
        """
        
        # Add existing domains instruction if provided
        if existing_domains:
            domains_str = json.dumps(existing_domains)
            existing_domains_instruction = f"""
            Here are the existing domains that need to be augmented:
            {domains_str}
            
            Please:
            1. Review these domains for quality and consistency
            2. Keep any good domains
            3. Remove or revise problematic domains
            4. Add new domains that are SEMANTICALLY DIFFERENT from existing ones
               (not just different wording for similar concepts)
            5. Ensure no conceptual overlap with existing domains
            """
        else:
            existing_domains_instruction = "Generate a comprehensive list of preference domains from scratch."
            
        # Format the prompt
        formatted_prompt = prompt.format(existing_domains_instruction=existing_domains_instruction)
        
        response = self.llm_client.generate(formatted_prompt, response_format="json")
        generated = json.loads(response["response_text"])

        if not isinstance(generated, dict) or "domains" not in generated:
            logger.error("Invalid response format from LLM")
            return existing_domains or []

        new_domains = generated["domains"]

        if existing_domains:
            existing_set = set(existing_domains)
            new_set = set(new_domains)
            all_domains = list(existing_set.union(new_set))

            added = new_set - existing_set
            removed = existing_set - new_set
            if added:
                logger.info(f"Added new domains: {added}")
            if removed:
                logger.info(f"Removed domains: {removed}")

            return all_domains
        else:
            logger.info(f"Generated {len(new_domains)} preference domains")
            return new_domains