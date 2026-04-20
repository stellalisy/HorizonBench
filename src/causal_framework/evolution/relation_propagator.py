"""
relation_propagator.py

Implements relational interdependence for the Causal Preference Evolution Framework.
Handles how changes in one preference can influence related preferences.
"""

import logging
import random
from typing import Dict, Any, List, Optional, Union, Tuple, Set

logger = logging.getLogger(__name__)

class RelationPropagator:
    """
    Handles relational interdependence between preferences, ensuring changes in one
    preference appropriately influence related preferences.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the relation propagator.
        
        Args:
            config: Optional configuration parameters
        """
        # Set default configuration or update with provided config
        self.config = {
            "propagation_depth": 2,       # Maximum depth of propagation (direct + indirect)
            "min_strength_threshold": 0.2, # Minimum edge strength to consider for propagation
            "propagation_decay": 0.7,     # Decay factor for multi-hop propagation
            "max_propagation_changes": 5, # Maximum number of propagated changes to apply
            "random_seed": 42             # Seed for reproducibility
        }
        
        if config:
            self.config.update(config)
            
        random.seed(self.config["random_seed"])
        
        logger.debug("RelationPropagator initialized")
    
    def propagate_changes(
        self,
        dependency_graph: Dict[str, Dict[str, float]],
        source_changes: Dict[str, Dict[str, Dict[str, Any]]],
        all_preferences: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Propagate changes from source preferences to related preferences.
        
        Args:
            dependency_graph: Graph of preference dependencies (preference_id -> {related_id: strength})
            source_changes: Dictionary of changes in source preferences (preference_id -> {attr: {from, to}})
            all_preferences: Dictionary of all current preference values
            
        Returns:
            Dictionary of propagated changes to apply (preference_id -> {attr: {from, to}})
        """
        # Skip if no changes or empty dependency graph
        if not source_changes or not dependency_graph:
            return {}
            
        # Initialize propagated changes
        propagated_changes = {}
        
        # Process each source preference that changed
        for source_pref_id, source_attrs in source_changes.items():
            # Skip if this preference has no dependencies in the graph
            if source_pref_id not in dependency_graph:
                continue
                
            # Get direct dependencies
            direct_dependencies = dependency_graph[source_pref_id]
            
            # Filter by minimum strength threshold
            valid_dependencies = {
                target_id: strength for target_id, strength in direct_dependencies.items()
                if strength >= self.config["min_strength_threshold"]
            }
            
            # Skip if no valid dependencies
            if not valid_dependencies:
                continue
                
            # Calculate influence on each target preference
            self._calculate_direct_influences(
                source_pref_id=source_pref_id,
                source_attrs=source_attrs,
                valid_dependencies=valid_dependencies,
                all_preferences=all_preferences,
                propagated_changes=propagated_changes
            )
            
            # If configured for deeper propagation, calculate indirect influences
            if self.config["propagation_depth"] > 1:
                processed_prefs = {source_pref_id}
                indirect_changes = dict(propagated_changes)
                
                for depth in range(2, self.config["propagation_depth"] + 1):
                    # Get new indirect sources from the previous level
                    indirect_sources = {}
                    
                    for pref_id in indirect_changes:
                        if pref_id not in processed_prefs and pref_id in dependency_graph:
                            indirect_sources[pref_id] = indirect_changes[pref_id]
                            processed_prefs.add(pref_id)
                            
                    # Skip if no new indirect sources
                    if not indirect_sources:
                        break
                        
                    # Calculate influences from these indirect sources
                    decay_factor = self.config["propagation_decay"] ** (depth - 1)
                    
                    for indirect_source, indirect_attrs in indirect_sources.items():
                        # Skip if this preference has no dependencies in the graph
                        if indirect_source not in dependency_graph:
                            continue
                            
                        # Get dependencies with decay applied
                        indirect_dependencies = {
                            target_id: strength * decay_factor
                            for target_id, strength in dependency_graph[indirect_source].items()
                            if strength * decay_factor >= self.config["min_strength_threshold"]
                        }
                        
                        # Calculate influence on each target preference
                        if indirect_dependencies:
                            self._calculate_indirect_influences(
                                source_pref_id=indirect_source,
                                source_attrs=indirect_attrs,
                                valid_dependencies=indirect_dependencies,
                                all_preferences=all_preferences,
                                propagated_changes=propagated_changes,
                                processed_prefs=processed_prefs
                            )
        
        # Limit the number of propagated changes
        if len(propagated_changes) > self.config["max_propagation_changes"]:
            # Keep only the strongest changes
            all_changes = []
            
            for pref_id, attrs in propagated_changes.items():
                for attr_name, change in attrs.items():
                    # Calculate change magnitude
                    magnitude = self._calculate_change_magnitude(change["from"], change["to"])
                    all_changes.append((pref_id, attr_name, change, magnitude))
                    
            # Sort by magnitude and keep only the top changes
            all_changes.sort(key=lambda x: x[3], reverse=True)
            top_changes = all_changes[:self.config["max_propagation_changes"]]
            
            # Rebuild propagated changes
            propagated_changes = {}
            for pref_id, attr_name, change, _ in top_changes:
                if pref_id not in propagated_changes:
                    propagated_changes[pref_id] = {}
                propagated_changes[pref_id][attr_name] = change
        
        logger.debug(f"Propagated {len(propagated_changes)} changes from {len(source_changes)} source changes")
        return propagated_changes
    
    def _calculate_direct_influences(
        self,
        source_pref_id: str,
        source_attrs: Dict[str, Dict[str, Any]],
        valid_dependencies: Dict[str, float],
        all_preferences: Dict[str, Dict[str, Any]],
        propagated_changes: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> None:
        """
        Calculate direct influences from a source preference to related preferences.
        
        Args:
            source_pref_id: ID of the source preference
            source_attrs: Changed attributes in the source preference
            valid_dependencies: Dictionary of target preferences with dependency strengths
            all_preferences: Dictionary of all current preference values
            propagated_changes: Dictionary to store propagated changes (modified in-place)
        """
        # For each target preference
        for target_pref_id, dependency_strength in valid_dependencies.items():
            # Skip if target is the same as source
            if target_pref_id == source_pref_id:
                continue
                
            # Skip if target preference doesn't exist
            if target_pref_id not in all_preferences:
                continue
                
            # Get target preference attributes
            target_pref = all_preferences[target_pref_id]
            target_attrs = target_pref.get("attributes", {})
            
            # Determine which target attributes to update based on source changes
            self._determine_target_updates(
                source_pref_id=source_pref_id,
                source_attrs=source_attrs,
                target_pref_id=target_pref_id,
                target_attrs=target_attrs,
                dependency_strength=dependency_strength,
                propagated_changes=propagated_changes
            )
    
    def _calculate_indirect_influences(
        self,
        source_pref_id: str,
        source_attrs: Dict[str, Dict[str, Any]],
        valid_dependencies: Dict[str, float],
        all_preferences: Dict[str, Dict[str, Any]],
        propagated_changes: Dict[str, Dict[str, Dict[str, Any]]],
        processed_prefs: Set[str]
    ) -> None:
        """
        Calculate indirect influences from a source preference to related preferences.
        
        Args:
            source_pref_id: ID of the source preference
            source_attrs: Changed attributes in the source preference
            valid_dependencies: Dictionary of target preferences with dependency strengths
            all_preferences: Dictionary of all current preference values
            propagated_changes: Dictionary to store propagated changes (modified in-place)
            processed_prefs: Set of already processed preferences (to avoid cycles)
        """
        # Similar to _calculate_direct_influences but with cycle detection
        for target_pref_id, dependency_strength in valid_dependencies.items():
            # Skip if target is the same as source or already processed
            if target_pref_id == source_pref_id or target_pref_id in processed_prefs:
                continue
                
            # Skip if target preference doesn't exist
            if target_pref_id not in all_preferences:
                continue
                
            # Get target preference attributes
            target_pref = all_preferences[target_pref_id]
            target_attrs = target_pref.get("attributes", {})
            
            # Determine which target attributes to update based on source changes
            self._determine_target_updates(
                source_pref_id=source_pref_id,
                source_attrs=source_attrs,
                target_pref_id=target_pref_id,
                target_attrs=target_attrs,
                dependency_strength=dependency_strength,
                propagated_changes=propagated_changes
            )
    
    def _determine_target_updates(
        self,
        source_pref_id: str,
        source_attrs: Dict[str, Dict[str, Any]],
        target_pref_id: str,
        target_attrs: Dict[str, Any],
        dependency_strength: float,
        propagated_changes: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> None:
        """
        Determine which target attributes to update based on source changes.
        
        Args:
            source_pref_id: ID of the source preference
            source_attrs: Changed attributes in the source preference
            target_pref_id: ID of the target preference
            target_attrs: Attributes of the target preference
            dependency_strength: Strength of the dependency relationship
            propagated_changes: Dictionary to store propagated changes (modified in-place)
        """
        # For each changed attribute in the source
        for source_attr, source_change in source_attrs.items():
            source_from = source_change.get("from")
            source_to = source_change.get("to")
            
            # Skip if the source values are the same
            if source_from == source_to:
                continue
                
            # Find semantically related attributes in target
            related_target_attrs = self._find_related_attributes(
                source_pref_id=source_pref_id,
                source_attr=source_attr,
                target_pref_id=target_pref_id,
                target_attrs=target_attrs
            )
            
            # For each related target attribute
            for target_attr, semantic_relatedness in related_target_attrs.items():
                # Skip if attribute doesn't exist in target
                if target_attr not in target_attrs:
                    continue
                    
                # Get current target value
                target_value = target_attrs[target_attr]
                
                # Calculate combined influence strength
                influence_strength = dependency_strength * semantic_relatedness
                
                # Skip if influence is too weak
                if influence_strength < self.config["min_strength_threshold"]:
                    continue
                    
                # Calculate new target value
                new_target_value = self._calculate_influenced_value(
                    source_from=source_from,
                    source_to=source_to,
                    target_value=target_value,
                    influence_strength=influence_strength
                )
                
                # Skip if the target value didn't change
                if new_target_value == target_value:
                    continue
                    
                # Add to propagated changes
                if target_pref_id not in propagated_changes:
                    propagated_changes[target_pref_id] = {}
                    
                propagated_changes[target_pref_id][target_attr] = {
                    "from": target_value,
                    "to": new_target_value
                }
    
    def _find_related_attributes(
        self,
        source_pref_id: str,
        source_attr: str,
        target_pref_id: str,
        target_attrs: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Find semantically related attributes between source and target preferences.
        
        Args:
            source_pref_id: ID of the source preference
            source_attr: Source attribute name
            target_pref_id: ID of the target preference
            target_attrs: Attributes of the target preference
            
        Returns:
            Dictionary mapping target attribute names to semantic relatedness scores (0-1)
        """
        # Initialize relatedness scores
        relatedness_scores = {}
        
        # Simple heuristic: attributes with similar names have high relatedness
        for target_attr in target_attrs:
            # Calculate string similarity between attribute names
            similarity = self._attribute_name_similarity(source_attr, target_attr)
            
            # Only include if similarity is high enough
            if similarity >= 0.5:
                relatedness_scores[target_attr] = similarity
                
        # If no good matches based on name, try some common known relationships
        if not relatedness_scores:
            # Example known relationships between different preference domains
            if source_pref_id == "music_preference" and target_pref_id == "fashion_preference":
                if source_attr == "genre_preference" and "style_preference" in target_attrs:
                    relatedness_scores["style_preference"] = 0.8
                elif source_attr == "intensity_preference" and "boldness" in target_attrs:
                    relatedness_scores["boldness"] = 0.7
                    
            elif source_pref_id == "food_preference" and target_pref_id == "health_preference":
                if source_attr == "spice_tolerance" and "risk_tolerance" in target_attrs:
                    relatedness_scores["risk_tolerance"] = 0.6
                elif source_attr == "dietary_restrictions" and "health_consciousness" in target_attrs:
                    relatedness_scores["health_consciousness"] = 0.9
        
        # If still no relatedness found, fall back to using the most similar attribute name
        if not relatedness_scores and target_attrs:
            best_attr = max(target_attrs, key=lambda attr: self._attribute_name_similarity(source_attr, attr))
            similarity = self._attribute_name_similarity(source_attr, best_attr)
            
            if similarity > 0.2:  # Minimum threshold for fallback
                relatedness_scores[best_attr] = similarity
        
        return relatedness_scores
    
    def _attribute_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between attribute names.
        
        Args:
            name1: First attribute name
            name2: Second attribute name
            
        Returns:
            Similarity score (0-1)
        """
        # Convert to lowercase and split into words
        words1 = set(name1.lower().replace('_', ' ').split())
        words2 = set(name2.lower().replace('_', ' ').split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _calculate_influenced_value(
        self,
        source_from: Any,
        source_to: Any,
        target_value: Any,
        influence_strength: float
    ) -> Any:
        """
        Calculate new target value based on source change and influence strength.
        
        Args:
            source_from: Original source value
            source_to: New source value
            target_value: Current target value
            influence_strength: Combined influence strength
            
        Returns:
            New target value
        """
        # Handle based on value types
        if isinstance(source_from, (int, float)) and isinstance(source_to, (int, float)) and isinstance(target_value, (int, float)):
            return self._calculate_numeric_influence(source_from, source_to, target_value, influence_strength)
        elif isinstance(source_from, bool) and isinstance(source_to, bool) and isinstance(target_value, bool):
            return self._calculate_boolean_influence(source_from, source_to, target_value, influence_strength)
        elif isinstance(source_from, list) and isinstance(source_to, list) and isinstance(target_value, list):
            return self._calculate_list_influence(source_from, source_to, target_value, influence_strength)
        else:
            # For other types, use simple probabilistic approach
            return self._calculate_categorical_influence(source_from, source_to, target_value, influence_strength)
    
    def _calculate_numeric_influence(
        self,
        source_from: Union[int, float],
        source_to: Union[int, float],
        target_value: Union[int, float],
        influence_strength: float
    ) -> Union[int, float]:
        """
        Calculate numeric influence between preferences.
        
        Args:
            source_from: Original source numeric value
            source_to: New source numeric value
            target_value: Current target numeric value
            influence_strength: Influence strength
            
        Returns:
            New target numeric value
        """
        # Calculate scaled change magnitude
        source_change = source_to - source_from
        
        # Calculate relative change (as proportion of original value)
        if source_from != 0:
            relative_change = source_change / abs(source_from)
        else:
            # If original value was zero, use absolute change
            relative_change = source_change
            
        # Apply influence to target
        if target_value != 0:
            target_change = target_value * relative_change * influence_strength
        else:
            # If target value is zero, use a small absolute change
            target_change = relative_change * influence_strength
            
        # Calculate new target value
        new_target_value = target_value + target_change
        
        # Return same type as original
        if isinstance(target_value, int):
            return int(round(new_target_value))
        return new_target_value
    
    def _calculate_boolean_influence(
        self,
        source_from: bool,
        source_to: bool,
        target_value: bool,
        influence_strength: float
    ) -> bool:
        """
        Calculate boolean influence between preferences.
        
        Args:
            source_from: Original source boolean value
            source_to: New source boolean value
            target_value: Current target boolean value
            influence_strength: Influence strength
            
        Returns:
            New target boolean value
        """
        # If source didn't change, target shouldn't change
        if source_from == source_to:
            return target_value
            
        # If target is already aligned with source, keep it
        if target_value == source_to:
            return target_value
            
        # Calculate change probability based on influence strength
        change_probability = influence_strength
        
        # Decide whether to change based on probability
        if random.random() < change_probability:
            return source_to
        else:
            return target_value
    
    def _calculate_list_influence(
        self,
        source_from: List[Any],
        source_to: List[Any],
        target_value: List[Any],
        influence_strength: float
    ) -> List[Any]:
        """
        Calculate list influence between preferences.
        
        Args:
            source_from: Original source list value
            source_to: New source list value
            target_value: Current target list value
            influence_strength: Influence strength
            
        Returns:
            New target list value
        """
        # Convert to sets for easier comparison
        source_from_set = set(str(item) for item in source_from)
        source_to_set = set(str(item) for item in source_to)
        target_set = set(str(item) for item in target_value)
        
        # Find additions and removals in source
        source_additions = [item for item in source_to if str(item) not in source_from_set]
        source_removals = [item for item in source_from if str(item) not in source_to_set]
        
        # Create a new target list
        new_target = target_value.copy()
        
        # Calculate number of items to influence based on influence strength
        add_count = max(1, int(len(source_additions) * influence_strength))
        remove_count = max(1, int(len(source_removals) * influence_strength))
        
        # Add random subset of source additions to target
        if source_additions:
            items_to_add = random.sample(source_additions, min(add_count, len(source_additions)))
            for item in items_to_add:
                if str(item) not in target_set:
                    new_target.append(item)
                    target_set.add(str(item))
                    
        # Remove random subset of source removals from target
        if source_removals:
            items_to_remove = []
            for item in new_target:
                str_item = str(item)
                if any(str_item == str(removal) for removal in source_removals):
                    items_to_remove.append(item)
                    
            if items_to_remove:
                remove_items = random.sample(items_to_remove, min(remove_count, len(items_to_remove)))
                new_target = [item for item in new_target if item not in remove_items]
                
        return new_target
    
    def _calculate_categorical_influence(
        self,
        source_from: Any,
        source_to: Any,
        target_value: Any,
        influence_strength: float
    ) -> Any:
        """
        Calculate categorical influence between preferences.
        
        Args:
            source_from: Original source categorical value
            source_to: New source categorical value
            target_value: Current target categorical value
            influence_strength: Influence strength
            
        Returns:
            New target categorical value
        """
        # If source didn't change, target shouldn't change
        if source_from == source_to:
            return target_value
            
        # If target is already aligned with source, keep it
        if target_value == source_to:
            return target_value
            
        # Calculate change probability based on influence strength
        change_probability = influence_strength
        
        # Decide whether to change based on probability
        if random.random() < change_probability:
            return source_to
        else:
            return target_value
    
    def _calculate_change_magnitude(self, from_value: Any, to_value: Any) -> float:
        """
        Calculate the magnitude of a change between two values.
        
        Args:
            from_value: Original value
            to_value: New value
            
        Returns:
            Change magnitude (0-1 scale)
        """
        # Handle based on value types
        if isinstance(from_value, (int, float)) and isinstance(to_value, (int, float)):
            # For numeric values, calculate relative change
            if from_value != 0:
                return min(1.0, abs(to_value - from_value) / abs(from_value))
            else:
                return min(1.0, abs(to_value - from_value))
        elif isinstance(from_value, bool) and isinstance(to_value, bool):
            # For boolean values, change is either 0 or 1
            return 1.0 if from_value != to_value else 0.0
        elif isinstance(from_value, list) and isinstance(to_value, list):
            # For lists, calculate proportion of items changed
            from_set = set(str(item) for item in from_value)
            to_set = set(str(item) for item in to_value)
            
            # Count changes (additions and removals)
            changes = len(from_set.symmetric_difference(to_set))
            total = len(from_set.union(to_set))
            
            return changes / total if total > 0 else 0.0
        else:
            # For other types, change is either 0 or 1
            return 1.0 if from_value != to_value else 0.0