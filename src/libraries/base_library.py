"""
base_library.py

Defines the abstract base class for all libraries in the Causal Preference Evolution Framework.
This provides common functionality for storing, retrieving, and sampling structures.
"""

import os
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union, Tuple

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for the structures stored in the library

class BaseLibrary(Generic[T], ABC):
    """
    Abstract base class for all libraries in the Causal Preference Evolution Framework.
    Provides common functionality for structure management.
    """
    
    def __init__(self, library_name: str):
        """
        Initialize a base library.
        
        Args:
            library_name: Name of the library
        """
        self.library_name = library_name
        self._structures: Dict[str, T] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Initialized {library_name}")
    
    def add_structure(self, structure_id: str, structure: T, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a structure to the library.
        
        Args:
            structure_id: Unique identifier for the structure
            structure: The structure to add
            metadata: Optional metadata about the structure
        """
        # Ensure structure has its ID field set
        if "id" not in structure:
            structure["id"] = structure_id
        self._structures[structure_id] = structure
        self._metadata[structure_id] = metadata or {}
        logger.debug(f"Added structure {structure_id} to {self.library_name}")
    
    def get_structure(self, structure_id: str) -> Optional[T]:
        """
        Get a structure from the library by ID.
        
        Args:
            structure_id: ID of the structure to retrieve
            
        Returns:
            The structure if found, None otherwise
        """
        return self._structures.get(structure_id)
    
    def get_structure_metadata(self, structure_id: str) -> Dict[str, Any]:
        """
        Get metadata for a structure.
        
        Args:
            structure_id: ID of the structure
            
        Returns:
            Dictionary of metadata, empty if structure not found
        """
        return self._metadata.get(structure_id, {})
    
    def remove_structure(self, structure_id: str) -> bool:
        """
        Remove a structure from the library.
        
        Args:
            structure_id: ID of the structure to remove
            
        Returns:
            True if removed, False if not found
        """
        if structure_id in self._structures:
            del self._structures[structure_id]
            if structure_id in self._metadata:
                del self._metadata[structure_id]
            logger.debug(f"Removed structure {structure_id} from {self.library_name}")
            return True
        return False
    
    def get_all_structure_ids(self) -> List[str]:
        """
        Get all structure IDs in the library.
        
        Returns:
            List of structure IDs
        """
        return list(self._structures.keys())
    
    def get_all_structures(self) -> Dict[str, T]:
        """
        Get all structures in the library.
        
        Returns:
            Dictionary mapping structure IDs to structures
        """
        return self._structures.copy()
    
    def get_structure_count(self) -> int:
        """
        Get the number of structures in the library.
        
        Returns:
            Count of structures
        """
        return len(self._structures)
    
    def sample_structure(self, filter_func: Optional[Callable[[T], bool]] = None) -> Optional[Tuple[str, T]]:
        """
        Sample a random structure from the library.
        
        Args:
            filter_func: Optional function to filter structures
            
        Returns:
            Tuple of (structure_id, structure) or None if no structures match
        """
        eligible_structures = self._structures
        
        if filter_func:
            eligible_structures = {
                sid: struct for sid, struct in self._structures.items()
                if filter_func(struct)
            }
            
        if not eligible_structures:
            return None
            
        structure_id = random.choice(list(eligible_structures.keys()))
        return structure_id, eligible_structures[structure_id]
    
    def filter_structures(self, filter_func: Callable[[T], bool]) -> Dict[str, T]:
        """
        Filter structures based on a predicate function.
        
        Args:
            filter_func: Function that takes a structure and returns a boolean
            
        Returns:
            Dictionary of filtered structures
        """
        return {
            sid: struct for sid, struct in self._structures.items()
            if filter_func(struct)
        }
    
    def load_from_file(self, file_path: str) -> int:
        """
        Load structures from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Number of structures loaded
            
        Raises:
            ValueError: If file format is invalid
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or "structures" not in data:
            raise ValueError(f"Invalid library file format: {file_path}")

        count = 0
        for structure_id, structure_data in data["structures"].items():
            parsed_structure = structure_data.get("data", {}) #self._parse_structure(structure_data)
            metadata = structure_data.get("metadata", {})
            self.add_structure(structure_id, parsed_structure, metadata)
            count += 1
            
        logger.info(f"Loaded {count} structures from {file_path} into {self.library_name}")
        return count
    
    def save_to_file(self, file_path: str) -> int:
        """
        Save structures to a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Number of structures saved
        """
        structures_data = {}
        
        for structure_id, structure in self._structures.items():
            structures_data[structure_id] = {
                "data": self._serialize_structure(structure),
                "metadata": self._metadata.get(structure_id, {})
            }
            
        data = {
            "library_name": self.library_name,
            "structures": structures_data
        }
        
        # make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved {len(structures_data)} structures from {self.library_name} to {file_path}")
        return len(structures_data)
    
    @abstractmethod
    def _parse_structure(self, data: Dict[str, Any]) -> T:
        """
        Parse a structure from its serialized form.
        
        Args:
            data: Serialized structure data
            
        Returns:
            Parsed structure object
        """
        pass
    
    @abstractmethod
    def _serialize_structure(self, structure: T) -> Dict[str, Any]:
        """
        Serialize a structure to a dictionary.
        
        Args:
            structure: Structure to serialize
            
        Returns:
            Serialized structure data
        """
        pass