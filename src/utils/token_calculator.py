"""
token_calculator.py

Utility for calculating token counts in generated conversations using a Llama tokenizer.
"""

import os
import json
import logging
from typing import Dict, List, Union
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class TokenCalculator:
    """
    Utility class for calculating token counts in conversations using a Llama tokenizer.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        """
        Initialize the token calculator with a tokenizer.
        
        Args:
            model_name: HuggingFace model name to use for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Initialized TokenCalculator with {model_name} tokenizer")
    
    def count_conversation_tokens(self, conversation: Dict) -> Dict[str, int]:
        """
        Count tokens in a conversation dictionary.
        
        Args:
            conversation: Dictionary containing the conversation data
            
        Returns:
            Dictionary with token counts for different parts of the conversation
        """
        token_counts = {
            "total_tokens": 0,
            "user_tokens": 0,
            "assistant_tokens": 0,
            "metadata_tokens": 0
        }
        
        # Count tokens in conversation turns
        for turn in conversation.get("turns", []):
            # Count user message tokens
            user_message = turn.get("user", "")
            user_tokens = len(self.tokenizer.encode(user_message))
            token_counts["user_tokens"] += user_tokens
            
            # Count assistant message tokens
            assistant_message = turn.get("assistant", "")
            assistant_tokens = len(self.tokenizer.encode(assistant_message))
            token_counts["assistant_tokens"] += assistant_tokens
        
        # Count tokens in metadata
        metadata = conversation.get("metadata", {})
        metadata_str = json.dumps(metadata)
        token_counts["metadata_tokens"] = len(self.tokenizer.encode(metadata_str))
        
        # Calculate total
        token_counts["total_tokens"] = (
            token_counts["user_tokens"] + 
            token_counts["assistant_tokens"] + 
            token_counts["metadata_tokens"]
        )
        
        return token_counts
    
    def analyze_conversation_file(self, filepath: str) -> Dict[str, Union[Dict[str, int], str]]:
        """
        Analyze a saved conversation file and return token counts.
        
        Args:
            filepath: Path to the conversation JSON file
            
        Returns:
            Dictionary containing token counts and file information
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Get the conversation part
            conversation = data.get("conversation", {})
            
            # Calculate token counts
            token_counts = self.count_conversation_tokens(conversation)
            
            return {
                "filepath": filepath,
                "token_counts": token_counts,
                "conversation_info": {
                    "num_turns": len(conversation.get("turns", [])),
                    "event_type": data.get("event", {}).get("type", "unknown")
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing file {filepath}: {e}")
            return {
                "filepath": filepath,
                "error": str(e)
            }
    
    def analyze_conversation_directory(self, directory: str) -> List[Dict]:
        """
        Analyze all conversation files in a directory.
        
        Args:
            directory: Path to directory containing conversation files
            
        Returns:
            List of analysis results for each file
        """
        results = []
        
        # Walk through directory
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith("_conversation.json"):
                    filepath = os.path.join(root, file)
                    result = self.analyze_conversation_file(filepath)
                    results.append(result)
        
        return results
    
    def generate_token_report(self, analysis_results: List[Dict]) -> Dict:
        """
        Generate a summary report from analysis results.
        
        Args:
            analysis_results: List of analysis results from analyze_conversation_directory
            
        Returns:
            Dictionary containing summary statistics
        """
        total_files = len(analysis_results)
        total_tokens = 0
        total_user_tokens = 0
        total_assistant_tokens = 0
        total_metadata_tokens = 0
        total_turns = 0
        
        # Calculate totals
        for result in analysis_results:
            if "token_counts" in result:
                token_counts = result["token_counts"]
                total_tokens += token_counts["total_tokens"]
                total_user_tokens += token_counts["user_tokens"]
                total_assistant_tokens += token_counts["assistant_tokens"]
                total_metadata_tokens += token_counts["metadata_tokens"]
                total_turns += result["conversation_info"]["num_turns"]
        
        # Calculate averages
        avg_tokens_per_file = total_tokens / total_files if total_files > 0 else 0
        avg_tokens_per_turn = total_tokens / total_turns if total_turns > 0 else 0
        
        return {
            "total_files_analyzed": total_files,
            "total_tokens": total_tokens,
            "total_user_tokens": total_user_tokens,
            "total_assistant_tokens": total_assistant_tokens,
            "total_metadata_tokens": total_metadata_tokens,
            "total_conversation_turns": total_turns,
            "average_tokens_per_file": avg_tokens_per_file,
            "average_tokens_per_turn": avg_tokens_per_turn
        }

def main():
    """
    Example usage of the TokenCalculator.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate token counts in conversation files")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing conversation files")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Llama model to use for tokenization")
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = TokenCalculator(model_name=args.model)
    
    # Analyze directory
    results = calculator.analyze_conversation_directory(args.directory)
    
    # Generate and print report
    report = calculator.generate_token_report(results)
    print("\nToken Usage Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main() 