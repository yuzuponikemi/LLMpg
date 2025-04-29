"""
Memory management system for the research agent.
Provides both working memory during plan execution and persistent long-term memory across sessions.
"""

import os
import json
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="memory_manager")

class MemoryManager:
    """
    Manages short-term and long-term memory for the research agent.
    
    Features:
    - Working memory for active plans
    - Long-term storage across sessions
    - Fact extraction and categorization
    - Memory retrieval based on relevance
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize the memory manager.
        
        Args:
            storage_dir: Directory to store persistent memories. If None, uses default.
        """
        # Set up storage directory
        if storage_dir is None:
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            storage_dir = os.path.join(base_dir, "data", "memory")
        
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Memory structures
        self.working_memory = {
            "facts": [],           # Extracted factual information
            "conclusions": [],     # Derived conclusions
            "web_content": [],     # Important content from web searches
            "code_snippets": [],   # Useful code snippets
            "context": {}          # General context about the current query/plan
        }
        
        # Long-term memory file
        self.memory_file = os.path.join(self.storage_dir, "long_term_memory.json")
        self.long_term_memory = self._load_long_term_memory()
        
        logger.info("Memory Manager initialized with storage at: %s", self.storage_dir)
    
    def _load_long_term_memory(self) -> Dict:
        """Load long-term memory from file or create default structure"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                logger.info(f"Loaded {len(memory_data.get('memories', []))} memories from storage")
                return memory_data
            except Exception as e:
                logger.error(f"Error loading long-term memory: {str(e)}")
                # Return default structure if loading fails
        
        # Default memory structure
        return {
            "memories": [],
            "metadata": {
                "last_updated": datetime.datetime.now().isoformat(),
                "version": "1.0"
            }
        }
    
    def save_long_term_memory(self) -> None:
        """Save current long-term memory to persistent storage"""
        try:
            # Update metadata
            self.long_term_memory["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
            
            # Save to file
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_memory, f, indent=2)
            
            logger.info(f"Saved {len(self.long_term_memory['memories'])} memories to persistent storage")
        except Exception as e:
            logger.error(f"Error saving long-term memory: {str(e)}")
    
    def clear_working_memory(self) -> None:
        """Clear the current working memory"""
        for key in self.working_memory:
            if isinstance(self.working_memory[key], list):
                self.working_memory[key] = []
            elif isinstance(self.working_memory[key], dict):
                self.working_memory[key] = {}
        
        logger.info("Working memory cleared")
    
    def store_step_result(self, step_instruction: str, step_result: str, goal: str = None) -> None:
        """
        Store the result of a step execution in working memory and extract key information.
        
        Args:
            step_instruction: The instruction for the step
            step_result: The result text from the step execution
            goal: Optional goal for context
        """
        # Extract key facts from step result (simplified version)
        facts = self._extract_facts(step_result)
        
        # Store information by category
        if any(keyword in step_instruction.lower() for keyword in ["search", "find", "look up"]):
            # This appears to be a search step, extract web content
            self.store_memory("web_content", {
                "step": step_instruction,
                "content": step_result,
                "timestamp": time.time(),
                "extracted_facts": facts
            })
            
        elif any(keyword in step_instruction.lower() for keyword in ["code", "script", "program", "implement"]):
            # This appears to be a code-related step
            self.store_memory("code_snippets", {
                "step": step_instruction,
                "content": self._extract_code_snippets(step_result),
                "timestamp": time.time()
            })
            
        elif any(keyword in step_instruction.lower() for keyword in ["analyze", "evaluate", "assess"]):
            # This appears to be an analysis step
            self.store_memory("conclusions", {
                "step": step_instruction,
                "content": step_result,
                "timestamp": time.time(),
                "key_conclusions": self._extract_conclusions(step_result)
            })
        
        # Store facts from all steps
        for fact in facts:
            self.store_memory("facts", fact)
        
        # Update context if goal provided
        if goal:
            self.working_memory["context"]["goal"] = goal
            self.working_memory["context"]["last_step"] = step_instruction
            self.working_memory["context"]["last_result"] = step_result
        
        logger.info(f"Stored step result with {len(facts)} extracted facts")
    
    def store_memory(self, memory_type: str, content: Any) -> None:
        """
        Store a memory item in the specified memory type.
        
        Args:
            memory_type: Type of memory (facts, conclusions, web_content, etc.)
            content: The content to store
        """
        if memory_type in self.working_memory:
            self.working_memory[memory_type].append(content)
        else:
            logger.warning(f"Unknown memory type: {memory_type}")
    
    def get_relevant_memories(self, query: str, memory_types: List[str] = None, limit: int = 5) -> List[Dict]:
        """
        Retrieve relevant memories based on the query.
        
        Args:
            query: The query to match against memories
            memory_types: Types of memories to search (if None, searches all)
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memory items
        """
        if memory_types is None:
            memory_types = ["facts", "conclusions", "web_content", "code_snippets"]
        
        relevant_memories = []
        
        # Search through working memory
        for memory_type in memory_types:
            if memory_type not in self.working_memory:
                continue
                
            for item in self.working_memory[memory_type]:
                relevance = self._calculate_relevance(query, item)
                if relevance > 0.3:  # Arbitrary threshold - would be better with semantic similarity
                    relevant_memories.append({
                        "type": memory_type,
                        "content": item,
                        "relevance": relevance,
                        "source": "working_memory"
                    })
        
        # Search through long-term memory
        for memory in self.long_term_memory.get("memories", []):
            relevance = self._calculate_relevance(query, memory)
            if relevance > 0.3:
                relevant_memories.append({
                    "type": memory.get("type", "unknown"),
                    "content": memory,
                    "relevance": relevance,
                    "source": "long_term_memory"
                })
        
        # Sort by relevance and limit results
        relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant_memories[:limit]
    
    def persist_important_memories(self) -> None:
        """
        Save important items from working memory to long-term memory.
        Called at the end of a plan execution.
        """
        # Identify important facts to remember long-term
        important_facts = [
            fact for fact in self.working_memory["facts"]
            if self._is_memory_important(fact)
        ]
        
        # Identify important conclusions
        important_conclusions = [
            conclusion for conclusion in self.working_memory["conclusions"]
            if self._is_memory_important(conclusion)
        ]
        
        # Add to long-term memory with metadata
        timestamp = time.time()
        
        for fact in important_facts:
            self.long_term_memory["memories"].append({
                "content": fact,
                "type": "fact",
                "timestamp": timestamp,
                "context": self.working_memory["context"].get("goal", ""),
                "importance": self._calculate_importance(fact)
            })
        
        for conclusion in important_conclusions:
            self.long_term_memory["memories"].append({
                "content": conclusion,
                "type": "conclusion",
                "timestamp": timestamp,
                "context": self.working_memory["context"].get("goal", ""),
                "importance": self._calculate_importance(conclusion)
            })
        
        # Save to disk
        self.save_long_term_memory()
        logger.info(f"Persisted {len(important_facts)} facts and {len(important_conclusions)} conclusions to long-term memory")
    
    def generate_memory_context(self, query: str, max_length: int = 2000) -> str:
        """
        Generate context from memory for inclusion in prompts.
        
        Args:
            query: The query to get relevant context for
            max_length: Maximum length of returned context
            
        Returns:
            String containing relevant memory context
        """
        relevant_memories = self.get_relevant_memories(query)
        
        if not relevant_memories:
            return ""
            
        context_parts = ["Here's what I know that might be relevant:"]
        
        for memory in relevant_memories:
            memory_content = memory["content"]
            memory_type = memory["type"]
            
            if memory_type == "facts":
                context_parts.append(f"- Fact: {memory_content}")
            elif memory_type == "conclusions":
                if isinstance(memory_content, dict) and "key_conclusions" in memory_content:
                    context_parts.append(f"- Conclusion: {memory_content['key_conclusions']}")
                else:
                    context_parts.append(f"- Conclusion: {memory_content}")
            elif memory_type == "web_content":
                if isinstance(memory_content, dict):
                    context_parts.append(f"- From web: {memory_content.get('content', '')[:200]}...")
            elif memory_type == "code_snippets":
                if isinstance(memory_content, dict):
                    context_parts.append(f"- Code example: {str(memory_content.get('content', ''))[:150]}...")
        
        # Join and truncate if necessary
        context = "\n".join(context_parts)
        if len(context) > max_length:
            context = context[:max_length] + "...(truncated)"
            
        return context
    
    def _extract_facts(self, text: str) -> List[str]:
        """
        Extract key facts from text.
        This is a simplified implementation - a real system would use more sophisticated NLP.
        """
        facts = []
        
        # Split into sentences (simple approach)
        sentences = text.split('. ')
        
        for sentence in sentences:
            # Skip very short sentences or ones that don't seem factual
            if len(sentence) < 15:
                continue
                
            # Apply simple heuristics to identify likely facts
            # In a real system, you'd use NLP to identify factual statements
            if any(keyword in sentence.lower() for keyword in [
                "is", "are", "was", "were", "has", "have", 
                "shows", "reveals", "indicates", "demonstrates", "according to"
            ]):
                facts.append(sentence.strip())
        
        return facts[:10]  # Limit to top 10 facts
    
    def _extract_code_snippets(self, text: str) -> List[str]:
        """Extract code snippets from text"""
        # Simple approach: look for text between code markdown blocks
        code_blocks = []
        lines = text.split('\n')
        in_code_block = False
        current_block = []
        
        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    # End of block
                    if current_block:
                        code_blocks.append("\n".join(current_block))
                    current_block = []
                    in_code_block = False
                else:
                    # Start of block
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)
        
        return code_blocks
    
    def _extract_conclusions(self, text: str) -> List[str]:
        """Extract key conclusions from analysis text"""
        conclusions = []
        
        # Simple approach: look for sentences that tend to express conclusions
        sentences = text.split('. ')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in [
                "therefore", "thus", "conclude", "conclusion", "summary", "overall", 
                "in summary", "to sum up", "key finding", "result indicates",
                "consequently", "as a result", "showing that", "proves that"
            ]):
                conclusions.append(sentence.strip())
        
        return conclusions
    
    def _calculate_relevance(self, query: str, memory_item: Any) -> float:
        """
        Calculate relevance score of a memory item to the query.
        This is a simplified implementation using keyword matching.
        A real system would use semantic similarity with embeddings.
        """
        query_tokens = set(query.lower().split())
        
        # Extract text to match against based on memory item type
        if isinstance(memory_item, str):
            memory_text = memory_item.lower()
        elif isinstance(memory_item, dict):
            if "content" in memory_item and isinstance(memory_item["content"], str):
                memory_text = memory_item["content"].lower()
            elif "content" in memory_item and isinstance(memory_item["content"], dict):
                # For nested structures, combine available text fields
                memory_text = " ".join([
                    str(v).lower() for k, v in memory_item["content"].items() 
                    if isinstance(v, (str, int, float))
                ])
            else:
                # Fall back to string representation
                memory_text = str(memory_item).lower()
        else:
            memory_text = str(memory_item).lower()
        
        # Count matching tokens
        memory_tokens = set(memory_text.split())
        matching_tokens = query_tokens.intersection(memory_tokens)
        
        # Calculate score - more matching tokens = higher relevance
        # In a real system, this would use vector similarity
        if not query_tokens:
            return 0
        
        return len(matching_tokens) / len(query_tokens)
    
    def _is_memory_important(self, memory_item: Any) -> bool:
        """
        Determine if a memory item is important enough to persist long-term.
        """
        # In a real system, this would be more sophisticated
        importance = self._calculate_importance(memory_item)
        return importance > 0.5  # Arbitrary threshold
    
    def _calculate_importance(self, memory_item: Any) -> float:
        """
        Calculate importance score for a memory item.
        This is a simplified placeholder implementation.
        """
        # For now, assign relatively high importance to all items
        # In a real system, importance would be calculated based on:
        # - Uniqueness of information
        # - Relevance to user's interests/goals
        # - Information density
        # - Confirmation from multiple sources
        return 0.75
