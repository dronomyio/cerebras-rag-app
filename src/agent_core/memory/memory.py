#!/usr/bin/env python3
"""
Memory Module - Responsible for storing and retrieving agent knowledge

This module provides memory capabilities for the autonomous agent, including
short-term working memory, long-term episodic memory, and semantic memory.
"""

import os
import uuid
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
import heapq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryModule:
    """
    Memory module for the autonomous agent.
    
    Responsible for storing and retrieving different types of memory:
    - Working memory: Short-term memory for current tasks
    - Episodic memory: Long-term memory of past events and interactions
    - Semantic memory: Factual knowledge and concepts
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory module with configuration settings.
        
        Args:
            config: Dictionary containing memory configuration
        """
        self.config = config
        
        # Initialize memory stores
        self.working_memory = {}  # Short-term memory
        self.episodic_memory = []  # Long-term memory of events
        self.semantic_memory = {}  # Factual knowledge
        
        # Memory capacity limits
        self.working_memory_capacity = config.get("working_memory_capacity", 100)
        self.episodic_memory_capacity = config.get("episodic_memory_capacity", 1000)
        
        logger.info("Memory module initialized")
    
    def store_working_memory(self, key: str, value: Any) -> bool:
        """
        Store an item in working memory.
        
        Args:
            key: Key to store the value under
            value: Value to store
            
        Returns:
            success: Whether the storage was successful
        """
        # Check if we need to make room
        if len(self.working_memory) >= self.working_memory_capacity:
            # Remove the oldest item
            oldest_key = min(self.working_memory.keys(), 
                            key=lambda k: self.working_memory[k].get("timestamp", 0))
            del self.working_memory[oldest_key]
        
        # Store the new item
        self.working_memory[key] = {
            "value": value,
            "timestamp": datetime.datetime.now().timestamp(),
            "access_count": 0
        }
        
        logger.debug(f"Stored item in working memory: {key}")
        return True
    
    def retrieve_working_memory(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from working memory.
        
        Args:
            key: Key to retrieve
            
        Returns:
            value: Retrieved value, or None if not found
        """
        if key not in self.working_memory:
            return None
        
        # Update access count and timestamp
        self.working_memory[key]["access_count"] += 1
        self.working_memory[key]["timestamp"] = datetime.datetime.now().timestamp()
        
        logger.debug(f"Retrieved item from working memory: {key}")
        return self.working_memory[key]["value"]
    
    def store_episodic_memory(self, event: Dict[str, Any]) -> str:
        """
        Store an event in episodic memory.
        
        Args:
            event: Event to store
            
        Returns:
            event_id: ID of the stored event
        """
        event_id = str(uuid.uuid4())
        
        # Add metadata to the event
        memory_event = {
            "id": event_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "importance": event.get("importance", 1),  # 1-5 scale
            "access_count": 0,
            "event": event
        }
        
        # Add to episodic memory
        self.episodic_memory.append(memory_event)
        
        # Check if we need to prune
        if len(self.episodic_memory) > self.episodic_memory_capacity:
            # Sort by importance and access count, remove least important
            self.episodic_memory.sort(key=lambda x: (x["importance"], x["access_count"]))
            self.episodic_memory = self.episodic_memory[1:]
        
        logger.debug(f"Stored event in episodic memory: {event_id}")
        return event_id
    
    def retrieve_episodic_memory(self, query: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve events from episodic memory based on a query.
        
        Args:
            query: Query parameters (e.g., time range, keywords)
            limit: Maximum number of events to retrieve
            
        Returns:
            events: List of matching events
        """
        # In a real implementation, this would use vector search or other techniques
        # For now, we'll use a simple filtering approach
        
        results = []
        
        for event in self.episodic_memory:
            match = True
            
            # Filter by time range if specified
            if "start_time" in query and "end_time" in query:
                event_time = datetime.datetime.fromisoformat(event["timestamp"])
                start_time = datetime.datetime.fromisoformat(query["start_time"])
                end_time = datetime.datetime.fromisoformat(query["end_time"])
                
                if not (start_time <= event_time <= end_time):
                    match = False
            
            # Filter by keywords if specified
            if "keywords" in query and isinstance(query["keywords"], list):
                event_str = json.dumps(event["event"]).lower()
                if not any(kw.lower() in event_str for kw in query["keywords"]):
                    match = False
            
            # Filter by type if specified
            if "type" in query and event["event"].get("type") != query["type"]:
                match = False
            
            if match:
                # Update access count
                event["access_count"] += 1
                results.append(event)
                
                # Break if we've reached the limit
                if len(results) >= limit:
                    break
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        
        logger.debug(f"Retrieved {len(results)} events from episodic memory")
        return results[:limit]
    
    def store_semantic_memory(self, concept: str, information: Dict[str, Any]) -> bool:
        """
        Store factual information in semantic memory.
        
        Args:
            concept: Concept or entity name
            information: Information about the concept
            
        Returns:
            success: Whether the storage was successful
        """
        # If concept already exists, update it
        if concept in self.semantic_memory:
            # Merge the information
            self.semantic_memory[concept].update(information)
            # Update metadata
            self.semantic_memory[concept]["_metadata"]["last_updated"] = datetime.datetime.now().isoformat()
            self.semantic_memory[concept]["_metadata"]["update_count"] += 1
        else:
            # Create new concept
            self.semantic_memory[concept] = information
            # Add metadata
            self.semantic_memory[concept]["_metadata"] = {
                "created_at": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat(),
                "access_count": 0,
                "update_count": 0
            }
        
        logger.debug(f"Stored information in semantic memory: {concept}")
        return True
    
    def retrieve_semantic_memory(self, concept: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve information about a concept from semantic memory.
        
        Args:
            concept: Concept or entity name
            
        Returns:
            information: Information about the concept, or None if not found
        """
        if concept not in self.semantic_memory:
            return None
        
        # Update access count
        self.semantic_memory[concept]["_metadata"]["access_count"] += 1
        
        logger.debug(f"Retrieved information from semantic memory: {concept}")
        return self.semantic_memory[concept]
    
    def search_semantic_memory(self, query: str) -> List[Tuple[str, float]]:
        """
        Search semantic memory for concepts related to the query.
        
        Args:
            query: Search query
            
        Returns:
            results: List of (concept, relevance) tuples
        """
        # In a real implementation, this would use vector search or other techniques
        # For now, we'll use a simple keyword matching approach
        
        results = []
        query_lower = query.lower()
        
        for concept, info in self.semantic_memory.items():
            # Check if query is in concept name
            if query_lower in concept.lower():
                relevance = 1.0
                results.append((concept, relevance))
                continue
            
            # Check if query is in concept information
            concept_str = json.dumps(info).lower()
            if query_lower in concept_str:
                # Calculate relevance based on number of occurrences
                relevance = 0.5 * concept_str.count(query_lower)
                results.append((concept, relevance))
        
        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Found {len(results)} concepts in semantic memory for query: {query}")
        return results
    
    def save_memory(self, directory: str) -> bool:
        """
        Save all memory to files.
        
        Args:
            directory: Directory to save the memory files
            
        Returns:
            success: Whether the save was successful
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save working memory
            with open(os.path.join(directory, "working_memory.json"), 'w') as f:
                json.dump(self.working_memory, f, indent=2)
            
            # Save episodic memory
            with open(os.path.join(directory, "episodic_memory.json"), 'w') as f:
                json.dump(self.episodic_memory, f, indent=2)
            
            # Save semantic memory
            with open(os.path.join(directory, "semantic_memory.json"), 'w') as f:
                json.dump(self.semantic_memory, f, indent=2)
            
            logger.info(f"Memory saved to {directory}")
            return True
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False
    
    def load_memory(self, directory: str) -> bool:
        """
        Load memory from files.
        
        Args:
            directory: Directory to load the memory files from
            
        Returns:
            success: Whether the load was successful
        """
        try:
            # Load working memory
            working_memory_path = os.path.join(directory, "working_memory.json")
            if os.path.exists(working_memory_path):
                with open(working_memory_path, 'r') as f:
                    self.working_memory = json.load(f)
            
            # Load episodic memory
            episodic_memory_path = os.path.join(directory, "episodic_memory.json")
            if os.path.exists(episodic_memory_path):
                with open(episodic_memory_path, 'r') as f:
                    self.episodic_memory = json.load(f)
            
            # Load semantic memory
            semantic_memory_path = os.path.join(directory, "semantic_memory.json")
            if os.path.exists(semantic_memory_path):
                with open(semantic_memory_path, 'r') as f:
                    self.semantic_memory = json.load(f)
            
            logger.info(f"Memory loaded from {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return False
