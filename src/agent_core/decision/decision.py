#!/usr/bin/env python3
"""
Decision Module - Responsible for action selection and decision making

This module provides decision-making capabilities for the autonomous agent,
including action selection, utility calculation, and reasoning.
"""

import os
import uuid
import json
import logging
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DecisionModule:
    """
    Decision module for the autonomous agent.
    
    Responsible for selecting actions, calculating utility, and reasoning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the decision module with configuration settings.
        
        Args:
            config: Dictionary containing decision configuration
        """
        self.config = config
        self.available_actions = {}  # Registry of available actions
        self.action_history = []  # History of actions taken
        
        # Load available actions
        self._load_actions()
        
        logger.info("Decision module initialized")
    
    def _load_actions(self):
        """Load available actions from configuration."""
        # In a real implementation, this would dynamically load action modules
        # For now, we'll define some basic actions
        
        self.available_actions = {
            "search_documents": {
                "description": "Search for information in documents",
                "parameters": ["query", "limit"],
                "constraints": {
                    "requires_documents": True
                }
            },
            "execute_code": {
                "description": "Execute code in a sandbox environment",
                "parameters": ["code", "language"],
                "constraints": {
                    "max_execution_time": 30  # seconds
                }
            },
            "generate_response": {
                "description": "Generate a response using the LLM",
                "parameters": ["prompt", "context", "max_tokens"],
                "constraints": {
                    "requires_context": True
                }
            },
            "save_to_memory": {
                "description": "Save information to memory",
                "parameters": ["key", "value", "memory_type"],
                "constraints": {}
            },
            "retrieve_from_memory": {
                "description": "Retrieve information from memory",
                "parameters": ["key", "memory_type"],
                "constraints": {}
            }
        }
    
    def select_action(self, state: Dict[str, Any], goal: Dict[str, Any], 
                     task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best action to take given the current state, goal, and task.
        
        Args:
            state: Current state of the agent
            goal: Current goal being pursued
            task: Current task being executed
            
        Returns:
            action: Selected action with parameters
        """
        # In a real implementation, this would use more sophisticated decision-making
        # For now, we'll use a simple rule-based approach
        
        # Get candidate actions
        candidates = self._get_candidate_actions(state, goal, task)
        
        if not candidates:
            logger.warning("No candidate actions available")
            return {
                "action": "no_op",
                "parameters": {},
                "reason": "No suitable actions available"
            }
        
        # Calculate utility for each candidate
        for candidate in candidates:
            candidate["utility"] = self._calculate_utility(candidate, state, goal, task)
        
        # Sort by utility (highest first)
        candidates.sort(key=lambda x: x["utility"], reverse=True)
        
        # Select the highest utility action
        selected = candidates[0]
        
        # Record the action in history
        self.action_history.append({
            "action": selected["action"],
            "parameters": selected["parameters"],
            "utility": selected["utility"],
            "goal_id": goal["id"],
            "task_id": task["id"],
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        logger.info(f"Selected action: {selected['action']} with utility {selected['utility']}")
        
        return {
            "action": selected["action"],
            "parameters": selected["parameters"],
            "reason": selected["reason"]
        }
    
    def _get_candidate_actions(self, state: Dict[str, Any], goal: Dict[str, Any], 
                              task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get candidate actions based on the current state, goal, and task.
        
        Args:
            state: Current state of the agent
            goal: Current goal being pursued
            task: Current task being executed
            
        Returns:
            candidates: List of candidate actions
        """
        candidates = []
        
        # Extract task description for analysis
        task_description = task["description"].lower()
        
        # Rule-based candidate generation
        if "search" in task_description or "find" in task_description or "research" in task_description:
            # Task involves searching for information
            search_query = task_description.replace("research", "").replace("search", "").replace("find", "").strip()
            candidates.append({
                "action": "search_documents",
                "parameters": {
                    "query": search_query,
                    "limit": 5
                },
                "reason": "Task involves searching for information"
            })
        
        if "analyze" in task_description or "calculate" in task_description or "compute" in task_description:
            # Task involves analysis or computation
            candidates.append({
                "action": "execute_code",
                "parameters": {
                    "code": f"# Analysis for: {task_description}\nprint('Analysis results')",
                    "language": "python"
                },
                "reason": "Task involves analysis or computation"
            })
        
        if "generate" in task_description or "create" in task_description or "write" in task_description:
            # Task involves generating content
            candidates.append({
                "action": "generate_response",
                "parameters": {
                    "prompt": f"Generate content for: {task_description}",
                    "context": [],
                    "max_tokens": 500
                },
                "reason": "Task involves generating content"
            })
        
        # Always consider retrieving from memory
        candidates.append({
            "action": "retrieve_from_memory",
            "parameters": {
                "key": task["id"],
                "memory_type": "working"
            },
            "reason": "Check if task information is in memory"
        })
        
        # Always consider saving to memory
        candidates.append({
            "action": "save_to_memory",
            "parameters": {
                "key": task["id"],
                "value": {
                    "task": task,
                    "goal": goal,
                    "timestamp": datetime.datetime.now().isoformat()
                },
                "memory_type": "working"
            },
            "reason": "Save task information to memory"
        })
        
        return candidates
    
    def _calculate_utility(self, candidate: Dict[str, Any], state: Dict[str, Any], 
                          goal: Dict[str, Any], task: Dict[str, Any]) -> float:
        """
        Calculate the utility of a candidate action.
        
        Args:
            candidate: Candidate action
            state: Current state of the agent
            goal: Current goal being pursued
            task: Current task being executed
            
        Returns:
            utility: Utility value of the action
        """
        # In a real implementation, this would use more sophisticated utility calculation
        # For now, we'll use a simple heuristic approach
        
        action_name = candidate["action"]
        base_utility = 0.5  # Default utility
        
        # Adjust utility based on action type
        if action_name == "search_documents":
            # Higher utility for search actions when task involves research
            if "research" in task["description"].lower() or "find" in task["description"].lower():
                base_utility = 0.8
        
        elif action_name == "execute_code":
            # Higher utility for code execution when task involves analysis
            if "analyze" in task["description"].lower() or "calculate" in task["description"].lower():
                base_utility = 0.8
        
        elif action_name == "generate_response":
            # Higher utility for response generation when task involves content creation
            if "generate" in task["description"].lower() or "create" in task["description"].lower():
                base_utility = 0.8
        
        elif action_name == "save_to_memory":
            # Medium utility for saving to memory
            base_utility = 0.6
        
        elif action_name == "retrieve_from_memory":
            # Medium utility for retrieving from memory
            base_utility = 0.6
        
        # Add some randomness to avoid getting stuck in local optima
        utility = base_utility + random.uniform(-0.1, 0.1)
        
        # Ensure utility is in [0, 1] range
        utility = max(0.0, min(1.0, utility))
        
        return utility
    
    def evaluate_action_result(self, action: Dict[str, Any], result: Dict[str, Any], 
                              goal: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the result of an action and determine next steps.
        
        Args:
            action: Action that was taken
            result: Result of the action
            goal: Current goal being pursued
            task: Current task being executed
            
        Returns:
            evaluation: Evaluation of the action result
        """
        # In a real implementation, this would use more sophisticated evaluation
        # For now, we'll use a simple rule-based approach
        
        action_name = action["action"]
        success = result.get("success", False)
        
        evaluation = {
            "action": action_name,
            "success": success,
            "task_complete": False,
            "next_action_needed": True,
            "reason": ""
        }
        
        if success:
            # Action was successful
            if action_name == "search_documents":
                # Check if search returned results
                if result.get("results") and len(result["results"]) > 0:
                    evaluation["reason"] = "Search returned results"
                    # If this was a research task, it might be complete
                    if "research" in task["description"].lower():
                        evaluation["task_complete"] = True
                else:
                    evaluation["reason"] = "Search returned no results"
                    evaluation["next_action_needed"] = True
            
            elif action_name == "execute_code":
                # Check if code execution was successful
                if result.get("error"):
                    evaluation["success"] = False
                    evaluation["reason"] = f"Code execution failed: {result['error']}"
                else:
                    evaluation["reason"] = "Code execution successful"
                    # If this was an analysis task, it might be complete
                    if "analyze" in task["description"].lower():
                        evaluation["task_complete"] = True
            
            elif action_name == "generate_response":
                # Check if response generation was successful
                if result.get("response"):
                    evaluation["reason"] = "Response generated successfully"
                    # If this was a generation task, it might be complete
                    if "generate" in task["description"].lower() or "create" in task["description"].lower():
                        evaluation["task_complete"] = True
                else:
                    evaluation["success"] = False
                    evaluation["reason"] = "Response generation failed"
            
            elif action_name == "save_to_memory" or action_name == "retrieve_from_memory":
                # Memory operations are usually intermediate steps
                evaluation["reason"] = f"{action_name} successful"
                evaluation["next_action_needed"] = True
        else:
            # Action failed
            evaluation["reason"] = result.get("error", "Action failed")
            evaluation["next_action_needed"] = True
        
        # Record the evaluation
        self.action_history[-1]["evaluation"] = evaluation
        
        logger.info(f"Action evaluation: {evaluation['reason']}")
        
        return evaluation
    
    def get_action_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of actions taken.
        
        Args:
            limit: Maximum number of actions to return
            
        Returns:
            history: List of actions taken
        """
        return self.action_history[-limit:]
    
    def save_action_history(self, file_path: str) -> bool:
        """
        Save action history to a file.
        
        Args:
            file_path: Path to save the history
            
        Returns:
            success: Whether the save was successful
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.action_history, f, indent=2)
            logger.info(f"Action history saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving action history: {e}")
            return False
    
    def load_action_history(self, file_path: str) -> bool:
        """
        Load action history from a file.
        
        Args:
            file_path: Path to load the history from
            
        Returns:
            success: Whether the load was successful
        """
        try:
            with open(file_path, 'r') as f:
                self.action_history = json.load(f)
            logger.info(f"Action history loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading action history: {e}")
            return False
