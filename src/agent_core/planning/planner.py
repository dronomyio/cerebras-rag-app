#!/usr/bin/env python3
"""
Planning Module - Responsible for goal-oriented planning and task decomposition

This module provides planning capabilities for the autonomous agent, including
goal representation, task decomposition, and plan execution monitoring.
"""

import os
import uuid
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlanningModule:
    """
    Planning module for the autonomous agent.
    
    Responsible for creating, managing, and executing plans to achieve goals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the planning module with configuration settings.
        
        Args:
            config: Dictionary containing planning configuration
        """
        self.config = config
        self.plans = {}  # Store active plans
        logger.info("Planning module initialized")
    
    def create_plan(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a plan to achieve the specified goal.
        
        Args:
            goal: Goal object with id, description, priority, etc.
            
        Returns:
            plan: The created plan
        """
        plan_id = str(uuid.uuid4())
        
        # Decompose the goal into tasks
        tasks = self._decompose_goal(goal["description"])
        
        plan = {
            "id": plan_id,
            "goal_id": goal["id"],
            "status": "active",
            "tasks": tasks,
            "current_task_index": 0,
            "created_at": datetime.datetime.now().isoformat(),
            "completed_at": None,
            "progress": 0.0
        }
        
        self.plans[plan_id] = plan
        logger.info(f"Created plan {plan_id} for goal {goal['id']}")
        
        return plan
    
    def _decompose_goal(self, goal_description: str) -> List[Dict[str, Any]]:
        """
        Decompose a goal into a sequence of tasks.
        
        Args:
            goal_description: Description of the goal
            
        Returns:
            tasks: List of task objects
        """
        # In a real implementation, this would use LLM to decompose the goal
        # For now, we'll use a simple placeholder implementation
        
        # Example task structure
        tasks = [
            {
                "id": str(uuid.uuid4()),
                "description": f"Analyze goal: {goal_description}",
                "status": "pending",
                "dependencies": [],
                "estimated_duration": 300,  # seconds
                "created_at": datetime.datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None
            },
            {
                "id": str(uuid.uuid4()),
                "description": f"Research information for: {goal_description}",
                "status": "pending",
                "dependencies": [0],  # Depends on the first task
                "estimated_duration": 600,  # seconds
                "created_at": datetime.datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None
            },
            {
                "id": str(uuid.uuid4()),
                "description": f"Execute actions for: {goal_description}",
                "status": "pending",
                "dependencies": [1],  # Depends on the second task
                "estimated_duration": 900,  # seconds
                "created_at": datetime.datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None
            },
            {
                "id": str(uuid.uuid4()),
                "description": f"Verify completion of: {goal_description}",
                "status": "pending",
                "dependencies": [2],  # Depends on the third task
                "estimated_duration": 300,  # seconds
                "created_at": datetime.datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None
            }
        ]
        
        return tasks
    
    def get_next_task(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the next task to execute from the plan.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            task: The next task to execute, or None if no tasks are ready
        """
        if plan_id not in self.plans:
            logger.error(f"Plan {plan_id} not found")
            return None
        
        plan = self.plans[plan_id]
        
        # If plan is not active, return None
        if plan["status"] != "active":
            return None
        
        # Find the next task that is ready to execute
        for i, task in enumerate(plan["tasks"]):
            if task["status"] == "pending":
                # Check if all dependencies are completed
                dependencies_met = True
                for dep_idx in task["dependencies"]:
                    if dep_idx >= len(plan["tasks"]) or plan["tasks"][dep_idx]["status"] != "completed":
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    # Update task status
                    task["status"] = "in_progress"
                    task["started_at"] = datetime.datetime.now().isoformat()
                    plan["current_task_index"] = i
                    return task
        
        # If we get here, no tasks are ready
        return None
    
    def update_task_status(self, plan_id: str, task_id: str, status: str, 
                          result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a task in the plan.
        
        Args:
            plan_id: ID of the plan
            task_id: ID of the task
            status: New status of the task
            result: Optional result of the task
            
        Returns:
            success: Whether the update was successful
        """
        if plan_id not in self.plans:
            logger.error(f"Plan {plan_id} not found")
            return False
        
        plan = self.plans[plan_id]
        
        # Find the task
        task_found = False
        for task in plan["tasks"]:
            if task["id"] == task_id:
                task["status"] = status
                
                if status == "completed":
                    task["completed_at"] = datetime.datetime.now().isoformat()
                    if result:
                        task["result"] = result
                
                task_found = True
                break
        
        if not task_found:
            logger.error(f"Task {task_id} not found in plan {plan_id}")
            return False
        
        # Update plan progress
        completed_tasks = sum(1 for task in plan["tasks"] if task["status"] == "completed")
        plan["progress"] = completed_tasks / len(plan["tasks"])
        
        # Check if plan is completed
        if all(task["status"] == "completed" for task in plan["tasks"]):
            plan["status"] = "completed"
            plan["completed_at"] = datetime.datetime.now().isoformat()
            logger.info(f"Plan {plan_id} completed")
        
        return True
    
    def adapt_plan(self, plan_id: str, new_information: Dict[str, Any]) -> bool:
        """
        Adapt the plan based on new information.
        
        Args:
            plan_id: ID of the plan
            new_information: New information to consider
            
        Returns:
            success: Whether the adaptation was successful
        """
        if plan_id not in self.plans:
            logger.error(f"Plan {plan_id} not found")
            return False
        
        plan = self.plans[plan_id]
        
        # In a real implementation, this would use LLM to adapt the plan
        # For now, we'll use a simple placeholder implementation
        logger.info(f"Adapting plan {plan_id} based on new information")
        
        # Example: Add a new task if certain conditions are met
        if "requires_additional_research" in new_information and new_information["requires_additional_research"]:
            new_task = {
                "id": str(uuid.uuid4()),
                "description": f"Additional research: {new_information.get('research_topic', 'unspecified')}",
                "status": "pending",
                "dependencies": [plan["current_task_index"]],
                "estimated_duration": 600,  # seconds
                "created_at": datetime.datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None
            }
            
            # Insert the new task after the current task
            plan["tasks"].insert(plan["current_task_index"] + 1, new_task)
            
            # Update dependencies for subsequent tasks
            for i in range(plan["current_task_index"] + 2, len(plan["tasks"])):
                for j, dep in enumerate(plan["tasks"][i]["dependencies"]):
                    if dep >= plan["current_task_index"] + 1:
                        plan["tasks"][i]["dependencies"][j] += 1
            
            logger.info(f"Added new task to plan {plan_id}")
            return True
        
        return False
    
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """
        Get the status of a plan.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            status: Status of the plan
        """
        if plan_id not in self.plans:
            logger.error(f"Plan {plan_id} not found")
            return {"error": "Plan not found"}
        
        plan = self.plans[plan_id]
        
        return {
            "id": plan["id"],
            "goal_id": plan["goal_id"],
            "status": plan["status"],
            "progress": plan["progress"],
            "current_task_index": plan["current_task_index"],
            "tasks_total": len(plan["tasks"]),
            "tasks_completed": sum(1 for task in plan["tasks"] if task["status"] == "completed"),
            "created_at": plan["created_at"],
            "completed_at": plan["completed_at"]
        }
    
    def save_plans(self, file_path: str) -> bool:
        """
        Save all plans to a file.
        
        Args:
            file_path: Path to save the plans
            
        Returns:
            success: Whether the save was successful
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.plans, f, indent=2)
            logger.info(f"Plans saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving plans: {e}")
            return False
    
    def load_plans(self, file_path: str) -> bool:
        """
        Load plans from a file.
        
        Args:
            file_path: Path to load the plans from
            
        Returns:
            success: Whether the load was successful
        """
        try:
            with open(file_path, 'r') as f:
                self.plans = json.load(f)
            logger.info(f"Plans loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading plans: {e}")
            return False
