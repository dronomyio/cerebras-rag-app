#!/usr/bin/env python3
"""
Monitoring Module - Responsible for self-monitoring and performance evaluation

This module provides monitoring capabilities for the autonomous agent,
including performance tracking, error detection, and self-improvement.
"""

import os
import uuid
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringModule:
    """
    Monitoring module for the autonomous agent.
    
    Responsible for tracking performance, detecting errors, and enabling
    self-improvement through reflection and adaptation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the monitoring module with configuration settings.
        
        Args:
            config: Dictionary containing monitoring configuration
        """
        self.config = config
        self.metrics = {}  # Performance metrics
        self.errors = []   # Detected errors
        self.reflections = []  # Self-reflections
        
        # Initialize performance metrics
        self._init_metrics()
        
        logger.info("Monitoring module initialized")
    
    def _init_metrics(self):
        """Initialize performance metrics."""
        self.metrics = {
            "goals": {
                "completed": 0,
                "failed": 0,
                "in_progress": 0,
                "completion_times": []  # in seconds
            },
            "tasks": {
                "completed": 0,
                "failed": 0,
                "in_progress": 0,
                "completion_times": []  # in seconds
            },
            "actions": {
                "successful": 0,
                "failed": 0,
                "types": {}  # Count by action type
            },
            "errors": {
                "count": 0,
                "types": {}  # Count by error type
            },
            "system": {
                "uptime": 0,
                "memory_usage": [],
                "cpu_usage": []
            }
        }
    
    def track_goal(self, goal: Dict[str, Any], status: str) -> None:
        """
        Track a goal's status.
        
        Args:
            goal: Goal object
            status: Status of the goal (completed, failed, in_progress)
        """
        # Update goal count
        if status == "completed":
            self.metrics["goals"]["completed"] += 1
            
            # Calculate completion time if possible
            if "created_at" in goal and "completed_at" in goal and goal["completed_at"]:
                created = datetime.datetime.fromisoformat(goal["created_at"])
                completed = datetime.datetime.fromisoformat(goal["completed_at"])
                completion_time = (completed - created).total_seconds()
                self.metrics["goals"]["completion_times"].append(completion_time)
                
                logger.info(f"Goal {goal['id']} completed in {completion_time:.2f} seconds")
        
        elif status == "failed":
            self.metrics["goals"]["failed"] += 1
            logger.warning(f"Goal {goal['id']} failed")
        
        elif status == "in_progress":
            self.metrics["goals"]["in_progress"] += 1
            logger.info(f"Goal {goal['id']} in progress")
    
    def track_task(self, task: Dict[str, Any], status: str) -> None:
        """
        Track a task's status.
        
        Args:
            task: Task object
            status: Status of the task (completed, failed, in_progress)
        """
        # Update task count
        if status == "completed":
            self.metrics["tasks"]["completed"] += 1
            
            # Calculate completion time if possible
            if "created_at" in task and "completed_at" in task and task["completed_at"]:
                created = datetime.datetime.fromisoformat(task["created_at"])
                completed = datetime.datetime.fromisoformat(task["completed_at"])
                completion_time = (completed - created).total_seconds()
                self.metrics["tasks"]["completion_times"].append(completion_time)
                
                logger.info(f"Task {task['id']} completed in {completion_time:.2f} seconds")
        
        elif status == "failed":
            self.metrics["tasks"]["failed"] += 1
            logger.warning(f"Task {task['id']} failed")
        
        elif status == "in_progress":
            self.metrics["tasks"]["in_progress"] += 1
            logger.info(f"Task {task['id']} in progress")
    
    def track_action(self, action: Dict[str, Any], success: bool) -> None:
        """
        Track an action's outcome.
        
        Args:
            action: Action object
            success: Whether the action was successful
        """
        action_type = action.get("action", "unknown")
        
        # Update action count
        if success:
            self.metrics["actions"]["successful"] += 1
        else:
            self.metrics["actions"]["failed"] += 1
        
        # Update action type count
        if action_type not in self.metrics["actions"]["types"]:
            self.metrics["actions"]["types"][action_type] = {
                "successful": 0,
                "failed": 0
            }
        
        if success:
            self.metrics["actions"]["types"][action_type]["successful"] += 1
        else:
            self.metrics["actions"]["types"][action_type]["failed"] += 1
        
        logger.debug(f"Action {action_type} {'succeeded' if success else 'failed'}")
    
    def track_error(self, error: Dict[str, Any]) -> None:
        """
        Track an error.
        
        Args:
            error: Error object with type, message, etc.
        """
        error_type = error.get("type", "unknown")
        
        # Add timestamp if not present
        if "timestamp" not in error:
            error["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add to errors list
        self.errors.append(error)
        
        # Update error count
        self.metrics["errors"]["count"] += 1
        
        # Update error type count
        if error_type not in self.metrics["errors"]["types"]:
            self.metrics["errors"]["types"][error_type] = 0
        
        self.metrics["errors"]["types"][error_type] += 1
        
        logger.error(f"Error detected: {error.get('message', 'No message')} (Type: {error_type})")
    
    def track_system_metrics(self, memory_usage: float, cpu_usage: float) -> None:
        """
        Track system metrics.
        
        Args:
            memory_usage: Memory usage in MB
            cpu_usage: CPU usage percentage
        """
        # Update system metrics
        self.metrics["system"]["uptime"] += 1  # Assuming called once per second
        self.metrics["system"]["memory_usage"].append(memory_usage)
        self.metrics["system"]["cpu_usage"].append(cpu_usage)
        
        # Keep only the last 3600 samples (1 hour at 1 sample per second)
        if len(self.metrics["system"]["memory_usage"]) > 3600:
            self.metrics["system"]["memory_usage"] = self.metrics["system"]["memory_usage"][-3600:]
        
        if len(self.metrics["system"]["cpu_usage"]) > 3600:
            self.metrics["system"]["cpu_usage"] = self.metrics["system"]["cpu_usage"][-3600:]
    
    def reflect(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform self-reflection based on current state and metrics.
        
        Args:
            state: Current state of the agent
            
        Returns:
            reflection: Self-reflection results
        """
        # In a real implementation, this would use more sophisticated reflection
        # For now, we'll use a simple rule-based approach
        
        reflection = {
            "timestamp": datetime.datetime.now().isoformat(),
            "observations": [],
            "improvements": []
        }
        
        # Check goal completion rate
        total_goals = self.metrics["goals"]["completed"] + self.metrics["goals"]["failed"]
        if total_goals > 0:
            completion_rate = self.metrics["goals"]["completed"] / total_goals
            reflection["observations"].append(f"Goal completion rate: {completion_rate:.2%}")
            
            if completion_rate < 0.5:
                reflection["improvements"].append("Improve goal completion rate by breaking goals into smaller, more manageable tasks")
        
        # Check task completion rate
        total_tasks = self.metrics["tasks"]["completed"] + self.metrics["tasks"]["failed"]
        if total_tasks > 0:
            completion_rate = self.metrics["tasks"]["completed"] / total_tasks
            reflection["observations"].append(f"Task completion rate: {completion_rate:.2%}")
            
            if completion_rate < 0.7:
                reflection["improvements"].append("Improve task completion rate by better task planning and execution")
        
        # Check action success rate
        total_actions = self.metrics["actions"]["successful"] + self.metrics["actions"]["failed"]
        if total_actions > 0:
            success_rate = self.metrics["actions"]["successful"] / total_actions
            reflection["observations"].append(f"Action success rate: {success_rate:.2%}")
            
            if success_rate < 0.8:
                reflection["improvements"].append("Improve action selection to increase success rate")
        
        # Check for frequent errors
        if self.metrics["errors"]["count"] > 0:
            most_common_error = max(self.metrics["errors"]["types"].items(), key=lambda x: x[1])
            reflection["observations"].append(f"Most common error: {most_common_error[0]} ({most_common_error[1]} occurrences)")
            
            reflection["improvements"].append(f"Implement better handling for {most_common_error[0]} errors")
        
        # Check for action type distribution
        if self.metrics["actions"]["types"]:
            action_counts = [(action_type, data["successful"] + data["failed"]) 
                            for action_type, data in self.metrics["actions"]["types"].items()]
            most_common_action = max(action_counts, key=lambda x: x[1])
            reflection["observations"].append(f"Most common action: {most_common_action[0]} ({most_common_action[1]} occurrences)")
            
            # Check if one action is dominating
            total_actions = sum(count for _, count in action_counts)
            if most_common_action[1] / total_actions > 0.8:
                reflection["improvements"].append("Diversify action selection to avoid over-reliance on a single action type")
        
        # Add reflection to history
        self.reflections.append(reflection)
        
        logger.info(f"Self-reflection completed with {len(reflection['improvements'])} improvement suggestions")
        
        return reflection
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            summary: Performance summary
        """
        summary = {
            "goals": {
                "completed": self.metrics["goals"]["completed"],
                "failed": self.metrics["goals"]["failed"],
                "in_progress": self.metrics["goals"]["in_progress"],
                "average_completion_time": None
            },
            "tasks": {
                "completed": self.metrics["tasks"]["completed"],
                "failed": self.metrics["tasks"]["failed"],
                "in_progress": self.metrics["tasks"]["in_progress"],
                "average_completion_time": None
            },
            "actions": {
                "successful": self.metrics["actions"]["successful"],
                "failed": self.metrics["actions"]["failed"],
                "success_rate": None
            },
            "errors": {
                "count": self.metrics["errors"]["count"],
                "most_common": None
            },
            "system": {
                "uptime": self.metrics["system"]["uptime"],
                "average_memory_usage": None,
                "average_cpu_usage": None
            }
        }
        
        # Calculate averages
        if self.metrics["goals"]["completion_times"]:
            summary["goals"]["average_completion_time"] = statistics.mean(self.metrics["goals"]["completion_times"])
        
        if self.metrics["tasks"]["completion_times"]:
            summary["tasks"]["average_completion_time"] = statistics.mean(self.metrics["tasks"]["completion_times"])
        
        total_actions = summary["actions"]["successful"] + summary["actions"]["failed"]
        if total_actions > 0:
            summary["actions"]["success_rate"] = summary["actions"]["successful"] / total_actions
        
        if self.metrics["errors"]["types"]:
            summary["errors"]["most_common"] = max(self.metrics["errors"]["types"].items(), key=lambda x: x[1])[0]
        
        if self.metrics["system"]["memory_usage"]:
            summary["system"]["average_memory_usage"] = statistics.mean(self.metrics["system"]["memory_usage"])
        
        if self.metrics["system"]["cpu_usage"]:
            summary["system"]["average_cpu_usage"] = statistics.mean(self.metrics["system"]["cpu_usage"])
        
        return summary
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in performance metrics.
        
        Returns:
            anomalies: List of detected anomalies
        """
        anomalies = []
        
        # Check for anomalies in goal completion time
        if len(self.metrics["goals"]["completion_times"]) >= 5:
            mean = statistics.mean(self.metrics["goals"]["completion_times"])
            stdev = statistics.stdev(self.metrics["goals"]["completion_times"])
            
            for i, time in enumerate(self.metrics["goals"]["completion_times"][-5:]):
                if abs(time - mean) > 2 * stdev:
                    anomalies.append({
                        "type": "goal_completion_time",
                        "value": time,
                        "mean": mean,
                        "stdev": stdev,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
        
        # Check for anomalies in task completion time
        if len(self.metrics["tasks"]["completion_times"]) >= 5:
            mean = statistics.mean(self.metrics["tasks"]["completion_times"])
            stdev = statistics.stdev(self.metrics["tasks"]["completion_times"])
            
            for i, time in enumerate(self.metrics["tasks"]["completion_times"][-5:]):
                if abs(time - mean) > 2 * stdev:
                    anomalies.append({
                        "type": "task_completion_time",
                        "value": time,
                        "mean": mean,
                        "stdev": stdev,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
        
        # Check for sudden increase in error rate
        if len(self.errors) >= 10:
            recent_errors = len([e for e in self.errors if 
                               datetime.datetime.fromisoformat(e["timestamp"]) > 
                               datetime.datetime.now() - datetime.timedelta(minutes=5)])
            
            if recent_errors >= 5:
                anomalies.append({
                    "type": "error_rate",
                    "value": recent_errors,
                    "threshold": 5,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        # Check for system resource anomalies
        if len(self.metrics["system"]["memory_usage"]) >= 60:
            recent_memory = self.metrics["system"]["memory_usage"][-60:]
            mean = statistics.mean(recent_memory)
            
            if recent_memory[-1] > mean * 1.5:
                anomalies.append({
                    "type": "memory_usage",
                    "value": recent_memory[-1],
                    "mean": mean,
                    "threshold": mean * 1.5,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        if len(self.metrics["system"]["cpu_usage"]) >= 60:
            recent_cpu = self.metrics["system"]["cpu_usage"][-60:]
            mean = statistics.mean(recent_cpu)
            
            if recent_cpu[-1] > mean * 1.5:
                anomalies.append({
                    "type": "cpu_usage",
                    "value": recent_cpu[-1],
                    "mean": mean,
                    "threshold": mean * 1.5,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        return anomalies
    
    def save_metrics(self, file_path: str) -> bool:
        """
        Save metrics to a file.
        
        Args:
            file_path: Path to save the metrics
            
        Returns:
            success: Whether the save was successful
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Metrics saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False
    
    def load_metrics(self, file_path: str) -> bool:
        """
        Load metrics from a file.
        
        Args:
            file_path: Path to load the metrics from
            
        Returns:
            success: Whether the load was successful
        """
        try:
            with open(file_path, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"Metrics loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return False
    
    def save_errors(self, file_path: str) -> bool:
        """
        Save errors to a file.
        
        Args:
            file_path: Path to save the errors
            
        Returns:
            success: Whether the save was successful
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.errors, f, indent=2)
            logger.info(f"Errors saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving errors: {e}")
            return False
    
    def load_errors(self, file_path: str) -> bool:
        """
        Load errors from a file.
        
        Args:
            file_path: Path to load the errors from
            
        Returns:
            success: Whether the load was successful
        """
        try:
            with open(file_path, 'r') as f:
                self.errors = json.load(f)
            logger.info(f"Errors loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading errors: {e}")
            return False
    
    def save_reflections(self, file_path: str) -> bool:
        """
        Save reflections to a file.
        
        Args:
            file_path: Path to save the reflections
            
        Returns:
            success: Whether the save was successful
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.reflections, f, indent=2)
            logger.info(f"Reflections saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving reflections: {e}")
            return False
    
    def load_reflections(self, file_path: str) -> bool:
        """
        Load reflections from a file.
        
        Args:
            file_path: Path to load the reflections from
            
        Returns:
            success: Whether the load was successful
        """
        try:
            with open(file_path, 'r') as f:
                self.reflections = json.load(f)
            logger.info(f"Reflections loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading reflections: {e}")
            return False
