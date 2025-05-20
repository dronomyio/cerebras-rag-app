#!/usr/bin/env python3
"""
Agent Core Integration - Connects the autonomous agent with the RAG application

This module provides the integration layer between the autonomous agent core
and the existing RAG application components.
"""

import os
import sys
import json
import logging
import datetime
import threading
import time
from typing import Dict, List, Any, Optional, Tuple

# Import agent core modules
from agent_core.agent import Agent
from agent_core.planning.planner import PlanningModule
from agent_core.memory.memory import MemoryModule
from agent_core.decision.decision import DecisionModule
from agent_core.monitoring.monitor import MonitoringModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentIntegration:
    """
    Integration layer between the autonomous agent and the RAG application.
    
    This class handles the communication between the agent core and the
    existing RAG components (document processor, webapp, code executor).
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the agent integration with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize agent
        self.agent = self._init_agent()
        
        # Initialize RAG connectors
        self.document_processor = None
        self.webapp = None
        self.code_executor = None
        
        # Initialize connectors
        self._init_connectors()
        
        # Agent control
        self.running = False
        self.agent_thread = None
        
        logger.info("Agent integration initialized")
    
    def _init_agent(self) -> Agent:
        """
        Initialize the agent with its components.
        
        Returns:
            agent: Initialized agent
        """
        agent_config = self.config.get("agent", {})
        agent = Agent(agent_config)
        
        # Initialize planning module
        planning_config = self.config.get("planning", {})
        agent.planning = PlanningModule(planning_config)
        
        # Initialize memory module
        memory_config = self.config.get("memory", {})
        agent.memory = MemoryModule(memory_config)
        
        # Initialize decision module
        decision_config = self.config.get("decision", {})
        agent.decision = DecisionModule(decision_config)
        
        # Initialize monitoring module
        monitoring_config = self.config.get("monitoring", {})
        agent.monitor = MonitoringModule(monitoring_config)
        
        logger.info("Agent initialized with all components")
        return agent
    
    def _init_connectors(self) -> None:
        """Initialize connectors to RAG components."""
        # These would be actual connectors to the RAG components
        # For now, we'll use placeholder implementations
        
        # Document processor connector
        self.document_processor = DocumentProcessorConnector(
            self.config.get("document_processor", {})
        )
        
        # Webapp connector
        self.webapp = WebappConnector(
            self.config.get("webapp", {})
        )
        
        # Code executor connector
        self.code_executor = CodeExecutorConnector(
            self.config.get("code_executor", {})
        )
        
        logger.info("RAG connectors initialized")
    
    def start(self) -> None:
        """Start the agent in a separate thread."""
        if self.running:
            logger.warning("Agent is already running")
            return
        
        self.running = True
        self.agent_thread = threading.Thread(target=self._agent_loop)
        self.agent_thread.daemon = True
        self.agent_thread.start()
        
        logger.info("Agent started")
    
    def stop(self) -> None:
        """Stop the agent."""
        if not self.running:
            logger.warning("Agent is not running")
            return
        
        self.running = False
        if self.agent_thread:
            self.agent_thread.join(timeout=5)
        
        logger.info("Agent stopped")
    
    def _agent_loop(self) -> None:
        """Main agent loop."""
        logger.info("Agent loop started")
        
        try:
            while self.running:
                # Execute agent cycle
                cycle_result = self.agent.execute_cycle()
                
                # Process cycle result
                self._process_cycle_result(cycle_result)
                
                # Check for new goals from webapp
                self._check_for_new_goals()
                
                # Sleep to avoid high CPU usage
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error in agent loop: {e}")
            self.running = False
        
        logger.info("Agent loop stopped")
    
    def _process_cycle_result(self, cycle_result: Dict[str, Any]) -> None:
        """
        Process the result of an agent cycle.
        
        Args:
            cycle_result: Result of the agent cycle
        """
        # Extract actions from the cycle result
        actions = cycle_result.get("act", {}).get("actions_taken", [])
        
        # Process each action
        for action in actions:
            self._execute_action(action)
    
    def _execute_action(self, action: Dict[str, Any]) -> None:
        """
        Execute an action using the appropriate connector.
        
        Args:
            action: Action to execute
        """
        action_type = action.get("action")
        parameters = action.get("parameters", {})
        
        try:
            if action_type == "search_documents":
                # Use document processor connector
                result = self.document_processor.search(
                    parameters.get("query", ""),
                    parameters.get("limit", 5)
                )
                
                # Store result in memory
                self.agent.memory.store_working_memory(
                    f"search_result_{datetime.datetime.now().isoformat()}",
                    result
                )
            
            elif action_type == "execute_code":
                # Use code executor connector
                result = self.code_executor.execute(
                    parameters.get("code", ""),
                    parameters.get("language", "python")
                )
                
                # Store result in memory
                self.agent.memory.store_working_memory(
                    f"code_result_{datetime.datetime.now().isoformat()}",
                    result
                )
            
            elif action_type == "generate_response":
                # Use webapp connector to generate response
                result = self.webapp.generate_response(
                    parameters.get("prompt", ""),
                    parameters.get("context", []),
                    parameters.get("max_tokens", 500)
                )
                
                # Store result in memory
                self.agent.memory.store_working_memory(
                    f"response_{datetime.datetime.now().isoformat()}",
                    result
                )
            
            # Other action types would be handled here
            
            logger.info(f"Executed action: {action_type}")
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            
            # Track error
            self.agent.monitor.track_error({
                "type": "action_execution",
                "action": action_type,
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            })
    
    def _check_for_new_goals(self) -> None:
        """Check for new goals from the webapp."""
        try:
            # Get new goals from webapp
            new_goals = self.webapp.get_new_goals()
            
            # Process each new goal
            for goal in new_goals:
                goal_id = self.agent.set_goal(
                    goal.get("description", ""),
                    goal.get("priority", 1),
                    goal.get("deadline")
                )
                
                logger.info(f"Added new goal from webapp: {goal_id}")
        except Exception as e:
            logger.error(f"Error checking for new goals: {e}")
    
    def set_goal(self, description: str, priority: int = 1, 
                deadline: Optional[str] = None) -> str:
        """
        Set a new goal for the agent.
        
        Args:
            description: Description of the goal
            priority: Priority level (1-5, with 5 being highest)
            deadline: Optional deadline in ISO format
            
        Returns:
            goal_id: ID of the created goal
        """
        return self.agent.set_goal(description, priority, deadline)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            status: Agent status
        """
        # Get performance summary
        performance = self.agent.monitor.get_performance_summary()
        
        # Get current goals
        goals = self.agent.state.get("current_goals", [])
        
        # Get current plan
        plan = self.agent.state.get("current_plan")
        
        return {
            "running": self.running,
            "goals": {
                "total": len(goals),
                "active": sum(1 for g in goals if g.get("status") == "active"),
                "completed": sum(1 for g in goals if g.get("status") == "completed"),
                "failed": sum(1 for g in goals if g.get("status") == "failed")
            },
            "performance": performance,
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def save_state(self, directory: str) -> bool:
        """
        Save the agent state to files.
        
        Args:
            directory: Directory to save the state
            
        Returns:
            success: Whether the save was successful
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save agent state
            self.agent.save_state(os.path.join(directory, "agent_state.json"))
            
            # Save planning state
            self.agent.planning.save_plans(os.path.join(directory, "plans.json"))
            
            # Save memory
            self.agent.memory.save_memory(directory)
            
            # Save decision history
            self.agent.decision.save_action_history(os.path.join(directory, "action_history.json"))
            
            # Save monitoring data
            self.agent.monitor.save_metrics(os.path.join(directory, "metrics.json"))
            self.agent.monitor.save_errors(os.path.join(directory, "errors.json"))
            self.agent.monitor.save_reflections(os.path.join(directory, "reflections.json"))
            
            logger.info(f"Agent state saved to {directory}")
            return True
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
            return False
    
    def load_state(self, directory: str) -> bool:
        """
        Load the agent state from files.
        
        Args:
            directory: Directory to load the state from
            
        Returns:
            success: Whether the load was successful
        """
        try:
            # Load agent state
            self.agent.load_state(os.path.join(directory, "agent_state.json"))
            
            # Load planning state
            self.agent.planning.load_plans(os.path.join(directory, "plans.json"))
            
            # Load memory
            self.agent.memory.load_memory(directory)
            
            # Load decision history
            self.agent.decision.load_action_history(os.path.join(directory, "action_history.json"))
            
            # Load monitoring data
            self.agent.monitor.load_metrics(os.path.join(directory, "metrics.json"))
            self.agent.monitor.load_errors(os.path.join(directory, "errors.json"))
            self.agent.monitor.load_reflections(os.path.join(directory, "reflections.json"))
            
            logger.info(f"Agent state loaded from {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading agent state: {e}")
            return False


class DocumentProcessorConnector:
    """Connector to the document processor component."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document processor connector.
        
        Args:
            config: Configuration for the connector
        """
        self.config = config
        logger.info("Document processor connector initialized")
    
    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            results: Search results
        """
        # In a real implementation, this would connect to the document processor
        # For now, we'll return a placeholder result
        
        logger.info(f"Searching for: {query} (limit: {limit})")
        
        return {
            "query": query,
            "limit": limit,
            "results": [
                {
                    "id": "doc1",
                    "title": "Sample Document 1",
                    "content": f"This is a sample document that matches the query: {query}",
                    "relevance": 0.95
                },
                {
                    "id": "doc2",
                    "title": "Sample Document 2",
                    "content": f"Another sample document for: {query}",
                    "relevance": 0.85
                }
            ],
            "timestamp": datetime.datetime.now().isoformat()
        }


class WebappConnector:
    """Connector to the webapp component."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the webapp connector.
        
        Args:
            config: Configuration for the connector
        """
        self.config = config
        self.new_goals_queue = []  # Queue of new goals from users
        logger.info("Webapp connector initialized")
    
    def generate_response(self, prompt: str, context: List[Dict[str, Any]], 
                         max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: Prompt for the LLM
            context: Context information
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            response: Generated response
        """
        # In a real implementation, this would connect to the webapp's LLM interface
        # For now, we'll return a placeholder result
        
        logger.info(f"Generating response for: {prompt[:50]}...")
        
        return {
            "prompt": prompt,
            "context_length": len(context),
            "response": f"This is a sample response to: {prompt}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_new_goals(self) -> List[Dict[str, Any]]:
        """
        Get new goals from users.
        
        Returns:
            goals: List of new goals
        """
        # In a real implementation, this would check for new goals from the webapp
        # For now, we'll return the queue and clear it
        
        goals = self.new_goals_queue
        self.new_goals_queue = []
        
        return goals
    
    def add_goal(self, description: str, priority: int = 1, 
                deadline: Optional[str] = None) -> None:
        """
        Add a new goal from a user.
        
        Args:
            description: Description of the goal
            priority: Priority level (1-5, with 5 being highest)
            deadline: Optional deadline in ISO format
        """
        self.new_goals_queue.append({
            "description": description,
            "priority": priority,
            "deadline": deadline,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        logger.info(f"Added new goal to queue: {description}")


class CodeExecutorConnector:
    """Connector to the code executor component."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the code executor connector.
        
        Args:
            config: Configuration for the connector
        """
        self.config = config
        logger.info("Code executor connector initialized")
    
    def execute(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code in a sandbox environment.
        
        Args:
            code: Code to execute
            language: Programming language
            
        Returns:
            result: Execution result
        """
        # In a real implementation, this would connect to the code executor
        # For now, we'll return a placeholder result
        
        logger.info(f"Executing {language} code: {code[:50]}...")
        
        return {
            "code": code,
            "language": language,
            "output": f"Sample output from executing {language} code",
            "error": None,
            "execution_time": 0.5,  # seconds
            "timestamp": datetime.datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # This would be the actual configuration file path
    config_path = "config/agent_config.json"
    
    # For testing, we'll use a dummy configuration
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        json.dump({
            "agent": {
                "name": "Cerebras RAG Agent"
            },
            "planning": {},
            "memory": {
                "working_memory_capacity": 100,
                "episodic_memory_capacity": 1000
            },
            "decision": {},
            "monitoring": {},
            "document_processor": {},
            "webapp": {},
            "code_executor": {}
        }, f)
        config_path = f.name
    
    # Initialize agent integration
    integration = AgentIntegration(config_path)
    
    # Set a goal
    goal_id = integration.set_goal("Research financial engineering concepts in Ruppert's book")
    print(f"Set goal with ID: {goal_id}")
    
    # Start the agent
    integration.start()
    
    # Run for a while
    try:
        for i in range(10):
            print(f"Agent running... ({i+1}/10)")
            time.sleep(1)
            
            # Get status
            status = integration.get_agent_status()
            print(f"Status: {json.dumps(status, indent=2)}")
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop the agent
        integration.stop()
        
        # Save state
        integration.save_state("agent_state")
        
        # Clean up the temporary config file
        os.unlink(config_path)
    
    print("Done")
