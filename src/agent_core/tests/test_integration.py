#!/usr/bin/env python3
"""
Agent Integration Test - Tests for the autonomous agent integration

This module provides tests for the integration between the autonomous agent
and the RAG application components.
"""

import os
import sys
import json
import logging
import unittest
import tempfile
from unittest.mock import MagicMock, patch

# Add parent directory to path to import agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent modules
from agent_core.integration import AgentIntegration
from agent_core.config import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestAgentIntegration(unittest.TestCase):
    """Test cases for agent integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file
        self.config_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        json.dump(DEFAULT_CONFIG, self.config_file)
        self.config_file.close()
        
        # Create agent integration with the config
        self.integration = AgentIntegration(self.config_file.name)
        
        # Mock the connectors
        self.integration.document_processor = MagicMock()
        self.integration.webapp = MagicMock()
        self.integration.code_executor = MagicMock()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the agent if running
        if self.integration.running:
            self.integration.stop()
        
        # Remove the temporary config file
        os.unlink(self.config_file.name)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        # Check that agent and components are initialized
        self.assertIsNotNone(self.integration.agent)
        self.assertIsNotNone(self.integration.agent.planning)
        self.assertIsNotNone(self.integration.agent.memory)
        self.assertIsNotNone(self.integration.agent.decision)
        self.assertIsNotNone(self.integration.agent.monitor)
        
        # Check that connectors are initialized
        self.assertIsNotNone(self.integration.document_processor)
        self.assertIsNotNone(self.integration.webapp)
        self.assertIsNotNone(self.integration.code_executor)
    
    def test_set_goal(self):
        """Test setting a goal."""
        # Set a goal
        goal_id = self.integration.set_goal("Test goal")
        
        # Check that the goal was added
        self.assertIsNotNone(goal_id)
        
        # Check that the goal is in the agent's state
        goals = self.integration.agent.state.get("current_goals", [])
        self.assertTrue(any(g.get("id") == goal_id for g in goals))
    
    def test_agent_start_stop(self):
        """Test starting and stopping the agent."""
        # Start the agent
        self.integration.start()
        
        # Check that the agent is running
        self.assertTrue(self.integration.running)
        self.assertIsNotNone(self.integration.agent_thread)
        
        # Stop the agent
        self.integration.stop()
        
        # Check that the agent is stopped
        self.assertFalse(self.integration.running)
    
    def test_execute_search_action(self):
        """Test executing a search action."""
        # Mock the search method
        self.integration.document_processor.search.return_value = {
            "query": "test",
            "results": [{"id": "doc1", "content": "Test content"}]
        }
        
        # Create a search action
        action = {
            "action": "search_documents",
            "parameters": {
                "query": "test",
                "limit": 5
            }
        }
        
        # Execute the action
        self.integration._execute_action(action)
        
        # Check that the search method was called
        self.integration.document_processor.search.assert_called_once_with("test", 5)
    
    def test_execute_code_action(self):
        """Test executing a code action."""
        # Mock the execute method
        self.integration.code_executor.execute.return_value = {
            "output": "Test output",
            "error": None
        }
        
        # Create a code action
        action = {
            "action": "execute_code",
            "parameters": {
                "code": "print('test')",
                "language": "python"
            }
        }
        
        # Execute the action
        self.integration._execute_action(action)
        
        # Check that the execute method was called
        self.integration.code_executor.execute.assert_called_once_with("print('test')", "python")
    
    def test_execute_generate_response_action(self):
        """Test executing a generate response action."""
        # Mock the generate_response method
        self.integration.webapp.generate_response.return_value = {
            "response": "Test response"
        }
        
        # Create a generate response action
        action = {
            "action": "generate_response",
            "parameters": {
                "prompt": "test prompt",
                "context": [],
                "max_tokens": 500
            }
        }
        
        # Execute the action
        self.integration._execute_action(action)
        
        # Check that the generate_response method was called
        self.integration.webapp.generate_response.assert_called_once_with("test prompt", [], 500)
    
    def test_check_for_new_goals(self):
        """Test checking for new goals."""
        # Mock the get_new_goals method
        self.integration.webapp.get_new_goals.return_value = [
            {
                "description": "Test goal from webapp",
                "priority": 3
            }
        ]
        
        # Check for new goals
        self.integration._check_for_new_goals()
        
        # Check that the get_new_goals method was called
        self.integration.webapp.get_new_goals.assert_called_once()
        
        # Check that the goal was added
        goals = self.integration.agent.state.get("current_goals", [])
        self.assertTrue(any(g.get("description") == "Test goal from webapp" for g in goals))
    
    def test_get_agent_status(self):
        """Test getting agent status."""
        # Get agent status
        status = self.integration.get_agent_status()
        
        # Check that the status contains expected fields
        self.assertIn("running", status)
        self.assertIn("goals", status)
        self.assertIn("performance", status)
        self.assertIn("last_updated", status)
    
    def test_save_load_state(self):
        """Test saving and loading agent state."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set a goal
            goal_id = self.integration.set_goal("Test goal for state saving")
            
            # Save state
            success = self.integration.save_state(temp_dir)
            self.assertTrue(success)
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "agent_state.json")))
            
            # Create a new integration
            new_integration = AgentIntegration(self.config_file.name)
            
            # Load state
            success = new_integration.load_state(temp_dir)
            self.assertTrue(success)
            
            # Check that the goal was loaded
            goals = new_integration.agent.state.get("current_goals", [])
            self.assertTrue(any(g.get("id") == goal_id for g in goals))


if __name__ == "__main__":
    unittest.main()
