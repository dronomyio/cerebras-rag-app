#!/usr/bin/env python3
"""
Agent Configuration - Default configuration for the autonomous agent

This module provides the default configuration for the autonomous agent core
and its integration with the RAG application.
"""

import json

# Default agent configuration
DEFAULT_CONFIG = {
    "agent": {
        "name": "Cerebras RAG Agent",
        "description": "Autonomous agent for the Cerebras RAG application",
        "version": "1.0.0"
    },
    "planning": {
        "max_plan_steps": 10,
        "max_plan_depth": 3,
        "adaptation_threshold": 0.7  # Threshold for plan adaptation
    },
    "memory": {
        "working_memory_capacity": 100,
        "episodic_memory_capacity": 1000,
        "semantic_memory_enabled": True
    },
    "decision": {
        "default_utility_threshold": 0.5,  # Minimum utility for action selection
        "exploration_rate": 0.1  # Rate of exploration vs. exploitation
    },
    "monitoring": {
        "performance_tracking_enabled": True,
        "anomaly_detection_enabled": True,
        "reflection_interval": 100  # Cycles between reflections
    },
    "document_processor": {
        "host": "document-processor",
        "port": 5000,
        "timeout": 30  # seconds
    },
    "webapp": {
        "host": "webapp",
        "port": 5000,
        "timeout": 30  # seconds
    },
    "code_executor": {
        "host": "code-executor",
        "port": 5000,
        "timeout": 30  # seconds
    },
    "weaviate": {
        "host": "weaviate",
        "port": 8080,
        "timeout": 30  # seconds
    },
    "cerebras": {
        "api_key_env": "CEREBRAS_API_KEY",
        "timeout": 60  # seconds
    },
    "unstructured": {
        "enabled": True,  # Can be disabled to use only local processors
        "api_url": "https://api.unstructured.io/general/v0/general",
        "timeout": 60  # seconds
    }
}

def generate_default_config(output_path):
    """
    Generate the default configuration file.
    
    Args:
        output_path: Path to save the configuration
    """
    with open(output_path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    
    print(f"Default configuration saved to {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "config/agent_config.json"
    
    generate_default_config(output_path)
