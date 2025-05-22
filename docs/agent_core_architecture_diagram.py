#!/usr/bin/env python3
"""
Agent Core Architecture Diagram with HuggingGPT Integration
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL
from diagrams.custom import Custom

# Set graph attributes
graph_attr = {
    "fontsize": "30",
    "bgcolor": "white",
    "rankdir": "TB",
    "pad": "0.5",
    "splines": "ortho",
    "nodesep": "0.8",
    "ranksep": "1.0"
}

# Create diagram
with Diagram("Autonomous Agent Architecture with HuggingGPT Integration", 
             show=True, 
             filename="agent_core_architecture_diagram",
             outformat="png", 
             graph_attr=graph_attr):

    # External Systems
    with Cluster("External Systems"):
        webapp = Python("Web Application")
        doc_processor = Python("Document Processor")
        weaviate = Custom("Weaviate", "./icons/database.png")
        llm_factory = Python("LLM Provider Factory")
    
    # Agent Core
    with Cluster("Agent Core"):
        agent_core = Python("Agent Core")
        
        # Original Agent Components
        with Cluster("Original Agent Components"):
            planner = Python("Planning Module")
            memory = Python("Memory Module")
            decision = Python("Decision Module")
            monitor = Python("Monitoring Module")
        
        # HuggingGPT Integration
        with Cluster("HuggingGPT Integration"):
            model_orchestrator = Python("Model Orchestrator")
            
            with Cluster("Specialized Models"):
                vision_model = Custom("Vision Model", "./icons/vision.png")
                code_model = Custom("Code Model", "./icons/code.png")
                math_model = Custom("Math Model", "./icons/math.png")
                audio_model = Custom("Audio Model", "./icons/audio.png")
    
    # Hierarchical Planning
    with Cluster("Hierarchical Planning"):
        strategic_planner = Python("Strategic Planner")
        tactical_planner = Python("Tactical Planner")
        
    # Multimodal Memory
    with Cluster("Multimodal Memory"):
        text_memory = SQL("Text Memory")
        image_memory = Custom("Image Memory", "./icons/image.png")
        code_memory = Custom("Code Memory", "./icons/code_memory.png")
        
    # Enhanced Decision System
    with Cluster("Enhanced Decision System"):
        strategy_decision = Python("Strategy Decision")
        model_selection = Python("Model Selection")
        
    # Comprehensive Monitoring
    with Cluster("Comprehensive Monitoring"):
        system_monitor = Python("System Monitor")
        model_monitor = Python("Model Monitor")
    
    # Define connections
    
    # External connections
    webapp >> agent_core
    agent_core >> doc_processor
    agent_core >> weaviate
    agent_core >> llm_factory
    
    # Core to components
    agent_core >> planner
    agent_core >> memory
    agent_core >> decision
    agent_core >> monitor
    agent_core >> model_orchestrator
    
    # Model orchestrator to models
    model_orchestrator >> vision_model
    model_orchestrator >> code_model
    model_orchestrator >> math_model
    model_orchestrator >> audio_model
    
    # Hierarchical planning connections
    planner >> strategic_planner
    planner >> tactical_planner
    strategic_planner >> tactical_planner
    
    # Memory connections
    memory >> text_memory
    memory >> image_memory
    memory >> code_memory
    
    # Decision connections
    decision >> strategy_decision
    decision >> model_selection
    strategy_decision >> model_selection
    
    # Monitoring connections
    monitor >> system_monitor
    monitor >> model_monitor
    
    # Integration connections
    model_orchestrator >> tactical_planner
    model_selection >> model_orchestrator
    model_monitor >> model_orchestrator
    
    # Data flow connections
    vision_model >> image_memory
    code_model >> code_memory
    math_model >> text_memory
    audio_model >> text_memory
