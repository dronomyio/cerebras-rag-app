#!/usr/bin/env python3
"""
Final Architecture Diagram - Updated diagram including autonomous agent

This script generates an updated architecture diagram for the Cerebras RAG application
that includes the new autonomous agent components.
"""

import os
from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.analytics import Spark
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.network import Nginx
from diagrams.onprem.queue import Kafka
from diagrams.programming.framework import Flask
from diagrams.programming.language import Python
from diagrams.generic.compute import Rack
from diagrams.generic.storage import Storage
from diagrams.generic.os import Ubuntu
from diagrams.generic.place import Datacenter
from diagrams.aws.compute import Lambda
from diagrams.custom import Custom

# Set diagram output directory
output_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(output_dir, "autonomous_agent_architecture")

# Create the diagram
with Diagram("Cerebras RAG with Autonomous Agent Architecture", filename=output_file, show=False):
    
    # External user
    user = Custom("User", "./icons/user.png")
    
    # Entry point
    with Cluster("Frontend"):
        nginx = Nginx("NGINX")
        
    # Web Application
    with Cluster("Web Application"):
        webapp = Flask("Web App")
        auth = Python("Authentication")
        chat_ui = Python("Chat Interface")
        upload_ui = Python("Document Upload")
    
    # Agent Core
    with Cluster("Autonomous Agent Core"):
        agent = Custom("Agent Core", "./icons/agent.png")
        
        with Cluster("Agent Modules"):
            planning = Python("Planning Module")
            memory = Python("Memory Module")
            decision = Python("Decision Module")
            monitoring = Python("Monitoring Module")
        
        agent_integration = Python("Agent Integration")
    
    # Document Processing
    with Cluster("Document Processing"):
        doc_processor = Python("Document Processor")
        
        with Cluster("Processor Plugins"):
            pdf_plugin = Python("PDF Processor")
            docx_plugin = Python("DOCX Processor")
            text_plugin = Python("Text Processor")
        
        with Cluster("External Services"):
            unstructured = Custom("Unstructured.io", "./icons/api.png")
    
    # Vector Database
    with Cluster("Vector Storage"):
        weaviate = Custom("Weaviate", "./icons/database.png")
        t2v = Custom("Text2Vec", "./icons/embedding.png")
    
    # Code Execution
    with Cluster("Code Execution"):
        code_executor = Python("Code Executor")
        docker = Custom("Docker", "./icons/docker.png")
        r_runtime = Custom("R Runtime", "./icons/r.png")
        py_runtime = Python("Python Runtime")
    
    # Session Storage
    with Cluster("Session Storage"):
        redis = Redis("Redis")
    
    # Inference
    with Cluster("Inference"):
        cerebras = Custom("Cerebras API", "./icons/cerebras.png")
    
    # Flow connections
    
    # User interactions
    user >> Edge(label="HTTP") >> nginx
    nginx >> Edge(label="Proxy") >> webapp
    
    # Web app components
    webapp >> auth
    webapp >> chat_ui
    webapp >> upload_ui
    
    # Agent connections
    webapp >> agent_integration
    agent_integration >> agent
    
    # Agent modules
    agent >> planning
    agent >> memory
    agent >> decision
    agent >> monitoring
    
    # Document processing
    agent_integration >> doc_processor
    doc_processor >> pdf_plugin
    doc_processor >> docx_plugin
    doc_processor >> text_plugin
    doc_processor >> unstructured
    
    # Vector storage
    doc_processor >> weaviate
    weaviate >> t2v
    agent_integration >> weaviate
    
    # Code execution
    agent_integration >> code_executor
    code_executor >> docker
    docker >> r_runtime
    docker >> py_runtime
    
    # Session storage
    webapp >> redis
    agent_integration >> redis
    
    # Inference
    agent_integration >> cerebras
    webapp >> cerebras
