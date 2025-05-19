#!/usr/bin/env python3
"""
Main document processor module for Cerebras RAG application.
Handles document processing with configurable Unstructured.io integration.
"""

import os
import yaml
import json
import logging
import importlib.util
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessorService:
    """
    Service for processing documents with configurable Unstructured.io integration.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the document processor service.
        
        Args:
            config_path: Path to the configuration file
        """
        # Set default config path if not provided
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'config',
                'document_processor.yaml'
            )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize processors
        self.processors = []
        self._load_plugins()
        
        # Unstructured.io settings
        self.enable_unstructured_io = self.config.get('enable_unstructured_io', False)
        if self.enable_unstructured_io:
            self.unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
            unstructured_config = self.config.get('unstructured_io', {})
            self.unstructured_api_url = unstructured_config.get(
                'api_url', 
                'https://api.unstructured.io/general/v0/general'
            )
            self.unstructured_params = unstructured_config.get('params', {})
            self.unstructured_timeout = unstructured_config.get('timeout_seconds', 60)
            
            logger.info("Unstructured.io integration is ENABLED")
        else:
            logger.info("Unstructured.io integration is DISABLED")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            return {
                'enable_unstructured_io': False,
                'default_chunk_size': 1000,
                'default_chunk_overlap': 200
            }
    
    def _load_plugins(self):
        """
        Dynamically load all processor plugins.
        """
        try:
            # Get the plugins directory
            plugins_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'plugins'
            )
            
            # Import all processor modules
            for file in Path(plugins_dir).glob("*_processor.py"):
                if file.name == "base_processor.py":
                    continue
                
                try:
                    # Import the module
                    module_name = file.stem
                    spec = importlib.util.spec_from_file_location(
                        module_name, 
                        file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find processor classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            attr.__name__ != "BaseDocumentProcessor" and
                            "Processor" in attr.__name__):
                            
                            # Initialize the processor with config
                            processor = attr(self.config)
                            self.processors.append(processor)
                            logger.info(f"Loaded processor plugin: {attr.__name__}")
                            
                except Exception as e:
                    logger.error(f"Error loading plugin {file.name}: {e}")
            
            logger.info(f"Loaded {len(self.processors)} processor plugins")
            
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a document using the appropriate processor or Unstructured.io.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            List of chunks, where each chunk is a dictionary with at least
            'content' and 'metadata' keys
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return [{
                "content": f"Error: File not found: {file_path}",
                "metadata": {
                    "source": file_path.name,
                    "error": "File not found"
                }
            }]
        
        # Check file size
        max_file_size = self.config.get('max_file_size_mb', 50) * 1024 * 1024  # Convert to bytes
        if file_path.stat().st_size > max_file_size:
            logger.error(f"File too large: {file_path} ({file_path.stat().st_size} bytes)")
            return [{
                "content": f"Error: File too large: {file_path.name}",
                "metadata": {
                    "source": file_path.name,
                    "error": "File too large"
                }
            }]
        
        # Try plugin processors first
        for processor in self.processors:
            if processor.can_process(file_path):
                logger.info(f"Processing {file_path} with {processor.__class__.__name__}")
                return processor.process(file_path)
        
        # If no plugin can process the file and Unstructured.io is enabled, use it
        if self.enable_unstructured_io:
            logger.info(f"No plugin found for {file_path}, using Unstructured.io")
            return self._process_with_unstructured(file_path)
        else:
            logger.warning(f"No plugin found for {file_path} and Unstructured.io is disabled")
            return [{
                "content": f"Error: No processor available for file type: {file_path.suffix}",
                "metadata": {
                    "source": file_path.name,
                    "file_type": file_path.suffix[1:] if file_path.suffix else "unknown",
                    "error": "Unsupported file type"
                }
            }]
    
    def _process_with_unstructured(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process document using Unstructured.io API.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            List of chunks from Unstructured.io
        """
        try:
            with open(file_path, 'rb') as f:
                files = {'files': (file_path.name, f)}
                headers = {'Accept': 'application/json'}
                
                if self.unstructured_api_key:
                    headers['unstructured-api-key'] = self.unstructured_api_key
                
                logger.info(f"Sending {file_path} to Unstructured.io API")
                response = requests.post(
                    self.unstructured_api_url,
                    headers=headers,
                    files=files,
                    params=self.unstructured_params,
                    timeout=self.unstructured_timeout
                )
                
                if response.status_code != 200:
                    logger.error(f"Unstructured.io API error: {response.text}")
                    return [{
                        "content": f"Error from Unstructured.io API: {response.text}",
                        "metadata": {
                            "source": file_path.name,
                            "file_type": file_path.suffix[1:] if file_path.suffix else "unknown",
                            "error": f"API error: {response.status_code}"
                        }
                    }]
                
                # Process the response into chunks
                elements = response.json()
                logger.info(f"Received {len(elements)} elements from Unstructured.io API")
                return self._convert_elements_to_chunks(elements, file_path)
                
        except Exception as e:
            logger.error(f"Error processing with Unstructured.io: {e}")
            return [{
                "content": f"Error processing with Unstructured.io: {str(e)}",
                "metadata": {
                    "source": file_path.name,
                    "file_type": file_path.suffix[1:] if file_path.suffix else "unknown",
                    "error": str(e)
                }
            }]
    
    def _convert_elements_to_chunks(self, elements: List[Dict[str, Any]], file_path: Path) -> List[Dict[str, Any]]:
        """
        Convert Unstructured.io elements to document chunks.
        
        Args:
            elements: List of elements from Unstructured.io
            file_path: Path to the original document
            
        Returns:
            List of chunks
        """
        chunks = []
        metadata = {
            "source": file_path.name,
            "file_type": file_path.suffix[1:] if file_path.suffix else "unknown",
            "processor": "unstructured.io"
        }
        
        current_chunk = {"content": "", "metadata": metadata.copy()}
        current_section = None
        
        for element in elements:
            element_type = element.get("type")
            text = element.get("text", "")
            
            # Skip empty elements
            if not text.strip():
                continue
            
            # Handle different element types
            if element_type == "Title":
                # Start a new chunk for titles
                if current_chunk["content"]:
                    chunks.append(current_chunk)
                
                current_section = text
                current_chunk = {"content": text, "metadata": metadata.copy()}
                current_chunk["metadata"]["title"] = text
                current_chunk["metadata"]["element_type"] = "title"
                
            elif element_type in ["NarrativeText", "Text"]:
                # Add to current chunk
                if current_chunk["content"]:
                    current_chunk["content"] += "\n\n"
                current_chunk["content"] += text
                
                if current_section:
                    current_chunk["metadata"]["section"] = current_section
                
            elif element_type == "Table":
                # Tables as separate chunks
                table_chunk = {
                    "content": text, 
                    "metadata": metadata.copy()
                }
                table_chunk["metadata"]["element_type"] = "table"
                
                if current_section:
                    table_chunk["metadata"]["section"] = current_section
                
                chunks.append(table_chunk)
                
            elif element_type == "Image":
                # Handle image captions
                if "caption" in element:
                    img_chunk = {
                        "content": f"Image: {element.get('caption', '')}", 
                        "metadata": metadata.copy()
                    }
                    img_chunk["metadata"]["element_type"] = "image"
                    
                    if current_section:
                        img_chunk["metadata"]["section"] = current_section
                    
                    chunks.append(img_chunk)
            
            elif element_type == "ListItem":
                # Add list items to current chunk
                if current_chunk["content"]:
                    current_chunk["content"] += "\n"
                current_chunk["content"] += f"• {text}"
                
            else:
                # Handle other element types
                if current_chunk["content"]:
                    current_chunk["content"] += "\n\n"
                current_chunk["content"] += text
                current_chunk["metadata"]["element_type"] = element_type.lower()
        
        # Add the last chunk if not empty
        if current_chunk["content"]:
            chunks.append(current_chunk)
            
        return chunks
    
    def ingest_to_weaviate(self, chunks: List[Dict[str, Any]], class_name: str = None) -> bool:
        """
        Ingest processed chunks to Weaviate.
        
        Args:
            chunks: List of chunks to ingest
            class_name: Weaviate class name (defaults to config value)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import weaviate
            
            # Get Weaviate configuration
            weaviate_config = self.config.get('weaviate', {})
            weaviate_url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
            weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
            
            # Use provided class name or get from config
            if class_name is None:
                class_name = weaviate_config.get('class_name', 'DocumentContent')
            
            # Connect to Weaviate
            auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
            client = weaviate.Client(
                url=weaviate_url,
                auth_client_secret=auth_config
            )
            
            # Check if class exists, create if not
            if not client.schema.exists(class_name):
                logger.info(f"Creating Weaviate class: {class_name}")
                
                # Define class schema
                class_schema = {
                    "class": class_name,
                    "description": "Content chunks from processed documents",
                    "vectorizer": "text2vec-transformers",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "vectorizeClassName": False
                        }
                    },
                    "properties": [
                        {
                            "name": "content",
                            "description": "The text content of the chunk",
                            "dataType": ["text"],
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": False,
                                    "vectorizePropertyName": False
                                }
                            }
                        },
                        {
                            "name": "source",
                            "description": "Source document filename",
                            "dataType": ["string"],
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "fileType",
                            "description": "File type of the source document",
                            "dataType": ["string"],
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "title",
                            "description": "Title or heading",
                            "dataType": ["string"],
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "section",
                            "description": "Section name",
                            "dataType": ["string"],
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "elementType",
                            "description": "Type of content element",
                            "dataType": ["string"],
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "pageNumber",
                            "description": "Page number in the source document",
                            "dataType": ["int"],
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "processor",
                            "description": "Processor used to extract the content",
                            "dataType": ["string"],
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        }
                    ]
                }
                
                # Create the class
                client.schema.create_class(class_schema)
            
            # Batch import chunks
            batch_size = weaviate_config.get('batch_size', 50)
            with client.batch as batch:
                batch.batch_size = batch_size
                
                for chunk in chunks:
                    # Extract metadata
                    metadata = chunk.get('metadata', {})
                    
                    # Prepare properties
                    properties = {
                        "content": chunk.get('content', ''),
                        "source": metadata.get('source', ''),
                        "fileType": metadata.get('file_type', ''),
                        "processor": metadata.get('processor', '')
                    }
                    
                    # Add optional properties if present
                    if 'title' in metadata:
                        properties["title"] = metadata['title']
                    
                    if 'section' in metadata:
                        properties["section"] = metadata['section']
                    
                    if 'element_type' in metadata:
                        properties["elementType"] = metadata['element_type']
                    
                    if 'page_number' in metadata:
                        properties["pageNumber"] = metadata['page_number']
                    
                    # Add to batch
                    batch.add_data_object(
                        data_object=properties,
                        class_name=class_name
                    )
            
            logger.info(f"Successfully ingested {len(chunks)} chunks into Weaviate class {class_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting to Weaviate: {e}")
            return False


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents with configurable Unstructured.io integration")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output", help="Path to output JSON file")
    parser.add_argument("--ingest", action="store_true", help="Ingest to Weaviate after processing")
    parser.add_argument("--class-name", help="Weaviate class name for ingestion")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DocumentProcessorService(args.config)
    
    # Process document
    chunks = processor.process_document(args.file_path)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {args.output}")
    else:
        logger.info(f"Processed {len(chunks)} chunks from {args.file_path}")
    
    # Ingest to Weaviate if requested
    if args.ingest:
        processor.ingest_to_weaviate(chunks, args.class_name)
