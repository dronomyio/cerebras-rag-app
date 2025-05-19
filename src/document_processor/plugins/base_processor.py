#!/usr/bin/env python3
"""
Base Document Processor Plugin
------------------------------
Abstract base class for document processor plugins.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseDocumentProcessor(ABC):
    """
    Abstract base class for document processor plugins.
    All document processor plugins must inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document processor with configuration.
        
        Args:
            config: Configuration dictionary from the main config file
        """
        self.config = config
        self.document_type_config = {}
        
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this processor can handle the file, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process the document and return chunks.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of chunks, where each chunk is a dictionary with at least
            'content' and 'metadata' keys
        """
        pass
    
    def get_document_type(self) -> str:
        """
        Get the document type this processor handles.
        
        Returns:
            String identifier for the document type (e.g., 'pdf', 'docx')
        """
        return self.__class__.__name__.lower().replace('processor', '')
    
    def is_enabled(self) -> bool:
        """
        Check if this processor is enabled in the configuration.
        
        Returns:
            True if enabled, False otherwise
        """
        doc_type = self.get_document_type()
        if doc_type in self.config.get('document_types', {}):
            return self.config['document_types'][doc_type].get('enabled', True)
        return True
    
    def get_chunk_size(self) -> int:
        """
        Get the chunk size for this document type.
        
        Returns:
            Chunk size in characters
        """
        doc_type = self.get_document_type()
        if doc_type in self.config.get('document_types', {}):
            return self.config['document_types'][doc_type].get(
                'chunk_size', 
                self.config.get('default_chunk_size', 1000)
            )
        return self.config.get('default_chunk_size', 1000)
    
    def get_chunk_overlap(self) -> int:
        """
        Get the chunk overlap for this document type.
        
        Returns:
            Chunk overlap in characters
        """
        doc_type = self.get_document_type()
        if doc_type in self.config.get('document_types', {}):
            return self.config['document_types'][doc_type].get(
                'chunk_overlap', 
                self.config.get('default_chunk_overlap', 200)
            )
        return self.config.get('default_chunk_overlap', 200)
    
    def get_setting(self, setting_name: str, default: Any = None) -> Any:
        """
        Get a specific setting for this document type.
        
        Args:
            setting_name: Name of the setting to retrieve
            default: Default value if setting is not found
            
        Returns:
            Setting value or default
        """
        doc_type = self.get_document_type()
        if doc_type in self.config.get('document_types', {}):
            return self.config['document_types'][doc_type].get(setting_name, default)
        return default
