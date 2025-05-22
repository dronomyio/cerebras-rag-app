#!/usr/bin/env python3
"""
Text Document Processor Plugin
-----------------------------
Plugin for processing plain text documents.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import base processor
from .base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)

class TextProcessor(BaseDocumentProcessor):
    """
    Processor for plain text documents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text processor with configuration.
        
        Args:
            config: Configuration dictionary from the main config file
        """
        super().__init__(config)
        
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this processor can handle the file, False otherwise
        """
        if not self.is_enabled():
            return False
            
        file_path = Path(file_path)
        return file_path.suffix.lower() in ['.txt', '.md', '.csv', '.json', '.xml', '.html']
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process the text document and return chunks.
        
        Args:
            file_path: Path to the text file to process
            
        Returns:
            List of chunks, where each chunk is a dictionary with at least
            'content' and 'metadata' keys
        """
        try:
            file_path = Path(file_path)
            chunks = []
            
            # Get document metadata
            metadata = {
                "source": file_path.name,
                "file_type": file_path.suffix[1:],
                "processor": "text_processor"
            }
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process with chunking
            chunk_size = self.get_chunk_size()
            chunk_overlap = self.get_chunk_overlap()
            
            # Special handling for different file types
            if file_path.suffix.lower() == '.md':
                # For markdown, try to split by headers
                sections = re.split(r'(#+\s+.*)', content)
                current_chunk = {"content": "", "metadata": metadata.copy()}
                current_chunk_size = 0
                current_header = None
                
                for i, section in enumerate(sections):
                    # Check if this is a header
                    if i % 2 == 1 and re.match(r'#+\s+.*', section):
                        # Start a new chunk for headers
                        if current_chunk["content"]:
                            chunks.append(current_chunk)
                        
                        current_header = section.strip()
                        current_chunk = {
                            "content": section,
                            "metadata": metadata.copy()
                        }
                        current_chunk["metadata"]["header"] = current_header
                        current_chunk_size = len(section)
                    else:
                        # If adding this section would exceed chunk size, save current chunk and start a new one
                        if current_chunk_size + len(section) > chunk_size and current_chunk["content"]:
                            chunks.append(current_chunk)
                            
                            # Start new chunk with overlap
                            overlap_text = current_chunk["content"][-chunk_overlap:] if chunk_overlap > 0 else ""
                            current_chunk = {
                                "content": overlap_text,
                                "metadata": metadata.copy()
                            }
                            if current_header:
                                current_chunk["metadata"]["header"] = current_header
                            current_chunk_size = len(overlap_text)
                        
                        # Add section to current chunk
                        if current_chunk["content"] and section:
                            current_chunk["content"] += "\n\n"
                        current_chunk["content"] += section
                        current_chunk_size = len(current_chunk["content"])
                
                # Add the last chunk if not empty
                if current_chunk["content"]:
                    chunks.append(current_chunk)
                    
            elif file_path.suffix.lower() == '.csv':
                # For CSV, chunk by rows if configured
                if self.get_setting("chunk_by_row", True):
                    lines = content.split('\n')
                    header = lines[0] if lines else ""
                    
                    max_rows = self.get_setting("max_rows_per_chunk", 50)
                    for i in range(1, len(lines), max_rows):
                        chunk_lines = [header] + lines[i:i+max_rows]
                        chunk_content = '\n'.join(chunk_lines)
                        
                        chunk_metadata = metadata.copy()
                        chunk_metadata["row_range"] = f"{i}-{min(i+max_rows-1, len(lines)-1)}"
                        
                        chunks.append({
                            "content": chunk_content,
                            "metadata": chunk_metadata
                        })
                else:
                    # Default chunking for CSV
                    self._chunk_text(content, metadata, chunks, chunk_size, chunk_overlap)
            else:
                # Default chunking for other text files
                self._chunk_text(content, metadata, chunks, chunk_size, chunk_overlap)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            # Return a single chunk with error information
            return [{
                "content": f"Error processing text file: {str(e)}",
                "metadata": {
                    "source": Path(file_path).name,
                    "file_type": Path(file_path).suffix[1:],
                    "error": str(e)
                }
            }]
    
    def _chunk_text(self, content: str, metadata: Dict[str, Any], 
                   chunks: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int):
        """
        Helper method to chunk text content.
        
        Args:
            content: Text content to chunk
            metadata: Metadata dictionary
            chunks: List to append chunks to
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
        """
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = {"content": "", "metadata": metadata.copy()}
        current_chunk_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size, save current chunk and start a new one
            if current_chunk_size + len(para) > chunk_size and current_chunk["content"]:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_text = current_chunk["content"][-chunk_overlap:] if chunk_overlap > 0 else ""
                current_chunk = {
                    "content": overlap_text,
                    "metadata": metadata.copy()
                }
                current_chunk_size = len(overlap_text)
            
            # Add paragraph to current chunk
            if current_chunk["content"]:
                current_chunk["content"] += "\n\n"
            current_chunk["content"] += para
            current_chunk_size = len(current_chunk["content"])
        
        # Add the last chunk if not empty
        if current_chunk["content"]:
            chunks.append(current_chunk)
