#!/usr/bin/env python3
"""
PDF Document Processor Plugin
-----------------------------
Plugin for processing PDF documents.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import base processor
from .base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)

class PDFProcessor(BaseDocumentProcessor):
    """
    Processor for PDF documents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PDF processor with configuration.
        
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
        return file_path.suffix.lower() == '.pdf'
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process the PDF document and return chunks.
        
        Args:
            file_path: Path to the PDF file to process
            
        Returns:
            List of chunks, where each chunk is a dictionary with at least
            'content' and 'metadata' keys
        """
        try:
            # Import PyMuPDF here to avoid dependency issues if not installed
            import fitz  # PyMuPDF
            
            file_path = Path(file_path)
            chunks = []
            
            # Extract text from PDF
            doc = fitz.open(file_path)
            
            # Get document metadata
            metadata = {
                "source": file_path.name,
                "file_type": "pdf",
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "processor": "pdf_processor"
            }
            
            # Process each page
            current_chunk = {"content": "", "metadata": metadata.copy()}
            current_chunk_size = 0
            chunk_size = self.get_chunk_size()
            chunk_overlap = self.get_chunk_overlap()
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # Skip empty pages
                if not text.strip():
                    continue
                
                # Update metadata with page number
                page_metadata = metadata.copy()
                page_metadata["page_number"] = page_num + 1
                
                # Extract images if configured
                if self.get_setting("extract_images", False):
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image:
                            # Here we could save the image or process it further
                            # For now, just note it in the metadata
                            page_metadata[f"has_image_{img_index}"] = True
                
                # Process text with chunking
                paragraphs = re.split(r'\n\s*\n', text)
                
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
                            "metadata": page_metadata.copy()
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
            
            # Apply OCR if enabled and if PyTesseract is available
            if self.get_setting("ocr_enabled", False):
                try:
                    import pytesseract
                    from PIL import Image
                    
                    # Process each page with OCR
                    for page_num, page in enumerate(doc):
                        # Convert page to image
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # Perform OCR
                        ocr_text = pytesseract.image_to_string(img)
                        
                        if ocr_text.strip():
                            ocr_metadata = metadata.copy()
                            ocr_metadata["page_number"] = page_num + 1
                            ocr_metadata["source_type"] = "ocr"
                            
                            chunks.append({
                                "content": ocr_text,
                                "metadata": ocr_metadata
                            })
                except ImportError:
                    logger.warning("OCR is enabled but pytesseract is not installed. Skipping OCR.")
            
            return chunks
            
        except ImportError:
            logger.error("PyMuPDF (fitz) is not installed. Cannot process PDF.")
            return [{
                "content": "Error: PyMuPDF is not installed. Cannot process PDF.",
                "metadata": {
                    "source": Path(file_path).name,
                    "file_type": "pdf",
                    "error": "Missing dependency: PyMuPDF"
                }
            }]
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            # Return a single chunk with error information
            return [{
                "content": f"Error processing PDF: {str(e)}",
                "metadata": {
                    "source": Path(file_path).name,
                    "file_type": "pdf",
                    "error": str(e)
                }
            }]
