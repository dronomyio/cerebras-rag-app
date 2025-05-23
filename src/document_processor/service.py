#!/usr/bin/env python3
"""
Document Processor Service - Minimal Version
"""
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalDocumentProcessorService:
    def __init__(self):
        logger.info("Initializing Minimal Document Processor Service")
    
    def process_document(self, file_path):
        logger.info(f"Processing document: {file_path}")
        return [{"content": "Demo content", "source": file_path}]
    
    def ingest_to_weaviate(self, chunks):
        logger.info(f"Ingesting {len(chunks)} chunks to Weaviate")
        return True

if __name__ == "__main__":
    logger.info("Starting Document Processor Service")
    
    # Initialize the document processor service
    processor = MinimalDocumentProcessorService()
    
    # Keep the service running
    try:
        while True:
            logger.info("Document Processor Service running...")
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Document Processor Service stopping...")