# Cerebras RAG Application

A comprehensive Retrieval-Augmented Generation (RAG) system using Cerebras inference and Weaviate vector database, with configurable Unstructured.io integration for processing various document types.

## Features

- **Document Processing**
  - Configurable Unstructured.io integration (can be enabled/disabled)
  - Pluggable processor system with specialized handlers for PDF, DOCX, and text files
  - Easy to extend with custom processors for additional file types

- **Vector Storage and Retrieval**
  - Weaviate for semantic search and storage
  - Text2Vec Transformers for generating embeddings

- **Web Application**
  - Authentication system with user management
  - Interactive chat interface with conversation history
  - Document upload functionality
  - Code execution capabilities for Python and R

- **Inference Pipeline**
  - Integration with Cerebras API for high-quality responses
  - Context-aware response generation with source citations

## Directory Structure

```
cerebras-rag-app/
├── config/
│   └── document_processor.yaml
├── data/
│   └── uploads/  (created when users upload documents)
├── docs/
│   └── final_architecture_diagram.png
├── src/
│   ├── document_processor/
│   │   ├── plugins/
│   │   │   ├── base_processor.py
│   │   │   ├── pdf_processor.py
│   │   │   ├── docx_processor.py
│   │   │   └── text_processor.py
│   │   └── processor.py
│   ├── webapp/
│   │   ├── templates/
│   │   │   ├── login.html
│   │   │   ├── register.html
│   │   │   └── chat.html
│   │   └── app.py
│   ├── code_executor/
│   │   └── app.py
│   └── utils/
│       └── helpers.py
└── docker-compose.yml
```

## Prerequisites

- Docker and Docker Compose
- Cerebras API key
- Unstructured.io API key (optional)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cerebras-rag-app.git
   cd cerebras-rag-app
   ```

2. Create a `.env` file in the root directory with the following variables:
   ```
   # Required
   CEREBRAS_API_KEY=your_cerebras_api_key
   WEAVIATE_ADMIN_KEY=choose_a_secure_random_string
   REDIS_PASSWORD=choose_a_secure_random_string
   WEBAPP_SECRET_KEY=choose_a_secure_random_string
   
   # Optional
   UNSTRUCTURED_API_KEY=your_unstructured_api_key
   CEREBRAS_API_URL=https://api.cerebras.ai/v1/completions
   ADMIN_EMAIL=admin@example.com
   ADMIN_PASSWORD=adminpassword
   ```

3. Start the application:
   ```bash
   docker-compose up -d
   ```

4. Access the web interface at http://localhost

## Usage

### Authentication

- Default admin credentials:
  - Email: admin@example.com
  - Password: adminpassword (as specified in your .env file)

### Document Upload

1. Log in to the web interface
2. Click the "Upload Document" button in the sidebar
3. Select a document to upload (PDF, DOCX, TXT, etc.)
4. Wait for processing to complete

### Asking Questions

1. Type your question in the input field at the bottom of the chat interface
2. Press Enter or click the send button
3. View the response with source citations

### Executing Code

1. When code blocks appear in responses, click the "Execute" button
2. View the execution results below the code block

## Configuration

### Document Processor

Edit `config/document_processor.yaml` to configure the document processor:

```yaml
# Main configuration
enable_unstructured_io: true  # Set to false to disable Unstructured.io
default_chunk_size: 1000
default_chunk_overlap: 200

# Document type specific settings
document_types:
  pdf:
    enabled: true
    chunk_size: 1000
    chunk_overlap: 200
    extract_images: true
    
  docx:
    enabled: true
    chunk_size: 800
    chunk_overlap: 150
    
  # Add more document types as needed
```

### Adding Custom Processors

1. Create a new processor in `src/document_processor/plugins/`
2. Inherit from `BaseDocumentProcessor`
3. Implement the required methods
4. Add configuration in `document_processor.yaml`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Cerebras](https://www.cerebras.ai/) for the inference API
- [Weaviate](https://weaviate.io/) for vector storage
- [Unstructured.io](https://unstructured.io/) for document processing
- [Flask](https://flask.palletsprojects.com/) for the web framework
