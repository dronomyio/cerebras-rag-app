"""
Updated webapp application with pluggable LLM provider support.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import requests
import weaviate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import document processor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from document_processor.processor import DocumentProcessorService

# Import LLM provider factory
from llm_providers import LLMProviderFactory, BaseLLMProvider
from llm_providers.cerebras import CerebrasProvider
from llm_providers.openai import OpenAIProvider
from llm_providers.anthropic import AnthropicProvider
from llm_providers.huggingface import HuggingFaceProvider

# Initialize Flask app
app = Flask(__name__)
app.config.from_pyfile(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'webapp.cfg'))

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize document processor
document_processor = DocumentProcessorService()

# Initialize Weaviate client
weaviate_url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
weaviate_client = weaviate.Client(
    url=weaviate_url,
    auth_client_secret=auth_config
)

# Initialize LLM providers
def initialize_llm_providers():
    """Initialize and register all LLM providers."""
    # Register provider classes
    LLMProviderFactory.register_provider("cerebras", CerebrasProvider)
    LLMProviderFactory.register_provider("openai", OpenAIProvider)
    LLMProviderFactory.register_provider("anthropic", AnthropicProvider)
    LLMProviderFactory.register_provider("huggingface", HuggingFaceProvider)
    
    # Load providers from environment variables
    LLMProviderFactory.load_providers_from_config()
    
    # Log available providers
    available_providers = LLMProviderFactory.list_available_providers()
    active_provider = os.getenv("DEFAULT_LLM_PROVIDER", "cerebras")
    logger.info(f"Available LLM providers: {available_providers}")
    logger.info(f"Active LLM provider: {active_provider}")

# Initialize LLM providers
initialize_llm_providers()

# User model for authentication
class User(UserMixin):
    def __init__(self, id, username, email, password_hash, is_admin=False):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.is_admin = is_admin

# In-memory user database (replace with a real database in production)
users = {}

# Load initial admin user from environment
admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
admin_password = os.getenv("ADMIN_PASSWORD", "adminpassword")
admin_id = str(uuid.uuid4())
users[admin_id] = User(
    id=admin_id,
    username="Admin",
    email=admin_email,
    password_hash=generate_password_hash(admin_password),
    is_admin=True
)

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Find user by email
        user = next((u for u in users.values() if u.email == email), None)
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('chat'))
        
        flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if email already exists
        if any(u.email == email for u in users.values()):
            flash('Email already registered')
            return render_template('register.html')
        
        # Create new user
        user_id = str(uuid.uuid4())
        users[user_id] = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        login_user(users[user_id])
        return redirect(url_for('chat'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/chat')
@login_required
def chat():
    # Get available LLM providers for the UI
    available_providers = LLMProviderFactory.list_available_providers()
    active_provider = LLMProviderFactory._active_provider or os.getenv("DEFAULT_LLM_PROVIDER", "cerebras")
    
    return render_template(
        'chat.html', 
        username=current_user.username,
        available_providers=available_providers,
        active_provider=active_provider,
        enable_runtime_switching=os.getenv("ENABLE_RUNTIME_SWITCHING", "true").lower() == "true"
    )

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data = request.json
    query = data.get('message', '')
    conversation_history = data.get('history', [])
    provider_name = data.get('provider')  # Optional provider override
    
    # Get user session ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    # Query Weaviate for relevant chunks
    try:
        results = query_weaviate(query)
        context = format_context(results)
        
        # Generate response with selected LLM provider
        response = generate_llm_response(query, context, conversation_history, provider_name)
        
        return jsonify({
            'answer': response.get('text', 'Sorry, I could not generate a response.'),
            'sources': response.get('sources', []),
            'provider': response.get('provider', 'unknown')
        })
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        return jsonify({
            'answer': f"An error occurred: {str(e)}",
            'sources': [],
            'provider': 'error'
        }), 500

@app.route('/api/switch_provider', methods=['POST'])
@login_required
def api_switch_provider():
    """API endpoint to switch the active LLM provider."""
    if os.getenv("ENABLE_RUNTIME_SWITCHING", "true").lower() != "true":
        return jsonify({
            'success': False,
            'error': 'Runtime provider switching is disabled'
        }), 403
    
    data = request.json
    provider_name = data.get('provider')
    
    if not provider_name:
        return jsonify({
            'success': False,
            'error': 'No provider specified'
        }), 400
    
    success = LLMProviderFactory.set_active_provider(provider_name)
    
    if success:
        return jsonify({
            'success': True,
            'provider': provider_name
        })
    else:
        return jsonify({
            'success': False,
            'error': f'Failed to switch to provider: {provider_name}'
        }), 500

@app.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        # Process the document
        try:
            chunks = document_processor.process_document(file_path)
            
            # Ingest to Weaviate
            success = document_processor.ingest_to_weaviate(chunks)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Successfully processed and ingested {len(chunks)} chunks from {filename}'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to ingest chunks to Weaviate'
                }), 500
                
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return jsonify({
                'success': False,
                'error': f'Error processing file: {str(e)}'
            }), 500

@app.route('/api/execute_code', methods=['POST'])
@login_required
def api_execute_code():
    data = request.json
    code = data.get('code', '')
    language = data.get('language', 'python')
    
    # Call code executor service
    try:
        code_executor_url = os.getenv("CODE_EXECUTOR_URL", "http://code-executor:5001/execute")
        response = requests.post(
            code_executor_url,
            json={
                'code': code,
                'language': language
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                'success': False,
                'error': f'Code execution failed: {response.text}'
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        return jsonify({
            'success': False,
            'error': f'Error executing code: {str(e)}'
        }), 500

def query_weaviate(query, limit=5):
    """
    Query Weaviate for relevant chunks.
    
    Args:
        query: User query
        limit: Maximum number of results
        
    Returns:
        List of relevant chunks
    """
    try:
        # Get class name from config
        class_name = document_processor.config.get('weaviate', {}).get('class_name', 'DocumentContent')
        
        # Perform semantic search
        result = (
            weaviate_client.query
            .get(class_name, ["content", "source", "fileType", "title", "section", "pageNumber"])
            .with_near_text({"concepts": [query]})
            .with_limit(limit)
            .do()
        )
        
        if "data" in result and "Get" in result["data"] and class_name in result["data"]["Get"]:
            return result["data"]["Get"][class_name]
        
        return []
        
    except Exception as e:
        logger.error(f"Error querying Weaviate: {e}")
        return []

def format_context(results):
    """
    Format Weaviate results into context for LLM.
    
    Args:
        results: List of Weaviate results
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, result in enumerate(results):
        content = result.get("content", "")
        source = result.get("source", "Unknown")
        
        # Add metadata if available
        metadata = []
        if "title" in result and result["title"]:
            metadata.append(f"Title: {result['title']}")
        if "section" in result and result["section"]:
            metadata.append(f"Section: {result['section']}")
        if "pageNumber" in result and result["pageNumber"]:
            metadata.append(f"Page: {result['pageNumber']}")
            
        metadata_str = " | ".join(metadata) if metadata else ""
        
        context_parts.append(f"[{i+1}] {content}\nSource: {source} {metadata_str}\n")
    
    return "\n".join(context_parts)

def generate_llm_response(query, context, conversation_history, provider_name=None):
    """
    Generate response using the configured LLM provider with fallback.
    
    Args:
        query: User query
        context: Retrieved context
        conversation_history: Previous conversation
        provider_name: Optional provider override
        
    Returns:
        Generated response with metadata
    """
    try:
        # Format conversation history
        formatted_history = ""
        for entry in conversation_history[-5:]:  # Use last 5 exchanges
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role and content:
                formatted_history += f"{role.capitalize()}: {content}\n"
        
        # Create prompt
        prompt = f"""You are an AI assistant for answering questions about financial engineering and statistics based on Ruppert's book.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

CONTEXT:
{context}

CONVERSATION HISTORY:
{formatted_history}

QUESTION: {query}

ANSWER:"""

        # For OpenAI and Anthropic, format as messages
        messages = [
            {"role": "system", "content": "You are an AI assistant for answering questions about financial engineering and statistics based on Ruppert's book."},
            {"role": "user", "content": f"""Use the following pieces of context to answer my question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

CONTEXT:
{context}

CONVERSATION HISTORY:
{formatted_history}

QUESTION: {query}"""}
        ]

        # Generate response with specified or active provider
        if provider_name:
            # Try to use specified provider
            provider = LLMProviderFactory.get_provider(provider_name)
            if provider and provider.is_available():
                if provider_name in ["openai", "anthropic"]:
                    response = provider.generate("", messages=messages)
                else:
                    response = provider.generate(prompt)
            else:
                # Fall back to factory with fallback
                response_with_meta = LLMProviderFactory.generate_with_fallback(prompt, messages=messages)
                response = response_with_meta.get("result", {})
                provider_name = response_with_meta.get("provider")
        else:
            # Use factory with fallback
            response_with_meta = LLMProviderFactory.generate_with_fallback(prompt, messages=messages)
            response = response_with_meta.get("result", {})
            provider_name = response_with_meta.get("provider")
        
        # Extract sources from context
        sources = []
        for result in results:
            source = result.get("source", "")
            if source and source not in sources:
                sources.append(source)
        
        # Add sources to response
        response["sources"] = sources
        response["provider"] = provider_name
        
        return response
            
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return {
            "text": f"An error occurred: {str(e)}",
            "sources": [],
            "provider": "error"
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
