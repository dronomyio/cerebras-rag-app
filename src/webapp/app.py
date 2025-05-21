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

# Import paths for modules
import sys
# The document_processor and llm_providers modules will be mounted as volumes
from document_processor.processor import DocumentProcessorService

# Import LLM provider factory
from llm_providers import LLMProviderFactory, BaseLLMProvider, MockProvider
from llm_providers.cerebras import CerebrasProvider
from llm_providers.openai import OpenAIProvider
from llm_providers.anthropic import AnthropicProvider
from llm_providers.huggingface import HuggingFaceProvider

# Import agent integration
try:
    from agent_core.integration import AgentIntegration
except Exception as e:
    logger.warning(f"Failed to import AgentIntegration: {e}")
    AgentIntegration = None

# Initialize Flask app
app = Flask(__name__)
# Use environment variables directly instead of config file
app.config['SECRET_KEY'] = os.getenv("WEBAPP_SECRET_KEY", "default_secret_key")
app.config['DEBUG'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'cerebras_rag_'
app.config['PERMANENT_SESSION_LIFETIME'] = 604800  # 7 days in seconds
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload size
app.config['UPLOAD_FOLDER'] = '/app/data/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt', 'md', 'csv', 'html'}

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize document processor with a simple mock class if import fails
try:
    document_processor = DocumentProcessorService()
except Exception as e:
    logger.warning(f"Failed to initialize DocumentProcessorService: {e}")
    # Create a mock class
    class MockDocumentProcessor:
        def __init__(self):
            self.config = {"weaviate": {"class_name": "DocumentContent"}}
        
        def process_document(self, file_path):
            logger.warning(f"Mock processing document: {file_path}")
            return []
            
        def ingest_to_weaviate(self, chunks):
            logger.warning(f"Mock ingesting {len(chunks)} chunks")
            return True
    
    document_processor = MockDocumentProcessor()

# Initialize Weaviate client
weaviate_url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
try:
    auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
    weaviate_client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=auth_config,
        timeout_config=(5, 15)  # 5 second connect timeout, 15 second read timeout
    )
    # Test connection
    weaviate_client.cluster.get_nodes_status()
    logger.info(f"Successfully connected to Weaviate at {weaviate_url}")
except Exception as e:
    logger.warning(f"Failed to initialize Weaviate client: {e}")
    # Create a mock client with minimal functionality
    class MockWeaviateClient:
        def __init__(self):
            self.query = type('obj', (object,), {
                'get': lambda *args, **kwargs: MockQueryBuilder()
            })
        
    class MockQueryBuilder:
        def get(self, *args, **kwargs):
            return self
            
        def with_near_text(self, *args, **kwargs):
            return self
            
        def with_limit(self, *args, **kwargs):
            return self
            
        def do(self, *args, **kwargs):
            return {"data": {"Get": {"DocumentContent": []}}}
    
    weaviate_client = MockWeaviateClient()
    logger.warning("Initialized mock Weaviate client")

# Initialize LLM providers
def initialize_llm_providers():
    """Initialize and register all LLM providers."""
    try:
        # Register provider classes
        LLMProviderFactory.register_provider("cerebras", CerebrasProvider)
        LLMProviderFactory.register_provider("openai", OpenAIProvider)
        LLMProviderFactory.register_provider("anthropic", AnthropicProvider)
        LLMProviderFactory.register_provider("huggingface", HuggingFaceProvider)
        
        # Register the mock provider for development/testing
        LLMProviderFactory.register_provider("mock", MockProvider)
        
        # Create a mock provider instance that's always available
        mock_provider = MockProvider({})
        LLMProviderFactory._provider_instances["mock"] = mock_provider
        
        # Load providers from environment variables
        LLMProviderFactory.load_providers_from_config()
        
        # Force the active provider to be "mock" for development
        if not LLMProviderFactory.set_active_provider("mock"):
            logger.warning("Failed to set mock as active provider")
        
        # Set mock as first in fallback order
        LLMProviderFactory.set_fallback_order(["mock", "cerebras", "openai", "anthropic", "huggingface"])
        
        # Log available providers
        available_providers = LLMProviderFactory.list_available_providers()
        active_provider = LLMProviderFactory._active_provider or "mock"
        logger.info(f"Available LLM providers: {available_providers}")
        logger.info(f"Active LLM provider: {active_provider}")
        return True
    except Exception as e:
        logger.error(f"Error initializing LLM providers: {e}")
        return False

# Initialize LLM providers
llm_providers_initialized = initialize_llm_providers()
if not llm_providers_initialized:
    logger.warning("Using fallback LLM generation. Responses will be mock data.")

# Initialize agent integration
agent_integration = None
try:
    if AgentIntegration:
        agent_config_path = os.getenv("AGENT_CONFIG_PATH", "/app/config/agent_config.json")
        # Use an absolute path with the mounted volumes in mind
        agent_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "config", "agent_config.json")
        if os.path.exists(agent_config_path):
            agent_integration = AgentIntegration(agent_config_path)
            logger.info(f"Agent integration initialized from {agent_config_path}")
        else:
            logger.warning(f"Agent config file not found at {agent_config_path}")
except Exception as e:
    logger.warning(f"Failed to initialize agent integration: {e}")

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
        try:
            results = query_weaviate(query)
            context = format_context(results)
        except Exception as e:
            logger.warning(f"Failed to query Weaviate: {e}")
            results = []
            context = "No context available due to Weaviate query failure."
        
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
        logger.error("No file part in the request")
        return jsonify({'success': False, 'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    if file:
        try:
            # Log file info for debugging
            logger.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")
            
            # Try to get file extension, very permissively
            try:
                file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
                allowed_extensions = app.config['ALLOWED_EXTENSIONS']
                
                if file_ext not in allowed_extensions:
                    logger.warning(f"Potentially invalid file extension: {file_ext}, but we'll try to process it anyway")
                else:
                    logger.info(f"Detected valid file extension: {file_ext}")
            except Exception as e:
                logger.warning(f"Error checking file extension: {e}, continuing anyway")
                file_ext = ""
            
            # Create a simplified, safe filename without using secure_filename
            try:
                # Just use a timestamp and the original extension to avoid any pattern issues
                import time
                safe_name = f"upload_{int(time.time())}"
                if file_ext:
                    safe_name = f"{safe_name}.{file_ext}"
                    
                logger.info(f"Using safe filename: {safe_name} instead of {file.filename}")
                filename = safe_name
            except Exception as e:
                logger.warning(f"Error creating safe filename: {e}, using secure_filename as fallback")
                filename = secure_filename(file.filename) or "document.pdf"
            
            # Ensure uploads directory exists with proper permissions
            upload_dir = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_dir):
                try:
                    os.makedirs(upload_dir, mode=0o755, exist_ok=True)
                    logger.info(f"Created uploads directory: {upload_dir}")
                except Exception as e:
                    logger.error(f"Failed to create uploads directory: {e}")
                    return jsonify({
                        'success': False,
                        'error': f'Failed to create uploads directory: {str(e)}'
                    }), 500
            
            # Save the file
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            logger.info(f"Saved uploaded file to {file_path}")
            
            # Check if file was actually saved
            if not os.path.exists(file_path):
                logger.error(f"File not saved: {file_path}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to save the uploaded file'
                }), 500
            
            # Process the document
            try:
                logger.info(f"Processing document: {file_path}")
                chunks = document_processor.process_document(file_path)
                logger.info(f"Generated {len(chunks)} chunks from {file_path}")
                
                # Check if any chunks were generated
                if not chunks:
                    logger.warning(f"No chunks were generated from {file_path}")
                    return jsonify({
                        'success': False,
                        'error': 'No content could be extracted from the document'
                    }), 400
                
                # Ingest to Weaviate
                try:
                    logger.info(f"Ingesting {len(chunks)} chunks into Weaviate")
                    success = document_processor.ingest_to_weaviate(chunks)
                    
                    if success:
                        logger.info(f"Successfully ingested chunks into Weaviate")
                        return jsonify({
                            'success': True,
                            'message': f'Successfully processed and ingested {len(chunks)} chunks from {filename}'
                        })
                    else:
                        # Try to proceed even if Weaviate ingestion fails
                        logger.warning(f"Failed to ingest chunks to Weaviate but document was processed")
                        return jsonify({
                            'success': True,
                            'message': f'Document processed with {len(chunks)} chunks. Note: Weaviate ingestion failed but you can still ask questions.'
                        })
                except Exception as e:
                    logger.error(f"Error ingesting to Weaviate: {e}")
                    # Still return success since processing worked
                    return jsonify({
                        'success': True,
                        'message': f'Document processed with {len(chunks)} chunks. Note: Weaviate ingestion failed but you can still ask questions.'
                    })
                    
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Error processing file: {str(e)}'
                }), 500
                
        except Exception as e:
            logger.error(f"Error in file upload: {e}")
            return jsonify({
                'success': False,
                'error': f'Upload error: {str(e)}'
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

# Agent API endpoints
@app.route('/api/agent/status', methods=['GET'])
@login_required
def api_agent_status():
    """Get the current status of the agent."""
    if not agent_integration:
        return jsonify({
            'success': False,
            'error': 'Agent integration not available'
        }), 503
    
    try:
        status = agent_integration.get_agent_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        return jsonify({
            'success': False,
            'error': f'Error getting agent status: {str(e)}'
        }), 500

@app.route('/api/agent/start', methods=['POST'])
@login_required
def api_agent_start():
    """Start the agent."""
    if not agent_integration:
        return jsonify({
            'success': False,
            'error': 'Agent integration not available'
        }), 503
    
    try:
        agent_integration.start()
        return jsonify({
            'success': True,
            'message': 'Agent started successfully'
        })
    except Exception as e:
        logger.error(f"Error starting agent: {e}")
        return jsonify({
            'success': False,
            'error': f'Error starting agent: {str(e)}'
        }), 500

@app.route('/api/agent/stop', methods=['POST'])
@login_required
def api_agent_stop():
    """Stop the agent."""
    if not agent_integration:
        return jsonify({
            'success': False,
            'error': 'Agent integration not available'
        }), 503
    
    try:
        agent_integration.stop()
        return jsonify({
            'success': True,
            'message': 'Agent stopped successfully'
        })
    except Exception as e:
        logger.error(f"Error stopping agent: {e}")
        return jsonify({
            'success': False,
            'error': f'Error stopping agent: {str(e)}'
        }), 500

@app.route('/api/agent/goal', methods=['POST'])
@login_required
def api_agent_goal():
    """Set a new goal for the agent."""
    if not agent_integration:
        return jsonify({
            'success': False,
            'error': 'Agent integration not available'
        }), 503
    
    data = request.json
    description = data.get('description', '')
    priority = data.get('priority', 1)
    deadline = data.get('deadline')
    
    if not description:
        return jsonify({
            'success': False,
            'error': 'Goal description is required'
        }), 400
    
    try:
        goal_id = agent_integration.set_goal(description, priority, deadline)
        return jsonify({
            'success': True,
            'goal_id': goal_id,
            'message': 'Goal set successfully'
        })
    except Exception as e:
        logger.error(f"Error setting goal: {e}")
        return jsonify({
            'success': False,
            'error': f'Error setting goal: {str(e)}'
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

        # Check if autonomous agent should handle this query
        use_agent = False
        if agent_integration and "agent" in query.lower():
            # This is a simple heuristic to check if the query is asking about agent capabilities
            # In a real system, you would use more sophisticated NLU to detect agent-related queries
            use_agent = True

        if use_agent and agent_integration:
            try:
                # Set a goal for the agent to handle this query
                goal_description = f"Answer user query: {query}"
                goal_id = agent_integration.set_goal(goal_description, priority=3)
                logger.info(f"Created agent goal {goal_id} for query: {query}")
                
                # Wait for the agent to process the goal (in a real system, this would be async)
                # For now, we'll just return a placeholder response
                response = {
                    "text": f"I've asked our autonomous agent to research this question for you. The agent will work on this in the background and provide a detailed answer soon. Your request has been assigned ID: {goal_id}",
                    "provider": "agent",
                    "goal_id": goal_id
                }
                
                return response
            except Exception as e:
                logger.error(f"Error using agent for query: {e}")
                # Fall back to regular LLM generation
                pass
        
        if llm_providers_initialized:
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
        else:
            # Use mock response
            logger.warning("Using mock LLM response as fallback")
            response = {
                "text": f"This is a mock response for query: '{query}'. LLM providers are not available.",
                "provider": "mock"
            }
            provider_name = "mock"
        
        # Extract sources from context if results is defined
        sources = []
        try:
            for result in results:
                source = result.get("source", "")
                if source and source not in sources:
                    sources.append(source)
            # Add sources to response
            response["sources"] = sources
        except NameError:
            # results not defined
            response["sources"] = []
        
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
