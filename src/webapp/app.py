#!/usr/bin/env python3
"""
Cerebras RAG Web Application - Minimal Version
"""

import os
import json
import logging
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('WEBAPP_SECRET_KEY', 'default-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
app.config['UPLOAD_FOLDER'] = '/app/data/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    # BREAKPOINT: Add a breakpoint here to debug the index route
    logger.debug("Accessing index route")
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # BREAKPOINT: Add a breakpoint here to debug login logic
    logger.debug(f"Login request method: {request.method}")
    
    if request.method == 'POST':
        # For demo, always allow login
        return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # BREAKPOINT: Add a breakpoint here to debug registration
    logger.debug(f"Register request method: {request.method}")
    
    if request.method == 'POST':
        # For demo, always redirect to login after registration
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    # BREAKPOINT: Add a breakpoint here to debug logout
    logger.debug("Logging out user")
    
    # For demo, simply redirect to login page
    return redirect(url_for('login'))

@app.route('/chat')
def chat():
    # BREAKPOINT: Add a breakpoint here to debug chat page
    logger.debug("Accessing chat page")
    
    return render_template('chat.html', username="Demo User")

@app.route('/api/chat', methods=['POST'])
def api_chat():
    # BREAKPOINT: Add a breakpoint here to debug chat API
    data = request.json
    query = data.get('message', '')
    logger.debug(f"Chat API received query: {query}")
    
    # Demo response
    response = {
        'answer': f"This is a demo response to your question: '{query}'. In a real application, this would call the Cerebras API or another LLM provider to generate a response based on the document context.",
        'sources': ['Demo Source 1', 'Demo Source 2']
    }
    
    return jsonify(response)

@app.route('/api/upload', methods=['POST'])
def api_upload():
    # BREAKPOINT: Add a breakpoint here to debug file upload
    logger.debug("File upload request received")
    
    try:
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({'success': False, 'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'success': False, 'error': 'No selected file'})
        
        # Log file details
        logger.info(f"Received file: {file.filename}, Content-Type: {file.content_type}, Size: {request.content_length} bytes")
        
        # BREAKPOINT: Add a breakpoint here to inspect the file object
        
        # Save the file (in a real app)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        # file.save(file_path)
        # logger.info(f"File saved to {file_path}")
        
        # Demo successful upload
        return jsonify({
            'success': True,
            'message': f'Successfully processed {file.filename}'
        })
    except Exception as e:
        logger.exception(f"Error during file upload: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred during upload: {str(e)}'})

@app.route('/api/execute_code', methods=['POST'])
def api_execute_code():
    # BREAKPOINT: Add a breakpoint here to debug code execution
    data = request.json
    code = data.get('code', '')
    language = data.get('language', 'python')
    logger.debug(f"Executing {language} code")
    
    # Demo code execution
    return jsonify({
        'success': True,
        'stdout': 'This is a demo output. In a real application, this would execute the code in a sandboxed environment.'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)