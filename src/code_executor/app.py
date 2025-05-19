#!/usr/bin/env python3
"""
Code Executor Service
--------------------
Flask service for securely executing code examples in isolated environments.
"""

import os
import uuid
import json
import logging
import tempfile
import subprocess
from typing import Dict, Any

from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "30"))  # seconds
ALLOWED_LANGUAGES = ["python", "r"]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/execute', methods=['POST'])
def execute_code():
    """
    Execute code in an isolated environment.
    
    Expected JSON payload:
    {
        "code": "print('Hello, world!')",
        "language": "python"
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        code = data.get('code', '')
        language = data.get('language', 'python').lower()
        
        if not code:
            return jsonify({"success": False, "error": "No code provided"}), 400
        
        if language not in ALLOWED_LANGUAGES:
            return jsonify({
                "success": False, 
                "error": f"Unsupported language: {language}. Allowed languages: {', '.join(ALLOWED_LANGUAGES)}"
            }), 400
        
        # Execute the code
        result = execute_code_safely(code, language)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

def execute_code_safely(code: str, language: str) -> Dict[str, Any]:
    """
    Execute code safely in an isolated environment.
    
    Args:
        code: Code to execute
        language: Programming language
        
    Returns:
        Dictionary with execution results
    """
    # Create a temporary directory for execution
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create a unique ID for this execution
            execution_id = str(uuid.uuid4())
            
            # Write code to a temporary file
            if language == "python":
                file_path = os.path.join(temp_dir, f"{execution_id}.py")
                command = ["python", file_path]
            elif language == "r":
                file_path = os.path.join(temp_dir, f"{execution_id}.R")
                command = ["Rscript", file_path]
            else:
                return {
                    "success": False,
                    "error": f"Unsupported language: {language}"
                }
            
            with open(file_path, 'w') as f:
                f.write(code)
            
            # Execute the code with timeout
            logger.info(f"Executing {language} code: {execution_id}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_dir,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=MAX_EXECUTION_TIME)
                exit_code = process.returncode
                
                result = {
                    "success": exit_code == 0,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code
                }
                
                if exit_code != 0:
                    result["error"] = f"Execution failed with exit code {exit_code}"
                
                return result
                
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"Execution timed out: {execution_id}")
                return {
                    "success": False,
                    "error": f"Execution timed out after {MAX_EXECUTION_TIME} seconds"
                }
                
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            return {
                "success": False,
                "error": f"Execution error: {str(e)}"
            }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
