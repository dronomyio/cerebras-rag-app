#!/usr/bin/env python3
"""
Debug entry point for the Cerebras RAG Web Application
This file sets up debugpy and runs the Flask app in debug mode
"""

import os
import debugpy
from app import app

# Enable debugpy
debugpy.listen(("0.0.0.0", 5678))
print("⚡️ Debugpy is listening on 0.0.0.0:5678")
print("🔍 VS Code can attach to this process (don't need to wait)")

# Enable Flask's debug mode
app.config['DEBUG'] = True
app.run(host='0.0.0.0', port=5000, use_reloader=False)