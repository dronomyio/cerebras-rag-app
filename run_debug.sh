#!/bin/bash

# Stop containers if they are running
docker-compose stop webapp nginx

# Comment out the production CMD and uncomment the debug CMD in Dockerfile
sed -i.bak 's/^CMD \["gunicorn", "--bind", "0.0.0.0:5000", "app:app"\]/#CMD \["gunicorn", "--bind", "0.0.0.0:5000", "app:app"\]/g' src/webapp/Dockerfile
sed -i.bak 's/^# CMD \["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "app.py"\]/CMD \["python", "debug_app.py"\]/g' src/webapp/Dockerfile

# Build and start the containers
docker-compose up -d --build webapp nginx

echo "🚀 Debug mode enabled!"
echo "✅ The webapp is now running in debug mode"
echo "⚡️ Debugpy is listening on port 5678"
echo "🔍 Open VS Code, add breakpoints, and start debugging with the 'Python: Remote Attach' configuration"