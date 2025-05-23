#!/bin/bash

# Stop containers if they are running
docker-compose stop webapp nginx

# Comment out the debug CMD and uncomment the production CMD in Dockerfile
sed -i.bak 's/^#CMD \["gunicorn", "--bind", "0.0.0.0:5000", "app:app"\]/CMD \["gunicorn", "--bind", "0.0.0.0:5000", "app:app"\]/g' src/webapp/Dockerfile
sed -i.bak 's/^CMD \["python", "debug_app.py"\]/# CMD \["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "app.py"\]/g' src/webapp/Dockerfile

# Build and start the containers
docker-compose up -d --build webapp nginx

echo "🚀 Production mode enabled!"
echo "✅ The webapp is now running in production mode"