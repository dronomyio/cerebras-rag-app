# Cerebras RAG App Troubleshooting Summary

## Docker Commands Used

### Container Inspection and Debugging
```bash
# Check container status
docker-compose ps

# View logs for webapp container
docker-compose logs webapp
docker-compose logs --tail=50 webapp

# View logs for nginx container
docker-compose logs nginx

# Run shell commands in containers
docker-compose exec webapp /bin/sh
docker-compose exec webapp /bin/bash
docker-compose exec -T webapp ls -la
docker-compose exec -T webapp cat app.py
docker-compose exec -T webapp ls -la templates
docker-compose exec -T webapp cat templates/login.html
docker-compose exec -T webapp cat templates/chat.html
docker-compose exec -T webapp cat templates/register.html
docker-compose exec -T nginx cat /etc/nginx/conf.d/default.conf
docker-compose exec -T nginx cat /etc/nginx/nginx.conf
docker-compose exec -T webapp mkdir -p /app/static

# Test connectivity between containers
docker-compose exec -T nginx ping -c 2 webapp
docker-compose exec -T nginx sh -c "apk add --no-cache busybox-extras && telnet webapp 5000"
```

### Container Management
```bash
# Stop specific containers
docker-compose stop webapp
docker-compose stop nginx

# Rebuild containers
docker-compose build webapp

# Start specific containers
docker-compose up -d webapp
docker-compose up -d nginx

# Stop and restart webapp and nginx together
docker-compose stop webapp nginx && docker-compose build webapp && docker-compose up -d webapp nginx

# Restart the entire application
docker-compose down && docker-compose up -d
```

### Testing Connectivity
```bash
# Test HTTP connectivity
curl -v http://localhost:8088
curl -v http://localhost:8088/chat
curl -v http://localhost:5000
```

## Changes Made to Fix Issues

### 1. Fixed Missing Routes in Flask App
- Added `/register` route to handle both GET and POST requests
- Added `/logout` route to redirect to the login page

### 2. Fixed File Upload Size Limitations
- Added `client_max_body_size 100M;` to Nginx configuration to allow larger uploads
- Added `MAX_CONTENT_LENGTH = 100 * 1024 * 1024` (100MB) to Flask app configuration
- Added upload directory configuration and creation
- Added improved error handling and logging for file uploads

### 3. Configured Static Files Directory
- Created static directory in webapp container: `/app/static`

## Modified Files

### 1. `/Users/macmachine/tools/drone_project_idea/Blogs/Cerebras/cerebras-rag-app/src/webapp/app.py`
- Added missing routes (register, logout)
- Added file upload size configuration
- Added improved error handling and logging for uploads
- Added upload directory configuration

### 2. `/Users/macmachine/tools/drone_project_idea/Blogs/Cerebras/cerebras-rag-app/config/nginx.conf`
- Added `client_max_body_size 100M;` directive to increase the upload limit

## Troubleshooting Process
1. Identified missing routes in Flask app causing 502 errors
2. Fixed app.py by adding missing routes
3. Created static directory for nginx
4. Restarted all containers to fix connectivity issues
5. Identified file upload size limitations
6. Modified nginx and Flask configurations to allow larger uploads
7. Added better error handling and logging
8. Rebuilt and restarted the affected containers

The application is now fully functional with all routes working properly and supporting file uploads up to 100MB.