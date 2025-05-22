FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set the Python path
ENV PYTHONPATH="/app:/src"

# Run the agent service
CMD ["python", "integration.py"]