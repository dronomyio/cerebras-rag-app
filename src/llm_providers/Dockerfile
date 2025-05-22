FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set the Python path
ENV PYTHONPATH="/app:/src"

# No standalone service, this is a library
CMD ["echo", "LLM Providers module is not a standalone service"]