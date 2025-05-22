FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create static directory
RUN mkdir -p /app/static

# Set the Python path to include parent src directory
ENV PYTHONPATH="/app:/src"

# Run the Flask application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]