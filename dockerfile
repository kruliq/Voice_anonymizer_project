# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY base_functionality/ ./base_functionality/
COPY Speech_to_text_demo/ ./Speech_to_text_demo/
COPY tests/ ./tests/
COPY LICENSE README.md ./

# Create necessary directories
RUN mkdir -p /app/base_functionality/web/uploads /app/base_functionality/web/processed && \
    chmod 755 /app/base_functionality/web/uploads /app/base_functionality/web/processed

# Set environment variables
ENV FLASK_APP=/app/base_functionality/web/web_test.py
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 5000

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]