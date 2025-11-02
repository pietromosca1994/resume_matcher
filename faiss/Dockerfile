# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional: for FAISS CPU)
RUN apt-get update && apt-get install -y build-essential libopenblas-dev libomp-dev && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port
EXPOSE 8000

# Default environment variables
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Start the API with Uvicorn
CMD ["python", "-m", "app.api"]
