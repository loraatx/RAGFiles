FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY query_rag.py .

# Create directory for ChromaDB (will be mounted or copied)
RUN mkdir -p /app/chroma_db

# Copy chroma_db if it exists in the build context
# If not present, the app will start but return 503 until DB is mounted
COPY chroma_db/ /app/chroma_db/ 2>/dev/null || true

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application with optimized settings for Cloud Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
