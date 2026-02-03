FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and lib
COPY main.py .
COPY lib/ lib/

# Set Python path to include lib
ENV PYTHONPATH=/app:/app/lib

# Expose port (Railway provides PORT env var)
EXPOSE 8001

# Run with PORT from environment
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}
