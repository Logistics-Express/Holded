FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Copy holded client from claude-tools
# Note: In production, this should be a proper package or copied at build time
RUN mkdir -p /app/lib
COPY lib/ /app/lib/

# Set Python path to include lib
ENV PYTHONPATH=/app:/app/lib

# Expose port
EXPOSE 8001

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
