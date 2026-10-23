
FROM python:3.11-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements.txt

# Copy application code
COPY . .

# Default command (overridden by Docker Compose)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
