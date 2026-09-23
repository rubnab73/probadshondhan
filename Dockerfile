FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with UID 1000 (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install python packages
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose Hugging Face Space port
EXPOSE 7860

# Start Flask application using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "app:app"]
