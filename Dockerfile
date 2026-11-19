# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime container
FROM python:3.11-slim

WORKDIR /app

# Install runtime system packages (FFmpeg and OpenGL libraries for headlessly running OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project source files
COPY . .

# Expose Streamlit default dashboard port
EXPOSE 8501

# Run the CLI entrypoint script by default
ENTRYPOINT ["python3", "guardian.py"]
CMD ["ui"]
