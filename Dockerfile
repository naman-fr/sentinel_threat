# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    cmake \
    build-essential \
    git \
    wget \
    libopencv-dev \
    libspdlog-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build C++ components
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# Create non-root user
RUN useradd -m -s /bin/bash sentinel
RUN chown -R sentinel:sentinel /app
USER sentinel

# Set environment variables for the application
ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Expose ports
EXPOSE 8000 9090

# Set entrypoint
ENTRYPOINT ["python3", "src/main.py"] 