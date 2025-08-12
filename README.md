# Flask Transformers API Wrapper - Deployment Guide

## Overview
This Flask application provides an OpenAI-compatible API wrapper around HuggingFace Transformers models, allowing you to serve language models remotely without installing them locally.

## Features
- **Implicit model loading**: Specify model in request, auto-loads if needed
- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`)
- Model discovery via `/v1/models` endpoint
- Seamless model switching between requests
- Memory-efficient model management
- GPU acceleration support
- Health monitoring
- Streaming support (basic implementation)
- Token usage tracking

## User Workflow
1. **List models**: Call `/v1/models` to see available models
2. **Make requests**: Use any model name in chat/completion requests
3. **Auto-loading**: Server automatically loads the specified model
4. **Seamless switching**: Change models between requests without manual loading

## Server Setup

### 1. Prerequisites
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# For GPU support (optional but recommended)
# Follow NVIDIA CUDA installation guide for your system
```

### 2. Installation
```bash
# Clone or create your project directory
mkdir transformers-api
cd transformers-api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Basic Deployment

#### Option A: Development Server
```bash
# Start the development server
python app.py

# The API will be available at http://your-server:5000
```

#### Option B: Production with Gunicorn
```bash
# Install gunicorn
pip install gunicorn

# Start production server
gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 300 app:app

# For GPU servers, use single worker to avoid memory conflicts
gunicorn --bind 0.0.0.0:5000 --workers 1 --worker-class sync --timeout 600 app:app
```

#### Option C: Using systemd (Linux)
Create a systemd service file:

```bash
sudo nano /etc/systemd/system/transformers-api.service
```

```ini
[Unit]
Description=Transformers API
After=network.target

[Service]
Type=exec
User=your-username
Group=your-group
WorkingDirectory=/path/to/transformers-api
Environment=PATH=/path/to/transformers-api/venv/bin
ExecStart=/path/to/transformers-api/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 600 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable transformers-api
sudo systemctl start transformers-api
```

### 4. Docker Deployment (Recommended for Production)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Create non-root user
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "600", "app:app"]
```

Build and run:
```bash
docker build -t transformers-api .
docker run -p 5000:5000 --gpus all transformers-api  # Add --gpus all for GPU support
```

### 5. Reverse Proxy with Nginx (Optional)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Important for streaming
        proxy_buffering off;
        proxy_cache off;
        
        # Increase timeouts for model loading
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

## API Usage

### 1. List Available Models
```bash
curl http://your-server:5000/v1/models
```

### 2. Chat Completion (Model Auto-loads)
```bash
curl -X POST http://your-server:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-small",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### 3. Text Completion (Model Auto-loads)
```bash
curl -X POST http://your-server:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 4. Switch Models Seamlessly
```bash
# First request with DialoGPT
curl -X POST http://your-server:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "microsoft/DialoGPT-small", "messages": [{"role": "user", "content": "Hi"}]}'

# Next request with GPT-2 (automatically switches)
curl -X POST http://your-server:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "prompt": "Hello world"}'
```

### 5. Health Check
```bash
curl http://your-server:5000/health
```

## Recommended Models

### Small Models (Good for testing)
- `microsoft/DialoGPT-small` (117M parameters)
- `distilgpt2` (82M parameters)
- `gpt2` (124M parameters)

### Medium Models
- `microsoft/DialoGPT-medium` (345M parameters)
- `gpt2-medium` (355M parameters)

### Large Models (Require significant GPU memory)
- `microsoft/DialoGPT-large` (762M parameters)
- `gpt2-large` (774M parameters)
- `facebook/opt-1.3b` (1.3B parameters)

## Performance Optimization

### Memory Management
- Use `torch.float16` for GPU inference (automatically enabled)
- Only load one model at a time (current implementation)
- Models are automatically cleared when loading new ones

### GPU Optimization
- Use CUDA-compatible PyTorch installation
- Set appropriate GPU memory fraction
- Consider using model parallelism for large models

### Scaling
- Use multiple instances behind a load balancer
- Each instance should run one model to avoid GPU memory conflicts
- Consider model-specific instances for different use cases

## Security Considerations

1. **Authentication**: Add API key authentication for production
2. **Rate Limiting**: Implement request rate limiting
3. **Input Validation**: Validate and sanitize all inputs
4. **Firewall**: Restrict access to necessary ports only
5. **HTTPS**: Use SSL/TLS certificates for encrypted communication

## Monitoring

### Health Endpoint
The `/health` endpoint provides:
- Service status
- Current loaded model
- CUDA availability
- Basic system info

### Logging
- All requests and errors are logged
- Monitor disk space (model downloads can be large)
- Monitor GPU/CPU usage
- Track response times

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce model size or increase system RAM/GPU memory
2. **CUDA Errors**: Check CUDA installation and PyTorch compatibility
3. **Model Loading Timeout**: Increase timeout values in configuration
4. **Slow Responses**: Consider using smaller models or GPU acceleration

### Model Loading Times
- First load: Downloads model from HuggingFace (can be slow)
- Subsequent loads: Loads from local cache (faster)
- Model switching: Previous model is cleared from memory

## Environment Variables

You can configure the application using environment variables:

```bash
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device
export TRANSFORMERS_CACHE=/path/to/cache  # Model cache directory
export FLASK_ENV=production
```

## Client Integration

Use the provided Python client or create your own in any language that can make HTTP requests. The API is compatible with OpenAI's client libraries with minor modifications to the base URL.