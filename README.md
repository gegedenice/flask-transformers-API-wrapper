# Flask Transformers Proxy - Simple API Gateway

## Overview
This Flask application acts as a **proxy gateway** that automatically launches and manages a `transformers serve` instance, then forwards API requests to it. It's designed to be a lightweight wrapper that makes the HuggingFace Transformers serving API easily accessible through a simple Flask interface.

## Architecture
```
Client Request → Flask Proxy (Port 5000) → Transformers Serve (Port 8000)
```

- **Flask Proxy**: Runs on port 5000, handles HTTP requests and forwards them
- **Transformers Serve**: Automatically launched on port 8000, handles the actual ML inference
- **Automatic Management**: The Flask app starts/stops the transformers server automatically

## Features
- **Automatic Server Launch**: `transformers serve` starts automatically when you run the Flask app
- **Simple Proxy**: Forwards all requests to the internal transformers server
- **Health Monitoring**: `/health` endpoint shows connectivity status
- **Graceful Shutdown**: Automatically cleans up subprocesses on exit
- **CORS Support**: Built-in CORS handling for web applications
- **Environment Configuration**: Configurable internal server URL

## Quick Start

### 1. Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or manually with pip
pip install flask flask_cors requests transformers[serving] torch accelerate
```

### 2. Run the Application
```bash
# Using uv (recommanded)
uv run app.py

# Or
python app.py
```

This will:
1. Start `transformers serve` on port 8000
2. Start Flask proxy on port 5000
3. Make the API available at `http://localhost:5000`

### 3. Test the API
```bash
# Health check
curl http://localhost:5000/health

# Generate text (forwarded to transformers serve)
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello, how are you?"}'
```

## API Endpoints

### `/health` (GET)
Health check endpoint that verifies both the Flask proxy and the internal transformers server.

**Response:**
```json
{
  "status": "ok",
  "internal_server": "http://localhost:8000",
  "internal_reachable": true
}
```

### `/generate` (POST)
Proxies POST requests to the internal transformers server's `/generate` endpoint.

**Usage:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Your prompt here"}'
```

### `/*` (Catch-all)
All other paths and HTTP methods are automatically forwarded to the internal transformers server.

## Configuration

### Environment Variables
```bash
# Override the internal server URL (default: http://localhost:8000)
export INTERNAL_SERVER_URL="http://localhost:8000"

# Or set a different port for transformers serve
export INTERNAL_SERVER_URL="http://localhost:9000"
```

### Port Configuration
- **Flask Proxy**: Port 5000 (configurable in `app.py`)
- **Transformers Serve**: Port 8000 (configurable in `start_transformers_server()`)

## Deployment Options

### Development
```bash
python app.py
```

### Production with Gunicorn
```bash
# Install gunicorn
pip install gunicorn

# Start with gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 1 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "app:app"]
```

### Systemd Service (Linux)
```ini
[Unit]
Description=Flask Transformers Proxy
After=network.target

[Service]
Type=exec
User=your-username
WorkingDirectory=/path/to/your/app
ExecStart=/path/to/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## How It Works

### Startup Process
1. **Check Dependencies**: Verifies `transformers` command is available
2. **Launch Transformers Serve**: Starts `transformers serve --port 8000` as subprocess
3. **Wait for Startup**: Waits 3 seconds for server to initialize
4. **Start Flask**: Launches Flask proxy on port 5000
5. **Ready**: Both servers are running and ready to handle requests

### Request Flow
```
Client → Flask Proxy (5000) → Transformers Serve (8000) → Response
```

### Shutdown Process
1. **Signal Handling**: Catches SIGINT/SIGTERM (Ctrl+C)
2. **Stop Transformers**: Terminates the subprocess gracefully
3. **Cleanup**: Ensures all resources are freed
4. **Exit**: Clean shutdown

## Troubleshooting

### Common Issues

#### "transformers command not found"
```bash
# Install transformers with serving support
pip install transformers[serving]
```

#### Port Already in Use
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process or change ports in app.py
```

#### Transformers Server Fails to Start
- Check if port 8000 is available
- Verify transformers[serving] is installed
- Check system resources (memory, disk space)

### Debug Mode
```python
# In app.py, change debug to True for more verbose output
app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
```

### Manual Server Management
If you prefer to manage the transformers server separately:

```bash
# Terminal 1: Start transformers serve
transformers serve --port 8000

# Terminal 2: Start Flask proxy (set environment variable)
export INTERNAL_SERVER_URL="http://localhost:8000"
python app.py
```

## Performance Considerations

### CPU vs GPU
- **CPU**: Works fine, slower inference, lower memory usage
- **GPU**: Faster inference, higher memory usage, requires CUDA setup

### Memory Management
- Transformers serve handles model loading/unloading
- Flask proxy is lightweight and adds minimal overhead
- Monitor memory usage during model loading

### Scaling
- Run multiple Flask instances behind a load balancer
- Each instance manages its own transformers serve subprocess
- Consider using different ports for each instance

## Security Notes

### Production Considerations
- Add authentication to the Flask proxy
- Use HTTPS in production
- Implement rate limiting
- Restrict network access to necessary ports only
- Consider running behind a reverse proxy (nginx, Apache)

### Network Security
- Flask proxy: External access (port 5000)
- Transformers serve: Internal only (port 8000)
- Use firewall rules to restrict access

## Monitoring

### Health Checks
- `/health` endpoint for basic connectivity
- Monitor subprocess status
- Check system resources

### Logging
- Flask logs: HTTP requests and proxy operations
- Transformers serve logs: Model operations and inference
- System logs: Process management and errors

## Examples

### Python Client
```python
import requests

# Health check
response = requests.get("http://localhost:5000/health")
print(response.json())

# Generate text
data = {"inputs": "Hello, world!"}
response = requests.post("http://localhost:5000/generate", json=data)
print(response.json())
```

### JavaScript/Node.js
```javascript
// Health check
fetch('http://localhost:5000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Generate text
fetch('http://localhost:5000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ inputs: 'Hello, world!' })
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL Examples
```bash
# Health check
curl http://localhost:5000/health

# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "The future of AI is"}'

# Any other transformers serve endpoint
curl http://localhost:5000/models
```

## License

[Your License Here]
