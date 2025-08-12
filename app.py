import os
import subprocess
import time
import signal
import sys
from flask import Flask, request, jsonify, Response
import requests

app = Flask(__name__)

# The internal server URL where 'transformers serve' is running
INTERNAL_SERVER_URL = os.environ.get("INTERNAL_SERVER_URL", "http://localhost:8000").rstrip("/")

# Global variable to store the subprocess
transformers_process = None


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health endpoint and internal connectivity check."""
    try:
        upstream = requests.get(f"{INTERNAL_SERVER_URL}/health", timeout=2)
        upstream_ok = upstream.ok
    except Exception:
        upstream_ok = False

    return jsonify({
        "status": "ok",
        "internal_server": INTERNAL_SERVER_URL,
        "internal_reachable": upstream_ok
    })


@app.route('/generate', methods=['POST'])
def proxy_generate():
    """Proxy POST /generate to the internal transformers server."""
    try:
        upstream_resp = requests.post(
            f"{INTERNAL_SERVER_URL}/generate",
            params=request.args,  # forward query params if any
            data=request.get_data(),
            headers={k: v for k, v in request.headers if k.lower() != 'host'},
            timeout=None,
        )
    except requests.RequestException as exc:
        return jsonify({"error": "Bad Gateway", "detail": str(exc)}), 502

    # Pass through JSON if possible, else raw content
    content_type = upstream_resp.headers.get('content-type', '')
    if 'application/json' in content_type:
        return jsonify(upstream_resp.json()), upstream_resp.status_code
    return Response(
        upstream_resp.content,
        status=upstream_resp.status_code,
        headers={k: v for k, v in upstream_resp.headers.items() if k.lower() not in {
            'content-encoding', 'transfer-encoding', 'connection'
        }}
    )


# Optional: generic catch-all proxy to forward any path/method to the internal server
@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
def proxy_all(path):
    # Avoid intercepting our explicit /health and /generate routes
    if path in {"health", "generate"}:
        return jsonify({"error": "Not Found"}), 404

    upstream_url = f"{INTERNAL_SERVER_URL}/{path}"
    try:
        upstream_resp = requests.request(
            method=request.method,
            url=upstream_url,
            params=request.args,
            data=request.get_data(),
            headers={k: v for k, v in request.headers if k.lower() != 'host'},
            allow_redirects=False,
            timeout=None,
        )
    except requests.RequestException as exc:
        return jsonify({"error": "Bad Gateway", "detail": str(exc)}), 502

    # Stream or return content based on content-type
    excluded_headers = {'content-encoding', 'transfer-encoding', 'connection'}
    headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in excluded_headers}
    return Response(upstream_resp.content, status=upstream_resp.status_code, headers=headers)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


def start_transformers_server():
    """Start the transformers serve subprocess"""
    global transformers_process
    
    try:
        # Check if transformers serve is available
        result = subprocess.run(['transformers', '--help'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Warning: 'transformers' command not found. Please install transformers[serving]")
            return False
        
        print("Starting transformers serve on port 8000...")
        transformers_process = subprocess.Popen(
            ['transformers', 'serve', '--port', '8000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for the server to start
        time.sleep(3)
        
        # Check if the process is still running
        if transformers_process.poll() is None:
            print(f"Transformers serve started successfully (PID: {transformers_process.pid})")
            return True
        else:
            print("Failed to start transformers serve")
            return False
            
    except Exception as e:
        print(f"Error starting transformers serve: {e}")
        return False

def stop_transformers_server():
    """Stop the transformers serve subprocess"""
    global transformers_process
    
    if transformers_process:
        print(f"Stopping transformers serve (PID: {transformers_process.pid})...")
        transformers_process.terminate()
        try:
            transformers_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            transformers_process.kill()
        print("Transformers serve stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutting down...")
    stop_transformers_server()
    sys.exit(0)

if __name__ == '__main__':
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start transformers serve
    if not start_transformers_server():
        print("Warning: Could not start transformers serve. The proxy may not work correctly.")
    
    try:
        print("Starting Flask proxy server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        # Ensure cleanup on exit
        stop_transformers_server()