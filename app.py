from flask import Flask, request, jsonify, stream_template
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import torch
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from threading import Lock
import gc

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model management
current_model = None
current_tokenizer = None
current_pipeline = None
current_model_name = None
model_lock = Lock()

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
    
    def load_model(self, model_name: str, task: str = "text-generation", **kwargs):
        """Load a model and tokenizer"""
        global current_model, current_tokenizer, current_pipeline, current_model_name
        
        with model_lock:
            if current_model_name == model_name:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            try:
                logger.info(f"Loading model: {model_name}")
                
                # Clear previous model from memory
                if current_model is not None:
                    del current_model
                    del current_tokenizer
                    del current_pipeline
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Load tokenizer
                current_tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Add padding token if not present
                if current_tokenizer.pad_token is None:
                    current_tokenizer.pad_token = current_tokenizer.eos_token
                
                # Load model based on task
                if task == "text-generation":
                    current_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        **kwargs
                    )
                    current_pipeline = pipeline(
                        "text-generation",
                        model=current_model,
                        tokenizer=current_tokenizer,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                elif task == "text2text-generation":
                    current_model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        **kwargs
                    )
                    current_pipeline = pipeline(
                        "text2text-generation",
                        model=current_model,
                        tokenizer=current_tokenizer,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                
                current_model_name = model_name
                logger.info(f"Successfully loaded model: {model_name} on {device}")
                return True
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                return False

model_manager = ModelManager()

def format_openai_response(text: str, model_name: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> Dict[str, Any]:
    """Format response in OpenAI-compatible format"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

def format_openai_stream_response(text: str, model_name: str, finish_reason: Optional[str] = None) -> str:
    """Format streaming response in OpenAI-compatible format"""
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "content": text
            } if text else {},
            "finish_reason": finish_reason
        }]
    }
    return f"data: {json.dumps(response)}\n\n"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": current_model_name is not None,
        "current_model": current_model_name,
        "cuda_available": torch.cuda.is_available()
    })

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models (OpenAI-compatible)"""
    # Popular models that work well with the API
    available_models = [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "distilgpt2",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "microsoft/CodeGPT-small-py",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B",
        "huggingface/CodeBERTa-small-v1"
    ]
    
    models = []
    for model_name in available_models:
        models.append({
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "huggingface",
            "permission": [],
            "root": model_name,
            "parent": None
        })
    
    return jsonify({"object": "list", "data": models})

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a specific model"""
    data = request.get_json()
    model_name = data.get('model_name')
    task = data.get('task', 'text-generation')
    
    if not model_name:
        return jsonify({"error": "model_name is required"}), 400
    
    success = model_manager.load_model(model_name, task)
    
    if success:
        return jsonify({
            "message": f"Model {model_name} loaded successfully",
            "model_name": model_name,
            "task": task
        })
    else:
        return jsonify({"error": f"Failed to load model {model_name}"}), 500

def ensure_model_loaded(model_name: str, task: str = "text-generation") -> bool:
    """Ensure the specified model is loaded, load it if not"""
    global current_model_name
    
    if current_model_name == model_name:
        return True
    
    # Determine task based on model name if not specified
    if task == "text-generation":
        if "flan-t5" in model_name.lower() or "t5" in model_name.lower():
            task = "text2text-generation"
    
    logger.info(f"Auto-loading model: {model_name}")
    success = model_manager.load_model(model_name, task)
    return success

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    data = request.get_json()
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 150)
    temperature = data.get('temperature', 0.7)
    stream = data.get('stream', False)
    model_name = data.get('model', 'microsoft/DialoGPT-small')  # Default model
    
    if not messages:
        return jsonify({"error": "messages are required"}), 400
    
    # Ensure the requested model is loaded
    if not ensure_model_loaded(model_name):
        return jsonify({"error": f"Failed to load model: {model_name}"}), 500
    
    # Convert messages to prompt
    if len(messages) == 1 and messages[0]['role'] == 'user':
        prompt = messages[0]['content']
    else:
        # Format conversation
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                prompt += f"Human: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
            elif role == 'system':
                prompt += f"System: {content}\n"
        prompt += "Assistant: "
    
    try:
        # Count input tokens (approximate)
        input_tokens = len(current_tokenizer.encode(prompt))
        
        # Generate response
        with torch.no_grad():
            response = current_pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=current_tokenizer.eos_token_id,
                return_full_text=False
            )
        
        generated_text = response[0]['generated_text'].strip()
        
        # Count output tokens (approximate)
        output_tokens = len(current_tokenizer.encode(generated_text))
        
        if stream:
            def generate_stream():
                # For simplicity, we'll send the entire response at once
                # In a more advanced implementation, you could generate token by token
                yield format_openai_stream_response(generated_text, model_name)
                yield format_openai_stream_response("", model_name, "stop")
                yield "data: [DONE]\n\n"
            
            return app.response_class(
                generate_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*'
                }
            )
        else:
            return jsonify(format_openai_response(
                generated_text, 
                current_model_name, 
                input_tokens, 
                output_tokens
            ))
    
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/v1/completions', methods=['POST'])
def completions():
    """OpenAI-compatible completions endpoint"""
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 150)
    temperature = data.get('temperature', 0.7)
    model_name = data.get('model', 'microsoft/DialoGPT-small')  # Default model
    
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    
    # Ensure the requested model is loaded
    if not ensure_model_loaded(model_name):
        return jsonify({"error": f"Failed to load model: {model_name}"}), 500
    
    try:
        # Count input tokens
        input_tokens = len(current_tokenizer.encode(prompt))
        
        with torch.no_grad():
            response = current_pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=current_tokenizer.eos_token_id,
                return_full_text=False
            )
        
        generated_text = response[0]['generated_text'].strip()
        output_tokens = len(current_tokenizer.encode(generated_text))
        
        return jsonify({
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,  # Use the requested model name
            "choices": [{
                "text": generated_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        })
    
    except Exception as e:
        logger.error(f"Error in completion: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Simple generation endpoint"""
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 150)
    temperature = data.get('temperature', 0.7)
    model_name = data.get('model', 'microsoft/DialoGPT-small')  # Default model
    
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    
    # Ensure the requested model is loaded
    if not ensure_model_loaded(model_name):
        return jsonify({"error": f"Failed to load model: {model_name}"}), 500
    
    try:
        with torch.no_grad():
            response = current_pipeline(
                prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=current_tokenizer.eos_token_id,
                return_full_text=False
            )
        
        return jsonify({
            "generated_text": response[0]['generated_text'].strip(),
            "prompt": prompt,
            "model": model_name  # Include model name in response
        })
    
    except Exception as e:
        logger.error(f"Error in generation: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # You can pre-load a model here if desired
    # model_manager.load_model("microsoft/DialoGPT-small", "text-generation")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)