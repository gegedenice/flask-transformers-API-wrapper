import requests
import json
from typing import Dict, Any, List

class TransformersAPIClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the client with the base URL of your Flask API
        Args:
            base_url: The base URL of your deployed Flask API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy and what model is loaded"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        response = self.session.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_name: str, task: str = "text-generation") -> Dict[str, Any]:
        """
        Load a model on the server (OPTIONAL - models are auto-loaded on first use)
        Args:
            model_name: HuggingFace model name (e.g., "microsoft/DialoGPT-small")
            task: Task type ("text-generation" or "text2text-generation")
        """
        data = {
            "model_name": model_name,
            "task": task
        }
        response = self.session.post(f"{self.base_url}/load_model", json=data)
        response.raise_for_status()
        return response.json()
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = "microsoft/DialoGPT-small",
                       max_tokens: int = 150,
                       temperature: float = 0.7,
                       stream: bool = False) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use (will be auto-loaded if needed)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
        """
        data = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        response = self.session.post(f"{self.base_url}/v1/chat/completions", json=data)
        response.raise_for_status()
        return response.json()
    
    def completion(self, 
                   prompt: str,
                   model: str = "microsoft/DialoGPT-small",
                   max_tokens: int = 150,
                   temperature: float = 0.7) -> Dict[str, Any]:
        """
        OpenAI-compatible text completion
        Args:
            prompt: Input prompt text
            model: Model name to use (will be auto-loaded if needed)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        data = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = self.session.post(f"{self.base_url}/v1/completions", json=data)
        response.raise_for_status()
        return response.json()
    
    def simple_generate(self, 
                       prompt: str,
                       model: str = "microsoft/DialoGPT-small",
                       max_length: int = 150,
                       temperature: float = 0.7) -> Dict[str, Any]:
        """
        Simple generation endpoint
        Args:
            prompt: Input prompt text
            model: Model name to use (will be auto-loaded if needed)
            max_length: Maximum length to generate
            temperature: Sampling temperature
        """
        data = {
            "prompt": prompt,
            "model": model,
            "max_length": max_length,
            "temperature": temperature
        }
        response = self.session.post(f"{self.base_url}/generate", json=data)
        response.raise_for_status()
        return response.json()

# Example usage
if __name__ == "__main__":
    # Replace with your server URL
    client = TransformersAPIClient("http://your-server:5000")
    
    # Check health
    health = client.health_check()
    print("Health check:", health)
    
    # List available models
    print("\n=== Available Models ===")
    models = client.list_models()
    for model in models["data"][:5]:  # Show first 5 models
        print(f"- {model['id']}")
    
    # Example 1: Simple generation (model auto-loads)
    print("\n=== Simple Generation ===")
    result = client.simple_generate(
        "Hello, how are you?", 
        model="microsoft/DialoGPT-small",
        max_length=50
    )
    print("Response:", result["generated_text"])
    
    # Example 2: OpenAI-style chat completion with different model
    print("\n=== Chat Completion ===")
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    result = client.chat_completion(
        messages, 
        model="gpt2",  # Different model - will auto-load
        max_tokens=100
    )
    print("Response:", result["choices"][0]["message"]["content"])
    
    # Example 3: OpenAI-style text completion
    print("\n=== Text Completion ===")
    result = client.completion(
        "The weather today is", 
        model="distilgpt2",  # Another model - will auto-load
        max_tokens=50
    )
    print("Response:", result["choices"][0]["text"])
    
    # Example 4: Multi-turn conversation
    print("\n=== Multi-turn Conversation ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke about programming."},
        {"role": "assistant", "content": "Why do programmers prefer dark mode? Because light attracts bugs!"},
        {"role": "user", "content": "That's funny! Tell me another one."}
    ]
    result = client.chat_completion(
        messages, 
        model="microsoft/DialoGPT-medium",  # Larger model for better conversations
        max_tokens=100
    )
    print("Response:", result["choices"][0]["message"]["content"])
    
    # Example 5: Using a T5 model for text-to-text generation
    print("\n=== T5 Model Example ===")
    result = client.completion(
        "translate English to German: Hello, how are you?",
        model="google/flan-t5-small",  # T5 model - will auto-detect task type
        max_tokens=50
    )
    print("Response:", result["choices"][0]["text"])