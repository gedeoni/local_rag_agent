import subprocess
import requests
import logging

logger = logging.getLogger(__name__)

def is_ollama_installed() -> bool:
    """Check if the Ollama CLI is installed on the system."""
    try:
        # Just running 'ollama --version' is a quick way to check if the binary exists
        result = subprocess.run(
            ["ollama", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_ollama_models() -> list[str]:
    """Check if Ollama daemon is running and retrieve list of pulled models."""
    try:
        # Hitting the local API directly is the safest way to verify the background service is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2.0)
        response.raise_for_status()
        
        models_data = response.json().get("models", [])
        models = [model["name"] for model in models_data]
        return models
    except requests.exceptions.RequestException as e:
        logger.warning(f"Ollama API not available (Service might be down): {e}")
        return []
