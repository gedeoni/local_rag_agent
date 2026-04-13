import subprocess
import requests
import logging
import streamlit as st

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

@st.cache_data(ttl=300)
def get_ollama_models(only_tools: bool = False) -> list[str]:
    """Check if Ollama daemon is running and retrieve list of pulled models.
    
    If only_tools is True, filters to show only models that support tool calling.
    """
    try:
        # Hitting the local API directly is the safest way to verify the background service is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2.0)
        response.raise_for_status()
        
        models_data = response.json().get("models", [])
        all_models = [model["name"] for model in models_data]
        
        if not only_tools:
            return all_models

        # Filter for tool compatibility
        tool_compatible_models = []
        for model_name in all_models:
            try:
                show_resp = requests.post(
                    "http://localhost:11434/api/show", 
                    json={"name": model_name}, 
                    timeout=2.0
                )
                if show_resp.status_code == 200:
                    capabilities = show_resp.json().get("capabilities", [])
                    if capabilities and "tools" in capabilities:
                        tool_compatible_models.append(model_name)
            except Exception as e:
                logger.warning(f"Could not check tools for {model_name}: {e}")
        
        return tool_compatible_models
    except requests.exceptions.RequestException as e:
        logger.warning(f"Ollama API not available (Service might be down): {e}")
        return []
