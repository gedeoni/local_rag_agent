import re

def clean_reasoning_output(text: str) -> str:
    """Removes <think>...</think> tags and returns the actual content."""
    if not text:
        return ""
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Also remove any markdown headers that might be added by model
    clean_text = re.sub(r'^.*?optimized_query:?\s*', '', clean_text, flags=re.IGNORECASE).strip()
    return clean_text
