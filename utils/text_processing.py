import re

def clean_reasoning_output(text: str) -> str:
    """Removes <think>...</think> tags and returns the actual content."""
    if not text:
        return ""
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Also remove any markdown headers that might be added by model
    clean_text = re.sub(r'^.*?optimized_query:?\s*', '', clean_text, flags=re.IGNORECASE).strip()
    return clean_text

def parse_model_response(content: str) -> tuple[str | None, str]:
    """Parses model output to extract thinking process and remove system tags."""
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    else:
        thinking, answer = None, content

    # Clean any leaked system tags like <additional_information> from the model's output
    answer = re.sub(r'<additional_information>.*?</additional_information>', '', answer, flags=re.DOTALL).strip()
    
    return thinking, answer
