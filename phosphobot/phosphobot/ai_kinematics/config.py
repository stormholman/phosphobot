# Configuration file for Kinematics AI
import os

def get_config():
    """
    Returns configuration dictionary with API keys and settings
    """
    return {
        # Anthropic API key for Claude vision analysis
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),
        
        # Optional: Other configuration settings
        'vision_model': 'claude-sonnet-4-20250514',
        'max_tokens': 1000,
        'confidence_threshold': 0.7,
    } 