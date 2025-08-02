# Configuration file for Kinematics AI

def get_config():
    """
    Returns configuration dictionary with API keys and settings
    """
    return {
        # Anthropic API key for Claude vision analysis
        'anthropic_api_key': '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',
        
        # Optional: Other configuration settings
        'vision_model': 'claude-sonnet-4-20250514',
        'max_tokens': 1000,
        'confidence_threshold': 0.7,
    } 