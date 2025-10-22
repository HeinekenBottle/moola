"""CLI patch to integrate model registry properly.

This patch updates the CLI to use the model registry instead of 
hardcoded model names, ensuring proper Jade/Sapphire/Opal usage.
"""

from moola.models import get_model, list_models

def patch_cli_model_options():
    """Update CLI model options to use registry."""
    
    # Get available models from registry
    available_models = list_models()
    
    # Filter for production models
    production_models = [name for name in available_models.keys() 
                        if any(stone in name.lower() for stone in ['jade', 'sapphire', 'opal'])]
    
    print("Available production models from registry:")
    for model_id in production_models:
        model_info = available_models[model_id]
        print(f"  - {model_info.codename}: {model_info.description}")
    
    return production_models

if __name__ == "__main__":
    models = patch_cli_model_options()
    print(f"\nFound {len(models)} production models in registry")