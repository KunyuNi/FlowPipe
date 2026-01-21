#!/usr/bin/env python3
"""
Embedder configuration file
Used to configure which embedder to use
"""

# Embedder configuration
EMBEDDER_CONFIG = {
    # Currently used embedder type
    "type": "llm",  # Options: "gte", "llm", "tapas"
    
    # LLM Embedder configuration
    "llm": {
        "dataset": "NY",  # Dataset: "NY", "SG", "TKY"
        "llm": "llama2",  # LLM Model: "llama2", "llama3", "chatglm2", "chatglm3", "gpt2", "gpt2_medium", "gpt2_large", "gpt2_xl"
        "prompt_type": "time",  # Prompt type: "address", "time", "cat_nearby"
    },
    
    # GTE Embedder configuration
    "gte": {
        "model_path": "/mnt/mydisk2/anoy/flowpipe/embed/gte-large-en-v1-5",
    },
    
    # Tapas Embedder configuration
    "tapas": {
        "model_path": "google/tapas-base",
    }
}

def get_embedder_config():
    """Get embedder configuration"""
    return EMBEDDER_CONFIG

def update_embedder_config(embedder_type: str, **kwargs):
    """
    Update embedder configuration
    
    Args:
        embedder_type: Embedder type
        **kwargs: Configuration parameters
    """
    global EMBEDDER_CONFIG
    
    EMBEDDER_CONFIG["type"] = embedder_type
    
    if embedder_type == "llm":
        EMBEDDER_CONFIG["llm"].update(kwargs)
    elif embedder_type == "gte":
        EMBEDDER_CONFIG["gte"].update(kwargs)
    elif embedder_type == "tapas":
        EMBEDDER_CONFIG["tapas"].update(kwargs)
    else:
        raise ValueError(f"Unsupported embedder type: {embedder_type}")

# Preset configurations
PRESET_CONFIGS = {
    "llama2_time": {
        "type": "llm",
        "llm": {"dataset": "NY", "llm": "llama2", "prompt_type": "time"}
    },
    "llama2_address": {
        "type": "llm", 
        "llm": {"dataset": "NY", "llm": "llama2", "prompt_type": "address"}
    },
    "llama3_time": {
        "type": "llm",
        "llm": {"dataset": "NY", "llm": "llama3", "prompt_type": "time"}
    },
    "chatglm3_time": {
        "type": "llm",
        "llm": {"dataset": "NY", "llm": "chatglm3", "prompt_type": "time"}
    },
    "gte": {
        "type": "gte"
    }
}

def use_preset_config(preset_name: str):
    """
    Use preset configuration
    
    Args:
        preset_name: Preset configuration name
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Preset configuration does not exist: {preset_name}")
    
    global EMBEDDER_CONFIG
    EMBEDDER_CONFIG = PRESET_CONFIGS[preset_name].copy()

if __name__ == "__main__":
    # Example usage
    print("Current config:")
    print(EMBEDDER_CONFIG)
    
    print("\nAvailable preset configs:")
    for name in PRESET_CONFIGS.keys():
        print(f"  - {name}")
    
    # Switch to LLaMA3 config
    print("\nSwitching to LLaMA3 config...")
    use_preset_config("llama3_time")
    print("New config:")
    print(EMBEDDER_CONFIG) 