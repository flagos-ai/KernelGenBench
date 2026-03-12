#!/usr/bin/env python3
"""
List available models from panda API server.
"""

import os
import sys
from openai import OpenAI

# Check for API key
api_key = os.environ.get("PANDA_API_KEY")
if not api_key:
    print("Error: PANDA_API_KEY not found in environment")
    print("Please set it with: export PANDA_API_KEY=your_key")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.pandalla.ai/v1",
    timeout=30,
    max_retries=1,
)

print("Querying panda API for available models...\n")

# Try to list models using OpenAI-compatible API
try:
    models = client.models.list()
    print("=" * 60)
    print("Available models from panda API:")
    print("=" * 60)
    for model in models.data:
        print(f"  - {model.id}")
        if hasattr(model, 'owned_by'):
            print(f"    (owned_by: {model.owned_by})")
    print(f"\nTotal: {len(models.data)} models")
except Exception as e:
    print(f"Error listing models via models.list(): {e}")
    print("\nTrying alternative method...")
    
    # Try to get error message with a test request
    test_models = [
        "gpt-5-2025-08-07",
        "gpt-5",
        "claude-opus-4-1-20250805",
        "gpt-4o-2024-08-06",
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "deepseek-v3-0324",
    ]
    
    print("\nTesting common model names:")
    print("=" * 60)
    for model_name in test_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            print(f"  ✓ {model_name} - Available")
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg or "无可用渠道" in error_msg:
                print(f"  ✗ {model_name} - Not available")
            else:
                print(f"  ? {model_name} - Error: {error_msg[:100]}")

