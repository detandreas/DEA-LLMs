#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to find all models from models2.json that have
artificial_analysis_intelligence_index above 50
"""

import json
from pathlib import Path


def find_models_with_high_intelligence_index(json_file: str, threshold: float = 50.0) -> list:
    """
    Finds all models with artificial_analysis_intelligence_index above threshold.
    
    Args:
        json_file: Path to the JSON file containing model data
        threshold: Minimum intelligence index value (default: 50.0)
    
    Returns:
        List of models matching the criteria
    """
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get models list
    models = data.get('data', [])
    
    # Filter models with intelligence_index > threshold
    matching_models = []
    for model in models:
        evaluations = model.get('evaluations', {})
        intelligence_index = evaluations.get('artificial_analysis_intelligence_index')
        
        # Check if intelligence_index exists and is above threshold
        if intelligence_index is not None and intelligence_index > threshold:
            matching_models.append(model)
    
    return matching_models


def find_models_with_high_intelligence_and_price(json_file: str, intelligence_threshold: float = 50.0, price_threshold: float = 0.6) -> list:
    """
    Finds all models with artificial_analysis_intelligence_index above threshold
    and price_1m_blended_3_to_1 above price threshold.
    
    Args:
        json_file: Path to the JSON file containing model data
        intelligence_threshold: Minimum intelligence index value (default: 50.0)
        price_threshold: Minimum blended price value (default: 0.6)
    
    Returns:
        List of models matching both criteria
    """
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get models list
    models = data.get('data', [])
    
    # Filter models with both criteria
    matching_models = []
    for model in models:
        evaluations = model.get('evaluations', {})
        pricing = model.get('pricing', {})
        
        intelligence_index = evaluations.get('artificial_analysis_intelligence_index')
        blended_price = pricing.get('price_1m_blended_3_to_1')
        
        # Check if both values exist and meet the thresholds
        if (intelligence_index is not None and intelligence_index > intelligence_threshold and
            blended_price is not None and blended_price > price_threshold):
            matching_models.append(model)
    
    return matching_models


def save_models_to_json(models: list, output_file: str, original_data: dict = None):
    """
    Saves filtered models to a new JSON file.
    
    Args:
        models: List of models to save
        output_file: Path to output JSON file
        original_data: Original JSON data to preserve metadata
    """
    # Prepare output data structure
    output_data = {
        "status": original_data.get("status", 200) if original_data else 200,
        "prompt_options": original_data.get("prompt_options", {}) if original_data else {},
        "data": models
    }
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nResults saved to: {output_file}")


def main():
    # Path to models2.json
    json_file = Path(__file__).parent / 'data' / 'models2.json'
    
    # Read original data to preserve metadata
    with open(json_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Find models with intelligence_index > 50
    matching_models = find_models_with_high_intelligence_index(str(json_file), threshold=50.0)
    
    # Print results
    print(f"Found {len(matching_models)} models with artificial_analysis_intelligence_index > 50\n")
    print("=" * 80)
    
    for i, model in enumerate(matching_models, 1):
        name = model.get('name', 'Unknown')
        creator = model.get('model_creator', {}).get('name', 'Unknown')
        intelligence_index = model.get('evaluations', {}).get('artificial_analysis_intelligence_index', 'N/A')
        release_date = model.get('release_date', 'N/A')
        
        print(f"\n{i}. {name}")
        print(f"   Creator: {creator}")
        print(f"   Intelligence Index: {intelligence_index}")
        print(f"   Release Date: {release_date}")
    
    print("\n" + "=" * 80)
    print(f"\nTotal: {len(matching_models)} models")
    
    # Save to new JSON file
    output_file = Path(__file__).parent / 'data' / 'models_high_intelligence.json'
    save_models_to_json(matching_models, str(output_file), original_data)
    
    # Find models with intelligence_index > 50 and price > 0.6
    print("\n" + "=" * 80)
    print("Searching for models with intelligence_index > 50 AND price_blended > 0.6...")
    print("=" * 80)
    
    matching_models_price = find_models_with_high_intelligence_and_price(
        str(json_file), 
        intelligence_threshold=50.0, 
        price_threshold=0.6
    )
    
    # Print results
    print(f"\nFound {len(matching_models_price)} models with intelligence_index > 50 AND price_blended > 0.6\n")
    print("=" * 80)
    
    for i, model in enumerate(matching_models_price, 1):
        name = model.get('name', 'Unknown')
        creator = model.get('model_creator', {}).get('name', 'Unknown')
        intelligence_index = model.get('evaluations', {}).get('artificial_analysis_intelligence_index', 'N/A')
        blended_price = model.get('pricing', {}).get('price_1m_blended_3_to_1', 'N/A')
        release_date = model.get('release_date', 'N/A')
        
        print(f"\n{i}. {name}")
        print(f"   Creator: {creator}")
        print(f"   Intelligence Index: {intelligence_index}")
        print(f"   Price Blended: {blended_price}")
        print(f"   Release Date: {release_date}")
    
    print("\n" + "=" * 80)
    print(f"\nTotal: {len(matching_models_price)} models")
    
    # Save to another JSON file
    output_file_price = Path(__file__).parent / 'data' / 'models_high_intelligence_price.json'
    save_models_to_json(matching_models_price, str(output_file_price), original_data)


if __name__ == "__main__":
    main()

