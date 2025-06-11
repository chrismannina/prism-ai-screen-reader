#!/usr/bin/env python3
"""
Test script for vision-based activity classification

This script tests the new VisionActivityClassifier to ensure it works
correctly with OpenAI's GPT-4 Vision API.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from prism.core.config import PrismConfig
from prism.agents.observer import VisionActivityClassifier


async def test_vision_classification():
    """Test the vision classification with a sample screenshot."""
    
    # Load config
    config = PrismConfig()
    
    # Check if OpenAI API key is available
    if not config.ml.openai_api_key:
        print("⚠️  No OpenAI API key found!")
        print("Please set your OpenAI API key in the environment variable OPENAI_API_KEY")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print(f"✅ OpenAI API key found (ending in ...{config.ml.openai_api_key[-4:]})")
    print(f"Vision classification enabled: {config.ml.use_vision_classification}")
    print(f"Vision model: {config.ml.vision_model}")
    print(f"Vision confidence threshold: {config.ml.vision_confidence_threshold}")
    
    # Initialize the vision classifier
    vision_classifier = VisionActivityClassifier(config)
    
    if not vision_classifier.client:
        print("❌ Failed to initialize OpenAI client")
        return
    
    print("✅ Vision classifier initialized successfully")
    
    # Test with a screenshot if available
    screenshot_path = project_root / "test_screenshot.png"
    
    if screenshot_path.exists():
        print(f"\n🖼️  Testing with screenshot: {screenshot_path}")
        
        try:
            # Read the screenshot file
            with open(screenshot_path, 'rb') as f:
                image_data = f.read()
            
            print(f"Screenshot size: {len(image_data)} bytes")
            
            # Classify the activity
            print("\n🔍 Analyzing screenshot...")
            result = await vision_classifier.classify_activity(image_data)
            
            print("\n📊 Classification Results:")
            print("=" * 40)
            print(f"Activity Type: {result['activity_type']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Source: {result.get('source', 'unknown')}")
            
            if result.get('reasoning'):
                print(f"\nReasoning: {result['reasoning']}")
            
            if result.get('details'):
                print(f"Details: {result['details']}")
            
            print("=" * 40)
            
        except Exception as e:
            print(f"❌ Error during classification: {e}")
    
    else:
        print(f"\n⚠️  No test screenshot found at {screenshot_path}")
        print("You can test the vision classification by:")
        print("1. Taking a screenshot and saving it as 'test_screenshot.png' in the project root")
        print("2. Running this script again")


async def test_activity_categories():
    """Display the available activity categories."""
    config = PrismConfig()
    vision_classifier = VisionActivityClassifier(config)
    
    print("\n📋 Available Activity Categories:")
    print("=" * 50)
    
    for category, description in vision_classifier.activity_categories.items():
        print(f"{category:12} : {description}")
    
    print("=" * 50)


if __name__ == "__main__":
    print("🧪 Testing Vision-Based Activity Classification")
    print("=" * 60)
    
    # Test activity categories
    asyncio.run(test_activity_categories())
    
    # Test vision classification
    asyncio.run(test_vision_classification())
    
    print("\n✅ Test completed!") 