# Vision-Based Activity Classification

## Overview

Prism now supports vision-based activity classification using OpenAI's GPT-4 Vision API. This feature significantly improves the accuracy of activity detection by analyzing screenshots directly, rather than relying solely on text-based pattern matching.

## Features

### Enhanced Classification Accuracy
- **Visual Context Understanding**: Analyzes UI elements, application interfaces, and content layout
- **Better IDE Detection**: Correctly identifies coding activities in IDEs like Cursor, VS Code, etc.
- **Multi-Modal Analysis**: Combines visual analysis with traditional text-based classification

### Supported Activity Categories
- **coding**: Writing, editing, or reviewing code in IDEs, editors, or terminals
- **browsing**: Web browsing, reading websites, social media, or online research
- **writing**: Creating documents, articles, emails, or other text content
- **communication**: Video calls, messaging, email, or other communication tools
- **design**: Creating or editing visual content, graphics, or design work
- **research**: Reading academic papers, documentation, or research materials
- **presentation**: Creating or viewing presentations, slides, or visual materials
- **media**: Watching videos, listening to music, or consuming media content
- **gaming**: Playing games or gaming-related activities
- **productivity**: Task management, planning, or organizational activities
- **learning**: Educational content, tutorials, courses, or learning materials
- **unknown**: Unable to determine specific activity type

## Setup

### 1. Install Dependencies
```bash
pip install openai>=1.3.0
```

### 2. Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Enable Vision Classification
The feature is enabled by default. You can configure it in your Prism config:

```python
# In your config or environment
use_vision_classification = True
vision_model = "gpt-4-vision-preview"
vision_confidence_threshold = 0.7
```

## Configuration Options

### ML Settings
- `use_vision_classification`: Enable/disable vision-based classification (default: True)
- `openai_api_key`: Your OpenAI API key (set via environment variable)
- `vision_model`: OpenAI model to use (default: "gpt-4-vision-preview")
- `vision_confidence_threshold`: Minimum confidence for vision classification (default: 0.7)
- `fallback_to_text_classification`: Use text classification as fallback (default: True)

### Classification Strategy
The system uses a hybrid approach:

1. **Primary**: Vision-based classification for visual context
2. **Fallback**: Text-based classification for keyword detection
3. **Selection Logic**: 
   - Use vision classification if confidence ≥ threshold and not 'unknown'
   - Fall back to text classification if vision fails or has low confidence
   - Compare both methods and choose the most reliable result

## Testing

### Test the Vision Classification
```bash
python test_vision_classification.py
```

This script will:
- Check if your OpenAI API key is configured
- Test the vision classifier with a sample screenshot (if available)
- Display classification results with confidence scores and reasoning

### Sample Test Output
```
🧪 Testing Vision-Based Activity Classification
============================================================

📋 Available Activity Categories:
==================================================
coding       : Writing, editing, or reviewing code in IDEs, editors, or terminals
browsing     : Web browsing, reading websites, social media, or online research
writing      : Creating documents, articles, emails, or other text content
...

✅ OpenAI API key found (ending in ...XYZ)
Vision classification enabled: True
Vision model: gpt-4-vision-preview
Vision confidence threshold: 0.7

✅ Vision classifier initialized successfully

🖼️  Testing with screenshot: test_screenshot.png
Screenshot size: 8394572 bytes

🔍 Analyzing screenshot...

📊 Classification Results:
========================================
Activity Type: coding
Confidence: 92%
Source: vision_classifier

Reasoning: The screenshot shows a code editor (Cursor) with Python code visible, indicating active coding work.
Details: Cursor IDE interface with Python code visible, file explorer on left, terminal at bottom
========================================
```

## Benefits

### Improved Accuracy
- **Better Context**: Understands visual layout and application UI
- **Reduced False Positives**: Less likely to misclassify based on text alone
- **IDE Recognition**: Correctly identifies coding activities across different editors

### Real-World Example
**Before (Text-only)**: Cursor IDE might be classified as "browsing" due to web-like interface
**After (Vision+Text)**: Correctly classified as "coding" by recognizing the IDE interface and code content

### Performance Considerations
- **API Costs**: Uses OpenAI Vision API (paid service)
- **Latency**: Slight increase due to API calls
- **Fallback**: Gracefully falls back to text classification if vision fails
- **Optimization**: Uses "low" detail setting for faster processing

## Privacy & Security

### Data Handling
- **Encryption Support**: Works with encrypted screenshots
- **Temporary Decryption**: Only decrypts screenshots temporarily for analysis
- **No Data Storage**: Images are not stored by OpenAI (as per their API policy)
- **Local Fallback**: Falls back to local text classification if vision fails

### Configuration for Privacy
```python
# Disable vision classification for extra privacy
use_vision_classification = False

# Or only enable for specific applications
exclude_apps = ["Banking", "Private", "Keychain Access"]
```

## Troubleshooting

### Common Issues

#### API Key Not Found
```
⚠️  No OpenAI API key found!
```
**Solution**: Set the environment variable `OPENAI_API_KEY`

#### Vision Classification Failing
Check logs for:
- Network connectivity issues
- API rate limits
- Invalid API key
- Model availability

#### Low Confidence Scores
- Ensure screenshots have clear UI elements
- Check if the activity fits the defined categories
- Consider adjusting `vision_confidence_threshold`

### Debug Mode
Enable debug logging to see detailed classification information:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features
- **Custom Categories**: User-defined activity categories
- **Learning Mode**: Improve classification based on user feedback
- **Batch Processing**: Classify multiple screenshots efficiently
- **Local Models**: Support for local vision models for privacy

### Contributing
To improve the vision classification:
1. Test with different application types
2. Report misclassifications with screenshots
3. Suggest new activity categories
4. Optimize the classification prompt

## API Costs

### OpenAI Pricing
- **GPT-4 Vision**: ~$0.01-0.03 per image depending on size
- **Optimization**: Uses "low" detail setting to reduce costs
- **Frequency**: Runs on each screenshot (configurable interval)

### Cost Estimation
- **Screenshots per hour**: 120 (30-second intervals)
- **Daily cost**: ~$1-4 depending on usage
- **Monthly cost**: ~$30-120

### Cost Optimization Tips
- Increase screenshot intervals
- Disable for certain applications
- Use text classification only for less critical periods
- Set up usage limits in OpenAI dashboard 