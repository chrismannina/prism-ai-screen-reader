#!/usr/bin/env python3
"""
Live Screenshot Test

Test script to capture and analyze a screenshot to see if we're capturing
actual application content or just desktop background.
"""

import time
from prism.agents.observer import ScreenCapture
from prism.core.config import PrismConfig  
from prism.core.security import SecurityManager
from PIL import Image
import numpy as np
import io

def analyze_screenshot_content(image_data, is_encrypted=False):
    """Analyze screenshot to determine content type."""
    
    # Decrypt if needed
    if is_encrypted:
        security = SecurityManager(PrismConfig())
        image_data = security.decrypt_data(image_data)
        if not image_data:
            print("❌ Failed to decrypt image")
            return
    
    # Load image
    image = Image.open(io.BytesIO(image_data))
    img_array = np.array(image)
    
    # Calculate various metrics
    total_pixels = img_array.shape[0] * img_array.shape[1]
    unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
    color_diversity = unique_colors / total_pixels
    
    # Check for uniform areas (like desktop background)
    # Convert to grayscale for edge detection
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Calculate edge density (more edges = more UI elements)
    from scipy import ndimage
    edges = ndimage.sobel(gray)
    edge_density = np.mean(np.abs(edges))
    
    # Color histogram analysis
    hist, _ = np.histogram(img_array.flatten(), bins=50)
    hist_variance = np.var(hist)
    
    print(f"📊 Screenshot Analysis:")
    print(f"   Resolution: {image.size}")
    print(f"   Unique colors: {unique_colors:,} ({color_diversity:.6f} diversity)")
    print(f"   Edge density: {edge_density:.2f}")
    print(f"   Histogram variance: {hist_variance:.2f}")
    
    # Determine likely content type
    if color_diversity < 0.0001:
        print("🎨 Content type: Likely solid color or very simple background")
    elif color_diversity < 0.001:
        print("🎨 Content type: Likely desktop background or minimal content")
    elif edge_density < 5:
        print("🎨 Content type: Likely background with some elements")
    else:
        print("🎨 Content type: Likely application UI with text/controls")
    
    return {
        'color_diversity': color_diversity,
        'edge_density': edge_density,
        'histogram_variance': hist_variance,
        'unique_colors': unique_colors
    }

def main():
    print("🔍 Live Screenshot Test")
    print("=" * 40)
    print("Make sure you have an application open (Safari, Cursor, etc.)")
    print("Press Enter when ready to capture...")
    input()
    
    # Initialize capture
    config = PrismConfig()
    security = SecurityManager(config)
    capture = ScreenCapture(config, security)
    
    print("\n📸 Capturing screenshot...")
    result = capture.capture_screenshot()
    
    if result:
        print(f"✅ Screenshot captured: {result['resolution']}")
        analyze_screenshot_content(result['image_data'], result['is_encrypted'])
        
        # Save for manual inspection
        if result['is_encrypted']:
            image_data = security.decrypt_data(result['image_data'])
        else:
            image_data = result['image_data']
            
        if image_data:
            with open('live_test_screenshot.png', 'wb') as f:
                f.write(image_data)
            print(f"\n💾 Screenshot saved as 'live_test_screenshot.png'")
            print("You can open it to verify the content was captured correctly")
    else:
        print("❌ Screenshot capture failed")

if __name__ == "__main__":
    main() 