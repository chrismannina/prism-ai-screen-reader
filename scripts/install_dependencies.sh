#!/bin/bash

# Prism Dependency Installation Script
# Installs required system dependencies for Prism AI Screen Reader

set -e

echo "🔍 Installing Prism System Dependencies"
echo "======================================"

# Detect operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    echo "📦 Installing Tesseract OCR..."
    brew install tesseract
    
    echo "📦 Installing additional language packs..."
    brew install tesseract-lang
    
    echo "✅ macOS dependencies installed successfully!"
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Detected Linux"
    
    # Check distribution
    if command -v apt-get &> /dev/null; then
        echo "📦 Installing dependencies with apt..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
        sudo apt-get install -y python3-tk python3-dev
        sudo apt-get install -y libssl-dev libffi-dev
        
    elif command -v yum &> /dev/null; then
        echo "📦 Installing dependencies with yum..."
        sudo yum install -y tesseract tesseract-langpack-eng
        sudo yum install -y python3-tkinter python3-devel
        sudo yum install -y openssl-devel libffi-devel
        
    elif command -v pacman &> /dev/null; then
        echo "📦 Installing dependencies with pacman..."
        sudo pacman -S tesseract tesseract-data-eng
        sudo pacman -S tk python-pip
        
    else
        echo "❌ Unsupported Linux distribution"
        echo "Please install the following packages manually:"
        echo "  - tesseract-ocr"
        echo "  - python3-tk"
        echo "  - python3-dev"
        exit 1
    fi
    
    echo "✅ Linux dependencies installed successfully!"
    
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "🪟 Detected Windows"
    echo "Please install the following dependencies manually:"
    echo "1. Download and install Tesseract OCR from:"
    echo "   https://github.com/UB-Mannheim/tesseract/wiki"
    echo "2. Add Tesseract to your PATH environment variable"
    echo "3. Install Visual C++ Build Tools if not already installed"
    
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

echo ""
echo "🎉 System dependencies installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Install Python dependencies: pip install -r requirements.txt"
echo "2. Initialize Prism: python -m prism.main init"
echo "3. Grant screen recording permissions in System Preferences (macOS)"
echo "4. Run Prism: python -m prism.main start" 