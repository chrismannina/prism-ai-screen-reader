.PHONY: help install dev-install clean test format lint run dashboard init setup permissions

# Default target
help:
	@echo "🔍 Prism - AI-Powered Screen Reader & Time Tracker"
	@echo "=================================================="
	@echo ""
	@echo "Available commands:"
	@echo "  setup        - Complete setup (install dependencies and initialize)"
	@echo "  install      - Install Prism in production mode"
	@echo "  dev-install  - Install Prism in development mode"
	@echo "  init         - Initialize Prism configuration and database"
	@echo "  run          - Run Prism in observer mode"
	@echo "  dashboard    - Launch the Prism dashboard"
	@echo "  status       - Show Prism status"
	@echo "  permissions  - Check and setup required permissions"
	@echo ""
	@echo "Development commands:"
	@echo "  test         - Run tests"
	@echo "  format       - Format code with black"
	@echo "  lint         - Run linting with flake8"
	@echo "  clean        - Clean up temporary files"
	@echo "  cleanup      - Clean up old Prism data (7 days)"

# Setup everything
setup: permissions install init
	@echo "✅ Prism setup complete!"
	@echo ""
	@echo "📋 Next steps:"
	@echo "  1. Run 'make run' to start monitoring"
	@echo "  2. Run 'make dashboard' to view your data"

# Check and setup permissions
permissions:
	@echo "🔐 Checking system permissions..."
	@echo ""
	@echo "📝 Required permissions for Prism:"
	@echo "  • Screen Recording (for screenshots)"
	@echo "  • Accessibility (for window detection)"
	@echo ""
	@echo "🍎 On macOS:"
	@echo "  1. Go to System Preferences → Security & Privacy → Privacy"
	@echo "  2. Select 'Screen Recording' from the left sidebar"
	@echo "  3. Add Terminal (or your Python environment) to the list"
	@echo "  4. Select 'Accessibility' and add Terminal as well"
	@echo ""
	@echo "After granting permissions, you may need to restart your terminal."

# Install for production use
install:
	@echo "📦 Installing Prism..."
	pip install -e .

# Install for development
dev-install:
	@echo "🛠️  Installing Prism for development..."
	pip install -e ".[dev]"
	@echo "✅ Development installation complete!"

# Initialize Prism
init:
	@echo "🔧 Initializing Prism..."
	python -m prism.main init

# Run Prism
run:
	@echo "🚀 Starting Prism observer..."
	python -m prism.main start

# Launch dashboard
dashboard:
	@echo "📊 Launching Prism dashboard..."
	python -m prism.dashboard

# Show status
status:
	@echo "📋 Prism status:"
	python -m prism.main status

# Clean up old data
cleanup:
	@echo "🧹 Cleaning up old Prism data..."
	python -m prism.main cleanup --days 7

# Development commands

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=prism --cov-report=html --cov-report=term

# Format code
format:
	@echo "🎨 Formatting code..."
	black prism/ tests/
	@echo "✅ Code formatting complete!"

# Lint code
lint:
	@echo "🔍 Linting code..."
	flake8 prism/ tests/
	mypy prism/
	@echo "✅ Linting complete!"

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/
	@echo "✅ Cleanup complete!"

# Install system dependencies (macOS)
install-deps-macos:
	@echo "🍎 Installing macOS dependencies..."
	@if command -v brew >/dev/null 2>&1; then \
		echo "Installing Tesseract OCR..."; \
		brew install tesseract; \
		echo "✅ Dependencies installed!"; \
	else \
		echo "❌ Homebrew not found. Please install Homebrew first:"; \
		echo "  /bin/bash -c \"\$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""; \
	fi

# Install system dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	@echo "🐧 Installing Ubuntu/Debian dependencies..."
	sudo apt-get update
	sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
	sudo apt-get install -y python3-tk
	@echo "✅ Dependencies installed!"

# Create virtual environment
venv:
	@echo "🐍 Creating virtual environment..."
	python3 -m venv venv
	@echo "✅ Virtual environment created!"
	@echo ""
	@echo "Activate with: source venv/bin/activate"

# Development workflow
dev: dev-install format lint test
	@echo "✅ Development workflow complete!"

# Build package
build: clean
	@echo "📦 Building package..."
	python setup.py sdist bdist_wheel
	@echo "✅ Package built in dist/"

# Install pre-commit hooks
pre-commit:
	@echo "🪝 Installing pre-commit hooks..."
	pip install pre-commit
	pre-commit install
	@echo "✅ Pre-commit hooks installed!" 