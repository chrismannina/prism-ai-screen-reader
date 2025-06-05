#!/usr/bin/env python3
"""
Setup script for Prism - AI-Powered Screen Reader & Time Tracker
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="prism-ai-screen-reader",
    version="0.1.0",
    author="Prism Team",
    author_email="team@prism.ai",
    description="AI-Powered Screen Reader & Time Tracker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/prism-ai-screen-reader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Office/Business :: Scheduling",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.1",
        ],
        "gui": [
            "tkinter-modern>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prism=prism.main:cli",
            "prism-dashboard=prism.dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "prism": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="screen-reader time-tracker ai productivity privacy",
    project_urls={
        "Bug Reports": "https://github.com/your-username/prism-ai-screen-reader/issues",
        "Source": "https://github.com/your-username/prism-ai-screen-reader",
        "Documentation": "https://github.com/your-username/prism-ai-screen-reader/wiki",
    },
) 