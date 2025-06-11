"""
Prism Configuration Management

Handles all configuration settings for the Prism system including
privacy controls, capture settings, and storage options.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()


@dataclass
class PrivacySettings:
    """Privacy and security configuration."""
    encrypt_screenshots: bool = True
    encrypt_text_data: bool = True
    auto_delete_screenshots_days: int = 7
    exclude_apps: list[str] = field(default_factory=lambda: [
        "Keychain Access",
        "1Password",
        "Banking",
        "Private"
    ])
    exclude_windows_containing: list[str] = field(default_factory=lambda: [
        "password",
        "private",
        "incognito",
        "private browsing"
    ])
    blur_sensitive_areas: bool = True


@dataclass
class CaptureSettings:
    """Screen capture and monitoring configuration."""
    screenshot_interval_seconds: int = 30
    window_detection_interval_seconds: int = 5
    ocr_enabled: bool = True
    ocr_confidence_threshold: float = 0.7
    capture_resolution_scale: float = 0.5  # Scale down for performance
    max_screenshot_size_mb: int = 5
    enable_mouse_tracking: bool = False
    enable_keyboard_tracking: bool = False


@dataclass
class MLSettings:
    """Machine learning and classification configuration."""
    activity_classification_enabled: bool = True
    model_update_interval_hours: int = 24
    confidence_threshold: float = 0.8
    local_model_path: str = "models/"
    use_cloud_models: bool = False
    batch_classification: bool = True
    
    # Vision-based classification settings
    use_vision_classification: bool = True
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    vision_model: str = "gpt-4o-mini"
    vision_confidence_threshold: float = 0.7
    fallback_to_text_classification: bool = True


@dataclass
class StorageSettings:
    """Data storage and database configuration."""
    database_path: str = "data/prism.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_storage_gb: int = 5
    compress_old_data: bool = True
    data_retention_days: int = 365


class PrismConfig:
    """Main configuration manager for Prism."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self.privacy = PrivacySettings()
        self.capture = CaptureSettings()
        self.ml = MLSettings()
        self.storage = StorageSettings()
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Load existing config if available
        self.load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        home_dir = Path.home()
        config_dir = home_dir / ".prism"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "config.json")
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            Path(self.storage.database_path).parent,
            Path(self.ml.local_model_path),
            Path(self.config_file).parent,
            Path.home() / ".prism" / "logs",
            Path.home() / ".prism" / "temp",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update settings from config data
                if 'privacy' in config_data:
                    self._update_dataclass(self.privacy, config_data['privacy'])
                if 'capture' in config_data:
                    self._update_dataclass(self.capture, config_data['capture'])
                if 'ml' in config_data:
                    self._update_dataclass(self.ml, config_data['ml'])
                if 'storage' in config_data:
                    self._update_dataclass(self.storage, config_data['storage'])
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load config file: {e}")
                print("Using default configuration.")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        config_data = {
            'privacy': self._dataclass_to_dict(self.privacy),
            'capture': self._dataclass_to_dict(self.capture),
            'ml': self._dataclass_to_dict(self.ml),
            'storage': self._dataclass_to_dict(self.storage),
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        if hasattr(obj, '__dataclass_fields__'):
            return {
                field.name: getattr(obj, field.name)
                for field in obj.__dataclass_fields__.values()
            }
        return {}
    
    def get_data_directory(self) -> Path:
        """Get the main data directory path."""
        return Path(self.storage.database_path).parent
    
    def get_logs_directory(self) -> Path:
        """Get the logs directory path."""
        return Path.home() / ".prism" / "logs"
    
    def get_models_directory(self) -> Path:
        """Get the models directory path."""
        return Path(self.ml.local_model_path)
    
    def is_app_excluded(self, app_name: str) -> bool:
        """Check if an application should be excluded from monitoring."""
        app_lower = app_name.lower()
        return any(
            excluded.lower() in app_lower 
            for excluded in self.privacy.exclude_apps
        )
    
    def is_window_excluded(self, window_title: str) -> bool:
        """Check if a window should be excluded based on its title."""
        title_lower = window_title.lower()
        return any(
            excluded.lower() in title_lower 
            for excluded in self.privacy.exclude_windows_containing
        )
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"PrismConfig(config_file='{self.config_file}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return (
            f"PrismConfig(\n"
            f"  privacy={self.privacy},\n"
            f"  capture={self.capture},\n"
            f"  ml={self.ml},\n"
            f"  storage={self.storage}\n"
            f")"
        ) 