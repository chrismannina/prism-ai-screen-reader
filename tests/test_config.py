"""
Tests for Prism configuration system.
"""

import pytest
import tempfile
import os
from pathlib import Path

from prism.core.config import PrismConfig, PrivacySettings, CaptureSettings


class TestPrismConfig:
    """Test the main configuration class."""
    
    def test_default_config_creation(self):
        """Test creating a config with default values."""
        config = PrismConfig()
        
        # Check that all sections exist
        assert config.privacy is not None
        assert config.capture is not None
        assert config.ml is not None
        assert config.storage is not None
        
        # Check some default values
        assert config.privacy.encrypt_screenshots is True
        assert config.capture.screenshot_interval_seconds == 30
        assert config.capture.ocr_enabled is True
    
    def test_config_file_handling(self):
        """Test saving and loading configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"
            
            # Create and save config
            config1 = PrismConfig(str(config_file))
            config1.capture.screenshot_interval_seconds = 60
            config1.privacy.encrypt_screenshots = False
            config1.save_config()
            
            # Load config from file
            config2 = PrismConfig(str(config_file))
            
            # Verify values were loaded
            assert config2.capture.screenshot_interval_seconds == 60
            assert config2.privacy.encrypt_screenshots is False
    
    def test_directory_creation(self):
        """Test that required directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "subdir" / "config.json"
            config = PrismConfig(str(config_file))
            
            # Check that directories were created
            assert config_file.parent.exists()
            assert config.get_data_directory().exists()
            assert config.get_logs_directory().exists()
    
    def test_app_exclusion(self):
        """Test application exclusion functionality."""
        config = PrismConfig()
        
        # Test default exclusions
        assert config.is_app_excluded("1Password")
        assert config.is_app_excluded("Keychain Access")
        assert not config.is_app_excluded("Google Chrome")
        
        # Test case insensitive
        assert config.is_app_excluded("keychain access")
    
    def test_window_exclusion(self):
        """Test window title exclusion functionality."""
        config = PrismConfig()
        
        # Test default exclusions
        assert config.is_window_excluded("Enter your password")
        assert config.is_window_excluded("Private browsing mode")
        assert not config.is_window_excluded("Normal window title")
        
        # Test case insensitive
        assert config.is_window_excluded("PASSWORD entry")


class TestPrivacySettings:
    """Test privacy settings."""
    
    def test_default_privacy_settings(self):
        """Test default privacy configuration."""
        privacy = PrivacySettings()
        
        assert privacy.encrypt_screenshots is True
        assert privacy.encrypt_text_data is True
        assert privacy.auto_delete_screenshots_days == 7
        assert len(privacy.exclude_apps) > 0
        assert "1Password" in privacy.exclude_apps


class TestCaptureSettings:
    """Test capture settings."""
    
    def test_default_capture_settings(self):
        """Test default capture configuration."""
        capture = CaptureSettings()
        
        assert capture.screenshot_interval_seconds == 30
        assert capture.window_detection_interval_seconds == 5
        assert capture.ocr_enabled is True
        assert capture.ocr_confidence_threshold == 0.7
        assert capture.capture_resolution_scale == 0.5


if __name__ == "__main__":
    pytest.main([__file__]) 