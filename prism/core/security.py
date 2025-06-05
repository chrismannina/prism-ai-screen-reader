"""
Security and Privacy Management for Prism

Handles encryption/decryption of sensitive data, privacy filtering,
and secure data handling throughout the Prism system.
"""

import os
import hashlib
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import re
from loguru import logger

from .config import PrismConfig


class SecurityManager:
    """Manages security and privacy operations for Prism."""
    
    def __init__(self, config: PrismConfig):
        self.config = config
        self._fernet: Optional[Fernet] = None
        self._key_file = self.config.get_data_directory() / ".encryption_key"
        
        # Privacy filters
        self._sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP addresses
            r'\b[A-F0-9]{32}\b',  # MD5 hashes
            r'\b[A-F0-9]{40}\b',  # SHA1 hashes
            r'\b[A-F0-9]{64}\b',  # SHA256 hashes
            r'password[:=]\s*\S+',  # Password fields
            r'api[_\s]?key[:=]\s*\S+',  # API keys
            r'token[:=]\s*\S+',  # Tokens
        ]
        
        self._initialize_encryption()
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption system."""
        if self.config.privacy.encrypt_screenshots or self.config.privacy.encrypt_text_data:
            if self._key_file.exists():
                # Load existing key
                with open(self._key_file, 'rb') as key_file:
                    key = key_file.read()
                self._fernet = Fernet(key)
                logger.debug("Loaded existing encryption key")
            else:
                # Generate new key
                key = Fernet.generate_key()
                self._fernet = Fernet(key)
                
                # Save key securely
                self._key_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self._key_file, 'wb') as key_file:
                    key_file.write(key)
                
                # Set restrictive permissions
                os.chmod(self._key_file, 0o600)
                logger.info("Generated new encryption key")
    
    def encrypt_data(self, data: bytes) -> Optional[bytes]:
        """
        Encrypt data using Fernet encryption.
        
        Args:
            data: Raw data to encrypt
            
        Returns:
            Encrypted data or None if encryption fails
        """
        if not self._fernet:
            logger.warning("Encryption not initialized")
            return data
        
        try:
            encrypted_data = self._fernet.encrypt(data)
            logger.debug(f"Encrypted {len(data)} bytes to {len(encrypted_data)} bytes")
            return encrypted_data
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: bytes) -> Optional[bytes]:
        """
        Decrypt data using Fernet encryption.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            
        Returns:
            Decrypted data or None if decryption fails
        """
        if not self._fernet:
            logger.warning("Encryption not initialized")
            return encrypted_data
        
        try:
            decrypted_data = self._fernet.decrypt(encrypted_data)
            logger.debug(f"Decrypted {len(encrypted_data)} bytes to {len(decrypted_data)} bytes")
            return decrypted_data
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def encrypt_text(self, text: str) -> Optional[str]:
        """
        Encrypt text and return as base64 encoded string.
        
        Args:
            text: Text to encrypt
            
        Returns:
            Base64 encoded encrypted text or None if encryption fails
        """
        if not text:
            return text
        
        text_bytes = text.encode('utf-8')
        encrypted_bytes = self.encrypt_data(text_bytes)
        
        if encrypted_bytes:
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        return None
    
    def decrypt_text(self, encrypted_text: str) -> Optional[str]:
        """
        Decrypt base64 encoded encrypted text.
        
        Args:
            encrypted_text: Base64 encoded encrypted text
            
        Returns:
            Decrypted text or None if decryption fails
        """
        if not encrypted_text:
            return encrypted_text
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
            decrypted_bytes = self.decrypt_data(encrypted_bytes)
            
            if decrypted_bytes:
                return decrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Text decryption failed: {e}")
        
        return None
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> str:
        """
        Create a SHA256 hash of data with optional salt.
        
        Args:
            data: Data to hash
            salt: Optional salt for the hash
            
        Returns:
            Hexadecimal hash string
        """
        if salt:
            data = f"{data}{salt}"
        
        hash_object = hashlib.sha256(data.encode('utf-8'))
        return hash_object.hexdigest()
    
    def is_sensitive_text(self, text: str) -> bool:
        """
        Check if text contains sensitive information.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text contains sensitive patterns
        """
        text_lower = text.lower()
        
        # Check for sensitive patterns
        for pattern in self._sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.debug(f"Sensitive pattern detected: {pattern}")
                return True
        
        # Check for excluded keywords
        for keyword in self.config.privacy.exclude_windows_containing:
            if keyword.lower() in text_lower:
                logger.debug(f"Excluded keyword detected: {keyword}")
                return True
        
        return False
    
    def sanitize_text(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Sanitize text by replacing sensitive information.
        
        Args:
            text: Text to sanitize
            replacement: String to replace sensitive data with
            
        Returns:
            Sanitized text
        """
        sanitized = text
        
        # Replace sensitive patterns
        for pattern in self._sensitive_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def should_exclude_app(self, app_name: str) -> bool:
        """
        Check if an application should be excluded from monitoring.
        
        Args:
            app_name: Name of the application
            
        Returns:
            True if app should be excluded
        """
        return self.config.is_app_excluded(app_name)
    
    def should_exclude_window(self, window_title: str) -> bool:
        """
        Check if a window should be excluded from monitoring.
        
        Args:
            window_title: Title of the window
            
        Returns:
            True if window should be excluded
        """
        return self.config.is_window_excluded(window_title)
    
    def analyze_privacy_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data for privacy risks and return a risk assessment.
        
        Args:
            data: Data to analyze
            
        Returns:
            Privacy risk assessment
        """
        risk_assessment = {
            'risk_level': 'low',
            'sensitive_data_found': False,
            'risk_factors': [],
            'recommendations': []
        }
        
        # Check text data for sensitive content
        text_fields = ['window_title', 'ocr_text', 'app_name']
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                if self.is_sensitive_text(data[field]):
                    risk_assessment['sensitive_data_found'] = True
                    risk_assessment['risk_factors'].append(f"Sensitive data in {field}")
                    risk_assessment['risk_level'] = 'high'
        
        # Check for excluded applications
        if 'app_name' in data:
            if self.should_exclude_app(data['app_name']):
                risk_assessment['risk_factors'].append("Application in exclusion list")
                risk_assessment['risk_level'] = 'high'
        
        # Check for excluded window titles
        if 'window_title' in data:
            if self.should_exclude_window(data['window_title']):
                risk_assessment['risk_factors'].append("Window in exclusion list")
                risk_assessment['risk_level'] = 'high'
        
        # Generate recommendations based on risk level
        if risk_assessment['risk_level'] == 'high':
            risk_assessment['recommendations'].extend([
                "Consider excluding this application/window",
                "Enable data encryption",
                "Review privacy settings"
            ])
        elif risk_assessment['sensitive_data_found']:
            risk_assessment['risk_level'] = 'medium'
            risk_assessment['recommendations'].extend([
                "Enable text sanitization",
                "Increase screenshot encryption"
            ])
        
        return risk_assessment
    
    def secure_delete_file(self, file_path: Path) -> bool:
        """
        Securely delete a file by overwriting it before deletion.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successfully deleted
        """
        try:
            if not file_path.exists():
                return True
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Overwrite file with random data multiple times
            with open(file_path, 'r+b') as file:
                for _ in range(3):  # Multiple passes
                    file.seek(0)
                    file.write(os.urandom(file_size))
                    file.flush()
                    os.fsync(file.fileno())
            
            # Delete the file
            file_path.unlink()
            logger.debug(f"Securely deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Secure file deletion failed: {e}")
            return False
    
    def create_data_fingerprint(self, data: Dict[str, Any]) -> str:
        """
        Create a unique fingerprint for data to detect duplicates.
        
        Args:
            data: Data to fingerprint
            
        Returns:
            SHA256 hash fingerprint
        """
        # Create a normalized string representation of the data
        normalized_data = str(sorted(data.items()))
        return self.hash_data(normalized_data)
    
    def validate_data_integrity(self, data: bytes, expected_hash: str) -> bool:
        """
        Validate data integrity using hash comparison.
        
        Args:
            data: Data to validate
            expected_hash: Expected SHA256 hash
            
        Returns:
            True if data integrity is valid
        """
        actual_hash = hashlib.sha256(data).hexdigest()
        return actual_hash == expected_hash
    
    def get_security_status(self) -> Dict[str, Any]:
        """
        Get current security configuration status.
        
        Returns:
            Security status information
        """
        return {
            'encryption_enabled': self._fernet is not None,
            'key_file_exists': self._key_file.exists(),
            'screenshot_encryption': self.config.privacy.encrypt_screenshots,
            'text_encryption': self.config.privacy.encrypt_text_data,
            'excluded_apps_count': len(self.config.privacy.exclude_apps),
            'excluded_patterns_count': len(self.config.privacy.exclude_windows_containing),
            'auto_delete_days': self.config.privacy.auto_delete_screenshots_days,
            'blur_sensitive_areas': self.config.privacy.blur_sensitive_areas
        }
    
    def export_security_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive security report.
        
        Returns:
            Security report with recommendations
        """
        status = self.get_security_status()
        
        report = {
            'timestamp': str(datetime.now()),
            'status': status,
            'recommendations': [],
            'security_score': 0
        }
        
        # Calculate security score and recommendations
        score = 0
        
        if status['encryption_enabled']:
            score += 30
        else:
            report['recommendations'].append("Enable data encryption")
        
        if status['screenshot_encryption']:
            score += 20
        else:
            report['recommendations'].append("Enable screenshot encryption")
        
        if status['text_encryption']:
            score += 20
        else:
            report['recommendations'].append("Enable text encryption")
        
        if status['excluded_apps_count'] > 0:
            score += 15
        else:
            report['recommendations'].append("Configure application exclusions")
        
        if status['auto_delete_days'] <= 30:
            score += 15
        else:
            report['recommendations'].append("Reduce data retention period")
        
        report['security_score'] = min(score, 100)
        
        return report
    
    def cleanup_security_data(self) -> None:
        """Clean up security-related temporary data."""
        temp_dir = self.config.get_data_directory() / "temp"
        if temp_dir.exists():
            for file_path in temp_dir.glob("*"):
                if file_path.is_file():
                    self.secure_delete_file(file_path)
        
        logger.info("Security data cleanup completed") 