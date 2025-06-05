"""
Observer Agent for Prism

The core agent responsible for screen monitoring, window detection,
OCR processing, and basic activity classification.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import io
import os

# Screen capture and image processing
import pyautogui
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np

# OCR processing
import pytesseract

# System monitoring
import psutil

# macOS specific imports
try:
    from AppKit import NSWorkspace, NSApplication
    MACOS_AVAILABLE = True
except ImportError:
    MACOS_AVAILABLE = False

from loguru import logger

from ..core.config import PrismConfig
from ..core.event_bus import EventBus, EventType
from ..core.database import DatabaseManager
from ..core.security import SecurityManager


class WindowDetector:
    """Handles window detection and application monitoring."""
    
    def __init__(self, config: PrismConfig):
        self.config = config
        self.current_window = None
        self.current_app = None
    
    def get_active_window_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently active window."""
        try:
            if MACOS_AVAILABLE:
                return self._get_macos_window_info()
            else:
                return self._get_cross_platform_window_info()
        except Exception as e:
            logger.error(f"Error getting window info: {e}")
            return None
    
    def _get_macos_window_info(self) -> Optional[Dict[str, Any]]:
        """Get window info on macOS using AppKit."""
        try:
            workspace = NSWorkspace.sharedWorkspace()
            active_app = workspace.activeApplication()
            
            if not active_app:
                return None
            
            app_name = active_app.get('NSApplicationName', 'Unknown')
            bundle_id = active_app.get('NSApplicationBundleIdentifier', '')
            
            # Try to get window title (limited on macOS for security)
            window_title = app_name  # Fallback to app name
            
            return {
                'window_title': window_title,
                'app_name': app_name,
                'bundle_id': bundle_id,
                'is_active': True,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"macOS window detection error: {e}")
            return None
    
    def _get_cross_platform_window_info(self) -> Optional[Dict[str, Any]]:
        """Get window info using cross-platform methods."""
        try:
            # Use psutil to get process information
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_info = proc.info
                    if proc_info['name']:
                        return {
                            'window_title': proc_info['name'],
                            'app_name': proc_info['name'],
                            'bundle_id': None,
                            'is_active': True,
                            'timestamp': datetime.now()
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"Cross-platform window detection error: {e}")
        
        return None


class ScreenCapture:
    """Handles screen capture and image processing."""
    
    def __init__(self, config: PrismConfig, security_manager: SecurityManager):
        self.config = config
        self.security_manager = security_manager
    
    def capture_screenshot(self) -> Optional[Dict[str, Any]]:
        """Capture a screenshot and return processed data."""
        try:
            # Capture screenshot
            screenshot = pyautogui.screenshot()
            
            # Scale down if configured
            if self.config.capture.capture_resolution_scale < 1.0:
                new_size = (
                    int(screenshot.width * self.config.capture.capture_resolution_scale),
                    int(screenshot.height * self.config.capture.capture_resolution_scale)
                )
                screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            img_byte_array = io.BytesIO()
            screenshot.save(img_byte_array, format='PNG')
            img_bytes = img_byte_array.getvalue()
            
            # Check file size
            file_size_mb = len(img_bytes) / (1024 * 1024)
            if file_size_mb > self.config.capture.max_screenshot_size_mb:
                logger.warning(f"Screenshot too large ({file_size_mb:.1f}MB), skipping")
                return None
            
            # Encrypt if configured
            if self.config.privacy.encrypt_screenshots:
                img_bytes = self.security_manager.encrypt_data(img_bytes)
                if img_bytes is None:
                    logger.error("Failed to encrypt screenshot")
                    return None
            
            return {
                'image_data': img_bytes,
                'resolution': (screenshot.width, screenshot.height),
                'file_size': len(img_bytes),
                'is_encrypted': self.config.privacy.encrypt_screenshots,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def blur_sensitive_areas(self, image: Image.Image, ocr_data: Dict[str, Any]) -> Image.Image:
        """Blur areas containing sensitive information."""
        if not self.config.privacy.blur_sensitive_areas:
            return image
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply blur to entire image (simple approach)
        # TODO: Implement intelligent area detection
        blurred = cv2.GaussianBlur(img_cv, (15, 15), 0)
        
        # Convert back to PIL
        img_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


class OCRProcessor:
    """Handles OCR processing of screenshots."""
    
    def __init__(self, config: PrismConfig, security_manager: SecurityManager):
        self.config = config
        self.security_manager = security_manager
        
        # Configure Tesseract
        self._configure_tesseract()
    
    def _configure_tesseract(self) -> None:
        """Configure Tesseract OCR settings."""
        # Set Tesseract configuration
        self.tesseract_config = '--oem 3 --psm 6'  # Use LSTM OCR Engine with uniform text blocks
    
    def process_image(self, image_data: bytes, is_encrypted: bool = False) -> Optional[Dict[str, Any]]:
        """Process image with OCR and return text data."""
        if not self.config.capture.ocr_enabled:
            return None
        
        try:
            # Decrypt if necessary
            if is_encrypted:
                image_data = self.security_manager.decrypt_data(image_data)
                if image_data is None:
                    logger.error("Failed to decrypt image for OCR")
                    return None
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image for better OCR
            image = self._preprocess_image(image)
            
            # Perform OCR
            ocr_data = pytesseract.image_to_data(
                image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_blocks = []
            total_confidence = 0
            valid_blocks = 0
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                confidence = ocr_data['conf'][i]
                
                if text and confidence > 0:
                    if confidence >= self.config.capture.ocr_confidence_threshold * 100:
                        text_blocks.append({
                            'text': text,
                            'confidence': confidence / 100.0,
                            'bbox': {
                                'x': ocr_data['left'][i],
                                'y': ocr_data['top'][i],
                                'width': ocr_data['width'][i],
                                'height': ocr_data['height'][i]
                            }
                        })
                        total_confidence += confidence
                        valid_blocks += 1
            
            # Combine all text
            full_text = ' '.join([block['text'] for block in text_blocks])
            avg_confidence = (total_confidence / valid_blocks / 100.0) if valid_blocks > 0 else 0.0
            
            # Check for sensitive content
            if self.security_manager.is_sensitive_text(full_text):
                logger.warning("Sensitive text detected in OCR, applying privacy filters")
                if self.config.privacy.encrypt_text_data:
                    full_text = self.security_manager.sanitize_text(full_text)
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'text_blocks': text_blocks,
                'block_count': len(text_blocks),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return None
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply slight sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        return image


class ActivityClassifier:
    """Basic activity classification based on window info and OCR text."""
    
    def __init__(self, config: PrismConfig):
        self.config = config
        
        # Activity patterns
        self.activity_patterns = {
            'coding': [
                'code', 'programming', 'development', 'github', 'vscode', 'intellij',
                'sublime', 'atom', 'vim', 'emacs', 'terminal', 'command', 'python',
                'javascript', 'java', 'cpp', 'function', 'class', 'variable'
            ],
            'writing': [
                'document', 'word', 'pages', 'notion', 'draft', 'writing', 'article',
                'essay', 'blog', 'manuscript', 'text', 'paragraph', 'sentence'
            ],
            'browsing': [
                'browser', 'chrome', 'firefox', 'safari', 'internet', 'web', 'website',
                'search', 'google', 'facebook', 'twitter', 'reddit', 'youtube'
            ],
            'communication': [
                'email', 'slack', 'discord', 'zoom', 'teams', 'skype', 'message',
                'chat', 'call', 'meeting', 'conference'
            ],
            'design': [
                'photoshop', 'illustrator', 'figma', 'sketch', 'design', 'graphic',
                'image', 'photo', 'edit', 'canvas', 'artboard'
            ],
            'research': [
                'research', 'paper', 'article', 'study', 'analysis', 'data',
                'pdf', 'academic', 'journal', 'scholar'
            ]
        }
    
    def classify_activity(self, 
                         window_info: Optional[Dict[str, Any]] = None,
                         ocr_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify current activity based on available data."""
        
        activity_scores = {activity: 0.0 for activity in self.activity_patterns.keys()}
        
        # Analyze window information
        if window_info:
            app_name = window_info.get('app_name', '').lower()
            window_title = window_info.get('window_title', '').lower()
            
            for activity, patterns in self.activity_patterns.items():
                for pattern in patterns:
                    if pattern in app_name:
                        activity_scores[activity] += 0.8
                    if pattern in window_title:
                        activity_scores[activity] += 0.6
        
        # Analyze OCR text
        if ocr_data and ocr_data.get('text'):
            text = ocr_data['text'].lower()
            
            for activity, patterns in self.activity_patterns.items():
                for pattern in patterns:
                    if pattern in text:
                        activity_scores[activity] += 0.4
        
        # Find best match
        best_activity = max(activity_scores.keys(), key=lambda k: activity_scores[k])
        confidence = min(activity_scores[best_activity], 1.0)
        
        # Default to 'unknown' if confidence is too low
        if confidence < self.config.ml.confidence_threshold:
            best_activity = 'unknown'
            confidence = 0.5
        
        return {
            'activity_type': best_activity,
            'confidence': confidence,
            'all_scores': activity_scores,
            'timestamp': datetime.now()
        }


class ObserverAgent:
    """Main Observer Agent that coordinates all monitoring activities."""
    
    def __init__(self, config: PrismConfig, event_bus: EventBus, 
                 database: DatabaseManager, security_manager: SecurityManager):
        self.config = config
        self.event_bus = event_bus
        self.database = database
        self.security_manager = security_manager
        
        # Initialize components
        self.window_detector = WindowDetector(config)
        self.screen_capture = ScreenCapture(config, security_manager)
        self.ocr_processor = OCRProcessor(config, security_manager)
        self.activity_classifier = ActivityClassifier(config)
        
        # State
        self.is_running = False
        self.last_screenshot_time = None
        self.last_window_check_time = None
        self.current_session_id = None
        
        # Tasks
        self._screenshot_task = None
        self._window_monitoring_task = None
        
        logger.info("Observer Agent initialized")
    
    async def start(self) -> None:
        """Start the Observer Agent."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring tasks
        self._screenshot_task = asyncio.create_task(self._screenshot_loop())
        self._window_monitoring_task = asyncio.create_task(self._window_monitoring_loop())
        
        # Start a new time session
        self.current_session_id = self.database.start_time_session()
        
        # Emit startup event
        await self.event_bus.emit(
            EventType.SYSTEM_STARTUP,
            data={'agent': 'observer'},
            source='observer_agent'
        )
        
        logger.info("Observer Agent started")
    
    async def stop(self) -> None:
        """Stop the Observer Agent."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel tasks
        if self._screenshot_task:
            self._screenshot_task.cancel()
        if self._window_monitoring_task:
            self._window_monitoring_task.cancel()
        
        # End current session
        if self.current_session_id:
            self.database.end_time_session(self.current_session_id)
        
        # Emit shutdown event
        await self.event_bus.emit(
            EventType.SYSTEM_SHUTDOWN,
            data={'agent': 'observer'},
            source='observer_agent'
        )
        
        logger.info("Observer Agent stopped")
    
    async def _screenshot_loop(self) -> None:
        """Main screenshot capture loop."""
        while self.is_running:
            try:
                await self._capture_and_process_screenshot()
                await asyncio.sleep(self.config.capture.screenshot_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in screenshot loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _window_monitoring_loop(self) -> None:
        """Main window monitoring loop."""
        while self.is_running:
            try:
                await self._check_active_window()
                await asyncio.sleep(self.config.capture.window_detection_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in window monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _capture_and_process_screenshot(self) -> None:
        """Capture screenshot and process it."""
        # Check if we should skip this window/app
        window_info = self.window_detector.get_active_window_info()
        if window_info:
            if (self.security_manager.should_exclude_app(window_info['app_name']) or
                self.security_manager.should_exclude_window(window_info['window_title'])):
                logger.debug(f"Skipping screenshot for excluded app/window: {window_info['app_name']}")
                return
        
        # Capture screenshot
        screenshot_data = self.screen_capture.capture_screenshot()
        if not screenshot_data:
            return
        
        # Store screenshot in database
        screenshot_id = self.database.store_screenshot(
            image_data=screenshot_data['image_data'],
            resolution=screenshot_data['resolution'],
            file_size=screenshot_data['file_size'],
            is_encrypted=screenshot_data['is_encrypted']
        )
        
        # Emit screenshot captured event
        await self.event_bus.emit(
            EventType.SCREENSHOT_CAPTURED,
            data={
                'screenshot_id': screenshot_id,
                'resolution': screenshot_data['resolution'],
                'file_size': screenshot_data['file_size']
            },
            source='observer_agent'
        )
        
        # Process OCR asynchronously
        if screenshot_id:
            asyncio.create_task(self._process_screenshot_ocr(screenshot_id, screenshot_data))
        
        self.last_screenshot_time = datetime.now()
    
    async def _process_screenshot_ocr(self, screenshot_id: int, screenshot_data: Dict[str, Any]) -> None:
        """Process OCR for a screenshot."""
        try:
            ocr_data = self.ocr_processor.process_image(
                screenshot_data['image_data'],
                screenshot_data['is_encrypted']
            )
            
            if ocr_data:
                # Update screenshot with OCR data
                # (This would require updating the database schema)
                
                # Emit OCR completed event
                await self.event_bus.emit(
                    EventType.OCR_COMPLETED,
                    data={
                        'screenshot_id': screenshot_id,
                        'text_length': len(ocr_data['text']),
                        'confidence': ocr_data['confidence'],
                        'block_count': ocr_data['block_count']
                    },
                    source='observer_agent'
                )
                
                # Classify activity
                window_info = self.window_detector.get_active_window_info()
                activity_data = self.activity_classifier.classify_activity(window_info, ocr_data)
                
                # Store activity
                activity_id = self.database.store_activity(
                    activity_type=activity_data['activity_type'],
                    confidence=activity_data['confidence'],
                    screenshot_id=screenshot_id,
                    metadata=activity_data
                )
                
                # Emit activity classified event
                await self.event_bus.emit(
                    EventType.ACTIVITY_CLASSIFIED,
                    data={
                        'activity_id': activity_id,
                        'activity_type': activity_data['activity_type'],
                        'confidence': activity_data['confidence']
                    },
                    source='observer_agent'
                )
                
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
    
    async def _check_active_window(self) -> None:
        """Check and update active window information."""
        window_info = self.window_detector.get_active_window_info()
        if not window_info:
            return
        
        # Store window info
        window_info_id = self.database.store_window_info(
            window_title=window_info['window_title'],
            app_name=window_info['app_name'],
            app_bundle_id=window_info.get('bundle_id'),
            is_active=window_info['is_active']
        )
        
        # Check if window changed
        current_window_key = f"{window_info['app_name']}:{window_info['window_title']}"
        if current_window_key != self.window_detector.current_window:
            # Window changed
            await self.event_bus.emit(
                EventType.WINDOW_CHANGED,
                data={
                    'window_info_id': window_info_id,
                    'app_name': window_info['app_name'],
                    'window_title': window_info['window_title'],
                    'previous_window': self.window_detector.current_window
                },
                source='observer_agent'
            )
            
            self.window_detector.current_window = current_window_key
        
        self.last_window_check_time = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the Observer Agent."""
        return {
            'is_running': self.is_running,
            'last_screenshot_time': self.last_screenshot_time.isoformat() if self.last_screenshot_time else None,
            'last_window_check_time': self.last_window_check_time.isoformat() if self.last_window_check_time else None,
            'current_session_id': self.current_session_id,
            'current_window': self.window_detector.current_window,
            'config': {
                'screenshot_interval': self.config.capture.screenshot_interval_seconds,
                'window_check_interval': self.config.capture.window_detection_interval_seconds,
                'ocr_enabled': self.config.capture.ocr_enabled,
                'encryption_enabled': self.config.privacy.encrypt_screenshots
            }
        } 