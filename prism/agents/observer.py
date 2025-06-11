"""
Observer Agent for Prism

The core agent responsible for screen monitoring, window detection,
OCR processing, and basic activity classification.
"""

import asyncio
import threading
import time
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import io
import os
import base64
import json

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

# Add OpenAI import
import openai

from ..core.config import PrismConfig
from ..core.event_bus import EventBus, EventType
from ..core.database import DatabaseManager
from ..core.security import SecurityManager
from ..core.api_monitor import APIMonitor


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
                'timestamp': datetime.now().isoformat()  # Convert to ISO string for JSON serialization
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
                            'timestamp': datetime.now().isoformat()  # Convert to ISO string for JSON serialization
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
        self._test_capture_method()
    
    def _test_capture_method(self):
        """Test which screenshot capture method works best."""
        self.use_macos_screencapture = False
        
        # On macOS, test if we can capture more than just desktop
        if MACOS_AVAILABLE:
            try:
                # Test pyautogui capture
                test_shot = pyautogui.screenshot()
                test_array = np.array(test_shot)
                unique_colors = len(np.unique(test_array.reshape(-1, test_array.shape[-1]), axis=0))
                color_diversity = unique_colors / (test_array.shape[0] * test_array.shape[1])
                
                # If color diversity is very low, try alternative method
                if color_diversity < 0.001:
                    logger.warning("Low color diversity in pyautogui capture, will try macOS screencapture")
                    self.use_macos_screencapture = True
                else:
                    logger.info(f"Using pyautogui capture (color diversity: {color_diversity:.6f})")
                    
            except Exception as e:
                logger.warning(f"Failed to test capture method: {e}")
    
    def capture_screenshot(self) -> Optional[Dict[str, Any]]:
        """Capture a screenshot and return processed data."""
        try:
            # Try alternative macOS method first if needed
            if self.use_macos_screencapture and MACOS_AVAILABLE:
                screenshot = self._capture_screenshot_macos()
            else:
                screenshot = self._capture_screenshot_pyautogui()
            
            if not screenshot:
                return None
            
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
                'timestamp': datetime.now().isoformat()  # Convert to ISO string for JSON serialization
            }
            
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def _capture_screenshot_pyautogui(self) -> Optional[Image.Image]:
        """Capture screenshot using pyautogui."""
        try:
            return pyautogui.screenshot()
        except Exception as e:
            logger.error(f"pyautogui screenshot failed: {e}")
            return None
    
    def _capture_screenshot_macos(self) -> Optional[Image.Image]:
        """Capture screenshot using macOS screencapture command."""
        try:
            import subprocess
            import tempfile
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Use macOS screencapture command
                result = subprocess.run([
                    'screencapture', 
                    '-x',  # Don't play camera sound
                    '-t', 'png',  # PNG format
                    tmp_path
                ], capture_output=True, timeout=10)
                
                if result.returncode == 0:
                    # Load the image
                    screenshot = Image.open(tmp_path)
                    # Make a copy since we'll delete the temp file
                    screenshot_copy = screenshot.copy()
                    screenshot.close()
                    return screenshot_copy
                else:
                    logger.error(f"screencapture failed: {result.stderr.decode()}")
                    return None
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"macOS screencapture failed: {e}")
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
                'timestamp': datetime.now().isoformat()  # Convert to ISO string for JSON serialization
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
            'timestamp': datetime.now().isoformat()  # Convert to ISO string for JSON serialization
        }


class VisionActivityClassifier:
    """Vision-based activity classification using OpenAI GPT-4 Vision."""
    
    def __init__(self, config: PrismConfig):
        self.config = config
        self.client = None
        self.api_monitor = APIMonitor(config)
        
        if config.ml.openai_api_key:
            self.client = openai.OpenAI(api_key=config.ml.openai_api_key)
        
        # Define activity categories and descriptions
        self.activity_categories = {
            'coding': 'Writing, editing, or reviewing code in IDEs, editors, or terminals',
            'browsing': 'Web browsing, reading websites, social media, or online research',
            'writing': 'Creating documents, articles, emails, or other text content',
            'communication': 'Video calls, messaging, email, or other communication tools',
            'productivity': 'Task management, spreadsheets, presentations, or general productivity apps',
            'entertainment': 'Gaming, watching videos, streaming, or other entertainment activities',
            'design': 'Graphic design, photo editing, or creative work',
            'unknown': 'Activity cannot be clearly classified into other categories'
        }
        
        logger.info(f"Vision Activity Classifier initialized with {len(self.activity_categories)} categories")
    
    async def classify_activity(self, image_data: bytes) -> Dict[str, Any]:
        """Classify activity from screenshot image data."""
        if not self.client:
            logger.warning("OpenAI client not available, returning fallback classification")
            return self._fallback_classification()
        
        try:
            start_time = time.time()
            
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create classification prompt
            prompt = self._create_classification_prompt()
            
            # Call OpenAI Vision API
            response = await self._call_openai_vision(base64_image, prompt)
            response_time_ms = int((time.time() - start_time) * 1000)
            
            if response:
                # Parse response and get classification
                result = self._parse_response(response)
                
                # Log successful API usage (estimate tokens)
                input_tokens = self._estimate_input_tokens(prompt, base64_image)
                output_tokens = self._estimate_output_tokens(response)
                
                self.api_monitor.log_usage(
                    provider='openai',
                    model=self.config.ml.vision_model,
                    operation='vision_classification',
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_time_ms=response_time_ms,
                    success=True
                )
                
                return result
            else:
                # Log failed API usage
                self.api_monitor.log_usage(
                    provider='openai',
                    model=self.config.ml.vision_model,
                    operation='vision_classification',
                    input_tokens=0,
                    output_tokens=0,
                    response_time_ms=response_time_ms,
                    success=False,
                    error_message="No response from OpenAI API"
                )
                
                return self._fallback_classification()
                
        except Exception as e:
            logger.error(f"Vision classification error: {e}")
            
            # Log failed API usage
            response_time_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
            self.api_monitor.log_usage(
                provider='openai',
                model=self.config.ml.vision_model,
                operation='vision_classification',
                input_tokens=0,
                output_tokens=0,
                response_time_ms=response_time_ms,
                success=False,
                error_message=str(e)
            )
            
            return self._fallback_classification()
    
    def _estimate_input_tokens(self, prompt: str, base64_image: str) -> int:
        """Estimate input tokens for the API call."""
        # Text tokens (roughly 4 characters per token)
        text_tokens = len(prompt) // 4
        
        # Image tokens (GPT-4 Vision pricing)
        # For images, OpenAI charges based on image size
        # Rough estimate: ~765 tokens for a typical screenshot
        image_tokens = 765
        
        return text_tokens + image_tokens
    
    def _estimate_output_tokens(self, response: str) -> int:
        """Estimate output tokens from the response."""
        # Roughly 4 characters per token
        return len(response) // 4
    
    def _create_classification_prompt(self) -> str:
        """Create the prompt for activity classification."""
        categories_list = "\n".join([f"- {cat}: {desc}" for cat, desc in self.activity_categories.items()])
        
        return f"""
Analyze this screenshot and classify the user's primary activity. Look at the interface, applications, content, and context clues.

Available categories:
{categories_list}

Please respond with a JSON object containing:
1. "activity_type": the most appropriate category from the list above
2. "confidence": a confidence score from 0.0 to 1.0
3. "reasoning": a brief explanation of why you chose this classification
4. "details": specific observations that led to this classification (e.g., "VS Code interface with Python code visible")

Focus on the dominant activity and what the user is actively doing, not just what's visible on screen.
"""
    
    async def _call_openai_vision(self, base64_image: str, prompt: str) -> Optional[str]:
        """Call OpenAI Vision API with the image and prompt."""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.ml.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "low"  # Use "low" for faster processing
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent classification
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the OpenAI response and extract classification data."""
        try:
            # Try to extract JSON from the response
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            
            data = json.loads(response_clean)
            
            # Validate the response
            activity_type = data.get('activity_type', 'unknown')
            if activity_type not in self.activity_categories:
                activity_type = 'unknown'
            
            confidence = float(data.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            
            return {
                'activity_type': activity_type,
                'confidence': confidence,
                'reasoning': data.get('reasoning', ''),
                'details': data.get('details', ''),
                'source': 'vision_classifier',
                'timestamp': datetime.now().isoformat()
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse vision classification response: {e}")
            logger.debug(f"Raw response: {response}")
            return self._fallback_classification()
    
    def _fallback_classification(self) -> Dict[str, Any]:
        """Return a fallback classification when vision analysis fails."""
        return {
            'activity_type': 'unknown',
            'confidence': 0.3,
            'reasoning': 'Vision classification failed',
            'details': 'Could not analyze screenshot',
            'source': 'vision_classifier_fallback',
            'timestamp': datetime.now().isoformat()
        }


class ObserverAgent:
    """
    Observer Agent
    
    Continuously monitors system state through screenshot capture, window detection,
    and OCR processing. Coordinates activity classification and data storage.
    """
    
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
        self.vision_activity_classifier = VisionActivityClassifier(config)
        
        # State tracking
        self.is_running = False
        self.screenshot_task = None
        self.window_monitoring_task = None
        self.last_screenshot_time = None
        self.last_window_check_time = None
        self.current_session_id = None
        
        # Activity session tracking
        self.current_activity_session = None
        self.last_activity_type = None
        self.activity_session_start_time = None
        self._activity_change_threshold = 0.1  # Confidence threshold for activity changes
        
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
        """Process OCR for a screenshot and classify activity using both text and vision methods."""
        try:
            # Process OCR if enabled
            ocr_data = None
            if self.config.capture.ocr_enabled:
                ocr_data = self.ocr_processor.process_image(
                    screenshot_data['image_data'],
                    screenshot_data['is_encrypted']
                )
                
                if ocr_data:
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
            
            # Get window information
            window_info = self.window_detector.get_active_window_info()
            
            # Try vision-based classification first (if enabled and API key available)
            vision_activity_data = None
            if (self.config.ml.use_vision_classification and 
                self.config.ml.openai_api_key and 
                self.vision_activity_classifier.client):
                
                try:
                    # Get unencrypted image data for vision analysis
                    image_data_for_vision = screenshot_data['image_data']
                    if screenshot_data['is_encrypted']:
                        # Decrypt for analysis (but keep original encrypted for storage)
                        image_data_for_vision = self.security_manager.decrypt_data(image_data_for_vision)
                    
                    if image_data_for_vision:
                        vision_activity_data = await self.vision_activity_classifier.classify_activity(image_data_for_vision)
                        logger.info(f"Vision classification: {vision_activity_data['activity_type']} "
                                   f"(confidence: {vision_activity_data['confidence']:.2f})")
                
                except Exception as e:
                    logger.warning(f"Vision classification failed, falling back to text-based: {e}")
            
            # Text-based classification as fallback or comparison
            text_activity_data = self.activity_classifier.classify_activity(window_info, ocr_data)
            
            # Choose the best classification result
            final_activity_data = self._select_best_classification(vision_activity_data, text_activity_data)
            
            # Calculate duration for this activity
            duration_seconds = await self._calculate_activity_duration(final_activity_data)
            
            # Store activity
            activity_id = self.database.store_activity(
                activity_type=final_activity_data['activity_type'],
                confidence=final_activity_data['confidence'],
                duration_seconds=duration_seconds,
                screenshot_id=screenshot_id,
                metadata=final_activity_data
            )
            
            # Emit activity classified event
            await self.event_bus.emit(
                EventType.ACTIVITY_CLASSIFIED,
                data={
                    'activity_id': activity_id,
                    'activity_type': final_activity_data['activity_type'],
                    'confidence': final_activity_data['confidence'],
                    'duration_seconds': duration_seconds,
                    'classification_source': final_activity_data.get('source', 'unknown')
                },
                source='observer_agent'
            )
                
        except Exception as e:
            logger.error(f"Screenshot processing error: {e}")
    
    async def _calculate_activity_duration(self, activity_data: Dict[str, Any]) -> int:
        """Calculate duration for the current activity based on session tracking."""
        current_time = datetime.now()
        activity_type = activity_data['activity_type']
        confidence = activity_data['confidence']
        
        # If this is the first activity or a different activity type
        if (self.last_activity_type != activity_type or 
            self.activity_session_start_time is None):
            
            # End previous session if it exists
            if self.last_activity_type and self.activity_session_start_time:
                previous_duration = int((current_time - self.activity_session_start_time).total_seconds())
                logger.debug(f"Ending {self.last_activity_type} session after {previous_duration}s")
            
            # Start new session
            self.last_activity_type = activity_type
            self.activity_session_start_time = current_time
            
            # For the first occurrence of an activity, duration is the screenshot interval
            duration_seconds = self.config.capture.screenshot_interval_seconds
            logger.debug(f"Starting new {activity_type} session")
            
        else:
            # Continuing same activity - calculate duration since session started
            duration_seconds = int((current_time - self.activity_session_start_time).total_seconds())
            logger.debug(f"Continuing {activity_type} session - total duration: {duration_seconds}s")
        
        return duration_seconds
    
    def _select_best_classification(self, 
                                   vision_data: Optional[Dict[str, Any]], 
                                   text_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best classification result from vision and text methods."""
        
        # If vision classification is not available, use text
        if not vision_data:
            text_data['source'] = 'text_classifier'
            return text_data
        
        # Compare confidence scores and method reliability
        vision_confidence = vision_data.get('confidence', 0.0)
        text_confidence = text_data.get('confidence', 0.0)
        
        # Vision classification tends to be more accurate for UI detection
        # But text classification might catch specific keywords
        
        # Use vision if it has reasonable confidence and is not 'unknown'
        if (vision_confidence >= self.config.ml.vision_confidence_threshold and 
            vision_data.get('activity_type') != 'unknown'):
            
            # Add comparison info to metadata
            vision_data['text_classification'] = {
                'activity_type': text_data['activity_type'],
                'confidence': text_data['confidence']
            }
            vision_data['source'] = 'vision_classifier'
            return vision_data
        
        # Use text classification if vision failed or has low confidence
        elif text_confidence >= self.config.ml.confidence_threshold:
            text_data['vision_classification'] = {
                'activity_type': vision_data.get('activity_type', 'unknown'),
                'confidence': vision_data.get('confidence', 0.0)
            }
            text_data['source'] = 'text_classifier'
            return text_data
        
        # If both have low confidence, prefer vision if available
        elif vision_confidence > text_confidence:
            vision_data['text_classification'] = {
                'activity_type': text_data['activity_type'],
                'confidence': text_data['confidence']
            }
            vision_data['source'] = 'vision_classifier_low_confidence'
            return vision_data
        
        else:
            text_data['vision_classification'] = {
                'activity_type': vision_data.get('activity_type', 'unknown'),
                'confidence': vision_data.get('confidence', 0.0)
            }
            text_data['source'] = 'text_classifier_fallback'
            return text_data
    
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