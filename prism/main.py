"""
Prism Main Application

Entry point for the Prism AI-powered screen reader and time tracker.
Coordinates all components and provides CLI interface.
"""

import asyncio
import signal
import sys
import os
import psutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from .core.config import PrismConfig
from .core.database import DatabaseManager, WindowInfo, Screenshot
from .core.event_bus import EventBus, EventType, get_event_bus
from .core.security import SecurityManager
from .core.api_monitor import APIMonitor
from .agents.observer import ObserverAgent


class PrismApp:
    """Main Prism application coordinator."""
    
    def __init__(self, config_file: Optional[str] = None):
        # Initialize core components
        self.config = PrismConfig(config_file)
        self.event_bus = get_event_bus()
        self.security_manager = SecurityManager(self.config)
        self.database = DatabaseManager(self.config)
        self.api_monitor = APIMonitor(self.config)
        
        # Initialize agents
        self.observer_agent = ObserverAgent(
            self.config, 
            self.event_bus, 
            self.database, 
            self.security_manager
        )
        
        # Setup logging
        self._setup_logging()
        
        # State
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        self._pid_file = self.config.get_data_directory() / "prism.pid"
        
        logger.info("Prism application initialized")
    
    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        log_dir = self.config.get_logs_directory()
        log_file = log_dir / f"prism_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Remove default logger
        logger.remove()
        
        # Add console logger
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Add file logger
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",
            retention="30 days",
            compression="zip"
        )
        
        logger.info(f"Logging configured - log file: {log_file}")
    
    async def start(self) -> None:
        """Start the Prism application."""
        if self.is_running:
            logger.warning("Prism is already running")
            return
        
        # Check if another instance is already running
        if self._is_already_running():
            logger.error("Another Prism instance is already running")
            raise RuntimeError("Another Prism instance is already running")
        
        try:
            logger.info("Starting Prism...")
            
            # Create PID file
            self._create_pid_file()
            
            # Start event bus
            await self.event_bus.start()
            
            # Subscribe to system events
            self._setup_event_handlers()
            
            # Start observer agent
            await self.observer_agent.start()
            
            self.is_running = True
            
            # Emit application startup event
            await self.event_bus.emit(
                EventType.SYSTEM_STARTUP,
                data={
                    'app': 'prism',
                    'version': '0.1.0',
                    'pid': os.getpid(),
                    'config': {
                        'screenshot_interval': self.config.capture.screenshot_interval_seconds,
                        'ocr_enabled': self.config.capture.ocr_enabled,
                        'encryption_enabled': self.config.privacy.encrypt_screenshots
                    }
                },
                source='prism_app'
            )
            
            logger.info("✅ Prism started successfully!")
            
        except Exception as e:
            logger.error(f"Failed to start Prism: {e}")
            self._cleanup_pid_file()
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the Prism application."""
        if not self.is_running:
            return
        
        logger.info("Stopping Prism...")
        
        try:
            # Stop observer agent
            await self.observer_agent.stop()
            
            # Stop event bus
            await self.event_bus.stop()
            
            # Close database connections
            self.database.close()
            
            # Cleanup security data
            self.security_manager.cleanup_security_data()
            
            # Remove PID file
            self._cleanup_pid_file()
            
            self.is_running = False
            self._shutdown_event.set()
            
            logger.info("✅ Prism stopped successfully!")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _create_pid_file(self) -> None:
        """Create a PID file to track the running process."""
        try:
            self._pid_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logger.debug(f"Created PID file: {self._pid_file}")
        except Exception as e:
            logger.warning(f"Could not create PID file: {e}")
    
    def _cleanup_pid_file(self) -> None:
        """Remove the PID file."""
        try:
            if self._pid_file.exists():
                self._pid_file.unlink()
                logger.debug(f"Removed PID file: {self._pid_file}")
        except Exception as e:
            logger.warning(f"Could not remove PID file: {e}")
    
    def _is_already_running(self) -> bool:
        """Check if another Prism instance is already running."""
        if not self._pid_file.exists():
            return False
        
        try:
            with open(self._pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if the process is still running
            if psutil.pid_exists(pid):
                try:
                    process = psutil.Process(pid)
                    # Check if it's actually a Prism process
                    if 'prism' in ' '.join(process.cmdline()).lower():
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # PID file exists but process is dead, clean it up
            self._cleanup_pid_file()
            return False
            
        except (ValueError, FileNotFoundError):
            # Invalid PID file, clean it up
            self._cleanup_pid_file()
            return False
    
    @classmethod
    def is_running_externally(cls, config_file: Optional[str] = None) -> dict:
        """Check if Prism is running without creating a new instance."""
        config = PrismConfig(config_file)
        pid_file = config.get_data_directory() / "prism.pid"
        
        result = {
            'is_running': False,
            'pid': None,
            'process_info': None
        }
        
        if not pid_file.exists():
            return result
        
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            if psutil.pid_exists(pid):
                try:
                    process = psutil.Process(pid)
                    if 'prism' in ' '.join(process.cmdline()).lower():
                        result['is_running'] = True
                        result['pid'] = pid
                        result['process_info'] = {
                            'command': ' '.join(process.cmdline()),
                            'create_time': datetime.fromtimestamp(process.create_time()),
                            'cpu_percent': process.cpu_percent(),
                            'memory_mb': process.memory_info().rss / 1024 / 1024
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        except (ValueError, FileNotFoundError):
            pass
        
        return result
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for system events."""
        
        def handle_system_event(event):
            """Handle system events."""
            logger.debug(f"System event: {event.event_type.value} from {event.source}")
            
            # Log significant events
            if event.event_type in [EventType.SYSTEM_ERROR, EventType.SENSITIVE_DATA_DETECTED]:
                logger.warning(f"Important event: {event.event_type.value} - {event.data}")
        
        def handle_activity_event(event):
            """Handle activity classification events."""
            if event.event_type == EventType.ACTIVITY_CLASSIFIED:
                activity_type = event.data.get('activity_type', 'unknown')
                confidence = event.data.get('confidence', 0)
                logger.info(f"Activity detected: {activity_type} (confidence: {confidence:.2f})")
        
        # Subscribe to events
        self.event_bus.subscribe(
            handle_system_event,
            event_types=[EventType.SYSTEM_ERROR, EventType.SENSITIVE_DATA_DETECTED],
            source_filter=None
        )
        
        self.event_bus.subscribe(
            handle_activity_event,
            event_types=[EventType.ACTIVITY_CLASSIFIED],
            source_filter=None
        )
    
    async def run_until_stopped(self) -> None:
        """Run the application until stopped."""
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for shutdown
        await self._shutdown_event.wait()
    
    def get_status(self) -> dict:
        """Get current application status."""
        return {
            'is_running': self.is_running,
            'observer_agent': self.observer_agent.get_status(),
            'event_bus': {
                'handler_count': self.event_bus.get_handler_count(),
                'running': self.event_bus._running
            },
            'database': self.database.get_database_stats(),
            'security': self.security_manager.get_security_status(),
            'config_file': self.config.config_file
        }


# CLI Interface

@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Prism - AI-Powered Screen Reader & Time Tracker"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")


@cli.command()
@click.option('--mode', default='observer', help='Running mode (observer, dashboard)')
@click.pass_context
def start(ctx, mode):
    """Start Prism in the specified mode."""
    config_file = ctx.obj.get('config')
    
    async def run_app():
        app = PrismApp(config_file)
        
        try:
            await app.start()
            
            if mode == 'observer':
                logger.info("Running in observer mode - monitoring screen activity...")
                await app.run_until_stopped()
            else:
                logger.error(f"Unknown mode: {mode}")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            await app.stop()
    
    # Check for required permissions
    if not _check_permissions():
        logger.error("Required permissions not granted. Please check README for setup instructions.")
        return
    
    # Run the application
    asyncio.run(run_app())


@cli.command()
@click.pass_context
def status(ctx):
    """Show current Prism status."""
    config_file = ctx.obj.get('config')
    
    try:
        # Check if Prism is running externally
        running_info = PrismApp.is_running_externally(config_file)
        
        # Get database and config info
        config = PrismConfig(config_file)
        database = DatabaseManager(config)
        security_manager = SecurityManager(config)
        api_monitor = APIMonitor(config)
        
        db_stats = database.get_database_stats()
        security_status = security_manager.get_security_status()
        api_status = api_monitor.get_status()
        
        click.echo("🔍 Prism Status")
        click.echo("=" * 50)
        
        # Show running status with process info
        if running_info['is_running']:
            process_info = running_info['process_info']
            click.echo(f"Running: ✅ Yes (PID: {running_info['pid']})")
            click.echo(f"Started: {process_info['create_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"CPU Usage: {process_info['cpu_percent']:.1f}%")
            click.echo(f"Memory: {process_info['memory_mb']:.1f} MB")
        else:
            click.echo("Running: ❌ No")
        
        click.echo(f"Config File: {config.config_file}")
        
        # Show database stats
        click.echo(f"\n🗄️  Database:")
        click.echo(f"  Screenshots: {db_stats.get('screenshots_count', 0):,}")
        click.echo(f"  Activities: {db_stats.get('activities_count', 0):,}")
        click.echo(f"  Window Records: {db_stats.get('window_info_count', 0):,}")
        click.echo(f"  Size: {db_stats.get('database_size_mb', 0):.1f} MB")
        
        # Show security status
        click.echo(f"\n🔐 Security:")
        click.echo(f"  Encryption: {'✅ Enabled' if security_status['encryption_enabled'] else '❌ Disabled'}")
        click.echo(f"  Excluded Apps: {security_status['excluded_apps_count']}")
        
        # Show API monitoring status
        if config.ml.api_monitoring_enabled:
            click.echo(f"\n💰 API Usage (OpenAI):")
            daily = api_status['daily_usage']
            monthly = api_status['monthly_usage']
            
            # Daily usage
            daily_icon = "🟢" if daily['percentage'] < 50 else "🟡" if daily['percentage'] < 80 else "🔴"
            click.echo(f"  Today: {daily_icon} ${daily['cost_usd']:.3f} / ${daily['limit_usd']:.2f} ({daily['percentage']:.1f}%)")
            click.echo(f"         {daily['calls']} calls, {daily['tokens']:,} tokens")
            
            # Monthly usage
            monthly_icon = "🟢" if monthly['percentage'] < 50 else "🟡" if monthly['percentage'] < 80 else "🔴"
            click.echo(f"  Month: {monthly_icon} ${monthly['cost_usd']:.3f} / ${monthly['limit_usd']:.2f} ({monthly['percentage']:.1f}%)")
            click.echo(f"         {monthly['calls']} calls, {monthly['tokens']:,} tokens")
            
            click.echo(f"  Success Rate: {api_status['success_rate_24h']:.1%}")
        
        # Close connections
        database.close()
        
    except Exception as e:
        click.echo(f"❌ Error getting status: {e}")
        logger.error(f"Status command error: {e}")


@cli.command()
@click.option('--days', default=7, help='Number of days to clean up')
@click.pass_context
def cleanup(ctx, days):
    """Clean up old data."""
    config_file = ctx.obj.get('config')
    
    try:
        app = PrismApp(config_file)
        app.database.cleanup_old_data(days)
        click.echo(f"✅ Cleaned up data older than {days} days")
    except Exception as e:
        click.echo(f"❌ Cleanup failed: {e}")


@cli.command()
def init():
    """Initialize Prism configuration and database."""
    try:
        click.echo("🔧 Initializing Prism...")
        
        # Initialize configuration
        config = PrismConfig()
        config.save_config()
        
        # Initialize database
        database = DatabaseManager(config)
        
        # Initialize security
        security_manager = SecurityManager(config)
        
        logger.info("Prism initialization completed successfully")
        click.echo("✅ Prism initialized successfully!")
        click.echo(f"Config saved to: {config.config_file}")
        click.echo(f"Database initialized at: {config.get_data_directory()}")
        click.echo()
        click.echo("📋 Next Steps:")
        click.echo("1. Grant screen recording permissions (System Preferences > Privacy)")
        click.echo("2. Install Tesseract OCR: brew install tesseract")
        click.echo("3. Run: prism start")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        click.echo(f"❌ Initialization failed: {e}")
        raise click.Abort()


@cli.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed activity breakdown')
@click.option('--hours', '-h', default=24, help='Hours of history to show (default: 24)')
@click.option('--limit', '-l', default=20, help='Limit number of activities shown (default: 20)')
@click.option('--show-ocr', is_flag=True, help='Show OCR text content')
def report(detailed, hours, limit, show_ocr):
    """Generate a detailed report of captured data."""
    try:
        config = PrismConfig()
        database = DatabaseManager(config)
        
        click.echo("📊 Prism Activity Report")
        click.echo("=" * 50)
        
        # Get database statistics
        db_stats = database.get_database_stats()
        click.echo(f"🗄️  Database Overview:")
        click.echo(f"   Screenshots: {db_stats.get('screenshots_count', 0):,}")
        click.echo(f"   Activities: {db_stats.get('activities_count', 0):,}")
        click.echo(f"   Window Records: {db_stats.get('window_info_count', 0):,}")
        click.echo(f"   Database Size: {db_stats.get('database_size_mb', 0):.1f} MB")
        click.echo()
        
        # Get recent activities
        activities = database.get_recent_activities(limit=limit, hours_back=hours)
        
        if not activities:
            click.echo("🔍 No activities found in the specified time range.")
            return
        
        click.echo(f"📈 Recent Activities (Last {hours} hours, showing {len(activities)} of {limit}):")
        click.echo("-" * 80)
        
        for i, activity in enumerate(activities, 1):
            timestamp = activity.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            confidence_icon = "🟢" if activity.confidence > 0.7 else "🟡" if activity.confidence > 0.5 else "🔴"
            
            click.echo(f"{i:2d}. {timestamp} | {confidence_icon} {activity.activity_type:<12} | {activity.confidence:.1%}")
            
            if detailed:
                # Show metadata if available
                if activity.activity_metadata:
                    import json
                    try:
                        metadata = json.loads(activity.activity_metadata) if isinstance(activity.activity_metadata, str) else activity.activity_metadata
                        if 'all_scores' in metadata:
                            scores = metadata['all_scores']
                            top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                            score_str = " | ".join([f"{k}: {v:.2f}" for k, v in top_3])
                            click.echo(f"     Scores: {score_str}")
                    except:
                        pass
                
                # Show window info if available
                try:
                    if hasattr(activity, 'window_info_id') and activity.window_info_id:
                        # Get window info separately to avoid session issues
                        with database.get_session() as session:
                            window_info = session.get(WindowInfo, activity.window_info_id)
                            if window_info:
                                click.echo(f"     App: {window_info.app_name}")
                                if window_info.window_title != window_info.app_name:
                                    click.echo(f"     Window: {window_info.window_title}")
                except:
                    pass
                
                # Show screenshot info if available
                try:
                    if hasattr(activity, 'screenshot_id') and activity.screenshot_id:
                        # Get screenshot info separately to avoid session issues
                        with database.get_session() as session:
                            screenshot = session.get(Screenshot, activity.screenshot_id)
                            if screenshot:
                                resolution = f"{screenshot.resolution_width}x{screenshot.resolution_height}"
                                size_kb = screenshot.file_size_bytes / 1024
                                click.echo(f"     Screenshot: {resolution} ({size_kb:.1f} KB)")
                                if show_ocr and screenshot.ocr_text:
                                    # Show first 200 characters of OCR text
                                    ocr_preview = screenshot.ocr_text[:200] + "..." if len(screenshot.ocr_text) > 200 else screenshot.ocr_text
                                    click.echo(f"     OCR Text: {ocr_preview}")
                                elif screenshot.ocr_text and not show_ocr:
                                    click.echo(f"     OCR Text: {len(screenshot.ocr_text)} characters (use --show-ocr to view)")
                except:
                    pass
                
                click.echo()
        
        # Get daily summary
        summary = database.get_daily_summary()
        if summary:
            click.echo("📊 Today's Summary:")
            click.echo("-" * 30)
            total_minutes = summary.get('total_duration_seconds', 0) // 60
            click.echo(f"Total Active Time: {total_minutes} minutes")
            click.echo(f"Total Activities: {summary.get('total_activities', 0)}")
            
            if summary.get('activity_breakdown'):
                click.echo("\nActivity Breakdown:")
                total_duration = summary.get('total_duration_seconds', 0)
                for activity_type, data in summary['activity_breakdown'].items():
                    duration_min = data['duration'] // 60
                    percentage = (data['duration'] / total_duration * 100) if total_duration > 0 else 0
                    click.echo(f"  {activity_type.title()}: {duration_min}m ({percentage:.1f}%) - {data['count']} sessions")
        
        database.close()
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        click.echo(f"❌ Report generation failed: {e}")
        raise click.Abort()


@cli.command()
@click.option('--limit', '-l', default=10, help='Number of screenshots to show (default: 10)')
@click.option('--save-to', '-s', help='Directory to save screenshots to')
@click.option('--open-viewer', '-o', is_flag=True, help='Open screenshots in default image viewer')
def screenshots(limit, save_to, open_viewer):
    """View and extract screenshots from the database."""
    try:
        config = PrismConfig()
        database = DatabaseManager(config)
        security_manager = SecurityManager(config)
        
        click.echo("📸 Prism Screenshots")
        click.echo("=" * 40)
        
        # Get recent screenshots
        with database.get_session() as session:
            screenshots_data = session.query(Screenshot).order_by(Screenshot.timestamp.desc()).limit(limit).all()
            
            if not screenshots_data:
                click.echo("🔍 No screenshots found.")
                return
            
            click.echo(f"Found {len(screenshots_data)} screenshots:")
            click.echo("-" * 50)
            
            for i, screenshot in enumerate(screenshots_data, 1):
                timestamp = screenshot.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                resolution = f"{screenshot.resolution_width}x{screenshot.resolution_height}"
                size_kb = screenshot.file_size_bytes / 1024
                encrypted_icon = "🔒" if screenshot.is_encrypted else "🔓"
                
                click.echo(f"{i:2d}. {timestamp} | {resolution} | {size_kb:6.1f} KB | {encrypted_icon}")
                
                if screenshot.ocr_text:
                    ocr_preview = screenshot.ocr_text[:60] + "..." if len(screenshot.ocr_text) > 60 else screenshot.ocr_text
                    click.echo(f"    OCR: {ocr_preview}")
                
                # Save screenshot if requested
                if save_to:
                    saved_path = _save_screenshot(screenshot, save_to, security_manager, i)
                    if saved_path:
                        click.echo(f"    💾 Saved: {saved_path}")
                        
                        # Open in viewer if requested
                        if open_viewer:
                            _open_screenshot(saved_path)
                
                click.echo()
        
        if save_to:
            click.echo(f"✅ Screenshots saved to: {save_to}")
            if open_viewer:
                click.echo("🖼️  Opening screenshots in default viewer...")
        
        database.close()
        
    except Exception as e:
        logger.error(f"Screenshot viewing failed: {e}")
        click.echo(f"❌ Screenshot viewing failed: {e}")
        raise click.Abort()


def _save_screenshot(screenshot, save_dir: str, security_manager, index: int) -> Optional[str]:
    """Save a screenshot to disk."""
    try:
        import os
        from PIL import Image
        import io
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Get image data
        image_data = screenshot.image_data
        if not image_data:
            logger.warning(f"No image data for screenshot {screenshot.id}")
            return None
        
        # Decrypt if necessary
        if screenshot.is_encrypted:
            image_data = security_manager.decrypt_data(image_data)
            if image_data is None:
                logger.error(f"Failed to decrypt screenshot {screenshot.id}")
                return None
        
        # Create filename
        timestamp_str = screenshot.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"prism_screenshot_{index:02d}_{timestamp_str}.png"
        file_path = save_path / filename
        
        # Load and save image
        image = Image.open(io.BytesIO(image_data))
        image.save(file_path, 'PNG')
        
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving screenshot: {e}")
        return None


def _open_screenshot(file_path: str) -> None:
    """Open screenshot in default image viewer."""
    try:
        import subprocess
        import sys
        
        if sys.platform.startswith('darwin'):  # macOS
            subprocess.run(['open', file_path])
        elif sys.platform.startswith('linux'):  # Linux
            subprocess.run(['xdg-open', file_path])
        elif sys.platform.startswith('win'):  # Windows
            subprocess.run(['start', file_path], shell=True)
        else:
            logger.warning(f"Unknown platform {sys.platform}, cannot open image viewer")
            
    except Exception as e:
        logger.error(f"Error opening screenshot: {e}")


def _check_permissions() -> bool:
    """Check if required permissions are granted."""
    try:
        # Try to take a test screenshot
        import pyautogui
        pyautogui.screenshot()
        return True
    except Exception as e:
        logger.error(f"Permission check failed: {e}")
        return False


@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
@click.option('--port', default=5000, help='Port to bind to (default: 5000)')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def dashboard(ctx, host, port, debug):
    """Launch the Prism web dashboard."""
    try:
        from .web_dashboard import PrismWebDashboard
        
        config_file = ctx.obj.get('config')
        web_dashboard = PrismWebDashboard(PrismConfig(config_file) if config_file else None)
        
        click.echo("🌐 Starting Prism Web Dashboard...")
        click.echo(f"📱 Dashboard will be available at: http://{host}:{port}")
        click.echo("🔄 Press Ctrl+C to stop the dashboard")
        
        web_dashboard.run(host=host, port=port, debug=debug)
        
    except ImportError as e:
        click.echo(f"❌ Web dashboard dependencies missing: {e}")
        click.echo("💡 Install with: pip install flask flask-socketio")
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Failed to start web dashboard: {e}")
        raise click.Abort()


@cli.command()
def diagnose():
    """Diagnose Prism permissions and screenshot capabilities."""
    try:
        click.echo("🔍 Prism Diagnostic Tool")
        click.echo("=" * 40)
        
        # Test basic screenshot capability
        click.echo("\n📸 Testing Screenshot Capability:")
        try:
            import pyautogui
            test_screenshot = pyautogui.screenshot()
            click.echo(f"✅ Basic screenshot successful: {test_screenshot.size}")
            
            # Check if screenshot contains actual content or just background
            # Convert to numpy array to analyze pixel diversity
            import numpy as np
            img_array = np.array(test_screenshot)
            
            # Calculate color diversity (more colors = likely capturing applications)
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            color_diversity = unique_colors / total_pixels
            
            click.echo(f"🎨 Color diversity: {color_diversity:.6f} ({unique_colors:,} unique colors)")
            
            if color_diversity < 0.001:
                click.echo("⚠️  WARNING: Very low color diversity - likely only capturing desktop background")
                click.echo("   This suggests insufficient screen recording permissions.")
            else:
                click.echo("✅ Good color diversity - appears to be capturing actual content")
                
        except Exception as e:
            click.echo(f"❌ Screenshot test failed: {e}")
        
        # Test window detection
        click.echo("\n🪟 Testing Window Detection:")
        try:
            from .agents.observer import WindowDetector
            config = PrismConfig()
            window_detector = WindowDetector(config)
            window_info = window_detector.get_active_window_info()
            
            if window_info:
                click.echo(f"✅ Active window detected:")
                click.echo(f"   App: {window_info['app_name']}")
                click.echo(f"   Window: {window_info['window_title']}")
                click.echo(f"   Bundle ID: {window_info.get('bundle_id', 'N/A')}")
            else:
                click.echo("❌ No active window detected")
                
        except Exception as e:
            click.echo(f"❌ Window detection test failed: {e}")
        
        # Test OCR capability
        click.echo("\n📝 Testing OCR Capability:")
        try:
            import pytesseract
            # Test with a simple image
            from PIL import Image, ImageDraw, ImageFont
            
            # Create test image with text
            img = Image.new('RGB', (200, 100), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Test OCR Text", fill='black')
            
            ocr_result = pytesseract.image_to_string(img).strip()
            if ocr_result:
                click.echo(f"✅ OCR working: '{ocr_result}'")
            else:
                click.echo("⚠️  OCR test returned empty result")
                
        except Exception as e:
            click.echo(f"❌ OCR test failed: {e}")
        
        # macOS-specific permission checks
        click.echo("\n🍎 macOS Permission Status:")
        try:
            import subprocess
            
            # Check screen recording permission using tccutil
            result = subprocess.run([
                'sqlite3', 
                f'{os.path.expanduser("~")}/Library/Application Support/com.apple.TCC/TCC.db', 
                "SELECT service, client, auth_value FROM access WHERE service='kTCCServiceScreenCapture';"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout:
                click.echo("✅ Screen recording permissions found in TCC database")
                # Parse and show relevant entries
                for line in result.stdout.strip().split('\n'):
                    if 'python' in line.lower() or 'terminal' in line.lower():
                        click.echo(f"   {line}")
            else:
                click.echo("⚠️  No screen recording permissions found in TCC database")
                
        except Exception as e:
            click.echo(f"⚠️  Could not check TCC permissions: {e}")
        
        # Recommendations
        click.echo("\n💡 Recommendations:")
        click.echo("1. Grant Screen Recording permission:")
        click.echo("   System Preferences → Security & Privacy → Privacy → Screen Recording")
        click.echo("   Add Terminal (or your Python environment) to the list")
        click.echo("")
        click.echo("2. If using a virtual environment, you may need to grant permissions to:")
        click.echo(f"   Python executable: {sys.executable}")
        click.echo("")
        click.echo("3. After granting permissions, restart Terminal and try again")
        click.echo("4. Test with: python -c \"import pyautogui; pyautogui.screenshot().show()\"")
        
    except Exception as e:
        click.echo(f"❌ Diagnostic failed: {e}")
        raise click.Abort()


if __name__ == "__main__":
    cli() 