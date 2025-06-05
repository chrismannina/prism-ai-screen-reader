"""
Prism Main Application

Entry point for the Prism AI-powered screen reader and time tracker.
Coordinates all components and provides CLI interface.
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from .core.config import PrismConfig
from .core.event_bus import EventBus, EventType, get_event_bus
from .core.database import DatabaseManager
from .core.security import SecurityManager
from .agents.observer import ObserverAgent


class PrismApp:
    """Main Prism application coordinator."""
    
    def __init__(self, config_file: Optional[str] = None):
        # Initialize core components
        self.config = PrismConfig(config_file)
        self.event_bus = get_event_bus()
        self.security_manager = SecurityManager(self.config)
        self.database = DatabaseManager(self.config)
        
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
        
        try:
            logger.info("Starting Prism...")
            
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
            
            self.is_running = False
            self._shutdown_event.set()
            
            logger.info("✅ Prism stopped successfully!")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
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
        app = PrismApp(config_file)
        status_info = app.get_status()
        
        click.echo("🔍 Prism Status")
        click.echo("=" * 50)
        click.echo(f"Running: {'✅ Yes' if status_info['is_running'] else '❌ No'}")
        click.echo(f"Config File: {status_info['config_file']}")
        
        if status_info['observer_agent']:
            agent_status = status_info['observer_agent']
            click.echo(f"\n📸 Observer Agent:")
            click.echo(f"  Running: {'✅ Yes' if agent_status['is_running'] else '❌ No'}")
            click.echo(f"  Last Screenshot: {agent_status['last_screenshot_time'] or 'Never'}")
            click.echo(f"  Current Window: {agent_status['current_window'] or 'None'}")
        
        if status_info['database']:
            db_stats = status_info['database']
            click.echo(f"\n🗄️  Database:")
            click.echo(f"  Screenshots: {db_stats.get('screenshots_count', 0)}")
            click.echo(f"  Activities: {db_stats.get('activities_count', 0)}")
            click.echo(f"  Size: {db_stats.get('database_size_mb', 0):.1f} MB")
        
        if status_info['security']:
            security_status = status_info['security']
            click.echo(f"\n🔐 Security:")
            click.echo(f"  Encryption: {'✅ Enabled' if security_status['encryption_enabled'] else '❌ Disabled'}")
            click.echo(f"  Excluded Apps: {security_status['excluded_apps_count']}")
            
    except Exception as e:
        click.echo(f"❌ Error getting status: {e}")


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
@click.pass_context
def init(ctx):
    """Initialize Prism configuration and database."""
    config_file = ctx.obj.get('config')
    
    try:
        click.echo("🔧 Initializing Prism...")
        
        # Create config
        config = PrismConfig(config_file)
        config.save_config()
        
        # Initialize database
        database = DatabaseManager(config)
        
        # Initialize security
        security_manager = SecurityManager(config)
        
        click.echo("✅ Prism initialized successfully!")
        click.echo(f"Config saved to: {config.config_file}")
        click.echo(f"Database initialized at: {config.get_data_directory()}")
        
        # Show next steps
        click.echo("\n📋 Next Steps:")
        click.echo("1. Grant screen recording permissions (System Preferences > Privacy)")
        click.echo("2. Install Tesseract OCR: brew install tesseract")
        click.echo("3. Run: python -m prism.main start")
        
    except Exception as e:
        click.echo(f"❌ Initialization failed: {e}")


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


if __name__ == "__main__":
    cli() 