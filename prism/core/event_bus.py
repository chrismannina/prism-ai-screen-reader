"""
Event Bus System for Prism

Provides a centralized event-driven communication system for all Prism agents
and components. Supports synchronous and asynchronous event handling.
"""

import asyncio
import threading
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from loguru import logger


class EventType(Enum):
    """Standard event types used throughout Prism."""
    
    # System Events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    
    # Observer Events
    SCREENSHOT_CAPTURED = "observer.screenshot_captured"
    WINDOW_CHANGED = "observer.window_changed"
    ACTIVITY_DETECTED = "observer.activity_detected"
    OCR_COMPLETED = "observer.ocr_completed"
    
    # Classification Events
    ACTIVITY_CLASSIFIED = "classifier.activity_classified"
    PROJECT_DETECTED = "classifier.project_detected"
    
    # Storage Events
    DATA_STORED = "storage.data_stored"
    DATA_RETRIEVED = "storage.data_retrieved"
    
    # Privacy Events
    SENSITIVE_DATA_DETECTED = "privacy.sensitive_data_detected"
    DATA_ENCRYPTED = "privacy.data_encrypted"
    
    # User Events
    USER_IDLE = "user.idle"
    USER_ACTIVE = "user.active"
    USER_BREAK = "user.break"


@dataclass
class Event:
    """Represents an event in the Prism system."""
    
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0  # Higher numbers = higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority
        }


class EventHandler:
    """Base class for event handlers."""
    
    def __init__(self, callback: Callable[[Event], Union[None, Any]], 
                 event_types: Optional[List[EventType]] = None,
                 source_filter: Optional[str] = None,
                 async_handler: bool = False):
        self.callback = callback
        self.event_types = event_types or []
        self.source_filter = source_filter
        self.async_handler = async_handler
        self.handler_id = str(uuid.uuid4())
    
    def should_handle(self, event: Event) -> bool:
        """Check if this handler should process the given event."""
        # Check event type filter
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Check source filter
        if self.source_filter and event.source != self.source_filter:
            return False
        
        return True
    
    async def handle(self, event: Event) -> Any:
        """Handle the event (async wrapper for sync callbacks)."""
        try:
            if self.async_handler:
                return await self.callback(event)
            else:
                return self.callback(event)
        except Exception as e:
            logger.error(f"Error in event handler {self.handler_id}: {e}")
            return None


class EventBus:
    """Central event bus for the Prism system."""
    
    def __init__(self):
        self._handlers: List[EventHandler] = []
        self._event_history: List[Event] = []
        self._running = False
        self._lock = threading.Lock()
        self._event_queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        
        logger.info("EventBus initialized")
    
    def subscribe(self, 
                  callback: Callable[[Event], Union[None, Any]],
                  event_types: Optional[List[EventType]] = None,
                  source_filter: Optional[str] = None,
                  async_handler: bool = False) -> str:
        """
        Subscribe to events with optional filtering.
        
        Args:
            callback: Function to call when event is received
            event_types: List of event types to listen for (None = all)
            source_filter: Only receive events from this source (None = all)
            async_handler: Whether the callback is async
            
        Returns:
            Handler ID for unsubscribing
        """
        handler = EventHandler(
            callback=callback,
            event_types=event_types,
            source_filter=source_filter,
            async_handler=async_handler
        )
        
        with self._lock:
            self._handlers.append(handler)
        
        logger.debug(f"Subscribed handler {handler.handler_id} for events {event_types}")
        return handler.handler_id
    
    def unsubscribe(self, handler_id: str) -> bool:
        """
        Unsubscribe an event handler.
        
        Args:
            handler_id: ID returned from subscribe()
            
        Returns:
            True if handler was found and removed
        """
        with self._lock:
            for i, handler in enumerate(self._handlers):
                if handler.handler_id == handler_id:
                    del self._handlers[i]
                    logger.debug(f"Unsubscribed handler {handler_id}")
                    return True
        
        logger.warning(f"Handler {handler_id} not found for unsubscribe")
        return False
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to all interested handlers.
        
        Args:
            event: Event to publish
        """
        # Add to history
        self._event_history.append(event)
        
        # Keep history reasonable size
        if len(self._event_history) > 1000:
            self._event_history = self._event_history[-500:]
        
        # Add to processing queue
        await self._event_queue.put(event)
        
        logger.debug(f"Published event {event.event_type.value} from {event.source}")
    
    def publish_sync(self, event: Event) -> None:
        """
        Publish an event synchronously (for non-async contexts).
        
        Args:
            event: Event to publish
        """
        # Create new event loop if none exists
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule the coroutine
                asyncio.create_task(self.publish(event))
            else:
                # If loop is not running, run the coroutine
                loop.run_until_complete(self.publish(event))
        except RuntimeError:
            # No event loop exists, create a new one
            asyncio.run(self.publish(event))
    
    async def emit(self, 
                   event_type: EventType,
                   data: Optional[Dict[str, Any]] = None,
                   source: str = "unknown",
                   priority: int = 0) -> None:
        """
        Convenience method to create and publish an event.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Source component/agent name
            priority: Event priority
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
            priority=priority
        )
        await self.publish(event)
    
    def emit_sync(self,
                  event_type: EventType,
                  data: Optional[Dict[str, Any]] = None,
                  source: str = "unknown",
                  priority: int = 0) -> None:
        """Synchronous version of emit()."""
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
            priority=priority
        )
        self.publish_sync(event)
    
    async def start(self) -> None:
        """Start the event bus worker."""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._event_worker())
        logger.info("EventBus started")
    
    async def stop(self) -> None:
        """Stop the event bus worker."""
        if not self._running:
            return
        
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("EventBus stopped")
    
    async def _event_worker(self) -> None:
        """Main event processing worker."""
        logger.info("Event worker started")
        
        while self._running:
            try:
                # Get next event with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event worker: {e}")
    
    async def _process_event(self, event: Event) -> None:
        """Process a single event by calling all interested handlers."""
        handlers_to_call = []
        
        with self._lock:
            for handler in self._handlers:
                if handler.should_handle(event):
                    handlers_to_call.append(handler)
        
        # Sort handlers by priority (higher first)
        handlers_to_call.sort(key=lambda h: event.priority, reverse=True)
        
        # Call handlers
        tasks = []
        for handler in handlers_to_call:
            tasks.append(handler.handle(event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_event_history(self, 
                          event_type: Optional[EventType] = None,
                          source: Optional[str] = None,
                          limit: int = 100) -> List[Event]:
        """
        Get recent events from history with optional filtering.
        
        Args:
            event_type: Filter by event type
            source: Filter by source
            limit: Maximum number of events to return
            
        Returns:
            List of events matching criteria
        """
        events = self._event_history
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if source:
            events = [e for e in events if e.source == source]
        
        # Sort by timestamp (newest first) and limit
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_handler_count(self) -> int:
        """Get the number of registered event handlers."""
        return len(self._handlers)
    
    def clear_history(self) -> None:
        """Clear the event history."""
        self._event_history.clear()
        logger.info("Event history cleared")


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus 