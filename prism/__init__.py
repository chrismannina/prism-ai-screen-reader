"""
Prism - AI-Powered Screen Reader & Time Tracker

An intelligent desktop ecosystem that understands your work patterns,
learns from your behavior, and evolves into a comprehensive productivity assistant.
"""

__version__ = "0.1.0"
__author__ = "Prism Team"
__email__ = "team@prism.ai"

from .core.config import PrismConfig
from .core.event_bus import EventBus
from .agents.observer import ObserverAgent

__all__ = [
    "PrismConfig",
    "EventBus", 
    "ObserverAgent",
] 