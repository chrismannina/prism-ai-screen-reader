"""
Database Management for Prism

Handles all data storage including screenshots, activities, window tracking,
and time sessions using SQLAlchemy with SQLite backend.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, 
    Float, Text, Boolean, ForeignKey, JSON, LargeBinary
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

from .config import PrismConfig

Base = declarative_base()


class Screenshot(Base):
    """Database model for screenshot data."""
    
    __tablename__ = "screenshots"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    file_path = Column(String(500), nullable=True)  # None if data stored in blob
    image_data = Column(LargeBinary, nullable=True)  # Encrypted image data
    resolution_width = Column(Integer, nullable=False)
    resolution_height = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    is_encrypted = Column(Boolean, default=True)
    ocr_text = Column(Text, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    
    # Relationships
    activities = relationship("Activity", back_populates="screenshot")


class WindowInfo(Base):
    """Database model for window information."""
    
    __tablename__ = "window_info"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    window_title = Column(String(500), nullable=False)
    app_name = Column(String(200), nullable=False)
    app_bundle_id = Column(String(200), nullable=True)  # macOS specific
    window_bounds = Column(JSON, nullable=True)  # {"x": 0, "y": 0, "width": 1920, "height": 1080}
    is_active = Column(Boolean, default=False)
    is_minimized = Column(Boolean, default=False)
    is_fullscreen = Column(Boolean, default=False)
    
    # Relationships
    activities = relationship("Activity", back_populates="window_info")


class Activity(Base):
    """Database model for classified activities."""
    
    __tablename__ = "activities"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    activity_type = Column(String(100), nullable=False)  # coding, writing, browsing, etc.
    confidence = Column(Float, nullable=False)
    duration_seconds = Column(Integer, default=0)
    
    # Foreign Keys
    screenshot_id = Column(Integer, ForeignKey("screenshots.id"), nullable=True)
    window_info_id = Column(Integer, ForeignKey("window_info.id"), nullable=True)
    
    # Additional data
    activity_metadata = Column(JSON, nullable=True)  # Additional classification data
    project_name = Column(String(200), nullable=True)
    tags = Column(JSON, nullable=True)  # List of tags
    
    # Relationships
    screenshot = relationship("Screenshot", back_populates="activities")
    window_info = relationship("WindowInfo", back_populates="activities")


class TimeSession(Base):
    """Database model for time tracking sessions."""
    
    __tablename__ = "time_sessions"
    
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, default=datetime.now, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    session_type = Column(String(50), default="work")  # work, break, idle
    
    # Session metadata
    primary_app = Column(String(200), nullable=True)
    primary_activity = Column(String(100), nullable=True)
    project_name = Column(String(200), nullable=True)
    focus_score = Column(Float, nullable=True)  # 0-1 focus rating
    interruptions_count = Column(Integer, default=0)
    
    # Additional data
    session_metadata = Column(JSON, nullable=True)


class SystemEvent(Base):
    """Database model for system events and logs."""
    
    __tablename__ = "system_events"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    event_type = Column(String(100), nullable=False)
    source = Column(String(100), nullable=False)
    message = Column(Text, nullable=True)
    level = Column(String(20), default="INFO")  # DEBUG, INFO, WARNING, ERROR
    event_data = Column(JSON, nullable=True)


class DatabaseManager:
    """Manages database operations for Prism."""
    
    def __init__(self, config: PrismConfig):
        self.config = config
        self.engine = None
        self.SessionLocal = None
        self._setup_database()
    
    def _setup_database(self) -> None:
        """Initialize database connection and create tables."""
        db_path = self.config.get_data_directory() / "prism.db"
        db_url = f"sqlite:///{db_path}"
        
        # Create engine with specific SQLite settings
        self.engine = create_engine(
            db_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            }
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        logger.info(f"Database initialized at {db_path}")
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def store_screenshot(self, 
                        file_path: Optional[str] = None,
                        image_data: Optional[bytes] = None,
                        resolution: tuple[int, int] = (0, 0),
                        file_size: int = 0,
                        is_encrypted: bool = True,
                        ocr_text: Optional[str] = None,
                        ocr_confidence: Optional[float] = None) -> Optional[int]:
        """
        Store screenshot information in database.
        
        Returns:
            Screenshot ID if successful, None otherwise
        """
        try:
            with self.get_session() as session:
                screenshot = Screenshot(
                    file_path=file_path,
                    image_data=image_data,
                    resolution_width=resolution[0],
                    resolution_height=resolution[1],
                    file_size_bytes=file_size,
                    is_encrypted=is_encrypted,
                    ocr_text=ocr_text,
                    ocr_confidence=ocr_confidence
                )
                
                session.add(screenshot)
                session.commit()
                
                logger.debug(f"Stored screenshot {screenshot.id}")
                return screenshot.id
                
        except SQLAlchemyError as e:
            logger.error(f"Error storing screenshot: {e}")
            return None
    
    def store_window_info(self,
                         window_title: str,
                         app_name: str,
                         app_bundle_id: Optional[str] = None,
                         window_bounds: Optional[Dict] = None,
                         is_active: bool = False,
                         is_minimized: bool = False,
                         is_fullscreen: bool = False) -> Optional[int]:
        """
        Store window information in database.
        
        Returns:
            Window info ID if successful, None otherwise
        """
        try:
            with self.get_session() as session:
                window_info = WindowInfo(
                    window_title=window_title,
                    app_name=app_name,
                    app_bundle_id=app_bundle_id,
                    window_bounds=window_bounds,
                    is_active=is_active,
                    is_minimized=is_minimized,
                    is_fullscreen=is_fullscreen
                )
                
                session.add(window_info)
                session.commit()
                
                logger.debug(f"Stored window info {window_info.id}")
                return window_info.id
                
        except SQLAlchemyError as e:
            logger.error(f"Error storing window info: {e}")
            return None
    
    def store_activity(self,
                      activity_type: str,
                      confidence: float,
                      duration_seconds: int = 0,
                      screenshot_id: Optional[int] = None,
                      window_info_id: Optional[int] = None,
                      metadata: Optional[Dict] = None,
                      project_name: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> Optional[int]:
        """
        Store activity classification in database.
        
        Returns:
            Activity ID if successful, None otherwise
        """
        try:
            with self.get_session() as session:
                activity = Activity(
                    activity_type=activity_type,
                    confidence=confidence,
                    duration_seconds=duration_seconds,
                    screenshot_id=screenshot_id,
                    window_info_id=window_info_id,
                    activity_metadata=metadata,
                    project_name=project_name,
                    tags=tags
                )
                
                session.add(activity)
                session.commit()
                
                logger.debug(f"Stored activity {activity.id}")
                return activity.id
                
        except SQLAlchemyError as e:
            logger.error(f"Error storing activity: {e}")
            return None
    
    def start_time_session(self,
                          session_type: str = "work",
                          primary_app: Optional[str] = None,
                          project_name: Optional[str] = None) -> Optional[int]:
        """
        Start a new time tracking session.
        
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            with self.get_session() as session:
                time_session = TimeSession(
                    session_type=session_type,
                    primary_app=primary_app,
                    project_name=project_name
                )
                
                session.add(time_session)
                session.commit()
                
                logger.debug(f"Started time session {time_session.id}")
                return time_session.id
                
        except SQLAlchemyError as e:
            logger.error(f"Error starting time session: {e}")
            return None
    
    def end_time_session(self, session_id: int, 
                        focus_score: Optional[float] = None,
                        interruptions_count: int = 0,
                        metadata: Optional[Dict] = None) -> bool:
        """
        End a time tracking session.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                time_session = session.get(TimeSession, session_id)
                if time_session and not time_session.end_time:
                    time_session.end_time = datetime.now()
                    time_session.duration_seconds = int(
                        (time_session.end_time - time_session.start_time).total_seconds()
                    )
                    time_session.focus_score = focus_score
                    time_session.interruptions_count = interruptions_count
                    time_session.session_metadata = metadata
                    
                    session.commit()
                    logger.debug(f"Ended time session {session_id}")
                    return True
                    
        except SQLAlchemyError as e:
            logger.error(f"Error ending time session: {e}")
        
        return False
    
    def get_recent_activities(self, 
                            limit: int = 100,
                            hours_back: int = 24) -> List[Activity]:
        """Get recent activities from the database."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            with self.get_session() as session:
                activities = session.query(Activity)\
                    .outerjoin(Activity.window_info)\
                    .outerjoin(Activity.screenshot)\
                    .filter(Activity.timestamp >= cutoff_time)\
                    .order_by(Activity.timestamp.desc())\
                    .limit(limit)\
                    .all()
                
                # Detach from session to avoid lazy loading issues
                session.expunge_all()
                return activities
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting recent activities: {e}")
            return []
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get daily activity summary."""
        if date is None:
            date = datetime.now()
        
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        try:
            with self.get_session() as session:
                # Get activities for the day
                activities = session.query(Activity)\
                    .filter(Activity.timestamp >= start_of_day)\
                    .filter(Activity.timestamp < end_of_day)\
                    .all()
                
                # Calculate summary statistics
                total_duration = sum(a.duration_seconds for a in activities)
                activity_counts = {}
                
                for activity in activities:
                    activity_type = activity.activity_type
                    if activity_type in activity_counts:
                        activity_counts[activity_type]['count'] += 1
                        activity_counts[activity_type]['duration'] += activity.duration_seconds
                    else:
                        activity_counts[activity_type] = {
                            'count': 1,
                            'duration': activity.duration_seconds
                        }
                
                return {
                    'date': date.date().isoformat(),
                    'total_duration_seconds': total_duration,
                    'total_activities': len(activities),
                    'activity_breakdown': activity_counts,
                    'most_common_activity': max(activity_counts.keys(), 
                                              key=lambda k: activity_counts[k]['duration']) 
                                              if activity_counts else None
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting daily summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Remove old data based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            with self.get_session() as session:
                # Delete old screenshots
                old_screenshots = session.query(Screenshot)\
                    .filter(Screenshot.timestamp < cutoff_date)\
                    .all()
                
                for screenshot in old_screenshots:
                    if screenshot.file_path and os.path.exists(screenshot.file_path):
                        os.remove(screenshot.file_path)
                    session.delete(screenshot)
                
                # Delete old activities
                session.query(Activity)\
                    .filter(Activity.timestamp < cutoff_date)\
                    .delete()
                
                # Delete old window info
                session.query(WindowInfo)\
                    .filter(WindowInfo.timestamp < cutoff_date)\
                    .delete()
                
                # Delete old time sessions
                session.query(TimeSession)\
                    .filter(TimeSession.start_time < cutoff_date)\
                    .delete()
                
                session.commit()
                logger.info(f"Cleaned up data older than {days_to_keep} days")
                
        except SQLAlchemyError as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.get_session() as session:
                stats = {
                    'screenshots_count': session.query(Screenshot).count(),
                    'activities_count': session.query(Activity).count(),
                    'window_info_count': session.query(WindowInfo).count(),
                    'time_sessions_count': session.query(TimeSession).count(),
                    'system_events_count': session.query(SystemEvent).count(),
                }
                
                # Get database file size
                db_path = self.config.get_data_directory() / "prism.db"
                if db_path.exists():
                    stats['database_size_mb'] = db_path.stat().st_size / (1024 * 1024)
                
                return stats
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed") 