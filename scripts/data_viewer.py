#!/usr/bin/env python3
"""
Simple Prism Data Viewer

Browse and explore captured data directly from the database.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import prism modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.core.config import PrismConfig
from prism.core.database import DatabaseManager, Screenshot, Activity, WindowInfo
from datetime import datetime, timedelta


def main():
    """Interactive data viewer."""
    config = PrismConfig()
    database = DatabaseManager(config)
    
    print("🔍 Prism Data Viewer")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Show recent screenshots")
        print("2. Show recent activities") 
        print("3. Show window tracking data")
        print("4. Search OCR text")
        print("5. Database statistics")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            show_screenshots(database)
        elif choice == '2':
            show_activities(database)
        elif choice == '3':
            show_windows(database)
        elif choice == '4':
            search_ocr(database)
        elif choice == '5':
            show_stats(database)
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")
    
    database.close()
    print("👋 Goodbye!")


def show_screenshots(database):
    """Show recent screenshots."""
    print("\n📸 Recent Screenshots:")
    print("-" * 50)
    
    try:
        with database.get_session() as session:
            screenshots = session.query(Screenshot).order_by(Screenshot.timestamp.desc()).limit(10).all()
            
            if not screenshots:
                print("No screenshots found.")
                return
            
            for i, screenshot in enumerate(screenshots, 1):
                timestamp = screenshot.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                resolution = f"{screenshot.resolution_width}x{screenshot.resolution_height}"
                size_kb = screenshot.file_size_bytes / 1024
                encrypted = "🔒" if screenshot.is_encrypted else "🔓"
                
                print(f"{i:2d}. {timestamp} | {resolution} | {size_kb:6.1f} KB | {encrypted}")
                
                if screenshot.ocr_text:
                    preview = screenshot.ocr_text[:80] + "..." if len(screenshot.ocr_text) > 80 else screenshot.ocr_text
                    print(f"    OCR: {preview}")
                
                print()
                
    except Exception as e:
        print(f"Error retrieving screenshots: {e}")


def show_activities(database):
    """Show recent activities."""
    print("\n🎯 Recent Activities:")
    print("-" * 50)
    
    try:
        activities = database.get_recent_activities(limit=10)
        
        if not activities:
            print("No activities found.")
            return
        
        for i, activity in enumerate(activities, 1):
            timestamp = activity.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            confidence_icon = "🟢" if activity.confidence > 0.7 else "🟡" if activity.confidence > 0.5 else "🔴"
            
            print(f"{i:2d}. {timestamp} | {confidence_icon} {activity.activity_type:<12} | {activity.confidence:.1%}")
            
            # Show metadata if available
            if activity.activity_metadata:
                import json
                try:
                    metadata = json.loads(activity.activity_metadata) if isinstance(activity.activity_metadata, str) else activity.activity_metadata
                    if 'all_scores' in metadata:
                        scores = metadata['all_scores']
                        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        score_str = " | ".join([f"{k}: {v:.2f}" for k, v in top_3])
                        print(f"    Classification scores: {score_str}")
                except:
                    pass
            
            print()
                
    except Exception as e:
        print(f"Error retrieving activities: {e}")


def show_windows(database):
    """Show recent window information."""
    print("\n🪟 Recent Window Tracking:")
    print("-" * 50)
    
    try:
        with database.get_session() as session:
            windows = session.query(WindowInfo).order_by(WindowInfo.timestamp.desc()).limit(15).all()
            
            if not windows:
                print("No window tracking data found.")
                return
            
            for i, window in enumerate(windows, 1):
                timestamp = window.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                active_icon = "🟢" if window.is_active else "⚪"
                
                print(f"{i:2d}. {timestamp} | {active_icon} {window.app_name}")
                if window.window_title != window.app_name:
                    print(f"    Window: {window.window_title}")
                if window.app_bundle_id:
                    print(f"    Bundle: {window.app_bundle_id}")
                print()
                
    except Exception as e:
        print(f"Error retrieving window data: {e}")


def search_ocr(database):
    """Search OCR text content."""
    search_term = input("\n🔍 Enter search term: ").strip()
    
    if not search_term:
        print("No search term provided.")
        return
    
    print(f"\nSearching for '{search_term}' in OCR text...")
    print("-" * 50)
    
    try:
        with database.get_session() as session:
            screenshots = session.query(Screenshot).filter(
                Screenshot.ocr_text.contains(search_term)
            ).order_by(Screenshot.timestamp.desc()).limit(10).all()
            
            if not screenshots:
                print("No matches found.")
                return
            
            for i, screenshot in enumerate(screenshots, 1):
                timestamp = screenshot.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                # Find the search term in context
                text = screenshot.ocr_text
                term_pos = text.lower().find(search_term.lower())
                if term_pos >= 0:
                    start = max(0, term_pos - 40)
                    end = min(len(text), term_pos + len(search_term) + 40)
                    context = text[start:end]
                    
                    print(f"{i:2d}. {timestamp}")
                    print(f"    Context: ...{context}...")
                    print()
                
    except Exception as e:
        print(f"Error searching OCR text: {e}")


def show_stats(database):
    """Show database statistics."""
    print("\n📊 Database Statistics:")
    print("-" * 30)
    
    try:
        stats = database.get_database_stats()
        
        print(f"Screenshots: {stats.get('screenshots_count', 0):,}")
        print(f"Activities: {stats.get('activities_count', 0):,}")
        print(f"Window Records: {stats.get('window_info_count', 0):,}")
        print(f"Time Sessions: {stats.get('time_sessions_count', 0):,}")
        print(f"System Events: {stats.get('system_events_count', 0):,}")
        print(f"Database Size: {stats.get('database_size_mb', 0):.1f} MB")
        
        # Recent activity summary
        summary = database.get_daily_summary()
        if summary:
            print(f"\n📅 Today's Activity:")
            print(f"Total Active Time: {summary.get('total_duration_seconds', 0) // 60} minutes")
            print(f"Total Activities: {summary.get('total_activities', 0)}")
            
            if summary.get('activity_breakdown'):
                print("\nActivity Types:")
                for activity_type, data in summary['activity_breakdown'].items():
                    print(f"  {activity_type.title()}: {data['count']} sessions")
        
    except Exception as e:
        print(f"Error retrieving statistics: {e}")


if __name__ == "__main__":
    main() 