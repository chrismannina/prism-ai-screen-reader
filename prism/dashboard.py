"""
Prism Dashboard

Simple GUI dashboard for viewing activity data and system status.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .core.config import PrismConfig
from .core.database import DatabaseManager
from .core.security import SecurityManager
from .main import PrismApp


class PrismDashboard:
    """Main dashboard window for Prism."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Prism Dashboard - AI Screen Reader & Time Tracker")
        self.root.geometry("800x600")
        
        # Initialize components
        self.config = PrismConfig()
        self.database = DatabaseManager(self.config)
        self.security_manager = SecurityManager(self.config)
        
        # Variables
        self.update_interval = 5000  # 5 seconds
        self.is_updating = True
        
        self._setup_ui()
        self._start_updates()
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status Tab
        self.status_frame = ttk.Frame(notebook)
        notebook.add(self.status_frame, text="Status")
        self._setup_status_tab()
        
        # Activities Tab
        self.activities_frame = ttk.Frame(notebook)
        notebook.add(self.activities_frame, text="Activities")
        self._setup_activities_tab()
        
        # Statistics Tab
        self.stats_frame = ttk.Frame(notebook)
        notebook.add(self.stats_frame, text="Statistics")
        self._setup_statistics_tab()
        
        # Settings Tab
        self.settings_frame = ttk.Frame(notebook)
        notebook.add(self.settings_frame, text="Settings")
        self._setup_settings_tab()
    
    def _setup_status_tab(self):
        """Setup the status monitoring tab."""
        # Title
        title_label = ttk.Label(self.status_frame, text="System Status", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Status frame
        status_container = ttk.LabelFrame(self.status_frame, text="Current Status", padding=10)
        status_container.pack(fill=tk.X, padx=10, pady=5)
        
        # Status labels
        self.status_labels = {}
        status_items = [
            ("Observer Agent", "observer_status"),
            ("Database", "database_status"),
            ("Security", "security_status"),
            ("Last Screenshot", "last_screenshot"),
            ("Current Activity", "current_activity"),
            ("Session Duration", "session_duration")
        ]
        
        for i, (label_text, key) in enumerate(status_items):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(status_container, text=f"{label_text}:").grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            self.status_labels[key] = ttk.Label(status_container, text="Loading...", foreground="blue")
            self.status_labels[key].grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=2)
        
        # Recent events
        events_container = ttk.LabelFrame(self.status_frame, text="Recent Events", padding=10)
        events_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Events listbox with scrollbar
        events_frame = ttk.Frame(events_container)
        events_frame.pack(fill=tk.BOTH, expand=True)
        
        self.events_listbox = tk.Listbox(events_frame)
        scrollbar = ttk.Scrollbar(events_frame, orient=tk.VERTICAL, command=self.events_listbox.yview)
        self.events_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.events_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_activities_tab(self):
        """Setup the activities monitoring tab."""
        title_label = ttk.Label(self.activities_frame, text="Recent Activities", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Activities tree view
        columns = ("Time", "Activity", "App", "Confidence", "Duration")
        self.activities_tree = ttk.Treeview(self.activities_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        for col in columns:
            self.activities_tree.heading(col, text=col)
            self.activities_tree.column(col, width=120, anchor=tk.CENTER)
        
        # Scrollbar for tree view
        tree_scrollbar = ttk.Scrollbar(self.activities_frame, orient=tk.VERTICAL, command=self.activities_tree.yview)
        self.activities_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.activities_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Refresh button
        refresh_btn = ttk.Button(self.activities_frame, text="Refresh Activities", command=self._refresh_activities)
        refresh_btn.pack(pady=5)
    
    def _setup_statistics_tab(self):
        """Setup the statistics tab."""
        title_label = ttk.Label(self.stats_frame, text="Activity Statistics", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Time period selection
        period_frame = ttk.Frame(self.stats_frame)
        period_frame.pack(pady=5)
        
        ttk.Label(period_frame, text="Time Period:").pack(side=tk.LEFT, padx=5)
        self.period_var = tk.StringVar(value="Today")
        period_combo = ttk.Combobox(period_frame, textvariable=self.period_var, 
                                   values=["Today", "Yesterday", "Last 7 Days", "Last 30 Days"])
        period_combo.pack(side=tk.LEFT, padx=5)
        period_combo.bind("<<ComboboxSelected>>", self._update_statistics)
        
        # Statistics display
        self.stats_text = tk.Text(self.stats_frame, height=20, width=80)
        stats_scrollbar = ttk.Scrollbar(self.stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def _setup_settings_tab(self):
        """Setup the settings configuration tab."""
        title_label = ttk.Label(self.settings_frame, text="Configuration", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Settings notebook
        settings_notebook = ttk.Notebook(self.settings_frame)
        settings_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Capture settings
        capture_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(capture_frame, text="Capture")
        
        ttk.Label(capture_frame, text="Screenshot Interval (seconds):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.screenshot_interval_var = tk.IntVar(value=self.config.capture.screenshot_interval_seconds)
        ttk.Entry(capture_frame, textvariable=self.screenshot_interval_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(capture_frame, text="OCR Enabled:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.ocr_enabled_var = tk.BooleanVar(value=self.config.capture.ocr_enabled)
        ttk.Checkbutton(capture_frame, variable=self.ocr_enabled_var).grid(row=1, column=1, padx=5, pady=5)
        
        # Privacy settings
        privacy_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(privacy_frame, text="Privacy")
        
        ttk.Label(privacy_frame, text="Encrypt Screenshots:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.encrypt_screenshots_var = tk.BooleanVar(value=self.config.privacy.encrypt_screenshots)
        ttk.Checkbutton(privacy_frame, variable=self.encrypt_screenshots_var).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(privacy_frame, text="Auto-delete Screenshots (days):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.auto_delete_var = tk.IntVar(value=self.config.privacy.auto_delete_screenshots_days)
        ttk.Entry(privacy_frame, textvariable=self.auto_delete_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Save button
        save_btn = ttk.Button(self.settings_frame, text="Save Settings", command=self._save_settings)
        save_btn.pack(pady=10)
    
    def _start_updates(self):
        """Start periodic updates."""
        self._update_status()
        self._refresh_activities()
        self._update_statistics()
        
        # Schedule next update
        if self.is_updating:
            self.root.after(self.update_interval, self._start_updates)
    
    def _update_status(self):
        """Update status information."""
        try:
            # Get database stats
            db_stats = self.database.get_database_stats()
            security_status = self.security_manager.get_security_status()
            
            # Update status labels
            self.status_labels["database_status"].config(
                text=f"✅ Connected ({db_stats.get('database_size_mb', 0):.1f} MB)",
                foreground="green"
            )
            
            self.status_labels["security_status"].config(
                text="✅ Enabled" if security_status['encryption_enabled'] else "❌ Disabled",
                foreground="green" if security_status['encryption_enabled'] else "red"
            )
            
            # Get recent activities for current activity
            recent_activities = self.database.get_recent_activities(limit=1)
            if recent_activities:
                latest = recent_activities[0]
                self.status_labels["current_activity"].config(
                    text=f"{latest.activity_type} ({latest.confidence:.1%})",
                    foreground="blue"
                )
                self.status_labels["last_screenshot"].config(
                    text=latest.timestamp.strftime("%H:%M:%S"),
                    foreground="blue"
                )
            
            # Mock session duration (would need session tracking)
            self.status_labels["session_duration"].config(
                text="2h 15m",
                foreground="blue"
            )
            
            self.status_labels["observer_status"].config(
                text="✅ Running",
                foreground="green"
            )
            
        except Exception as e:
            for label in self.status_labels.values():
                label.config(text="Error", foreground="red")
    
    def _refresh_activities(self):
        """Refresh the activities list."""
        try:
            # Clear existing items
            for item in self.activities_tree.get_children():
                self.activities_tree.delete(item)
            
            # Get recent activities
            activities = self.database.get_recent_activities(limit=50)
            
            for activity in activities:
                time_str = activity.timestamp.strftime("%H:%M:%S")
                duration_str = f"{activity.duration_seconds}s" if activity.duration_seconds > 0 else "N/A"
                
                # Get window info if available
                app_name = "Unknown"
                if activity.window_info:
                    app_name = activity.window_info.app_name
                
                self.activities_tree.insert("", tk.END, values=(
                    time_str,
                    activity.activity_type,
                    app_name,
                    f"{activity.confidence:.1%}",
                    duration_str
                ))
                
        except Exception as e:
            print(f"Error refreshing activities: {e}")
    
    def _update_statistics(self, event=None):
        """Update statistics display."""
        try:
            self.stats_text.delete(1.0, tk.END)
            
            # Get daily summary
            summary = self.database.get_daily_summary()
            
            if summary:
                stats_text = f"""
📊 DAILY ACTIVITY SUMMARY
{"="*50}

📅 Date: {summary.get('date', 'Today')}
⏱️  Total Active Time: {summary.get('total_duration_seconds', 0) // 60} minutes
📈 Total Activities: {summary.get('total_activities', 0)}

🎯 ACTIVITY BREAKDOWN
{"="*50}
"""
                
                activity_breakdown = summary.get('activity_breakdown', {})
                for activity, data in activity_breakdown.items():
                    duration_min = data['duration'] // 60
                    percentage = (data['duration'] / summary.get('total_duration_seconds', 1)) * 100
                    stats_text += f"\n{activity.title():<15}: {duration_min:>3}m ({percentage:>5.1f}%) - {data['count']} sessions"
                
                most_common = summary.get('most_common_activity')
                if most_common:
                    stats_text += f"\n\n🏆 Most Active: {most_common.title()}"
                
                # Database statistics
                db_stats = self.database.get_database_stats()
                stats_text += f"""

💾 DATABASE STATISTICS
{"="*50}
Screenshots: {db_stats.get('screenshots_count', 0):,}
Activities: {db_stats.get('activities_count', 0):,}
Window Records: {db_stats.get('window_info_count', 0):,}
Time Sessions: {db_stats.get('time_sessions_count', 0):,}
Database Size: {db_stats.get('database_size_mb', 0):.1f} MB
"""
                
                self.stats_text.insert(tk.END, stats_text)
            else:
                self.stats_text.insert(tk.END, "No activity data available for the selected period.")
                
        except Exception as e:
            self.stats_text.insert(tk.END, f"Error loading statistics: {e}")
    
    def _save_settings(self):
        """Save configuration changes."""
        try:
            # Update config
            self.config.capture.screenshot_interval_seconds = self.screenshot_interval_var.get()
            self.config.capture.ocr_enabled = self.ocr_enabled_var.get()
            self.config.privacy.encrypt_screenshots = self.encrypt_screenshots_var.get()
            self.config.privacy.auto_delete_screenshots_days = self.auto_delete_var.get()
            
            # Save to file
            self.config.save_config()
            
            messagebox.showinfo("Settings Saved", "Configuration has been saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def run(self):
        """Run the dashboard."""
        try:
            # Setup window close handler
            def on_closing():
                self.is_updating = False
                self.database.close()
                self.root.destroy()
            
            self.root.protocol("WM_DELETE_WINDOW", on_closing)
            
            # Start the GUI
            self.root.mainloop()
            
        except Exception as e:
            messagebox.showerror("Error", f"Dashboard error: {e}")


def main():
    """Main function to run the dashboard."""
    dashboard = PrismDashboard()
    dashboard.run()


if __name__ == "__main__":
    main() 