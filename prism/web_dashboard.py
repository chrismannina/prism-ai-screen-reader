"""
Prism Web Dashboard

Modern web-based dashboard for viewing and managing Prism data.
"""

import io
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO
from PIL import Image

from .core.config import PrismConfig
from .core.database import DatabaseManager, Screenshot, Activity, WindowInfo
from .core.security import SecurityManager


class PrismWebDashboard:
    """Flask-based web dashboard for Prism."""
    
    def __init__(self, config: Optional[PrismConfig] = None):
        self.config = config or PrismConfig()
        self.database = DatabaseManager(self.config)
        self.security_manager = SecurityManager(self.config)
        
        # Initialize Flask app
        self.app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
        self.app.secret_key = 'prism-dashboard-secret'  # Should be configurable
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self._setup_routes()
        self._setup_socketio()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            """Get system status."""
            try:
                db_stats = self.database.get_database_stats()
                security_status = self.security_manager.get_security_status()
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'database': db_stats,
                        'security': security_status,
                        'timestamp': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/screenshots')
        def api_screenshots():
            """Get screenshots list."""
            try:
                limit = request.args.get('limit', 20, type=int)
                
                with self.database.get_session() as session:
                    screenshots = session.query(Screenshot)\
                        .order_by(Screenshot.timestamp.desc())\
                        .limit(limit)\
                        .all()
                    
                    screenshots_data = []
                    for screenshot in screenshots:
                        screenshots_data.append({
                            'id': screenshot.id,
                            'timestamp': screenshot.timestamp.isoformat(),
                            'resolution': f"{screenshot.resolution_width}x{screenshot.resolution_height}",
                            'size_kb': round(screenshot.file_size_bytes / 1024, 1),
                            'is_encrypted': screenshot.is_encrypted,
                            'has_ocr': bool(screenshot.ocr_text),
                            'ocr_preview': screenshot.ocr_text[:100] + "..." if screenshot.ocr_text and len(screenshot.ocr_text) > 100 else screenshot.ocr_text
                        })
                
                return jsonify({
                    'status': 'success',
                    'data': screenshots_data
                })
                
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/screenshot/<int:screenshot_id>')
        def api_screenshot_image(screenshot_id):
            """Get screenshot image."""
            try:
                with self.database.get_session() as session:
                    screenshot = session.get(Screenshot, screenshot_id)
                    
                    if not screenshot or not screenshot.image_data:
                        return jsonify({'status': 'error', 'message': 'Screenshot not found'}), 404
                    
                    # Decrypt image data
                    image_data = screenshot.image_data
                    if screenshot.is_encrypted:
                        image_data = self.security_manager.decrypt_data(image_data)
                        if image_data is None:
                            return jsonify({'status': 'error', 'message': 'Failed to decrypt'}), 500
                    
                    # Convert to base64 for web display
                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                    
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'image': f"data:image/png;base64,{image_b64}",
                            'timestamp': screenshot.timestamp.isoformat(),
                            'resolution': f"{screenshot.resolution_width}x{screenshot.resolution_height}"
                        }
                    })
                    
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/activities')
        def api_activities():
            """Get activities list."""
            try:
                limit = request.args.get('limit', 50, type=int)
                hours = request.args.get('hours', 24, type=int)
                
                activities = self.database.get_recent_activities(limit=limit, hours_back=hours)
                
                activities_data = []
                for activity in activities:
                    activities_data.append({
                        'id': activity.id,
                        'timestamp': activity.timestamp.isoformat(),
                        'activity_type': activity.activity_type,
                        'confidence': activity.confidence,
                        'duration_seconds': activity.duration_seconds,
                        'screenshot_id': activity.screenshot_id,
                        'window_info_id': activity.window_info_id
                    })
                
                return jsonify({
                    'status': 'success',
                    'data': activities_data
                })
                
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/search')
        def api_search():
            """Search OCR text."""
            try:
                query = request.args.get('q', '').strip()
                limit = request.args.get('limit', 10, type=int)
                
                if not query:
                    return jsonify({'status': 'error', 'message': 'No search query provided'}), 400
                
                with self.database.get_session() as session:
                    screenshots = session.query(Screenshot)\
                        .filter(Screenshot.ocr_text.contains(query))\
                        .order_by(Screenshot.timestamp.desc())\
                        .limit(limit)\
                        .all()
                    
                    results = []
                    for screenshot in screenshots:
                        # Find context around the search term
                        text = screenshot.ocr_text or ""
                        term_pos = text.lower().find(query.lower())
                        if term_pos >= 0:
                            start = max(0, term_pos - 50)
                            end = min(len(text), term_pos + len(query) + 50)
                            context = text[start:end]
                            
                            results.append({
                                'id': screenshot.id,
                                'timestamp': screenshot.timestamp.isoformat(),
                                'context': context,
                                'resolution': f"{screenshot.resolution_width}x{screenshot.resolution_height}"
                            })
                
                return jsonify({
                    'status': 'success',
                    'data': results,
                    'query': query
                })
                
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/summary')
        def api_summary():
            """Get daily summary."""
            try:
                summary = self.database.get_daily_summary()
                return jsonify({
                    'status': 'success',
                    'data': summary
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _setup_socketio(self):
        """Setup WebSocket events for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print("Client connected to dashboard")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print("Client disconnected from dashboard")
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the web dashboard."""
        print(f"🌐 Starting Prism Web Dashboard...")
        print(f"📱 Open your browser to: http://{host}:{port}")
        print(f"🔄 Debug mode: {'enabled' if debug else 'disabled'}")
        
        self.socketio.run(self.app, host=host, port=port, debug=debug)
    
    def close(self):
        """Clean up resources."""
        self.database.close()


def create_app(config: Optional[PrismConfig] = None) -> Flask:
    """Create Flask app factory."""
    dashboard = PrismWebDashboard(config)
    return dashboard.app


if __name__ == "__main__":
    dashboard = PrismWebDashboard()
    dashboard.run() 