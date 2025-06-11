"""
API Usage Monitoring

Tracks API usage, costs, and provides alerts for quota management.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from .config import PrismConfig


@dataclass
class APIUsage:
    """Data class for API usage tracking."""
    timestamp: datetime
    provider: str  # 'openai', 'anthropic', etc.
    model: str  # 'gpt-4o-mini', 'gpt-4-vision-preview', etc.
    operation: str  # 'vision_classification', 'text_completion', etc.
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    response_time_ms: int
    success: bool
    error_message: Optional[str] = None


class APIMonitor:
    """Monitors and tracks API usage across different providers."""
    
    def __init__(self, config: PrismConfig):
        self.config = config
        self.db_path = config.get_data_directory() / "api_usage.db"
        self._setup_database()
        
        # Pricing information (USD per 1K tokens)
        self.pricing = {
            'openai': {
                'gpt-4o-mini': {
                    'input': 0.00015,   # $0.15 per 1M tokens
                    'output': 0.0006,   # $0.60 per 1M tokens
                },
                'gpt-4-vision-preview': {
                    'input': 0.01,      # $10 per 1M tokens
                    'output': 0.03,     # $30 per 1M tokens
                },
                'gpt-4': {
                    'input': 0.03,      # $30 per 1M tokens
                    'output': 0.06,     # $60 per 1M tokens
                }
            }
        }
        
        # Usage limits and alerts
        self.daily_limit_usd = getattr(config.ml, 'api_daily_limit_usd', 5.0)
        self.monthly_limit_usd = getattr(config.ml, 'api_monthly_limit_usd', 50.0)
        self.alert_threshold = getattr(config.ml, 'api_alert_threshold', 0.8)
        
        logger.info("API Monitor initialized")
    
    def _setup_database(self) -> None:
        """Setup the API usage tracking database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    response_time_ms INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON api_usage(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_provider_model ON api_usage(provider, model)
            """)
            
        logger.debug(f"API usage database initialized at {self.db_path}")
    
    def calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for an API call."""
        if provider not in self.pricing or model not in self.pricing[provider]:
            logger.warning(f"No pricing info for {provider}/{model}, using default rates")
            # Default rates for unknown models
            input_rate = 0.001
            output_rate = 0.002
        else:
            pricing_info = self.pricing[provider][model]
            input_rate = pricing_info['input']
            output_rate = pricing_info['output']
        
        # Convert to cost per token (from per 1K tokens)
        input_cost = (input_tokens / 1000) * input_rate
        output_cost = (output_tokens / 1000) * output_rate
        
        return input_cost + output_cost
    
    def log_usage(self, 
                  provider: str,
                  model: str,
                  operation: str,
                  input_tokens: int,
                  output_tokens: int,
                  response_time_ms: int,
                  success: bool = True,
                  error_message: Optional[str] = None) -> None:
        """Log an API usage event."""
        
        total_tokens = input_tokens + output_tokens
        cost_usd = self.calculate_cost(provider, model, input_tokens, output_tokens)
        
        usage = APIUsage(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            response_time_ms=response_time_ms,
            success=success,
            error_message=error_message
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_usage 
                (timestamp, provider, model, operation, input_tokens, output_tokens, 
                 total_tokens, cost_usd, response_time_ms, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                usage.timestamp.isoformat(),
                usage.provider,
                usage.model,
                usage.operation,
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
                usage.cost_usd,
                usage.response_time_ms,
                usage.success,
                usage.error_message
            ))
        
        # Check for usage alerts
        self._check_usage_alerts(usage)
        
        logger.info(f"API Usage: {provider}/{model} - {total_tokens} tokens - ${cost_usd:.4f}")
    
    def _check_usage_alerts(self, usage: APIUsage) -> None:
        """Check if usage exceeds alert thresholds."""
        # Check daily usage
        daily_usage = self.get_usage_summary(hours_back=24)
        daily_cost = daily_usage['total_cost_usd']
        
        if daily_cost >= self.daily_limit_usd * self.alert_threshold:
            logger.warning(f"Daily API usage alert: ${daily_cost:.2f} / ${self.daily_limit_usd:.2f}")
            
        if daily_cost >= self.daily_limit_usd:
            logger.error(f"Daily API limit exceeded: ${daily_cost:.2f} / ${self.daily_limit_usd:.2f}")
        
        # Check monthly usage
        monthly_usage = self.get_usage_summary(days_back=30)
        monthly_cost = monthly_usage['total_cost_usd']
        
        if monthly_cost >= self.monthly_limit_usd * self.alert_threshold:
            logger.warning(f"Monthly API usage alert: ${monthly_cost:.2f} / ${self.monthly_limit_usd:.2f}")
            
        if monthly_cost >= self.monthly_limit_usd:
            logger.error(f"Monthly API limit exceeded: ${monthly_cost:.2f} / ${self.monthly_limit_usd:.2f}")
    
    def get_usage_summary(self, 
                         hours_back: Optional[int] = None,
                         days_back: Optional[int] = None) -> Dict[str, Any]:
        """Get usage summary for a time period."""
        
        if hours_back:
            start_time = datetime.now() - timedelta(hours=hours_back)
        elif days_back:
            start_time = datetime.now() - timedelta(days=days_back)
        else:
            start_time = datetime.now() - timedelta(hours=24)  # Default to 24 hours
        
        with sqlite3.connect(self.db_path) as conn:
            # Get total usage
            result = conn.execute("""
                SELECT 
                    COUNT(*) as total_calls,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost_usd) as total_cost_usd,
                    AVG(response_time_ms) as avg_response_time_ms,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls
                FROM api_usage 
                WHERE timestamp >= ?
            """, (start_time.isoformat(),)).fetchone()
            
            summary = {
                'total_calls': result[0] or 0,
                'total_input_tokens': result[1] or 0,
                'total_output_tokens': result[2] or 0,
                'total_tokens': result[3] or 0,
                'total_cost_usd': result[4] or 0.0,
                'avg_response_time_ms': result[5] or 0,
                'successful_calls': result[6] or 0,
                'failed_calls': result[7] or 0,
                'success_rate': (result[6] or 0) / max(result[0] or 1, 1)
            }
            
            # Get breakdown by model
            model_results = conn.execute("""
                SELECT 
                    provider,
                    model,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    SUM(cost_usd) as cost_usd
                FROM api_usage 
                WHERE timestamp >= ?
                GROUP BY provider, model
                ORDER BY cost_usd DESC
            """, (start_time.isoformat(),)).fetchall()
            
            summary['by_model'] = [
                {
                    'provider': row[0],
                    'model': row[1],
                    'calls': row[2],
                    'tokens': row[3],
                    'cost_usd': row[4]
                }
                for row in model_results
            ]
            
        return summary
    
    def get_recent_usage(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent API usage records."""
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT 
                    timestamp, provider, model, operation, 
                    input_tokens, output_tokens, total_tokens,
                    cost_usd, response_time_ms, success, error_message
                FROM api_usage 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [
                {
                    'timestamp': row[0],
                    'provider': row[1],
                    'model': row[2],
                    'operation': row[3],
                    'input_tokens': row[4],
                    'output_tokens': row[5],
                    'total_tokens': row[6],
                    'cost_usd': row[7],
                    'response_time_ms': row[8],
                    'success': bool(row[9]),
                    'error_message': row[10]
                }
                for row in results
            ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current API monitoring status."""
        daily_summary = self.get_usage_summary(hours_back=24)
        monthly_summary = self.get_usage_summary(days_back=30)
        
        return {
            'daily_usage': {
                'cost_usd': daily_summary['total_cost_usd'],
                'limit_usd': self.daily_limit_usd,
                'percentage': (daily_summary['total_cost_usd'] / self.daily_limit_usd) * 100,
                'calls': daily_summary['total_calls'],
                'tokens': daily_summary['total_tokens']
            },
            'monthly_usage': {
                'cost_usd': monthly_summary['total_cost_usd'],
                'limit_usd': self.monthly_limit_usd,
                'percentage': (monthly_summary['total_cost_usd'] / self.monthly_limit_usd) * 100,
                'calls': monthly_summary['total_calls'],
                'tokens': monthly_summary['total_tokens']
            },
            'alert_threshold': self.alert_threshold,
            'success_rate_24h': daily_summary['success_rate']
        }
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> None:
        """Clean up old API usage data."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                DELETE FROM api_usage 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_count = result.rowcount
            
        logger.info(f"Cleaned up {deleted_count} old API usage records") 