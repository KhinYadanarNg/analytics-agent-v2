"""
Simplified analytics query processing service where LLM handles date extraction.
Main entry point that imports from organized modules.
"""
import logging
from app.core.analytics_service import AnalyticsService

# Setup logger
logger = logging.getLogger(__name__)

# Re-export the main service for backward compatibility
__all__ = ['AnalyticsService']