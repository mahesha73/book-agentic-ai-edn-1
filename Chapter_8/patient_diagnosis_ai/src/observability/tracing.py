
"""
Distributed tracing utilities for the Patient Diagnosis AI system.
"""

import logging
from typing import Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MockSpan:
    """Mock span for tracing when no tracer is available."""
    
    def __init__(self, name: str):
        self.name = name
        self.attributes = {}
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: dict = None):
        """Add event to span."""
        pass
    
    def get_span_context(self):
        """Get span context."""
        return MockSpanContext()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockSpanContext:
    """Mock span context."""
    
    def __init__(self):
        self.trace_id = 12345678901234567890123456789012
        self.span_id = 1234567890123456


class MockTracer:
    """Mock tracer for when no real tracer is available."""
    
    @contextmanager
    def start_as_current_span(self, name: str):
        """Start a new span."""
        span = MockSpan(name)
        yield span


def get_tracer(name: str) -> MockTracer:
    """Get tracer instance."""
    return MockTracer()
