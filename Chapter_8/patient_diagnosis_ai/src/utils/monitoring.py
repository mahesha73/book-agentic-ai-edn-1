
"""
Monitoring and performance tracking utilities for the Patient Diagnosis AI system.

This module provides decorators and utilities for monitoring system performance,
tracking metrics, and logging operational data.
"""

import logging
import time
import functools
import psutil
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict, deque
import json

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        
        # System metrics
        self.system_metrics = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'disk_usage_percent': 0.0,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0}
        }
        
        # Start system monitoring thread
        if settings.observability.enable_metrics:
            self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Start system monitoring in background thread."""
        def monitor_system():
            while True:
                try:
                    # CPU usage
                    self.system_metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.system_metrics['memory_percent'] = memory.percent
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.system_metrics['disk_usage_percent'] = (disk.used / disk.total) * 100
                    
                    # Network I/O
                    net_io = psutil.net_io_counters()
                    self.system_metrics['network_io'] = {
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv
                    }
                    
                    # Update gauges
                    with self._lock:
                        self.gauges['system.cpu_percent'] = self.system_metrics['cpu_percent']
                        self.gauges['system.memory_percent'] = self.system_metrics['memory_percent']
                        self.gauges['system.disk_usage_percent'] = self.system_metrics['disk_usage_percent']
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        logger.info("System monitoring started")
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            metric_name = self._format_metric_name(name, tags)
            self.counters[metric_name] += value
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            metric_name = self._format_metric_name(name, tags)
            self.gauges[metric_name] = value
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        with self._lock:
            metric_name = self._format_metric_name(name, tags)
            self.histograms[metric_name].append({
                'value': value,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def record_timing(self, name: str, duration_seconds: float, tags: Optional[Dict[str, str]] = None):
        """Record timing information."""
        self.record_histogram(f"{name}.duration", duration_seconds, tags)
        self.increment_counter(f"{name}.count", 1, tags)
    
    def _format_metric_name(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Format metric name with tags."""
        if not tags:
            return name
        
        tag_string = ",".join([f"{k}={v}" for k, v in sorted(tags.items())])
        return f"{name}[{tag_string}]"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {k: list(v) for k, v in self.histograms.items()},
                'system_metrics': self.system_metrics.copy(),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_summary_stats(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a histogram metric."""
        with self._lock:
            values = [item['value'] for item in self.histograms.get(metric_name, [])]
            
            if not values:
                return {}
            
            values.sort()
            n = len(values)
            
            return {
                'count': n,
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / n,
                'median': values[n // 2],
                'p95': values[int(n * 0.95)] if n > 0 else 0,
                'p99': values[int(n * 0.99)] if n > 0 else 0
            }
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
        
        logger.info("Metrics reset")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def track_performance(func: Callable) -> Callable:
    """
    Decorator to track function performance metrics.
    
    This decorator tracks:
    - Execution time
    - Call count
    - Success/failure rates
    - Resource usage
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Track function call
        performance_monitor.increment_counter(f"function.calls", 1, {'function': func_name})
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Track success
            performance_monitor.increment_counter(f"function.success", 1, {'function': func_name})
            
            return result
            
        except Exception as e:
            # Track failure
            performance_monitor.increment_counter(f"function.errors", 1, {
                'function': func_name,
                'error_type': type(e).__name__
            })
            raise
            
        finally:
            # Track execution time
            execution_time = time.time() - start_time
            performance_monitor.record_timing(f"function.execution_time", execution_time, {'function': func_name})
    
    return wrapper


def log_agent_action(func: Callable) -> Callable:
    """
    Decorator to log agent actions for monitoring and debugging.
    
    This decorator logs:
    - Agent execution start/end
    - Input/output data (sanitized)
    - Performance metrics
    - Error information
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        
        # Extract agent information
        agent_name = "unknown"
        agent_type = "unknown"
        
        if args and hasattr(args[0], 'name'):
            agent_name = args[0].name
        if args and hasattr(args[0], 'agent_type'):
            agent_type = args[0].agent_type
        
        # Log action start
        logger.info(
            f"Agent action started: {func_name}",
            extra={
                'agent_name': agent_name,
                'agent_type': agent_type,
                'action': func_name,
                'start_time': datetime.utcnow().isoformat(),
                'monitoring': True
            }
        )
        
        # Track agent metrics
        performance_monitor.increment_counter("agent.actions", 1, {
            'agent_type': agent_type,
            'action': func_name
        })
        
        try:
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Log successful completion
            logger.info(
                f"Agent action completed: {func_name}",
                extra={
                    'agent_name': agent_name,
                    'agent_type': agent_type,
                    'action': func_name,
                    'execution_time_seconds': execution_time,
                    'status': 'success',
                    'monitoring': True
                }
            )
            
            # Track success metrics
            performance_monitor.increment_counter("agent.success", 1, {
                'agent_type': agent_type,
                'action': func_name
            })
            performance_monitor.record_timing("agent.execution_time", execution_time, {
                'agent_type': agent_type,
                'action': func_name
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Agent action failed: {func_name}",
                extra={
                    'agent_name': agent_name,
                    'agent_type': agent_type,
                    'action': func_name,
                    'execution_time_seconds': execution_time,
                    'status': 'error',
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'monitoring': True
                },
                exc_info=True
            )
            
            # Track error metrics
            performance_monitor.increment_counter("agent.errors", 1, {
                'agent_type': agent_type,
                'action': func_name,
                'error_type': type(e).__name__
            })
            
            raise
    
    return wrapper


def monitor_resource_usage(func: Callable) -> Callable:
    """
    Decorator to monitor resource usage during function execution.
    
    This decorator tracks:
    - Memory usage before/after
    - CPU usage during execution
    - Execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Get final resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            execution_time = time.time() - start_time
            memory_delta = final_memory - initial_memory
            
            # Log resource usage
            logger.info(
                f"Resource usage for {func.__name__}",
                extra={
                    'function': func.__name__,
                    'execution_time_seconds': execution_time,
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_delta_mb': memory_delta,
                    'monitoring': True
                }
            )
            
            # Track metrics
            performance_monitor.record_histogram("resource.memory_usage", final_memory, {'function': func.__name__})
            performance_monitor.record_histogram("resource.memory_delta", memory_delta, {'function': func.__name__})
            
            return result
            
        except Exception as e:
            logger.error(f"Resource monitoring failed for {func.__name__}: {e}")
            raise
    
    return wrapper


class HealthChecker:
    """System health checking utilities."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks = {}
        self.last_check_time = None
        self.check_interval = settings.observability.health_check_interval
    
    def register_check(self, name: str, check_func: Callable[[], bool], description: str = ""):
        """Register a health check function."""
        self.checks[name] = {
            'function': check_func,
            'description': description,
            'last_result': None,
            'last_check': None
        }
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        for name, check_info in self.checks.items():
            try:
                start_time = time.time()
                result = check_info['function']()
                check_time = time.time() - start_time
                
                check_info['last_result'] = result
                check_info['last_check'] = datetime.utcnow()
                
                results['checks'][name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'description': check_info['description'],
                    'response_time_seconds': check_time,
                    'last_check': check_info['last_check'].isoformat()
                }
                
                if not result:
                    results['overall_status'] = 'unhealthy'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'description': check_info['description'],
                    'error': str(e),
                    'last_check': datetime.utcnow().isoformat()
                }
                results['overall_status'] = 'unhealthy'
        
        self.last_check_time = datetime.utcnow()
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'python_version': f"{psutil.version_info}",
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }


# Global health checker instance
health_checker = HealthChecker()


def get_monitoring_status() -> Dict[str, Any]:
    """
    Get current monitoring status and metrics.
    
    Returns:
        Dict[str, Any]: Monitoring status information
    """
    return {
        'performance_monitoring_enabled': settings.observability.enable_metrics,
        'metrics_collected': len(performance_monitor.counters) + len(performance_monitor.gauges) + len(performance_monitor.histograms),
        'system_metrics': performance_monitor.system_metrics,
        'health_checks_registered': len(health_checker.checks),
        'last_health_check': health_checker.last_check_time.isoformat() if health_checker.last_check_time else None,
        'monitoring_config': {
            'metrics_port': settings.observability.metrics_port,
            'health_check_interval': settings.observability.health_check_interval
        }
    }


# Register default health checks
def _check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits."""
    memory_percent = psutil.virtual_memory().percent
    return memory_percent < 90  # Alert if memory usage > 90%


def _check_disk_space() -> bool:
    """Check if disk space is sufficient."""
    disk_usage = psutil.disk_usage('/').percent
    return disk_usage < 85  # Alert if disk usage > 85%


def _check_cpu_usage() -> bool:
    """Check if CPU usage is within acceptable limits."""
    cpu_percent = psutil.cpu_percent(interval=1)
    return cpu_percent < 80  # Alert if CPU usage > 80%


# Register default health checks
health_checker.register_check('memory_usage', _check_memory_usage, 'Memory usage check')
health_checker.register_check('disk_space', _check_disk_space, 'Disk space check')
health_checker.register_check('cpu_usage', _check_cpu_usage, 'CPU usage check')
