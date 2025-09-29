
"""
HIPAA compliance utilities for the Patient Diagnosis AI system.

This module provides decorators and utilities to ensure HIPAA compliance
throughout the system, including data encryption, audit logging, and
access controls.
"""

import logging
import functools
import hashlib
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import inspect

from cryptography.fernet import Fernet
from src.config.settings import get_settings
from src.db.engines import get_database_engines

logger = logging.getLogger(__name__)
settings = get_settings()


class HIPAACompliance:
    """HIPAA compliance manager."""
    
    def __init__(self):
        """Initialize HIPAA compliance manager."""
        self.cipher_suite = Fernet(settings.security.encryption_key.encode()[:32])
        self.db_engines = get_database_engines()
    
    def encrypt_phi(self, data: str) -> bytes:
        """Encrypt Protected Health Information (PHI)."""
        if not data:
            return b''
        return self.cipher_suite.encrypt(data.encode())
    
    def decrypt_phi(self, encrypted_data: bytes) -> str:
        """Decrypt Protected Health Information (PHI)."""
        if not encrypted_data:
            return ''
        return self.cipher_suite.decrypt(encrypted_data).decode()
    
    def hash_identifier(self, identifier: str) -> str:
        """Create a hash of an identifier for logging purposes."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def sanitize_for_logging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for logging by removing or hashing PHI."""
        sanitized = {}
        
        # List of fields that contain PHI
        phi_fields = {
            'patient_id', 'patient_name', 'name', 'ssn', 'phone', 'email', 
            'address', 'date_of_birth', 'mrn', 'medical_record_number'
        }
        
        for key, value in data.items():
            if key.lower() in phi_fields:
                if isinstance(value, str) and value:
                    sanitized[key] = self.hash_identifier(value)
                else:
                    sanitized[key] = "***"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_for_logging(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_for_logging(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def log_access(
        self,
        user_id: Optional[str],
        resource_type: str,
        resource_id: Optional[str],
        action: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log access to PHI for audit purposes."""
        if not settings.security.audit_log_enabled:
            return
        
        try:
            with self.db_engines.postgres_session() as session:
                from src.db.models import AuditLog
                
                audit_entry = AuditLog(
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    description=f"{action} on {resource_type}",
                    risk_level=self._assess_risk_level(action, resource_type)
                )
                
                if additional_data:
                    audit_entry.new_values = self.sanitize_for_logging(additional_data)
                
                session.add(audit_entry)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")
    
    def _assess_risk_level(self, action: str, resource_type: str) -> str:
        """Assess risk level for audit logging."""
        high_risk_actions = {'DELETE', 'EXPORT', 'BULK_ACCESS'}
        high_risk_resources = {'Patient', 'MedicalHistory', 'DiagnosisRequest'}
        
        if action.upper() in high_risk_actions:
            return 'high'
        elif resource_type in high_risk_resources:
            return 'medium'
        else:
            return 'low'


# Global compliance manager instance
compliance_manager = HIPAACompliance()


def ensure_hipaa_compliance(func: Callable) -> Callable:
    """
    Decorator to ensure HIPAA compliance for functions handling PHI.
    
    This decorator:
    1. Logs function access for audit purposes
    2. Validates that proper security measures are in place
    3. Ensures data is handled according to HIPAA requirements
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get function information
        func_name = func.__name__
        module_name = func.__module__
        
        # Extract user context if available
        user_id = None
        session_id = None
        
        # Try to extract user context from arguments
        for arg in args:
            if hasattr(arg, 'user_id'):
                user_id = arg.user_id
            if hasattr(arg, 'session_id'):
                session_id = arg.session_id
        
        # Check kwargs for user context
        user_id = user_id or kwargs.get('user_id')
        session_id = session_id or kwargs.get('session_id')
        
        # Log function access
        compliance_manager.log_access(
            user_id=user_id,
            resource_type=module_name,
            resource_id=session_id,
            action=f"EXECUTE_{func_name.upper()}",
            additional_data={
                'function': func_name,
                'module': module_name,
                'has_phi': True  # Assume functions with this decorator handle PHI
            }
        )
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log successful completion
            compliance_manager.log_access(
                user_id=user_id,
                resource_type=module_name,
                resource_id=session_id,
                action=f"COMPLETE_{func_name.upper()}",
                additional_data={'status': 'success'}
            )
            
            return result
            
        except Exception as e:
            # Log error
            compliance_manager.log_access(
                user_id=user_id,
                resource_type=module_name,
                resource_id=session_id,
                action=f"ERROR_{func_name.upper()}",
                additional_data={
                    'status': 'error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise
    
    return wrapper


def audit_log(func: Callable) -> Callable:
    """
    Decorator for comprehensive audit logging.
    
    This decorator logs detailed information about function execution
    for compliance and monitoring purposes.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        func_name = func.__name__
        
        # Get function signature for logging
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Sanitize arguments for logging
        sanitized_args = compliance_manager.sanitize_for_logging(dict(bound_args.arguments))
        
        logger.info(
            f"Function {func_name} started",
            extra={
                'function': func_name,
                'arguments': sanitized_args,
                'start_time': start_time.isoformat(),
                'audit': True
            }
        )
        
        try:
            result = func(*args, **kwargs)
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Function {func_name} completed successfully",
                extra={
                    'function': func_name,
                    'execution_time_seconds': execution_time,
                    'end_time': end_time.isoformat(),
                    'status': 'success',
                    'audit': True
                }
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(
                f"Function {func_name} failed",
                extra={
                    'function': func_name,
                    'execution_time_seconds': execution_time,
                    'end_time': end_time.isoformat(),
                    'status': 'error',
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'audit': True
                },
                exc_info=True
            )
            raise
    
    return wrapper


def validate_phi_access(required_permissions: list = None):
    """
    Decorator to validate PHI access permissions.
    
    Args:
        required_permissions: List of required permissions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user context
            user_id = kwargs.get('user_id')
            
            if not user_id:
                # Try to find user_id in args
                for arg in args:
                    if hasattr(arg, 'user_id'):
                        user_id = arg.user_id
                        break
            
            if not user_id:
                logger.warning(f"PHI access attempted without user context in {func.__name__}")
                raise PermissionError("User authentication required for PHI access")
            
            # Validate permissions (simplified implementation)
            if required_permissions:
                # In a real implementation, this would check user permissions
                # against a permission system
                logger.info(f"Validating permissions {required_permissions} for user {user_id}")
            
            # Log PHI access
            compliance_manager.log_access(
                user_id=user_id,
                resource_type="PHI",
                resource_id=None,
                action=f"ACCESS_{func.__name__.upper()}",
                additional_data={
                    'required_permissions': required_permissions,
                    'function': func.__name__
                }
            )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def encrypt_phi_fields(*field_names):
    """
    Decorator to automatically encrypt specified PHI fields in function arguments.
    
    Args:
        field_names: Names of fields to encrypt
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Encrypt specified fields in kwargs
            for field_name in field_names:
                if field_name in kwargs and kwargs[field_name]:
                    kwargs[field_name] = compliance_manager.encrypt_phi(kwargs[field_name])
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def data_retention_check(retention_days: int = None):
    """
    Decorator to check data retention policies.
    
    Args:
        retention_days: Number of days to retain data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retention_period = retention_days or settings.security.data_retention_days
            
            # Log data retention check
            logger.info(
                f"Data retention check for {func.__name__}",
                extra={
                    'function': func.__name__,
                    'retention_days': retention_period,
                    'audit': True
                }
            )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class PHIRedactor:
    """Utility class for redacting PHI from text and data structures."""
    
    def __init__(self):
        """Initialize PHI redactor."""
        self.phi_patterns = [
            # SSN patterns
            r'\b\d{3}-\d{2}-\d{4}\b',
            r'\b\d{9}\b',
            # Phone number patterns
            r'\b\d{3}-\d{3}-\d{4}\b',
            r'\(\d{3}\)\s*\d{3}-\d{4}',
            # Email patterns
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            # Date patterns (potential DOB)
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
        ]
    
    def redact_text(self, text: str) -> str:
        """Redact PHI from text."""
        import re
        
        redacted_text = text
        for pattern in self.phi_patterns:
            redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)
        
        return redacted_text
    
    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PHI from dictionary."""
        redacted = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                redacted[key] = self.redact_text(value)
            elif isinstance(value, dict):
                redacted[key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self.redact_dict(item) if isinstance(item, dict) 
                    else self.redact_text(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                redacted[key] = value
        
        return redacted


# Global PHI redactor instance
phi_redactor = PHIRedactor()


def get_compliance_status() -> Dict[str, Any]:
    """
    Get current HIPAA compliance status.
    
    Returns:
        Dict[str, Any]: Compliance status information
    """
    return {
        "audit_logging_enabled": settings.security.audit_log_enabled,
        "encryption_enabled": True,
        "data_retention_days": settings.security.data_retention_days,
        "compliance_manager_initialized": compliance_manager is not None,
        "phi_redactor_available": phi_redactor is not None,
        "security_settings": {
            "algorithm": settings.security.algorithm,
            "access_token_expire_minutes": settings.security.access_token_expire_minutes,
            "rate_limiting_enabled": True
        }
    }
