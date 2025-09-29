
"""
SQLAlchemy models for the Patient Diagnosis AI system.

This module defines the database schema for PostgreSQL using SQLAlchemy ORM.
All models include HIPAA compliance features like encryption, audit trails,
and proper data retention policies.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, JSON, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    LargeBinary, Float, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from enum import Enum
import json
from cryptography.fernet import Fernet

from src.config.settings import get_settings

settings = get_settings()
Base = declarative_base()

# Initialize encryption for HIPAA compliance
cipher_suite = Fernet(settings.security.encryption_key.encode()[:32])


class AuditMixin:
    """Mixin for audit trail functionality."""
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(UUID(as_uuid=True), nullable=True)
    updated_by = Column(UUID(as_uuid=True), nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    deleted_by = Column(UUID(as_uuid=True), nullable=True)


class EncryptedMixin:
    """Mixin for encrypted fields (HIPAA compliance)."""
    
    @staticmethod
    def encrypt_data(data: str) -> bytes:
        """Encrypt sensitive data."""
        if not data:
            return b''
        return cipher_suite.encrypt(data.encode())
    
    @staticmethod
    def decrypt_data(encrypted_data: bytes) -> str:
        """Decrypt sensitive data."""
        if not encrypted_data:
            return ''
        return cipher_suite.decrypt(encrypted_data).decode()


class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    CLINICIAN = "clinician"
    NURSE = "nurse"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class DiagnosisStatus(str, Enum):
    """Status of diagnosis requests."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    """Types of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    PATIENT_HISTORY = "patient_history"
    MEDICAL_CODING = "medical_coding"
    DRUG_SAFETY = "drug_safety"
    LITERATURE_RESEARCH = "literature_research"
    IMAGE_ANALYSIS = "image_analysis"


class User(Base, AuditMixin):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Professional credentials
    license_number = Column(String(100), nullable=True)
    specialty = Column(String(100), nullable=True)
    institution = Column(String(255), nullable=True)
    
    # Relationships
    diagnosis_requests = relationship("DiagnosisRequest", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_username', 'username'),
        Index('idx_user_role', 'role'),
    )


class Patient(Base, AuditMixin, EncryptedMixin):
    """Patient model with HIPAA-compliant encryption."""
    
    __tablename__ = "patients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Encrypted personal information
    encrypted_name = Column(LargeBinary, nullable=False)
    encrypted_date_of_birth = Column(LargeBinary, nullable=True)
    encrypted_ssn = Column(LargeBinary, nullable=True)
    encrypted_phone = Column(LargeBinary, nullable=True)
    encrypted_email = Column(LargeBinary, nullable=True)
    encrypted_address = Column(LargeBinary, nullable=True)
    
    # Non-sensitive demographic data
    gender = Column(String(20), nullable=True)
    race = Column(String(50), nullable=True)
    ethnicity = Column(String(50), nullable=True)
    
    # Medical record number (encrypted)
    encrypted_mrn = Column(LargeBinary, nullable=True, unique=True)
    
    # Relationships
    diagnosis_requests = relationship("DiagnosisRequest", back_populates="patient")
    medical_history = relationship("MedicalHistory", back_populates="patient")
    
    @property
    def name(self) -> str:
        """Decrypt and return patient name."""
        return self.decrypt_data(self.encrypted_name)
    
    @name.setter
    def name(self, value: str):
        """Encrypt and store patient name."""
        self.encrypted_name = self.encrypt_data(value)
    
    @property
    def date_of_birth(self) -> Optional[str]:
        """Decrypt and return date of birth."""
        if self.encrypted_date_of_birth:
            return self.decrypt_data(self.encrypted_date_of_birth)
        return None
    
    @date_of_birth.setter
    def date_of_birth(self, value: Optional[str]):
        """Encrypt and store date of birth."""
        if value:
            self.encrypted_date_of_birth = self.encrypt_data(value)
    
    @property
    def mrn(self) -> Optional[str]:
        """Decrypt and return medical record number."""
        if self.encrypted_mrn:
            return self.decrypt_data(self.encrypted_mrn)
        return None
    
    @mrn.setter
    def mrn(self, value: Optional[str]):
        """Encrypt and store medical record number."""
        if value:
            self.encrypted_mrn = self.encrypt_data(value)
    
    __table_args__ = (
        Index('idx_patient_gender', 'gender'),
        Index('idx_patient_created_at', 'created_at'),
    )


class MedicalHistory(Base, AuditMixin, EncryptedMixin):
    """Medical history records for patients."""
    
    __tablename__ = "medical_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    
    # Encrypted medical data
    encrypted_condition = Column(LargeBinary, nullable=False)
    encrypted_description = Column(LargeBinary, nullable=True)
    encrypted_medications = Column(LargeBinary, nullable=True)
    
    # Medical codes
    icd10_code = Column(String(20), nullable=True)
    snomed_code = Column(String(50), nullable=True)
    
    # Dates
    onset_date = Column(DateTime(timezone=True), nullable=True)
    resolution_date = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    severity = Column(String(20), nullable=True)  # mild, moderate, severe
    
    # Relationships
    patient = relationship("Patient", back_populates="medical_history")
    
    @property
    def condition(self) -> str:
        """Decrypt and return condition."""
        return self.decrypt_data(self.encrypted_condition)
    
    @condition.setter
    def condition(self, value: str):
        """Encrypt and store condition."""
        self.encrypted_condition = self.encrypt_data(value)
    
    __table_args__ = (
        Index('idx_medical_history_patient', 'patient_id'),
        Index('idx_medical_history_icd10', 'icd10_code'),
        Index('idx_medical_history_snomed', 'snomed_code'),
    )


class DiagnosisRequest(Base, AuditMixin):
    """Diagnosis request submitted to the AI system."""
    
    __tablename__ = "diagnosis_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True)
    
    # Request details
    symptoms = Column(JSONB, nullable=False)
    chief_complaint = Column(Text, nullable=True)
    additional_context = Column(JSONB, nullable=True)
    
    # Status and timing
    status = Column(SQLEnum(DiagnosisStatus), default=DiagnosisStatus.PENDING, nullable=False)
    priority = Column(String(20), default="normal", nullable=False)  # low, normal, high, urgent
    
    # Results
    diagnosis_results = Column(JSONB, nullable=True)
    confidence_score = Column(Float, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    
    # Agent workflow tracking
    agents_involved = Column(JSONB, nullable=True)
    workflow_state = Column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="diagnosis_requests")
    patient = relationship("Patient", back_populates="diagnosis_requests")
    agent_executions = relationship("AgentExecution", back_populates="diagnosis_request")
    
    __table_args__ = (
        Index('idx_diagnosis_request_user', 'user_id'),
        Index('idx_diagnosis_request_patient', 'patient_id'),
        Index('idx_diagnosis_request_status', 'status'),
        Index('idx_diagnosis_request_created_at', 'created_at'),
    )


class AgentExecution(Base, AuditMixin):
    """Record of individual agent executions within a diagnosis request."""
    
    __tablename__ = "agent_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    diagnosis_request_id = Column(UUID(as_uuid=True), ForeignKey("diagnosis_requests.id"), nullable=False)
    
    # Agent details
    agent_type = Column(SQLEnum(AgentType), nullable=False)
    agent_name = Column(String(100), nullable=False)
    agent_version = Column(String(20), nullable=True)
    
    # Execution details
    input_data = Column(JSONB, nullable=False)
    output_data = Column(JSONB, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)
    
    # Status and error handling
    status = Column(String(20), default="pending", nullable=False)  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Observability
    trace_id = Column(String(100), nullable=True)
    span_id = Column(String(100), nullable=True)
    
    # Relationships
    diagnosis_request = relationship("DiagnosisRequest", back_populates="agent_executions")
    
    __table_args__ = (
        Index('idx_agent_execution_request', 'diagnosis_request_id'),
        Index('idx_agent_execution_type', 'agent_type'),
        Index('idx_agent_execution_status', 'status'),
        Index('idx_agent_execution_trace', 'trace_id'),
    )


class KnowledgeBase(Base, AuditMixin):
    """Knowledge base entries for medical information."""
    
    __tablename__ = "knowledge_base"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Content
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False)  # guideline, research, drug_info, etc.
    
    # Medical codes and classification
    medical_codes = Column(JSONB, nullable=True)  # ICD-10, SNOMED, etc.
    keywords = Column(JSONB, nullable=True)
    
    # Source information
    source = Column(String(255), nullable=True)
    source_url = Column(String(500), nullable=True)
    publication_date = Column(DateTime(timezone=True), nullable=True)
    
    # Quality and relevance
    evidence_level = Column(String(20), nullable=True)  # A, B, C, D
    quality_score = Column(Float, nullable=True)
    
    # Vector embeddings for RAG
    embedding_vector = Column(LargeBinary, nullable=True)
    embedding_model = Column(String(100), nullable=True)
    
    __table_args__ = (
        Index('idx_knowledge_base_type', 'content_type'),
        Index('idx_knowledge_base_codes', 'medical_codes', postgresql_using='gin'),
        Index('idx_knowledge_base_keywords', 'keywords', postgresql_using='gin'),
        Index('idx_knowledge_base_evidence', 'evidence_level'),
    )


class AuditLog(Base):
    """Comprehensive audit log for HIPAA compliance."""
    
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # User and session information
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    session_id = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Action details
    action = Column(String(100), nullable=False)  # CREATE, READ, UPDATE, DELETE, LOGIN, etc.
    resource_type = Column(String(100), nullable=False)  # Patient, DiagnosisRequest, etc.
    resource_id = Column(String(255), nullable=True)
    
    # Change tracking
    old_values = Column(JSONB, nullable=True)
    new_values = Column(JSONB, nullable=True)
    
    # Additional context
    description = Column(Text, nullable=True)
    risk_level = Column(String(20), default="low", nullable=False)  # low, medium, high, critical
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index('idx_audit_log_timestamp', 'timestamp'),
        Index('idx_audit_log_user', 'user_id'),
        Index('idx_audit_log_action', 'action'),
        Index('idx_audit_log_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_log_risk', 'risk_level'),
    )


class SystemConfiguration(Base, AuditMixin):
    """System configuration and settings."""
    
    __tablename__ = "system_configuration"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Configuration details
    config_key = Column(String(255), unique=True, nullable=False)
    config_value = Column(JSONB, nullable=False)
    config_type = Column(String(50), nullable=False)  # agent, api, security, etc.
    
    # Metadata
    description = Column(Text, nullable=True)
    is_sensitive = Column(Boolean, default=False, nullable=False)
    requires_restart = Column(Boolean, default=False, nullable=False)
    
    # Validation
    validation_schema = Column(JSONB, nullable=True)
    
    __table_args__ = (
        Index('idx_system_config_key', 'config_key'),
        Index('idx_system_config_type', 'config_type'),
    )


class APIUsage(Base):
    """Track API usage for monitoring and billing."""
    
    __tablename__ = "api_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Request details
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    
    # Response details
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Float, nullable=False)
    
    # Resource usage
    tokens_used = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    
    # Additional metadata
    request_size_bytes = Column(Integer, nullable=True)
    response_size_bytes = Column(Integer, nullable=True)
    
    __table_args__ = (
        Index('idx_api_usage_timestamp', 'timestamp'),
        Index('idx_api_usage_user', 'user_id'),
        Index('idx_api_usage_endpoint', 'endpoint'),
    )


# Data retention policy implementation
class DataRetentionPolicy:
    """Implement HIPAA data retention policies."""
    
    @staticmethod
    def get_retention_date(record_type: str) -> datetime:
        """Get retention date for different record types."""
        retention_days = settings.security.data_retention_days
        
        # Different retention periods for different data types
        retention_periods = {
            "audit_logs": 2555,  # 7 years
            "diagnosis_requests": 2555,  # 7 years
            "medical_history": 2555,  # 7 years
            "api_usage": 365,  # 1 year
            "system_configuration": 1095,  # 3 years
        }
        
        days = retention_periods.get(record_type, retention_days)
        return datetime.utcnow() - timedelta(days=days)
    
    @staticmethod
    def should_retain(record, record_type: str) -> bool:
        """Check if a record should be retained."""
        retention_date = DataRetentionPolicy.get_retention_date(record_type)
        return record.created_at > retention_date


# Database initialization functions
def create_indexes():
    """Create additional database indexes for performance."""
    # This would be called during database setup
    pass


def create_constraints():
    """Create additional database constraints."""
    # This would be called during database setup
    pass
