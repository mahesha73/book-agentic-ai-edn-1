
"""
State management for the Patient Diagnosis AI orchestrator.

This module defines the shared state structure and management utilities
for the multi-agent diagnostic workflow.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Status of the diagnostic workflow."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    AGENT_EXECUTING = "agent_executing"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Priority levels for diagnostic requests."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


@dataclass
class PatientContext:
    """Patient context information for diagnosis."""
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: Optional[str] = None
    symptoms: List[str] = field(default_factory=list)
    medical_history: List[Dict[str, Any]] = field(default_factory=list)
    current_medications: List[Dict[str, Any]] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    vital_signs: Dict[str, Any] = field(default_factory=dict)
    lab_results: List[Dict[str, Any]] = field(default_factory=list)
    imaging_studies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentExecution:
    """Information about an agent execution."""
    agent_name: str
    agent_type: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    confidence_score: float = 0.0


class DiagnosticState(BaseModel):
    """
    Comprehensive state for the diagnostic workflow.
    
    This class maintains all information needed throughout the
    multi-agent diagnostic process, including patient context,
    agent outputs, and workflow status.
    """
    
    # Workflow identification
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    
    # Workflow status
    status: WorkflowStatus = WorkflowStatus.INITIALIZED
    priority: Priority = Priority.NORMAL
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Patient information
    patient_context: Optional[PatientContext] = None
    
    # Original request
    original_query: str = ""
    clinical_question: str = ""
    additional_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent execution tracking
    agent_executions: List[AgentExecution] = Field(default_factory=list)
    current_agent: Optional[str] = None
    next_agents: List[str] = Field(default_factory=list)
    completed_agents: List[str] = Field(default_factory=list)
    failed_agents: List[str] = Field(default_factory=list)
    
    # Agent outputs
    patient_history_output: Optional[Dict[str, Any]] = None
    medical_coding_output: Optional[Dict[str, Any]] = None
    drug_safety_output: Optional[Dict[str, Any]] = None
    literature_research_output: Optional[Dict[str, Any]] = None
    image_analysis_output: Optional[Dict[str, Any]] = None
    
    # Synthesis and results
    differential_diagnosis: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    safety_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality and confidence
    overall_confidence: float = 0.0
    quality_score: float = 0.0
    completeness_score: float = 0.0
    
    # Escalation and review
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None
    requires_human_review: bool = True
    urgent_findings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Workflow control
    workflow_path: List[str] = Field(default_factory=list)
    decision_points: List[Dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    # Observability
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            PatientContext: lambda v: v.__dict__ if v else None,
            AgentExecution: lambda v: v.__dict__ if v else None,
        }
    
    def add_agent_execution(self, execution: AgentExecution) -> None:
        """Add an agent execution to the state."""
        self.agent_executions.append(execution)
        self.updated_at = datetime.utcnow()
    
    def update_agent_output(self, agent_type: str, output: Dict[str, Any]) -> None:
        """Update agent output in the state."""
        if agent_type == "patient_history":
            self.patient_history_output = output
        elif agent_type == "medical_coding":
            self.medical_coding_output = output
        elif agent_type == "drug_safety":
            self.drug_safety_output = output
        elif agent_type == "literature_research":
            self.literature_research_output = output
        elif agent_type == "image_analysis":
            self.image_analysis_output = output
        
        self.updated_at = datetime.utcnow()
    
    def add_error(self, error_type: str, message: str, agent_name: Optional[str] = None) -> None:
        """Add an error to the state."""
        error = {
            "type": error_type,
            "message": message,
            "agent_name": agent_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.errors.append(error)
        self.updated_at = datetime.utcnow()
    
    def add_warning(self, warning_type: str, message: str, agent_name: Optional[str] = None) -> None:
        """Add a warning to the state."""
        warning = {
            "type": warning_type,
            "message": message,
            "agent_name": agent_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.warnings.append(warning)
        self.updated_at = datetime.utcnow()
    
    def mark_agent_completed(self, agent_name: str) -> None:
        """Mark an agent as completed."""
        if agent_name not in self.completed_agents:
            self.completed_agents.append(agent_name)
        if agent_name in self.next_agents:
            self.next_agents.remove(agent_name)
        self.current_agent = None
        self.updated_at = datetime.utcnow()
    
    def mark_agent_failed(self, agent_name: str, error_message: str) -> None:
        """Mark an agent as failed."""
        if agent_name not in self.failed_agents:
            self.failed_agents.append(agent_name)
        if agent_name in self.next_agents:
            self.next_agents.remove(agent_name)
        self.current_agent = None
        self.add_error("agent_failure", error_message, agent_name)
        self.updated_at = datetime.utcnow()
    
    def set_next_agents(self, agent_names: List[str]) -> None:
        """Set the next agents to execute."""
        self.next_agents = agent_names
        self.updated_at = datetime.utcnow()
    
    def get_next_agent(self) -> Optional[str]:
        """Get the next agent to execute."""
        if self.next_agents:
            return self.next_agents[0]
        return None
    
    def is_complete(self) -> bool:
        """Check if the workflow is complete."""
        return self.status == WorkflowStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if the workflow has failed."""
        return self.status == WorkflowStatus.FAILED
    
    def should_escalate(self) -> bool:
        """Check if the case should be escalated."""
        return self.requires_escalation or len(self.urgent_findings) > 0
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of agent executions."""
        return {
            "total_agents_executed": len(self.agent_executions),
            "completed_agents": self.completed_agents,
            "failed_agents": self.failed_agents,
            "current_agent": self.current_agent,
            "next_agents": self.next_agents,
            "total_execution_time": sum(
                exec.execution_time_seconds for exec in self.agent_executions
            ),
            "average_confidence": (
                sum(exec.confidence_score for exec in self.agent_executions) / 
                len(self.agent_executions) if self.agent_executions else 0
            )
        }
    
    def get_agent_outputs(self) -> Dict[str, Any]:
        """Get all agent outputs."""
        return {
            "patient_history": self.patient_history_output,
            "medical_coding": self.medical_coding_output,
            "drug_safety": self.drug_safety_output,
            "literature_research": self.literature_research_output,
            "image_analysis": self.image_analysis_output
        }
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score."""
        if not self.agent_executions:
            return 0.0
        
        # Weight different agents differently
        weights = {
            "patient_history": 0.2,
            "medical_coding": 0.15,
            "drug_safety": 0.25,
            "literature_research": 0.3,
            "image_analysis": 0.1
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for execution in self.agent_executions:
            agent_weight = weights.get(execution.agent_type, 0.1)
            weighted_confidence += execution.confidence_score * agent_weight
            total_weight += agent_weight
        
        self.overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        return self.overall_confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return self.dict()
    
    def to_summary(self) -> Dict[str, Any]:
        """Get a summary of the diagnostic state."""
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "patient_id": self.patient_context.patient_id if self.patient_context else None,
            "clinical_question": self.clinical_question,
            "execution_summary": self.get_execution_summary(),
            "overall_confidence": self.overall_confidence,
            "requires_escalation": self.requires_escalation,
            "requires_human_review": self.requires_human_review,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class StateManager:
    """
    Manager for diagnostic state persistence and retrieval.
    
    This class handles saving and loading diagnostic states,
    managing state transitions, and providing state queries.
    """
    
    def __init__(self, redis_client=None, mongodb_client=None):
        """
        Initialize the state manager.
        
        Args:
            redis_client: Redis client for caching
            mongodb_client: MongoDB client for persistence
        """
        self.redis_client = redis_client
        self.mongodb_client = mongodb_client
        self._states: Dict[str, DiagnosticState] = {}  # In-memory cache
    
    def create_state(
        self,
        patient_context: PatientContext,
        clinical_question: str,
        user_id: Optional[str] = None,
        priority: Priority = Priority.NORMAL
    ) -> DiagnosticState:
        """
        Create a new diagnostic state.
        
        Args:
            patient_context: Patient information
            clinical_question: The clinical question to answer
            user_id: ID of the requesting user
            priority: Priority level
            
        Returns:
            DiagnosticState: New diagnostic state
        """
        state = DiagnosticState(
            patient_context=patient_context,
            clinical_question=clinical_question,
            user_id=user_id,
            priority=priority,
            status=WorkflowStatus.INITIALIZED
        )
        
        # Cache the state
        self._states[state.session_id] = state
        
        # Persist to database if available
        if self.mongodb_client:
            self._persist_state(state)
        
        logger.info(f"Created new diagnostic state: {state.session_id}")
        return state
    
    def get_state(self, session_id: str) -> Optional[DiagnosticState]:
        """
        Retrieve a diagnostic state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            DiagnosticState: The diagnostic state or None if not found
        """
        # Check in-memory cache first
        if session_id in self._states:
            return self._states[session_id]
        
        # Try Redis cache
        if self.redis_client:
            state_data = self._load_from_redis(session_id)
            if state_data:
                state = DiagnosticState.parse_obj(state_data)
                self._states[session_id] = state
                return state
        
        # Try MongoDB
        if self.mongodb_client:
            state_data = self._load_from_mongodb(session_id)
            if state_data:
                state = DiagnosticState.parse_obj(state_data)
                self._states[session_id] = state
                return state
        
        return None
    
    def update_state(self, state: DiagnosticState) -> None:
        """
        Update a diagnostic state.
        
        Args:
            state: The updated diagnostic state
        """
        state.updated_at = datetime.utcnow()
        
        # Update in-memory cache
        self._states[state.session_id] = state
        
        # Update Redis cache
        if self.redis_client:
            self._cache_in_redis(state)
        
        # Persist to MongoDB
        if self.mongodb_client:
            self._persist_state(state)
        
        logger.debug(f"Updated diagnostic state: {state.session_id}")
    
    def delete_state(self, session_id: str) -> bool:
        """
        Delete a diagnostic state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if deleted successfully
        """
        # Remove from in-memory cache
        if session_id in self._states:
            del self._states[session_id]
        
        # Remove from Redis
        if self.redis_client:
            self._delete_from_redis(session_id)
        
        # Remove from MongoDB
        if self.mongodb_client:
            self._delete_from_mongodb(session_id)
        
        logger.info(f"Deleted diagnostic state: {session_id}")
        return True
    
    def list_active_states(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List active diagnostic states.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List[Dict[str, Any]]: List of state summaries
        """
        active_states = []
        
        for state in self._states.values():
            if state.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                if user_id is None or state.user_id == user_id:
                    active_states.append(state.to_summary())
        
        return active_states
    
    def get_state_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a diagnostic state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict[str, Any]: State metrics
        """
        state = self.get_state(session_id)
        if not state:
            return {}
        
        return {
            "session_id": session_id,
            "status": state.status.value,
            "execution_time": (datetime.utcnow() - state.created_at).total_seconds(),
            "agent_count": len(state.agent_executions),
            "error_count": len(state.errors),
            "warning_count": len(state.warnings),
            "confidence_score": state.overall_confidence,
            "completion_percentage": self._calculate_completion_percentage(state)
        }
    
    def cleanup_old_states(self, max_age_hours: int = 24) -> int:
        """
        Clean up old diagnostic states.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            int: Number of states cleaned up
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        # Clean up in-memory cache
        session_ids_to_remove = []
        for session_id, state in self._states.items():
            if state.updated_at < cutoff_time and state.is_complete():
                session_ids_to_remove.append(session_id)
        
        for session_id in session_ids_to_remove:
            del self._states[session_id]
            cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old diagnostic states")
        return cleaned_count
    
    # Private methods for persistence
    
    def _persist_state(self, state: DiagnosticState) -> None:
        """Persist state to MongoDB."""
        if not self.mongodb_client:
            return
        
        try:
            collection = self.mongodb_client.diagnostic_states
            state_dict = state.dict()
            state_dict["_id"] = state.session_id
            
            collection.replace_one(
                {"_id": state.session_id},
                state_dict,
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error persisting state to MongoDB: {e}")
    
    def _load_from_mongodb(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load state from MongoDB."""
        if not self.mongodb_client:
            return None
        
        try:
            collection = self.mongodb_client.diagnostic_states
            state_doc = collection.find_one({"_id": session_id})
            if state_doc:
                del state_doc["_id"]  # Remove MongoDB ID
                return state_doc
        except Exception as e:
            logger.error(f"Error loading state from MongoDB: {e}")
        
        return None
    
    def _delete_from_mongodb(self, session_id: str) -> None:
        """Delete state from MongoDB."""
        if not self.mongodb_client:
            return
        
        try:
            collection = self.mongodb_client.diagnostic_states
            collection.delete_one({"_id": session_id})
        except Exception as e:
            logger.error(f"Error deleting state from MongoDB: {e}")
    
    def _cache_in_redis(self, state: DiagnosticState) -> None:
        """Cache state in Redis."""
        if not self.redis_client:
            return
        
        try:
            key = f"diagnostic_state:{state.session_id}"
            value = state.json()
            self.redis_client.setex(key, 3600, value)  # 1 hour TTL
        except Exception as e:
            logger.error(f"Error caching state in Redis: {e}")
    
    def _load_from_redis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load state from Redis."""
        if not self.redis_client:
            return None
        
        try:
            key = f"diagnostic_state:{session_id}"
            value = self.redis_client.get(key)
            if value:
                return DiagnosticState.parse_raw(value).dict()
        except Exception as e:
            logger.error(f"Error loading state from Redis: {e}")
        
        return None
    
    def _delete_from_redis(self, session_id: str) -> None:
        """Delete state from Redis."""
        if not self.redis_client:
            return
        
        try:
            key = f"diagnostic_state:{session_id}"
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting state from Redis: {e}")
    
    def _calculate_completion_percentage(self, state: DiagnosticState) -> float:
        """Calculate completion percentage for a state."""
        total_agents = 5  # patient_history, medical_coding, drug_safety, literature_research, image_analysis
        completed_agents = len(state.completed_agents)
        return (completed_agents / total_agents) * 100
