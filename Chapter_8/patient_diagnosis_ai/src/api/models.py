
"""
Pydantic models for the Patient Diagnosis AI API.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class DiagnosisRequest(BaseModel):
    """Request model for diagnosis."""
    
    patient_id: str = Field(..., description="Patient identifier")
    user_id: Optional[str] = Field(None, description="User making the request")
    clinical_question: str = Field(..., description="Clinical question to answer")
    
    # Patient demographics
    patient_age: Optional[int] = Field(None, description="Patient age")
    patient_gender: Optional[str] = Field(None, description="Patient gender")
    
    # Clinical data
    chief_complaint: Optional[str] = Field(None, description="Chief complaint")
    symptoms: List[str] = Field(default_factory=list, description="List of symptoms")
    medical_history: Optional[List[Dict[str, Any]]] = Field(None, description="Medical history")
    current_medications: Optional[List[Dict[str, Any]]] = Field(None, description="Current medications")
    allergies: Optional[List[str]] = Field(None, description="Known allergies")
    vital_signs: Optional[Dict[str, Any]] = Field(None, description="Vital signs")
    lab_results: Optional[List[Dict[str, Any]]] = Field(None, description="Laboratory results")
    imaging_studies: Optional[List[Dict[str, Any]]] = Field(None, description="Imaging studies")
    
    # Request metadata
    priority: Optional[str] = Field("normal", description="Request priority")


class DiagnosisResponse(BaseModel):
    """Response model for diagnosis."""
    
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Diagnosis status")
    
    # Results
    differential_diagnosis: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    safety_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality metrics
    overall_confidence: float = Field(0.0, description="Overall confidence score")
    
    # Escalation flags
    requires_escalation: bool = Field(False, description="Requires escalation")
    requires_human_review: bool = Field(True, description="Requires human review")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
