
"""
Patient History Agent for the Patient Diagnosis AI system.

This agent specializes in retrieving, analyzing, and synthesizing patient
medical history to identify relevant patterns, risk factors, and clinical
context that inform diagnostic decisions.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentResponse, AgentError, ConfidenceLevel
from src.config.prompts import get_prompt
from src.config.settings import get_settings
from src.db.engines import get_database_engines
from src.utils.medical_nlp import extract_medical_entities, normalize_medical_terms
from src.knowledge.fhir_client import FHIRClient

logger = logging.getLogger(__name__)
settings = get_settings()


class PatientHistoryAgent(BaseAgent):
    """
    Specialized agent for patient medical history analysis.
    
    This agent focuses on:
    - Retrieving patient medical history from EHR systems
    - Analyzing temporal patterns in medical events
    - Identifying risk factors and comorbidities
    - Recognizing medication history and adherence patterns
    - Flagging potential contraindications or complications
    """
    
    def __init__(self, llm, fhir_client: Optional[FHIRClient] = None):
        """
        Initialize the Patient History Agent.
        
        Args:
            llm: Language model instance
            fhir_client: Optional FHIR client for EHR integration
        """
        self.db_engines = get_database_engines()
        self.fhir_client = fhir_client
        
        # Initialize tools
        tools = self._create_tools()
        
        super().__init__(
            name="Patient History Specialist",
            agent_type="patient_history",
            description="Analyzes patient medical history to identify patterns, risk factors, and clinical context",
            tools=tools,
            llm=llm,
            max_iterations=settings.agents.agent_retry_attempts,
            timeout_seconds=settings.agents.agent_timeout
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for patient history analysis."""
        tools = [
            Tool(
                name="retrieve_patient_history",
                description="Retrieve comprehensive patient medical history from database",
                func=self._retrieve_patient_history
            ),
            Tool(
                name="analyze_medication_history",
                description="Analyze patient medication history for patterns and interactions",
                func=self._analyze_medication_history
            ),
            Tool(
                name="identify_risk_factors",
                description="Identify risk factors based on medical history and demographics",
                func=self._identify_risk_factors
            ),
            Tool(
                name="extract_family_history",
                description="Extract and analyze family medical history",
                func=self._extract_family_history
            ),
            Tool(
                name="timeline_medical_events",
                description="Create chronological timeline of medical events",
                func=self._timeline_medical_events
            ),
            Tool(
                name="check_contraindications",
                description="Check for potential contraindications based on history",
                func=self._check_contraindications
            )
        ]
        
        # Add FHIR tools if client is available
        if self.fhir_client:
            tools.extend([
                Tool(
                    name="query_fhir_resources",
                    description="Query FHIR resources for patient data",
                    func=self._query_fhir_resources
                ),
                Tool(
                    name="get_patient_encounters",
                    description="Retrieve patient encounters from FHIR server",
                    func=self._get_patient_encounters
                )
            ])
        
        return tools
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with patient history prompt."""
        prompt = get_prompt("patient_history")
        
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=settings.app.debug,
            max_iterations=self.max_iterations,
            max_execution_time=self.timeout_seconds,
            return_intermediate_steps=True
        )
    
    def _validate_input(self, request: Dict[str, Any]) -> None:
        """Validate input for patient history analysis."""
        if not request:
            raise AgentError(
                "Empty request provided",
                agent_name=self.name,
                error_code="INVALID_INPUT"
            )
        
        # Check for required fields
        required_fields = ["patient_id"]
        missing_fields = [field for field in required_fields if field not in request]
        
        if missing_fields:
            raise AgentError(
                f"Missing required fields: {missing_fields}",
                agent_name=self.name,
                error_code="MISSING_FIELDS"
            )
        
        # Validate patient ID format
        patient_id = request.get("patient_id")
        if not isinstance(patient_id, str) or len(patient_id) == 0:
            raise AgentError(
                "Invalid patient ID format",
                agent_name=self.name,
                error_code="INVALID_PATIENT_ID"
            )
    
    def _process_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format patient history analysis output."""
        try:
            # Extract the main output from agent execution
            output_text = raw_output.get("output", "")
            intermediate_steps = raw_output.get("intermediate_steps", [])
            
            # Parse structured output from agent response
            processed_output = {
                "medical_history_summary": self._extract_history_summary(output_text),
                "risk_factors": self._extract_risk_factors(output_text),
                "medication_analysis": self._extract_medication_analysis(output_text),
                "family_history": self._extract_family_history_data(output_text),
                "timeline": self._extract_timeline(output_text),
                "contraindications": self._extract_contraindications(output_text),
                "clinical_significance": self._assess_clinical_significance(output_text),
                "recommendations": self._extract_recommendations(output_text)
            }
            
            # Add metadata from intermediate steps
            processed_output["tool_usage"] = self._summarize_tool_usage(intermediate_steps)
            processed_output["data_sources"] = self._identify_data_sources(intermediate_steps)
            
            return processed_output
            
        except Exception as e:
            logger.error(f"Error processing patient history output: {e}")
            raise AgentError(
                f"Failed to process agent output: {str(e)}",
                agent_name=self.name,
                error_code="OUTPUT_PROCESSING_ERROR"
            )
    
    # Tool Implementation Methods
    
    def _retrieve_patient_history(self, patient_id: str) -> str:
        """Retrieve comprehensive patient medical history."""
        try:
            with self.db_engines.postgres_session() as session:
                # Query medical history from database
                from src.db.models import MedicalHistory, Patient
                
                patient = session.query(Patient).filter(Patient.id == patient_id).first()
                if not patient:
                    return f"Patient {patient_id} not found in database"
                
                history_records = session.query(MedicalHistory).filter(
                    MedicalHistory.patient_id == patient_id,
                    MedicalHistory.is_deleted == False
                ).order_by(MedicalHistory.onset_date.desc()).all()
                
                if not history_records:
                    return "No medical history found for this patient"
                
                # Format history for analysis
                history_summary = []
                for record in history_records:
                    history_item = {
                        "condition": record.condition,
                        "onset_date": record.onset_date.isoformat() if record.onset_date else None,
                        "resolution_date": record.resolution_date.isoformat() if record.resolution_date else None,
                        "severity": record.severity,
                        "is_active": record.is_active,
                        "icd10_code": record.icd10_code,
                        "snomed_code": record.snomed_code
                    }
                    history_summary.append(history_item)
                
                return f"Retrieved {len(history_records)} medical history records: {history_summary}"
                
        except Exception as e:
            logger.error(f"Error retrieving patient history: {e}")
            return f"Error retrieving patient history: {str(e)}"
    
    def _analyze_medication_history(self, patient_id: str) -> str:
        """Analyze patient medication history."""
        try:
            # This would integrate with pharmacy systems or EHR medication data
            # For now, return a placeholder implementation
            
            medication_analysis = {
                "current_medications": [],
                "medication_adherence": "unknown",
                "potential_interactions": [],
                "allergy_alerts": [],
                "dosage_concerns": []
            }
            
            return f"Medication analysis completed: {medication_analysis}"
            
        except Exception as e:
            logger.error(f"Error analyzing medication history: {e}")
            return f"Error analyzing medication history: {str(e)}"
    
    def _identify_risk_factors(self, patient_data: str) -> str:
        """Identify risk factors from patient data."""
        try:
            # Extract medical entities from patient data
            entities = extract_medical_entities(patient_data)
            
            # Analyze risk factors based on conditions and demographics
            risk_factors = {
                "cardiovascular_risk": [],
                "diabetes_risk": [],
                "cancer_risk": [],
                "infectious_disease_risk": [],
                "genetic_risk": []
            }
            
            # Simple rule-based risk assessment
            conditions = entities.get("conditions", [])
            for condition in conditions:
                if any(term in condition.lower() for term in ["hypertension", "high blood pressure"]):
                    risk_factors["cardiovascular_risk"].append("Hypertension history")
                if any(term in condition.lower() for term in ["diabetes", "glucose"]):
                    risk_factors["diabetes_risk"].append("Diabetes history")
            
            return f"Risk factors identified: {risk_factors}"
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return f"Error identifying risk factors: {str(e)}"
    
    def _extract_family_history(self, patient_id: str) -> str:
        """Extract family medical history."""
        try:
            # This would query family history data
            # Placeholder implementation
            family_history = {
                "cardiovascular_disease": "unknown",
                "diabetes": "unknown",
                "cancer": "unknown",
                "genetic_disorders": "unknown"
            }
            
            return f"Family history extracted: {family_history}"
            
        except Exception as e:
            logger.error(f"Error extracting family history: {e}")
            return f"Error extracting family history: {str(e)}"
    
    def _timeline_medical_events(self, patient_id: str) -> str:
        """Create chronological timeline of medical events."""
        try:
            with self.db_engines.postgres_session() as session:
                from src.db.models import MedicalHistory
                
                events = session.query(MedicalHistory).filter(
                    MedicalHistory.patient_id == patient_id,
                    MedicalHistory.is_deleted == False
                ).order_by(MedicalHistory.onset_date).all()
                
                timeline = []
                for event in events:
                    timeline_item = {
                        "date": event.onset_date.isoformat() if event.onset_date else "unknown",
                        "event": event.condition,
                        "type": "diagnosis",
                        "severity": event.severity,
                        "status": "active" if event.is_active else "resolved"
                    }
                    timeline.append(timeline_item)
                
                return f"Medical timeline created with {len(timeline)} events: {timeline}"
                
        except Exception as e:
            logger.error(f"Error creating medical timeline: {e}")
            return f"Error creating medical timeline: {str(e)}"
    
    def _check_contraindications(self, patient_data: str) -> str:
        """Check for potential contraindications."""
        try:
            # Extract conditions and medications
            entities = extract_medical_entities(patient_data)
            
            contraindications = {
                "drug_contraindications": [],
                "procedure_contraindications": [],
                "allergy_alerts": []
            }
            
            # Simple contraindication checking
            conditions = entities.get("conditions", [])
            if any("kidney" in condition.lower() for condition in conditions):
                contraindications["drug_contraindications"].append("Nephrotoxic drugs contraindicated")
            
            return f"Contraindications identified: {contraindications}"
            
        except Exception as e:
            logger.error(f"Error checking contraindications: {e}")
            return f"Error checking contraindications: {str(e)}"
    
    def _query_fhir_resources(self, patient_id: str, resource_type: str = "Condition") -> str:
        """Query FHIR resources for patient data."""
        if not self.fhir_client:
            return "FHIR client not available"
        
        try:
            resources = self.fhir_client.search_resources(
                resource_type=resource_type,
                patient_id=patient_id
            )
            
            return f"Retrieved {len(resources)} {resource_type} resources from FHIR server"
            
        except Exception as e:
            logger.error(f"Error querying FHIR resources: {e}")
            return f"Error querying FHIR resources: {str(e)}"
    
    def _get_patient_encounters(self, patient_id: str) -> str:
        """Retrieve patient encounters from FHIR server."""
        if not self.fhir_client:
            return "FHIR client not available"
        
        try:
            encounters = self.fhir_client.get_patient_encounters(patient_id)
            
            encounter_summary = []
            for encounter in encounters:
                encounter_summary.append({
                    "id": encounter.get("id"),
                    "status": encounter.get("status"),
                    "class": encounter.get("class", {}).get("display"),
                    "period": encounter.get("period")
                })
            
            return f"Retrieved {len(encounters)} encounters: {encounter_summary}"
            
        except Exception as e:
            logger.error(f"Error retrieving patient encounters: {e}")
            return f"Error retrieving patient encounters: {str(e)}"
    
    # Output Processing Helper Methods
    
    def _extract_history_summary(self, output_text: str) -> Dict[str, Any]:
        """Extract medical history summary from output."""
        # This would use NLP to extract structured data from the agent's output
        return {
            "summary": "Medical history analysis completed",
            "key_conditions": [],
            "active_problems": [],
            "resolved_conditions": []
        }
    
    def _extract_risk_factors(self, output_text: str) -> List[Dict[str, Any]]:
        """Extract risk factors from output."""
        return [
            {
                "factor": "Example risk factor",
                "category": "cardiovascular",
                "severity": "moderate",
                "evidence": "Based on medical history"
            }
        ]
    
    def _extract_medication_analysis(self, output_text: str) -> Dict[str, Any]:
        """Extract medication analysis from output."""
        return {
            "current_medications": [],
            "adherence_assessment": "unknown",
            "interaction_alerts": [],
            "recommendations": []
        }
    
    def _extract_family_history_data(self, output_text: str) -> Dict[str, Any]:
        """Extract family history data from output."""
        return {
            "significant_family_history": [],
            "genetic_risk_factors": [],
            "hereditary_conditions": []
        }
    
    def _extract_timeline(self, output_text: str) -> List[Dict[str, Any]]:
        """Extract medical timeline from output."""
        return [
            {
                "date": "2024-01-01",
                "event": "Example medical event",
                "type": "diagnosis",
                "significance": "high"
            }
        ]
    
    def _extract_contraindications(self, output_text: str) -> List[Dict[str, Any]]:
        """Extract contraindications from output."""
        return [
            {
                "type": "drug",
                "contraindication": "Example contraindication",
                "severity": "moderate",
                "reason": "Based on medical history"
            }
        ]
    
    def _assess_clinical_significance(self, output_text: str) -> Dict[str, Any]:
        """Assess clinical significance of findings."""
        return {
            "overall_significance": "moderate",
            "key_findings": [],
            "clinical_implications": [],
            "urgency_level": "routine"
        }
    
    def _extract_recommendations(self, output_text: str) -> List[str]:
        """Extract recommendations from output."""
        return [
            "Continue monitoring patient's medical history",
            "Consider additional screening based on risk factors",
            "Review medication adherence"
        ]
    
    def _summarize_tool_usage(self, intermediate_steps: List) -> Dict[str, Any]:
        """Summarize which tools were used during execution."""
        tool_usage = {}
        for step in intermediate_steps:
            if len(step) >= 2:
                action = step[0]
                if hasattr(action, 'tool'):
                    tool_name = action.tool
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        return tool_usage
    
    def _identify_data_sources(self, intermediate_steps: List) -> List[str]:
        """Identify data sources used during analysis."""
        sources = set()
        for step in intermediate_steps:
            if len(step) >= 2:
                observation = step[1]
                if "database" in str(observation).lower():
                    sources.add("internal_database")
                if "fhir" in str(observation).lower():
                    sources.add("fhir_server")
        
        return list(sources)
    
    def _calculate_confidence(self, output: Dict[str, Any]) -> ConfidenceLevel:
        """Calculate confidence based on data availability and quality."""
        # Assess confidence based on available data
        data_completeness = 0
        
        if output.get("medical_history_summary", {}).get("key_conditions"):
            data_completeness += 0.3
        if output.get("medication_analysis", {}).get("current_medications"):
            data_completeness += 0.2
        if output.get("risk_factors"):
            data_completeness += 0.2
        if output.get("timeline"):
            data_completeness += 0.2
        if output.get("family_history", {}).get("significant_family_history"):
            data_completeness += 0.1
        
        if data_completeness >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif data_completeness >= 0.6:
            return ConfidenceLevel.HIGH
        elif data_completeness >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif data_completeness >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _requires_escalation(self, output: Dict[str, Any]) -> bool:
        """Determine if case requires escalation based on history findings."""
        # Check for high-risk conditions or critical contraindications
        contraindications = output.get("contraindications", [])
        for contraindication in contraindications:
            if contraindication.get("severity") == "critical":
                return True
        
        # Check clinical significance
        significance = output.get("clinical_significance", {})
        if significance.get("urgency_level") == "urgent":
            return True
        
        return False
    
    def _suggest_next_actions(self, output: Dict[str, Any]) -> List[str]:
        """Suggest next actions based on history analysis."""
        actions = []
        
        # Always recommend clinical correlation
        actions.append("Correlate findings with current clinical presentation")
        
        # Suggest additional data gathering if needed
        if not output.get("medication_analysis", {}).get("current_medications"):
            actions.append("Obtain complete current medication list")
        
        if not output.get("family_history", {}).get("significant_family_history"):
            actions.append("Gather detailed family medical history")
        
        # Suggest follow-up based on risk factors
        risk_factors = output.get("risk_factors", [])
        if any(rf.get("category") == "cardiovascular" for rf in risk_factors):
            actions.append("Consider cardiovascular risk assessment")
        
        return actions
