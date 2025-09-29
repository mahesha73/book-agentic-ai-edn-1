
"""
Drug Safety Agent for the Patient Diagnosis AI system.

This agent specializes in medication safety analysis, drug interactions,
contraindications, adverse event monitoring, and FDA safety alerts.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentResponse, AgentError, ConfidenceLevel
from src.config.prompts import get_prompt
from src.config.settings import get_settings
from src.knowledge.fda_api_client import FDAAPIClient
from src.utils.medical_nlp import extract_medical_entities

logger = logging.getLogger(__name__)
settings = get_settings()


class DrugSafetyAgent(BaseAgent):
    """
    Specialized agent for drug safety and pharmacovigilance.
    
    This agent focuses on:
    - Drug-drug interaction analysis
    - Drug-disease contraindication checking
    - Adverse event monitoring and analysis
    - Dosage appropriateness assessment
    - FDA safety alerts and recall monitoring
    - Pharmacokinetic and pharmacodynamic considerations
    """
    
    def __init__(self, llm, fda_client: Optional[FDAAPIClient] = None):
        """
        Initialize the Drug Safety Agent.
        
        Args:
            llm: Language model instance
            fda_client: FDA API client for safety data
        """
        self.fda_client = fda_client or FDAAPIClient()
        
        # Drug interaction database (simplified for demo)
        self.interaction_db = self._load_interaction_database()
        
        # Initialize tools
        tools = self._create_tools()
        
        super().__init__(
            name="Drug Safety Specialist",
            agent_type="drug_safety",
            description="Analyzes medication safety, interactions, contraindications, and adverse events",
            tools=tools,
            llm=llm,
            max_iterations=settings.agents.agent_retry_attempts,
            timeout_seconds=settings.agents.agent_timeout
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for drug safety analysis."""
        return [
            Tool(
                name="check_drug_interactions",
                description="Check for drug-drug interactions between medications",
                func=self._check_drug_interactions
            ),
            Tool(
                name="check_contraindications",
                description="Check for drug-disease contraindications",
                func=self._check_contraindications
            ),
            Tool(
                name="analyze_adverse_events",
                description="Analyze potential adverse events for medications",
                func=self._analyze_adverse_events
            ),
            Tool(
                name="check_dosage_appropriateness",
                description="Check if medication dosages are appropriate",
                func=self._check_dosage_appropriateness
            ),
            Tool(
                name="get_fda_safety_alerts",
                description="Get FDA safety alerts and recalls for medications",
                func=self._get_fda_safety_alerts
            ),
            Tool(
                name="assess_pregnancy_safety",
                description="Assess medication safety during pregnancy",
                func=self._assess_pregnancy_safety
            ),
            Tool(
                name="check_renal_dosing",
                description="Check renal dosing adjustments for medications",
                func=self._check_renal_dosing
            ),
            Tool(
                name="check_hepatic_dosing",
                description="Check hepatic dosing adjustments for medications",
                func=self._check_hepatic_dosing
            ),
            Tool(
                name="analyze_polypharmacy_risk",
                description="Analyze risks associated with polypharmacy",
                func=self._analyze_polypharmacy_risk
            ),
            Tool(
                name="get_drug_monitoring_requirements",
                description="Get monitoring requirements for medications",
                func=self._get_drug_monitoring_requirements
            )
        ]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with drug safety prompt."""
        prompt = get_prompt("drug_safety")
        
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
        """Validate input for drug safety analysis."""
        if not request:
            raise AgentError(
                "Empty request provided",
                agent_name=self.name,
                error_code="INVALID_INPUT"
            )
        
        # Check for medications
        medications = request.get("medications", [])
        if not medications:
            raise AgentError(
                "No medications provided for safety analysis",
                agent_name=self.name,
                error_code="NO_MEDICATIONS"
            )
        
        # Validate medication format
        for med in medications:
            if not isinstance(med, (str, dict)):
                raise AgentError(
                    "Invalid medication format",
                    agent_name=self.name,
                    error_code="INVALID_MEDICATION_FORMAT"
                )
    
    def _process_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format drug safety analysis output."""
        try:
            output_text = raw_output.get("output", "")
            intermediate_steps = raw_output.get("intermediate_steps", [])
            
            processed_output = {
                "safety_summary": self._extract_safety_summary(output_text),
                "drug_interactions": self._extract_drug_interactions(output_text),
                "contraindications": self._extract_contraindications(output_text),
                "adverse_events": self._extract_adverse_events(output_text),
                "dosage_recommendations": self._extract_dosage_recommendations(output_text),
                "monitoring_requirements": self._extract_monitoring_requirements(output_text),
                "fda_alerts": self._extract_fda_alerts(output_text),
                "risk_assessment": self._extract_risk_assessment(output_text),
                "safety_recommendations": self._extract_safety_recommendations(output_text)
            }
            
            # Add metadata
            processed_output["analysis_scope"] = self._determine_analysis_scope(intermediate_steps)
            processed_output["safety_score"] = self._calculate_safety_score(processed_output)
            
            return processed_output
            
        except Exception as e:
            logger.error(f"Error processing drug safety output: {e}")
            raise AgentError(
                f"Failed to process safety analysis: {str(e)}",
                agent_name=self.name,
                error_code="OUTPUT_PROCESSING_ERROR"
            )
    
    # Tool Implementation Methods
    
    def _check_drug_interactions(self, medications: str) -> str:
        """Check for drug-drug interactions."""
        try:
            # Parse medications
            med_list = self._parse_medication_list(medications)
            
            interactions = []
            for i, med1 in enumerate(med_list):
                for med2 in med_list[i+1:]:
                    interaction = self._find_interaction(med1, med2)
                    if interaction:
                        interactions.append(interaction)
            
            if not interactions:
                return f"No significant drug interactions found among: {med_list}"
            
            # Format interactions by severity
            critical_interactions = [i for i in interactions if i["severity"] == "critical"]
            major_interactions = [i for i in interactions if i["severity"] == "major"]
            moderate_interactions = [i for i in interactions if i["severity"] == "moderate"]
            
            result = {
                "total_interactions": len(interactions),
                "critical": critical_interactions,
                "major": major_interactions,
                "moderate": moderate_interactions
            }
            
            return f"Drug interaction analysis: {result}"
            
        except Exception as e:
            logger.error(f"Error checking drug interactions: {e}")
            return f"Error checking drug interactions: {str(e)}"
    
    def _check_contraindications(self, medications_and_conditions: str) -> str:
        """Check for drug-disease contraindications."""
        try:
            # Parse input to extract medications and conditions
            data = self._parse_medications_and_conditions(medications_and_conditions)
            medications = data["medications"]
            conditions = data["conditions"]
            
            contraindications = []
            for medication in medications:
                for condition in conditions:
                    contraindication = self._find_contraindication(medication, condition)
                    if contraindication:
                        contraindications.append(contraindication)
            
            if not contraindications:
                return "No contraindications found"
            
            # Categorize by severity
            absolute_contraindications = [c for c in contraindications if c["type"] == "absolute"]
            relative_contraindications = [c for c in contraindications if c["type"] == "relative"]
            
            result = {
                "total_contraindications": len(contraindications),
                "absolute": absolute_contraindications,
                "relative": relative_contraindications
            }
            
            return f"Contraindication analysis: {result}"
            
        except Exception as e:
            logger.error(f"Error checking contraindications: {e}")
            return f"Error checking contraindications: {str(e)}"
    
    def _analyze_adverse_events(self, medication: str) -> str:
        """Analyze potential adverse events for a medication."""
        try:
            # Query FDA adverse event database
            adverse_events = self.fda_client.get_adverse_events(medication)
            
            if not adverse_events:
                return f"No adverse event data found for {medication}"
            
            # Analyze adverse event patterns
            analysis = {
                "total_reports": len(adverse_events),
                "common_adverse_events": self._identify_common_adverse_events(adverse_events),
                "serious_adverse_events": self._identify_serious_adverse_events(adverse_events),
                "demographic_patterns": self._analyze_demographic_patterns(adverse_events),
                "temporal_patterns": self._analyze_temporal_patterns(adverse_events)
            }
            
            return f"Adverse event analysis for {medication}: {analysis}"
            
        except Exception as e:
            logger.error(f"Error analyzing adverse events: {e}")
            return f"Error analyzing adverse events: {str(e)}"
    
    def _check_dosage_appropriateness(self, medication_dosage: str) -> str:
        """Check if medication dosage is appropriate."""
        try:
            # Parse medication and dosage
            med_data = self._parse_medication_dosage(medication_dosage)
            medication = med_data["medication"]
            dosage = med_data["dosage"]
            patient_factors = med_data.get("patient_factors", {})
            
            # Get standard dosing information
            standard_dosing = self._get_standard_dosing(medication)
            
            # Assess appropriateness
            assessment = {
                "medication": medication,
                "prescribed_dosage": dosage,
                "standard_dosing": standard_dosing,
                "appropriateness": "appropriate",  # Default
                "recommendations": []
            }
            
            # Check for dosage issues
            if self._is_dosage_too_high(dosage, standard_dosing):
                assessment["appropriateness"] = "too_high"
                assessment["recommendations"].append("Consider dose reduction")
            elif self._is_dosage_too_low(dosage, standard_dosing):
                assessment["appropriateness"] = "too_low"
                assessment["recommendations"].append("Consider dose increase")
            
            # Consider patient factors
            if patient_factors.get("age") and patient_factors["age"] > 65:
                assessment["recommendations"].append("Consider geriatric dosing adjustments")
            
            return f"Dosage appropriateness assessment: {assessment}"
            
        except Exception as e:
            logger.error(f"Error checking dosage appropriateness: {e}")
            return f"Error checking dosage appropriateness: {str(e)}"
    
    def _get_fda_safety_alerts(self, medication: str) -> str:
        """Get FDA safety alerts and recalls for medication."""
        try:
            # Query FDA enforcement and recall data
            safety_alerts = self.fda_client.get_safety_alerts(medication)
            recalls = self.fda_client.get_drug_recalls(medication)
            
            result = {
                "medication": medication,
                "safety_alerts": safety_alerts,
                "recalls": recalls,
                "alert_count": len(safety_alerts),
                "recall_count": len(recalls)
            }
            
            # Assess urgency
            if any(alert.get("classification") == "Class I" for alert in recalls):
                result["urgency"] = "high"
            elif safety_alerts or recalls:
                result["urgency"] = "medium"
            else:
                result["urgency"] = "low"
            
            return f"FDA safety information for {medication}: {result}"
            
        except Exception as e:
            logger.error(f"Error getting FDA safety alerts: {e}")
            return f"Error getting FDA safety alerts: {str(e)}"
    
    def _assess_pregnancy_safety(self, medication: str) -> str:
        """Assess medication safety during pregnancy."""
        try:
            # Get pregnancy category information
            pregnancy_info = self._get_pregnancy_category(medication)
            
            assessment = {
                "medication": medication,
                "pregnancy_category": pregnancy_info.get("category", "Unknown"),
                "risk_summary": pregnancy_info.get("risk_summary", ""),
                "recommendations": []
            }
            
            # Provide recommendations based on category
            category = pregnancy_info.get("category", "")
            if category in ["D", "X"]:
                assessment["recommendations"].append("Contraindicated in pregnancy")
            elif category == "C":
                assessment["recommendations"].append("Use only if benefits outweigh risks")
            elif category in ["A", "B"]:
                assessment["recommendations"].append("Generally considered safe")
            
            return f"Pregnancy safety assessment for {medication}: {assessment}"
            
        except Exception as e:
            logger.error(f"Error assessing pregnancy safety: {e}")
            return f"Error assessing pregnancy safety: {str(e)}"
    
    def _check_renal_dosing(self, medication_and_function: str) -> str:
        """Check renal dosing adjustments."""
        try:
            # Parse medication and renal function
            data = self._parse_medication_and_renal_function(medication_and_function)
            medication = data["medication"]
            creatinine_clearance = data.get("creatinine_clearance")
            
            # Get renal dosing guidelines
            renal_dosing = self._get_renal_dosing_guidelines(medication)
            
            adjustment = {
                "medication": medication,
                "creatinine_clearance": creatinine_clearance,
                "dosing_adjustment": "none",
                "recommendations": []
            }
            
            # Determine if adjustment needed
            if creatinine_clearance and creatinine_clearance < 60:
                adjustment["dosing_adjustment"] = "required"
                adjustment["recommendations"].append("Reduce dose for renal impairment")
            
            return f"Renal dosing assessment: {adjustment}"
            
        except Exception as e:
            logger.error(f"Error checking renal dosing: {e}")
            return f"Error checking renal dosing: {str(e)}"
    
    def _check_hepatic_dosing(self, medication_and_function: str) -> str:
        """Check hepatic dosing adjustments."""
        try:
            # Parse medication and hepatic function
            data = self._parse_medication_and_hepatic_function(medication_and_function)
            medication = data["medication"]
            liver_function = data.get("liver_function", "normal")
            
            # Get hepatic dosing guidelines
            hepatic_dosing = self._get_hepatic_dosing_guidelines(medication)
            
            adjustment = {
                "medication": medication,
                "liver_function": liver_function,
                "dosing_adjustment": "none",
                "recommendations": []
            }
            
            # Determine if adjustment needed
            if liver_function in ["mild_impairment", "moderate_impairment", "severe_impairment"]:
                adjustment["dosing_adjustment"] = "required"
                adjustment["recommendations"].append("Adjust dose for hepatic impairment")
            
            return f"Hepatic dosing assessment: {adjustment}"
            
        except Exception as e:
            logger.error(f"Error checking hepatic dosing: {e}")
            return f"Error checking hepatic dosing: {str(e)}"
    
    def _analyze_polypharmacy_risk(self, medications: str) -> str:
        """Analyze risks associated with polypharmacy."""
        try:
            med_list = self._parse_medication_list(medications)
            
            risk_analysis = {
                "medication_count": len(med_list),
                "polypharmacy_risk": "low",
                "risk_factors": [],
                "recommendations": []
            }
            
            # Assess polypharmacy risk
            if len(med_list) >= 10:
                risk_analysis["polypharmacy_risk"] = "high"
                risk_analysis["risk_factors"].append("High medication count (≥10)")
            elif len(med_list) >= 5:
                risk_analysis["polypharmacy_risk"] = "moderate"
                risk_analysis["risk_factors"].append("Moderate medication count (5-9)")
            
            # Check for high-risk medication classes
            high_risk_classes = ["anticoagulants", "antipsychotics", "benzodiazepines"]
            for med in med_list:
                if any(risk_class in med.lower() for risk_class in high_risk_classes):
                    risk_analysis["risk_factors"].append(f"High-risk medication: {med}")
            
            # Provide recommendations
            if risk_analysis["polypharmacy_risk"] in ["moderate", "high"]:
                risk_analysis["recommendations"].extend([
                    "Review medication necessity",
                    "Consider deprescribing opportunities",
                    "Monitor for drug interactions"
                ])
            
            return f"Polypharmacy risk analysis: {risk_analysis}"
            
        except Exception as e:
            logger.error(f"Error analyzing polypharmacy risk: {e}")
            return f"Error analyzing polypharmacy risk: {str(e)}"
    
    def _get_drug_monitoring_requirements(self, medication: str) -> str:
        """Get monitoring requirements for medication."""
        try:
            # Get monitoring guidelines
            monitoring = self._get_monitoring_guidelines(medication)
            
            requirements = {
                "medication": medication,
                "laboratory_monitoring": monitoring.get("lab_tests", []),
                "clinical_monitoring": monitoring.get("clinical_signs", []),
                "frequency": monitoring.get("frequency", "as clinically indicated"),
                "duration": monitoring.get("duration", "throughout therapy")
            }
            
            return f"Monitoring requirements for {medication}: {requirements}"
            
        except Exception as e:
            logger.error(f"Error getting monitoring requirements: {e}")
            return f"Error getting monitoring requirements: {str(e)}"
    
    # Helper Methods
    
    def _load_interaction_database(self) -> Dict[str, Any]:
        """Load drug interaction database (simplified)."""
        return {
            "warfarin": {
                "interactions": [
                    {
                        "drug": "aspirin",
                        "severity": "major",
                        "mechanism": "increased bleeding risk",
                        "management": "monitor INR closely"
                    }
                ]
            }
        }
    
    def _parse_medication_list(self, medications: str) -> List[str]:
        """Parse medication list from string."""
        # Handle various formats
        if isinstance(medications, str):
            return [med.strip() for med in medications.split(",") if med.strip()]
        elif isinstance(medications, list):
            return medications
        else:
            return []
    
    def _find_interaction(self, med1: str, med2: str) -> Optional[Dict[str, Any]]:
        """Find interaction between two medications."""
        # Simplified interaction checking
        med1_lower = med1.lower()
        med2_lower = med2.lower()
        
        # Example interactions
        if ("warfarin" in med1_lower and "aspirin" in med2_lower) or \
           ("aspirin" in med1_lower and "warfarin" in med2_lower):
            return {
                "drug1": med1,
                "drug2": med2,
                "severity": "major",
                "mechanism": "increased bleeding risk",
                "management": "monitor INR closely",
                "evidence_level": "high"
            }
        
        return None
    
    def _parse_medications_and_conditions(self, text: str) -> Dict[str, List[str]]:
        """Parse medications and conditions from text."""
        # Use NLP to extract entities
        entities = extract_medical_entities(text)
        
        return {
            "medications": entities.get("medications", []),
            "conditions": entities.get("conditions", [])
        }
    
    def _find_contraindication(self, medication: str, condition: str) -> Optional[Dict[str, Any]]:
        """Find contraindication between medication and condition."""
        # Simplified contraindication checking
        med_lower = medication.lower()
        cond_lower = condition.lower()
        
        # Example contraindications
        if "metformin" in med_lower and "kidney" in cond_lower:
            return {
                "medication": medication,
                "condition": condition,
                "type": "relative",
                "severity": "moderate",
                "reason": "Risk of lactic acidosis in renal impairment"
            }
        
        return None
    
    # Additional helper methods would be implemented here...
    
    def _calculate_confidence(self, output: Dict[str, Any]) -> ConfidenceLevel:
        """Calculate confidence level for drug safety analysis."""
        safety_score = output.get("safety_score", 0.5)
        
        if safety_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif safety_score >= 0.75:
            return ConfidenceLevel.HIGH
        elif safety_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif safety_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _requires_escalation(self, output: Dict[str, Any]) -> bool:
        """Determine if safety findings require escalation."""
        # Check for critical interactions or contraindications
        interactions = output.get("drug_interactions", [])
        contraindications = output.get("contraindications", [])
        
        # Escalate for critical safety issues
        for interaction in interactions:
            if interaction.get("severity") == "critical":
                return True
        
        for contraindication in contraindications:
            if contraindication.get("type") == "absolute":
                return True
        
        return False
    
    # Placeholder methods for complex functionality
    def _identify_common_adverse_events(self, events): return []
    def _identify_serious_adverse_events(self, events): return []
    def _analyze_demographic_patterns(self, events): return {}
    def _analyze_temporal_patterns(self, events): return {}
    def _parse_medication_dosage(self, text): return {"medication": "", "dosage": ""}
    def _get_standard_dosing(self, med): return {}
    def _is_dosage_too_high(self, dosage, standard): return False
    def _is_dosage_too_low(self, dosage, standard): return False
    def _get_pregnancy_category(self, med): return {}
    def _parse_medication_and_renal_function(self, text): return {"medication": ""}
    def _get_renal_dosing_guidelines(self, med): return {}
    def _parse_medication_and_hepatic_function(self, text): return {"medication": ""}
    def _get_hepatic_dosing_guidelines(self, med): return {}
    def _get_monitoring_guidelines(self, med): return {}
    
    # Output extraction methods
    def _extract_safety_summary(self, text): return {}
    def _extract_drug_interactions(self, text): return []
    def _extract_contraindications(self, text): return []
    def _extract_adverse_events(self, text): return []
    def _extract_dosage_recommendations(self, text): return []
    def _extract_monitoring_requirements(self, text): return []
    def _extract_fda_alerts(self, text): return []
    def _extract_risk_assessment(self, text): return {}
    def _extract_safety_recommendations(self, text): return []
    def _determine_analysis_scope(self, steps): return []
    def _calculate_safety_score(self, output): return 0.8
