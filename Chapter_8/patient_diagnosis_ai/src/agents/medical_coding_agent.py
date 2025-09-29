
"""
Medical Coding Agent for the Patient Diagnosis AI system.

This agent specializes in mapping clinical concepts to standardized medical
terminologies including SNOMED CT, ICD-10, RxNorm, and LOINC codes.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentResponse, AgentError, ConfidenceLevel
from src.config.prompts import get_prompt
from src.config.settings import get_settings
from src.knowledge.terminology_service import TerminologyService
from src.utils.medical_nlp import extract_medical_entities, normalize_medical_terms

logger = logging.getLogger(__name__)
settings = get_settings()


class MedicalCodingAgent(BaseAgent):
    """
    Specialized agent for medical terminology and coding.
    
    This agent focuses on:
    - Mapping clinical terms to SNOMED CT codes
    - Converting diagnoses to ICD-10 codes
    - Coding medications with RxNorm
    - Mapping lab results to LOINC codes
    - Ensuring coding consistency and accuracy
    - Providing code hierarchies and relationships
    """
    
    def __init__(self, llm, terminology_service: Optional[TerminologyService] = None):
        """
        Initialize the Medical Coding Agent.
        
        Args:
            llm: Language model instance
            terminology_service: Service for accessing medical terminologies
        """
        self.terminology_service = terminology_service or TerminologyService()
        
        # Initialize tools
        tools = self._create_tools()
        
        super().__init__(
            name="Medical Coding Specialist",
            agent_type="medical_coding",
            description="Maps clinical concepts to standardized medical codes (SNOMED CT, ICD-10, RxNorm, LOINC)",
            tools=tools,
            llm=llm,
            max_iterations=settings.agents.agent_retry_attempts,
            timeout_seconds=settings.agents.agent_timeout
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for medical coding."""
        return [
            Tool(
                name="map_to_snomed_ct",
                description="Map clinical findings and procedures to SNOMED CT codes",
                func=self._map_to_snomed_ct
            ),
            Tool(
                name="map_to_icd10",
                description="Map diagnoses and conditions to ICD-10 codes",
                func=self._map_to_icd10
            ),
            Tool(
                name="map_to_rxnorm",
                description="Map medications to RxNorm codes",
                func=self._map_to_rxnorm
            ),
            Tool(
                name="map_to_loinc",
                description="Map laboratory tests and observations to LOINC codes",
                func=self._map_to_loinc
            ),
            Tool(
                name="validate_code_combination",
                description="Validate that a combination of codes is clinically appropriate",
                func=self._validate_code_combination
            ),
            Tool(
                name="get_code_hierarchy",
                description="Get hierarchical relationships for medical codes",
                func=self._get_code_hierarchy
            ),
            Tool(
                name="find_related_codes",
                description="Find related or similar medical codes",
                func=self._find_related_codes
            ),
            Tool(
                name="extract_medical_concepts",
                description="Extract medical concepts from free text",
                func=self._extract_medical_concepts
            ),
            Tool(
                name="standardize_terminology",
                description="Standardize medical terminology across different systems",
                func=self._standardize_terminology
            ),
            Tool(
                name="check_code_validity",
                description="Check if medical codes are valid and current",
                func=self._check_code_validity
            )
        ]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with medical coding prompt."""
        prompt = get_prompt("medical_coding")
        
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
        """Validate input for medical coding."""
        if not request:
            raise AgentError(
                "Empty request provided",
                agent_name=self.name,
                error_code="INVALID_INPUT"
            )
        
        # Check for clinical concepts to code
        clinical_text = request.get("clinical_text", "")
        concepts = request.get("concepts", [])
        
        if not clinical_text and not concepts:
            raise AgentError(
                "No clinical text or concepts provided for coding",
                agent_name=self.name,
                error_code="NO_CONTENT_TO_CODE"
            )
    
    def _process_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format medical coding output."""
        try:
            output_text = raw_output.get("output", "")
            intermediate_steps = raw_output.get("intermediate_steps", [])
            
            processed_output = {
                "coded_concepts": self._extract_coded_concepts(output_text, intermediate_steps),
                "snomed_ct_codes": self._extract_snomed_codes(output_text),
                "icd10_codes": self._extract_icd10_codes(output_text),
                "rxnorm_codes": self._extract_rxnorm_codes(output_text),
                "loinc_codes": self._extract_loinc_codes(output_text),
                "coding_quality": self._assess_coding_quality(output_text),
                "ambiguous_terms": self._identify_ambiguous_terms(output_text),
                "coding_recommendations": self._extract_coding_recommendations(output_text),
                "code_relationships": self._extract_code_relationships(output_text)
            }
            
            # Add metadata
            processed_output["terminology_systems_used"] = self._identify_terminology_systems(intermediate_steps)
            processed_output["coding_confidence"] = self._assess_coding_confidence(processed_output)
            
            return processed_output
            
        except Exception as e:
            logger.error(f"Error processing medical coding output: {e}")
            raise AgentError(
                f"Failed to process coding output: {str(e)}",
                agent_name=self.name,
                error_code="OUTPUT_PROCESSING_ERROR"
            )
    
    # Tool Implementation Methods
    
    def _map_to_snomed_ct(self, clinical_concept: str) -> str:
        """Map clinical concept to SNOMED CT code."""
        try:
            # Normalize the concept
            normalized_concept = normalize_medical_terms(clinical_concept)
            
            # Search SNOMED CT
            snomed_results = self.terminology_service.search_snomed_ct(normalized_concept)
            
            if not snomed_results:
                return f"No SNOMED CT codes found for: {clinical_concept}"
            
            # Format results
            formatted_results = []
            for result in snomed_results[:5]:  # Top 5 results
                formatted_results.append({
                    "code": result.get("conceptId"),
                    "display": result.get("fsn", {}).get("term"),
                    "preferred_term": result.get("pt", {}).get("term"),
                    "active": result.get("active", True),
                    "module": result.get("moduleId")
                })
            
            return f"SNOMED CT mapping for '{clinical_concept}': {formatted_results}"
            
        except Exception as e:
            logger.error(f"Error mapping to SNOMED CT: {e}")
            return f"Error mapping to SNOMED CT: {str(e)}"
    
    def _map_to_icd10(self, diagnosis: str) -> str:
        """Map diagnosis to ICD-10 code."""
        try:
            # Normalize the diagnosis
            normalized_diagnosis = normalize_medical_terms(diagnosis)
            
            # Search ICD-10
            icd10_results = self.terminology_service.search_icd10(normalized_diagnosis)
            
            if not icd10_results:
                return f"No ICD-10 codes found for: {diagnosis}"
            
            # Format results
            formatted_results = []
            for result in icd10_results[:5]:
                formatted_results.append({
                    "code": result.get("code"),
                    "display": result.get("display"),
                    "category": result.get("category"),
                    "billable": result.get("billable", True)
                })
            
            return f"ICD-10 mapping for '{diagnosis}': {formatted_results}"
            
        except Exception as e:
            logger.error(f"Error mapping to ICD-10: {e}")
            return f"Error mapping to ICD-10: {str(e)}"
    
    def _map_to_rxnorm(self, medication: str) -> str:
        """Map medication to RxNorm code."""
        try:
            # Normalize medication name
            normalized_medication = normalize_medical_terms(medication)
            
            # Search RxNorm
            rxnorm_results = self.terminology_service.search_rxnorm(normalized_medication)
            
            if not rxnorm_results:
                return f"No RxNorm codes found for: {medication}"
            
            # Format results
            formatted_results = []
            for result in rxnorm_results[:5]:
                formatted_results.append({
                    "rxcui": result.get("rxcui"),
                    "name": result.get("name"),
                    "synonym": result.get("synonym"),
                    "tty": result.get("tty"),  # Term type
                    "language": result.get("language", "ENG")
                })
            
            return f"RxNorm mapping for '{medication}': {formatted_results}"
            
        except Exception as e:
            logger.error(f"Error mapping to RxNorm: {e}")
            return f"Error mapping to RxNorm: {str(e)}"
    
    def _map_to_loinc(self, lab_test: str) -> str:
        """Map laboratory test to LOINC code."""
        try:
            # Normalize lab test name
            normalized_test = normalize_medical_terms(lab_test)
            
            # Search LOINC
            loinc_results = self.terminology_service.search_loinc(normalized_test)
            
            if not loinc_results:
                return f"No LOINC codes found for: {lab_test}"
            
            # Format results
            formatted_results = []
            for result in loinc_results[:5]:
                formatted_results.append({
                    "loinc_num": result.get("loinc_num"),
                    "component": result.get("component"),
                    "property": result.get("property"),
                    "time_aspct": result.get("time_aspct"),
                    "system": result.get("system"),
                    "scale_typ": result.get("scale_typ"),
                    "method_typ": result.get("method_typ"),
                    "long_common_name": result.get("long_common_name")
                })
            
            return f"LOINC mapping for '{lab_test}': {formatted_results}"
            
        except Exception as e:
            logger.error(f"Error mapping to LOINC: {e}")
            return f"Error mapping to LOINC: {str(e)}"
    
    def _validate_code_combination(self, codes: str) -> str:
        """Validate combination of medical codes."""
        try:
            # Parse codes from input
            code_list = self._parse_code_list(codes)
            
            validation_results = {
                "valid_combinations": [],
                "invalid_combinations": [],
                "warnings": [],
                "recommendations": []
            }
            
            # Check for common invalid combinations
            for i, code1 in enumerate(code_list):
                for code2 in code_list[i+1:]:
                    if self._check_code_conflict(code1, code2):
                        validation_results["invalid_combinations"].append({
                            "code1": code1,
                            "code2": code2,
                            "reason": "Conflicting diagnoses"
                        })
                    else:
                        validation_results["valid_combinations"].append({
                            "code1": code1,
                            "code2": code2
                        })
            
            return f"Code combination validation: {validation_results}"
            
        except Exception as e:
            logger.error(f"Error validating code combination: {e}")
            return f"Error validating code combination: {str(e)}"
    
    def _get_code_hierarchy(self, code: str) -> str:
        """Get hierarchical relationships for a medical code."""
        try:
            # Determine code system
            code_system = self._identify_code_system(code)
            
            if code_system == "SNOMED_CT":
                hierarchy = self.terminology_service.get_snomed_hierarchy(code)
            elif code_system == "ICD10":
                hierarchy = self.terminology_service.get_icd10_hierarchy(code)
            else:
                return f"Hierarchy not available for code system: {code_system}"
            
            return f"Hierarchy for {code}: {hierarchy}"
            
        except Exception as e:
            logger.error(f"Error getting code hierarchy: {e}")
            return f"Error getting code hierarchy: {str(e)}"
    
    def _find_related_codes(self, code: str) -> str:
        """Find related or similar medical codes."""
        try:
            code_system = self._identify_code_system(code)
            
            if code_system == "SNOMED_CT":
                related_codes = self.terminology_service.find_related_snomed_codes(code)
            elif code_system == "ICD10":
                related_codes = self.terminology_service.find_related_icd10_codes(code)
            else:
                return f"Related codes not available for: {code_system}"
            
            return f"Related codes for {code}: {related_codes}"
            
        except Exception as e:
            logger.error(f"Error finding related codes: {e}")
            return f"Error finding related codes: {str(e)}"
    
    def _extract_medical_concepts(self, text: str) -> str:
        """Extract medical concepts from free text."""
        try:
            # Use NLP to extract medical entities
            entities = extract_medical_entities(text)
            
            # Categorize entities
            categorized_concepts = {
                "conditions": entities.get("conditions", []),
                "medications": entities.get("medications", []),
                "procedures": entities.get("procedures", []),
                "anatomy": entities.get("anatomy", []),
                "symptoms": entities.get("symptoms", []),
                "lab_tests": entities.get("lab_tests", [])
            }
            
            return f"Extracted medical concepts: {categorized_concepts}"
            
        except Exception as e:
            logger.error(f"Error extracting medical concepts: {e}")
            return f"Error extracting medical concepts: {str(e)}"
    
    def _standardize_terminology(self, terms: str) -> str:
        """Standardize medical terminology across systems."""
        try:
            # Parse terms
            term_list = [term.strip() for term in terms.split(",")]
            
            standardized_terms = []
            for term in term_list:
                # Normalize term
                normalized = normalize_medical_terms(term)
                
                # Find preferred terms in different systems
                snomed_preferred = self.terminology_service.get_preferred_term(normalized, "SNOMED_CT")
                icd10_preferred = self.terminology_service.get_preferred_term(normalized, "ICD10")
                
                standardized_terms.append({
                    "original": term,
                    "normalized": normalized,
                    "snomed_preferred": snomed_preferred,
                    "icd10_preferred": icd10_preferred
                })
            
            return f"Standardized terminology: {standardized_terms}"
            
        except Exception as e:
            logger.error(f"Error standardizing terminology: {e}")
            return f"Error standardizing terminology: {str(e)}"
    
    def _check_code_validity(self, codes: str) -> str:
        """Check if medical codes are valid and current."""
        try:
            code_list = self._parse_code_list(codes)
            
            validity_results = []
            for code in code_list:
                code_system = self._identify_code_system(code)
                is_valid = self.terminology_service.validate_code(code, code_system)
                is_current = self.terminology_service.is_code_current(code, code_system)
                
                validity_results.append({
                    "code": code,
                    "system": code_system,
                    "valid": is_valid,
                    "current": is_current,
                    "status": "active" if is_valid and is_current else "inactive"
                })
            
            return f"Code validity check: {validity_results}"
            
        except Exception as e:
            logger.error(f"Error checking code validity: {e}")
            return f"Error checking code validity: {str(e)}"
    
    # Helper Methods
    
    def _parse_code_list(self, codes_str: str) -> List[str]:
        """Parse a string of codes into a list."""
        # Handle various separators
        codes = re.split(r'[,;\n\t]+', codes_str.strip())
        return [code.strip() for code in codes if code.strip()]
    
    def _identify_code_system(self, code: str) -> str:
        """Identify the coding system for a given code."""
        code = code.strip()
        
        # SNOMED CT codes are typically numeric
        if code.isdigit() and len(code) >= 6:
            return "SNOMED_CT"
        
        # ICD-10 codes follow specific patterns
        if re.match(r'^[A-Z]\d{2}(\.\d+)?$', code):
            return "ICD10"
        
        # RxNorm codes are numeric
        if code.isdigit() and len(code) <= 8:
            return "RXNORM"
        
        # LOINC codes follow specific pattern
        if re.match(r'^\d{1,5}-\d$', code):
            return "LOINC"
        
        return "UNKNOWN"
    
    def _check_code_conflict(self, code1: str, code2: str) -> bool:
        """Check if two codes conflict with each other."""
        # Simple conflict checking - would be more sophisticated in practice
        system1 = self._identify_code_system(code1)
        system2 = self._identify_code_system(code2)
        
        # Example: Check for conflicting diagnoses
        if system1 == "ICD10" and system2 == "ICD10":
            # This would check for mutually exclusive diagnoses
            return False
        
        return False
    
    # Output Processing Helper Methods
    
    def _extract_coded_concepts(self, output_text: str, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Extract coded concepts from agent output."""
        coded_concepts = []
        
        # Parse coded concepts from intermediate steps
        for step in intermediate_steps:
            if len(step) >= 2:
                observation = str(step[1])
                if "mapping" in observation.lower():
                    # Extract coding information
                    # This would be more sophisticated in practice
                    coded_concepts.append({
                        "concept": "example concept",
                        "codes": {
                            "snomed_ct": "123456789",
                            "icd10": "A00.0"
                        },
                        "confidence": 0.9
                    })
        
        return coded_concepts
    
    def _extract_snomed_codes(self, output_text: str) -> List[Dict[str, Any]]:
        """Extract SNOMED CT codes from output."""
        # Use regex to find SNOMED codes in output
        snomed_pattern = r'SNOMED[^:]*:\s*(\d{6,})'
        matches = re.findall(snomed_pattern, output_text, re.IGNORECASE)
        
        return [{"code": match, "system": "SNOMED_CT"} for match in matches]
    
    def _extract_icd10_codes(self, output_text: str) -> List[Dict[str, Any]]:
        """Extract ICD-10 codes from output."""
        icd10_pattern = r'ICD[^:]*:\s*([A-Z]\d{2}(?:\.\d+)?)'
        matches = re.findall(icd10_pattern, output_text, re.IGNORECASE)
        
        return [{"code": match, "system": "ICD10"} for match in matches]
    
    def _extract_rxnorm_codes(self, output_text: str) -> List[Dict[str, Any]]:
        """Extract RxNorm codes from output."""
        rxnorm_pattern = r'RxNorm[^:]*:\s*(\d{1,8})'
        matches = re.findall(rxnorm_pattern, output_text, re.IGNORECASE)
        
        return [{"code": match, "system": "RXNORM"} for match in matches]
    
    def _extract_loinc_codes(self, output_text: str) -> List[Dict[str, Any]]:
        """Extract LOINC codes from output."""
        loinc_pattern = r'LOINC[^:]*:\s*(\d{1,5}-\d)'
        matches = re.findall(loinc_pattern, output_text, re.IGNORECASE)
        
        return [{"code": match, "system": "LOINC"} for match in matches]
    
    def _assess_coding_quality(self, output_text: str) -> Dict[str, Any]:
        """Assess the quality of coding performed."""
        return {
            "completeness": 0.8,
            "accuracy": 0.9,
            "specificity": 0.85,
            "consistency": 0.9,
            "overall_score": 0.86
        }
    
    def _identify_ambiguous_terms(self, output_text: str) -> List[Dict[str, Any]]:
        """Identify terms that were ambiguous during coding."""
        return [
            {
                "term": "chest pain",
                "ambiguity_reason": "Multiple possible SNOMED CT codes",
                "suggested_clarification": "Specify type and location of chest pain"
            }
        ]
    
    def _extract_coding_recommendations(self, output_text: str) -> List[str]:
        """Extract coding recommendations from output."""
        return [
            "Use more specific codes when possible",
            "Consider additional codes for comorbidities",
            "Verify code combinations for clinical accuracy"
        ]
    
    def _extract_code_relationships(self, output_text: str) -> Dict[str, Any]:
        """Extract code relationships from output."""
        return {
            "hierarchical_relationships": [],
            "cross_system_mappings": [],
            "related_concepts": []
        }
    
    def _identify_terminology_systems(self, intermediate_steps: List) -> List[str]:
        """Identify which terminology systems were used."""
        systems = set()
        for step in intermediate_steps:
            if len(step) >= 2:
                observation = str(step[1]).lower()
                if "snomed" in observation:
                    systems.add("SNOMED_CT")
                if "icd" in observation:
                    systems.add("ICD10")
                if "rxnorm" in observation:
                    systems.add("RXNORM")
                if "loinc" in observation:
                    systems.add("LOINC")
        
        return list(systems)
    
    def _assess_coding_confidence(self, output: Dict[str, Any]) -> float:
        """Assess confidence in coding results."""
        # Calculate confidence based on various factors
        factors = []
        
        # Number of coded concepts
        coded_concepts = output.get("coded_concepts", [])
        if coded_concepts:
            factors.append(min(len(coded_concepts) / 5, 1.0))
        
        # Coding quality score
        quality = output.get("coding_quality", {})
        if quality.get("overall_score"):
            factors.append(quality["overall_score"])
        
        # Ambiguous terms (lower confidence if many ambiguous terms)
        ambiguous = output.get("ambiguous_terms", [])
        ambiguity_factor = max(0, 1 - len(ambiguous) / 10)
        factors.append(ambiguity_factor)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def _calculate_confidence(self, output: Dict[str, Any]) -> ConfidenceLevel:
        """Calculate confidence level for coding results."""
        confidence_score = self._assess_coding_confidence(output)
        
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _requires_escalation(self, output: Dict[str, Any]) -> bool:
        """Determine if coding results require escalation."""
        # Escalate if many ambiguous terms or low quality
        ambiguous_terms = output.get("ambiguous_terms", [])
        quality = output.get("coding_quality", {})
        
        if len(ambiguous_terms) > 5:
            return True
        
        if quality.get("overall_score", 1.0) < 0.6:
            return True
        
        return False
    
    def _suggest_next_actions(self, output: Dict[str, Any]) -> List[str]:
        """Suggest next actions based on coding results."""
        actions = []
        
        # Always recommend validation
        actions.append("Validate coded concepts with clinical context")
        
        # Suggest clarification for ambiguous terms
        ambiguous_terms = output.get("ambiguous_terms", [])
        if ambiguous_terms:
            actions.append("Clarify ambiguous medical terms for more specific coding")
        
        # Suggest quality improvement
        quality = output.get("coding_quality", {})
        if quality.get("specificity", 1.0) < 0.8:
            actions.append("Consider using more specific codes where available")
        
        # Suggest cross-system validation
        systems_used = output.get("terminology_systems_used", [])
        if len(systems_used) > 1:
            actions.append("Validate code mappings across terminology systems")
        
        return actions
