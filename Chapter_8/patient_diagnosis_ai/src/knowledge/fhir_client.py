
"""
FHIR client for healthcare data integration.

This module provides a client for accessing FHIR-compliant healthcare
data sources and Electronic Health Record (EHR) systems.
"""

import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FHIRClient:
    """
    Client for accessing FHIR-compliant healthcare data sources.
    
    This client provides methods to interact with FHIR servers,
    retrieve patient data, and search for healthcare resources.
    """
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """
        Initialize FHIR client.
        
        Args:
            base_url: Base URL of the FHIR server
            api_key: API key for authentication
        """
        self.base_url = base_url or "https://hapi.fhir.org/baseR4"  # Public test server
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set up authentication headers
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/fhir+json',
                'Accept': 'application/fhir+json'
            })
        else:
            self.session.headers.update({
                'Content-Type': 'application/fhir+json',
                'Accept': 'application/fhir+json'
            })
        
        logger.info(f"FHIR client initialized with base URL: {self.base_url}")
    
    def search_resources(
        self,
        resource_type: str,
        patient_id: Optional[str] = None,
        parameters: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for FHIR resources.
        
        Args:
            resource_type: Type of FHIR resource (e.g., 'Patient', 'Condition')
            patient_id: Optional patient ID to filter by
            parameters: Additional search parameters
            
        Returns:
            List[Dict[str, Any]]: List of FHIR resources
        """
        try:
            url = f"{self.base_url}/{resource_type}"
            params = parameters or {}
            
            if patient_id:
                params['patient'] = patient_id
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            bundle = response.json()
            
            # Extract entries from bundle
            resources = []
            if 'entry' in bundle:
                for entry in bundle['entry']:
                    if 'resource' in entry:
                        resources.append(entry['resource'])
            
            logger.info(f"Retrieved {len(resources)} {resource_type} resources")
            return resources
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching FHIR resources: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in FHIR search: {e}")
            return []
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get patient resource by ID.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict[str, Any]: Patient resource or None if not found
        """
        try:
            url = f"{self.base_url}/Patient/{patient_id}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 404:
                logger.warning(f"Patient {patient_id} not found")
                return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving patient {patient_id}: {e}")
            return None
    
    def get_patient_conditions(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get conditions for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List[Dict[str, Any]]: List of condition resources
        """
        return self.search_resources("Condition", patient_id=patient_id)
    
    def get_patient_medications(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get medications for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List[Dict[str, Any]]: List of medication resources
        """
        # Search both MedicationRequest and MedicationStatement
        medication_requests = self.search_resources("MedicationRequest", patient_id=patient_id)
        medication_statements = self.search_resources("MedicationStatement", patient_id=patient_id)
        
        return medication_requests + medication_statements
    
    def get_patient_observations(self, patient_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get observations for a patient.
        
        Args:
            patient_id: Patient identifier
            category: Optional category filter (e.g., 'vital-signs', 'laboratory')
            
        Returns:
            List[Dict[str, Any]]: List of observation resources
        """
        parameters = {}
        if category:
            parameters['category'] = category
        
        return self.search_resources("Observation", patient_id=patient_id, parameters=parameters)
    
    def get_patient_encounters(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get encounters for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List[Dict[str, Any]]: List of encounter resources
        """
        return self.search_resources("Encounter", patient_id=patient_id)
    
    def get_patient_allergies(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get allergies for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List[Dict[str, Any]]: List of allergy intolerance resources
        """
        return self.search_resources("AllergyIntolerance", patient_id=patient_id)
    
    def get_patient_procedures(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get procedures for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List[Dict[str, Any]]: List of procedure resources
        """
        return self.search_resources("Procedure", patient_id=patient_id)
    
    def get_patient_diagnostic_reports(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get diagnostic reports for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List[Dict[str, Any]]: List of diagnostic report resources
        """
        return self.search_resources("DiagnosticReport", patient_id=patient_id)
    
    def get_comprehensive_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """
        Get comprehensive patient data including all relevant resources.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict[str, Any]: Comprehensive patient data
        """
        logger.info(f"Retrieving comprehensive data for patient {patient_id}")
        
        patient_data = {
            "patient": self.get_patient(patient_id),
            "conditions": self.get_patient_conditions(patient_id),
            "medications": self.get_patient_medications(patient_id),
            "observations": self.get_patient_observations(patient_id),
            "encounters": self.get_patient_encounters(patient_id),
            "allergies": self.get_patient_allergies(patient_id),
            "procedures": self.get_patient_procedures(patient_id),
            "diagnostic_reports": self.get_patient_diagnostic_reports(patient_id)
        }
        
        # Add summary statistics
        patient_data["summary"] = {
            "conditions_count": len(patient_data["conditions"]),
            "medications_count": len(patient_data["medications"]),
            "observations_count": len(patient_data["observations"]),
            "encounters_count": len(patient_data["encounters"]),
            "allergies_count": len(patient_data["allergies"]),
            "procedures_count": len(patient_data["procedures"]),
            "diagnostic_reports_count": len(patient_data["diagnostic_reports"]),
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
        return patient_data
    
    def search_patients(self, search_params: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Search for patients based on criteria.
        
        Args:
            search_params: Search parameters (e.g., {'family': 'Smith', 'given': 'John'})
            
        Returns:
            List[Dict[str, Any]]: List of patient resources
        """
        return self.search_resources("Patient", parameters=search_params)
    
    def create_resource(self, resource_type: str, resource_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new FHIR resource.
        
        Args:
            resource_type: Type of FHIR resource
            resource_data: Resource data
            
        Returns:
            Dict[str, Any]: Created resource or None if failed
        """
        try:
            url = f"{self.base_url}/{resource_type}"
            response = self.session.post(url, json=resource_data, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Created {resource_type} resource")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating FHIR resource: {e}")
            return None
    
    def update_resource(self, resource_type: str, resource_id: str, resource_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing FHIR resource.
        
        Args:
            resource_type: Type of FHIR resource
            resource_id: Resource identifier
            resource_data: Updated resource data
            
        Returns:
            Dict[str, Any]: Updated resource or None if failed
        """
        try:
            url = f"{self.base_url}/{resource_type}/{resource_id}"
            response = self.session.put(url, json=resource_data, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Updated {resource_type} resource {resource_id}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error updating FHIR resource: {e}")
            return None
    
    def extract_patient_summary(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a summary from comprehensive patient data.
        
        Args:
            patient_data: Comprehensive patient data from get_comprehensive_patient_data
            
        Returns:
            Dict[str, Any]: Patient summary
        """
        summary = {
            "patient_id": None,
            "demographics": {},
            "active_conditions": [],
            "current_medications": [],
            "recent_observations": [],
            "allergies": [],
            "last_encounter": None
        }
        
        # Extract patient demographics
        if patient_data.get("patient"):
            patient = patient_data["patient"]
            summary["patient_id"] = patient.get("id")
            
            if "name" in patient and patient["name"]:
                name = patient["name"][0]
                summary["demographics"]["name"] = {
                    "family": name.get("family", ""),
                    "given": name.get("given", [])
                }
            
            summary["demographics"]["gender"] = patient.get("gender")
            summary["demographics"]["birth_date"] = patient.get("birthDate")
        
        # Extract active conditions
        for condition in patient_data.get("conditions", []):
            if condition.get("clinicalStatus", {}).get("coding", [{}])[0].get("code") == "active":
                condition_summary = {
                    "code": condition.get("code", {}).get("coding", [{}])[0].get("code"),
                    "display": condition.get("code", {}).get("coding", [{}])[0].get("display"),
                    "onset_date": condition.get("onsetDateTime")
                }
                summary["active_conditions"].append(condition_summary)
        
        # Extract current medications
        for medication in patient_data.get("medications", []):
            if medication.get("status") in ["active", "intended"]:
                med_summary = {
                    "medication": self._extract_medication_name(medication),
                    "dosage": self._extract_dosage_instruction(medication),
                    "status": medication.get("status")
                }
                summary["current_medications"].append(med_summary)
        
        # Extract recent vital signs
        vital_observations = [obs for obs in patient_data.get("observations", [])
                            if self._is_vital_sign(obs)]
        vital_observations.sort(key=lambda x: x.get("effectiveDateTime", ""), reverse=True)
        summary["recent_observations"] = vital_observations[:10]  # Last 10 vital signs
        
        # Extract allergies
        for allergy in patient_data.get("allergies", []):
            allergy_summary = {
                "substance": self._extract_allergy_substance(allergy),
                "reaction": self._extract_allergy_reaction(allergy),
                "severity": allergy.get("criticality")
            }
            summary["allergies"].append(allergy_summary)
        
        # Find most recent encounter
        encounters = patient_data.get("encounters", [])
        if encounters:
            encounters.sort(key=lambda x: x.get("period", {}).get("start", ""), reverse=True)
            summary["last_encounter"] = encounters[0]
        
        return summary
    
    def _extract_medication_name(self, medication: Dict[str, Any]) -> str:
        """Extract medication name from medication resource."""
        if "medicationCodeableConcept" in medication:
            coding = medication["medicationCodeableConcept"].get("coding", [])
            if coding:
                return coding[0].get("display", "Unknown medication")
        elif "medicationReference" in medication:
            return medication["medicationReference"].get("display", "Unknown medication")
        return "Unknown medication"
    
    def _extract_dosage_instruction(self, medication: Dict[str, Any]) -> str:
        """Extract dosage instruction from medication resource."""
        if "dosageInstruction" in medication and medication["dosageInstruction"]:
            dosage = medication["dosageInstruction"][0]
            return dosage.get("text", "See instructions")
        return "No dosage information"
    
    def _is_vital_sign(self, observation: Dict[str, Any]) -> bool:
        """Check if observation is a vital sign."""
        categories = observation.get("category", [])
        for category in categories:
            for coding in category.get("coding", []):
                if coding.get("code") == "vital-signs":
                    return True
        return False
    
    def _extract_allergy_substance(self, allergy: Dict[str, Any]) -> str:
        """Extract allergy substance from allergy intolerance resource."""
        if "code" in allergy:
            coding = allergy["code"].get("coding", [])
            if coding:
                return coding[0].get("display", "Unknown substance")
        return "Unknown substance"
    
    def _extract_allergy_reaction(self, allergy: Dict[str, Any]) -> List[str]:
        """Extract allergy reactions from allergy intolerance resource."""
        reactions = []
        for reaction in allergy.get("reaction", []):
            for manifestation in reaction.get("manifestation", []):
                for coding in manifestation.get("coding", []):
                    reactions.append(coding.get("display", "Unknown reaction"))
        return reactions
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on FHIR server.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Try to get server metadata
            url = f"{self.base_url}/metadata"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            metadata = response.json()
            
            return {
                "status": "healthy",
                "server_url": self.base_url,
                "fhir_version": metadata.get("fhirVersion"),
                "software": metadata.get("software", {}).get("name"),
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "server_url": self.base_url,
                "error": str(e)
            }
