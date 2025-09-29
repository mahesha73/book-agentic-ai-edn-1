
"""
Medical Natural Language Processing utilities for the Patient Diagnosis AI system.

This module provides NLP utilities specifically designed for medical text processing,
including medical entity extraction, terminology normalization, and clinical text analysis.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans

logger = logging.getLogger(__name__)


@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0
    normalized_form: Optional[str] = None
    codes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.codes is None:
            self.codes = {}


class MedicalNLP:
    """Medical NLP processor for clinical text analysis."""
    
    def __init__(self):
        """Initialize medical NLP processor."""
        try:
            # Load spaCy model (using en_core_web_sm as base)
            self.nlp = spacy.load("en_core_web_sm")
            
            # Add custom medical entity patterns
            self._add_medical_patterns()
            
            # Initialize medical terminology dictionaries
            self._load_medical_dictionaries()
            
            logger.info("Medical NLP processor initialized successfully")
            
        except OSError:
            logger.warning("spaCy model not found, using basic NLP processing")
            self.nlp = None
            self._load_medical_dictionaries()
    
    def _add_medical_patterns(self):
        """Add custom medical entity patterns to spaCy."""
        if not self.nlp:
            return
        
        # Create matcher for medical patterns
        self.matcher = Matcher(self.nlp.vocab)
        
        # Symptom patterns
        symptom_patterns = [
            [{"LOWER": {"IN": ["pain", "ache", "discomfort"]}},
             {"LOWER": {"IN": ["in", "at", "on"]}, "OP": "?"},
             {"POS": "NOUN", "OP": "?"}],
            [{"LOWER": {"IN": ["shortness", "difficulty"]}},
             {"LOWER": "of"},
             {"LOWER": "breath"}],
            [{"LOWER": "chest"}, {"LOWER": "pain"}],
            [{"LOWER": "abdominal"}, {"LOWER": "pain"}],
            [{"LOWER": {"IN": ["nausea", "vomiting", "dizziness", "fatigue", "fever"]}}],
            [{"LOWER": {"IN": ["high", "low"]}}, 
             {"LOWER": {"IN": ["blood", "glucose"]}},
             {"LOWER": {"IN": ["pressure", "sugar"]}}]
        ]
        
        for i, pattern in enumerate(symptom_patterns):
            self.matcher.add(f"SYMPTOM_{i}", [pattern])
        
        # Medication patterns
        medication_patterns = [
            [{"LOWER": {"REGEX": r".*pril$"}}],  # ACE inhibitors
            [{"LOWER": {"REGEX": r".*olol$"}}],  # Beta blockers
            [{"LOWER": {"REGEX": r".*statin$"}}],  # Statins
            [{"LOWER": "aspirin"}],
            [{"LOWER": "metformin"}],
            [{"LOWER": "insulin"}],
            [{"LOWER": {"REGEX": r".*cillin$"}}],  # Penicillins
        ]
        
        for i, pattern in enumerate(medication_patterns):
            self.matcher.add(f"MEDICATION_{i}", [pattern])
        
        # Condition patterns
        condition_patterns = [
            [{"LOWER": "diabetes"}, {"LOWER": "mellitus", "OP": "?"}],
            [{"LOWER": {"IN": ["hypertension", "hypotension"]}}],
            [{"LOWER": "myocardial"}, {"LOWER": "infarction"}],
            [{"LOWER": "heart"}, {"LOWER": {"IN": ["attack", "failure", "disease"]}}],
            [{"LOWER": {"IN": ["pneumonia", "bronchitis", "asthma"]}}],
            [{"LOWER": {"IN": ["cancer", "carcinoma", "tumor", "malignancy"]}}],
        ]
        
        for i, pattern in enumerate(condition_patterns):
            self.matcher.add(f"CONDITION_{i}", [pattern])
    
    def _load_medical_dictionaries(self):
        """Load medical terminology dictionaries."""
        # Symptom dictionary
        self.symptoms_dict = {
            "chest pain", "abdominal pain", "back pain", "headache",
            "shortness of breath", "difficulty breathing", "dyspnea",
            "nausea", "vomiting", "diarrhea", "constipation",
            "fever", "chills", "fatigue", "weakness",
            "dizziness", "lightheadedness", "syncope",
            "palpitations", "irregular heartbeat",
            "cough", "sore throat", "runny nose",
            "rash", "itching", "swelling",
            "joint pain", "muscle pain", "stiffness",
            "blurred vision", "double vision",
            "hearing loss", "tinnitus",
            "urinary frequency", "burning urination",
            "excessive thirst", "excessive urination"
        }
        
        # Medication dictionary
        self.medications_dict = {
            "aspirin", "acetaminophen", "ibuprofen", "naproxen",
            "lisinopril", "enalapril", "captopril",
            "metoprolol", "atenolol", "propranolol",
            "amlodipine", "nifedipine", "diltiazem",
            "simvastatin", "atorvastatin", "rosuvastatin",
            "metformin", "glipizide", "insulin",
            "warfarin", "heparin", "clopidogrel",
            "omeprazole", "ranitidine", "famotidine",
            "albuterol", "prednisone", "azithromycin",
            "amoxicillin", "ciprofloxacin", "doxycycline"
        }
        
        # Condition dictionary
        self.conditions_dict = {
            "diabetes mellitus", "type 1 diabetes", "type 2 diabetes",
            "hypertension", "high blood pressure", "hypotension",
            "myocardial infarction", "heart attack", "angina",
            "heart failure", "congestive heart failure",
            "atrial fibrillation", "arrhythmia",
            "stroke", "transient ischemic attack", "tia",
            "pneumonia", "bronchitis", "asthma", "copd",
            "cancer", "carcinoma", "tumor", "malignancy",
            "arthritis", "osteoarthritis", "rheumatoid arthritis",
            "depression", "anxiety", "bipolar disorder",
            "kidney disease", "renal failure", "dialysis",
            "liver disease", "hepatitis", "cirrhosis"
        }
        
        # Anatomy dictionary
        self.anatomy_dict = {
            "heart", "lung", "liver", "kidney", "brain",
            "stomach", "intestine", "colon", "pancreas",
            "thyroid", "adrenal", "prostate", "uterus",
            "chest", "abdomen", "pelvis", "extremities",
            "head", "neck", "back", "spine",
            "arm", "leg", "hand", "foot",
            "eye", "ear", "nose", "throat"
        }
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[MedicalEntity]]:
        """
        Extract medical entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict[str, List[MedicalEntity]]: Categorized medical entities
        """
        entities = {
            "symptoms": [],
            "medications": [],
            "conditions": [],
            "anatomy": [],
            "procedures": [],
            "lab_tests": []
        }
        
        if not text:
            return entities
        
        text_lower = text.lower()
        
        # Use spaCy if available
        if self.nlp:
            entities.update(self._extract_with_spacy(text))
        
        # Rule-based extraction as fallback or supplement
        entities["symptoms"].extend(self._extract_symptoms(text_lower))
        entities["medications"].extend(self._extract_medications(text_lower))
        entities["conditions"].extend(self._extract_conditions(text_lower))
        entities["anatomy"].extend(self._extract_anatomy(text_lower))
        
        # Remove duplicates and sort by confidence
        for category in entities:
            entities[category] = self._deduplicate_entities(entities[category])
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> Dict[str, List[MedicalEntity]]:
        """Extract entities using spaCy NLP."""
        entities = {
            "symptoms": [],
            "medications": [],
            "conditions": [],
            "anatomy": [],
            "procedures": [],
            "lab_tests": []
        }
        
        try:
            doc = self.nlp(text)
            
            # Use custom matcher
            matches = self.matcher(doc)
            spans = []
            
            for match_id, start, end in matches:
                label = self.nlp.vocab.strings[match_id]
                span = doc[start:end]
                spans.append((span, label))
            
            # Process matched spans
            for span, label in spans:
                entity = MedicalEntity(
                    text=span.text,
                    label=label,
                    start=span.start_char,
                    end=span.end_char,
                    confidence=0.8  # Default confidence for pattern matches
                )
                
                if "SYMPTOM" in label:
                    entities["symptoms"].append(entity)
                elif "MEDICATION" in label:
                    entities["medications"].append(entity)
                elif "CONDITION" in label:
                    entities["conditions"].append(entity)
            
            # Also extract named entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG"]:  # Skip non-medical entities
                    continue
                
                entity = MedicalEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.6  # Lower confidence for general NER
                )
                
                # Categorize based on context
                if any(symptom in ent.text.lower() for symptom in self.symptoms_dict):
                    entities["symptoms"].append(entity)
                elif any(med in ent.text.lower() for med in self.medications_dict):
                    entities["medications"].append(entity)
        
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
        
        return entities
    
    def _extract_symptoms(self, text: str) -> List[MedicalEntity]:
        """Extract symptoms using rule-based approach."""
        symptoms = []
        
        for symptom in self.symptoms_dict:
            if symptom in text:
                start_pos = text.find(symptom)
                entity = MedicalEntity(
                    text=symptom,
                    label="SYMPTOM",
                    start=start_pos,
                    end=start_pos + len(symptom),
                    confidence=0.9,
                    normalized_form=symptom
                )
                symptoms.append(entity)
        
        return symptoms
    
    def _extract_medications(self, text: str) -> List[MedicalEntity]:
        """Extract medications using rule-based approach."""
        medications = []
        
        for medication in self.medications_dict:
            if medication in text:
                start_pos = text.find(medication)
                entity = MedicalEntity(
                    text=medication,
                    label="MEDICATION",
                    start=start_pos,
                    end=start_pos + len(medication),
                    confidence=0.9,
                    normalized_form=medication
                )
                medications.append(entity)
        
        return medications
    
    def _extract_conditions(self, text: str) -> List[MedicalEntity]:
        """Extract medical conditions using rule-based approach."""
        conditions = []
        
        for condition in self.conditions_dict:
            if condition in text:
                start_pos = text.find(condition)
                entity = MedicalEntity(
                    text=condition,
                    label="CONDITION",
                    start=start_pos,
                    end=start_pos + len(condition),
                    confidence=0.9,
                    normalized_form=condition
                )
                conditions.append(entity)
        
        return conditions
    
    def _extract_anatomy(self, text: str) -> List[MedicalEntity]:
        """Extract anatomical references using rule-based approach."""
        anatomy = []
        
        for anatomical_part in self.anatomy_dict:
            if anatomical_part in text:
                start_pos = text.find(anatomical_part)
                entity = MedicalEntity(
                    text=anatomical_part,
                    label="ANATOMY",
                    start=start_pos,
                    end=start_pos + len(anatomical_part),
                    confidence=0.8,
                    normalized_form=anatomical_part
                )
                anatomy.append(entity)
        
        return anatomy
    
    def _deduplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove duplicate entities and sort by confidence."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create a key based on text and position
            key = (entity.text.lower(), entity.start, entity.end)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        # Sort by confidence (descending)
        unique_entities.sort(key=lambda x: x.confidence, reverse=True)
        return unique_entities
    
    def normalize_medical_terms(self, text: str) -> str:
        """
        Normalize medical terminology in text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text:
            return text
        
        normalized_text = text.lower().strip()
        
        # Common medical abbreviations and their expansions
        abbreviations = {
            "mi": "myocardial infarction",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "dm": "diabetes mellitus",
            "htn": "hypertension",
            "cad": "coronary artery disease",
            "uti": "urinary tract infection",
            "sob": "shortness of breath",
            "cp": "chest pain",
            "n/v": "nausea and vomiting",
            "r/o": "rule out",
            "h/o": "history of",
            "s/p": "status post",
            "w/": "with",
            "w/o": "without"
        }
        
        # Replace abbreviations
        for abbrev, expansion in abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            normalized_text = re.sub(pattern, expansion, normalized_text, flags=re.IGNORECASE)
        
        # Standardize common variations
        variations = {
            r'\bheart attack\b': 'myocardial infarction',
            r'\bhigh blood pressure\b': 'hypertension',
            r'\blow blood pressure\b': 'hypotension',
            r'\bsugar diabetes\b': 'diabetes mellitus',
            r'\bbreathing problems?\b': 'dyspnea',
            r'\bstomach ache\b': 'abdominal pain',
            r'\bheadaches?\b': 'cephalgia'
        }
        
        for pattern, replacement in variations.items():
            normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text
    
    def extract_dosage_information(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medication dosage information from text.
        
        Args:
            text: Input text containing dosage information
            
        Returns:
            List[Dict[str, Any]]: Extracted dosage information
        """
        dosage_info = []
        
        # Dosage patterns
        dosage_patterns = [
            r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|mcg|units?)\s*(?:(\d+)\s*times?\s*(?:per\s+)?(?:day|daily|week|weekly))?',
            r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|mcg|units?)\s+(?:every|q)\s*(\d+)\s*(hours?|hrs?|h)',
            r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|mcg|units?)\s+(bid|tid|qid|qd|prn)',
        ]
        
        for pattern in dosage_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                dosage_dict = {
                    'medication': groups[0],
                    'dose': groups[1],
                    'unit': groups[2],
                    'frequency': groups[3] if len(groups) > 3 else None,
                    'raw_text': match.group(0)
                }
                dosage_info.append(dosage_dict)
        
        return dosage_info
    
    def extract_vital_signs(self, text: str) -> Dict[str, Any]:
        """
        Extract vital signs from text.
        
        Args:
            text: Input text containing vital signs
            
        Returns:
            Dict[str, Any]: Extracted vital signs
        """
        vital_signs = {}
        
        # Blood pressure pattern
        bp_pattern = r'(?:bp|blood pressure)[\s:]*(\d{2,3})/(\d{2,3})'
        bp_match = re.search(bp_pattern, text, re.IGNORECASE)
        if bp_match:
            vital_signs['blood_pressure'] = {
                'systolic': int(bp_match.group(1)),
                'diastolic': int(bp_match.group(2))
            }
        
        # Heart rate pattern
        hr_pattern = r'(?:hr|heart rate|pulse)[\s:]*(\d{2,3})'
        hr_match = re.search(hr_pattern, text, re.IGNORECASE)
        if hr_match:
            vital_signs['heart_rate'] = int(hr_match.group(1))
        
        # Temperature pattern
        temp_pattern = r'(?:temp|temperature)[\s:]*(\d{2,3}(?:\.\d)?)\s*(?:°?f|fahrenheit|°?c|celsius)?'
        temp_match = re.search(temp_pattern, text, re.IGNORECASE)
        if temp_match:
            vital_signs['temperature'] = float(temp_match.group(1))
        
        # Respiratory rate pattern
        rr_pattern = r'(?:rr|respiratory rate|respiration)[\s:]*(\d{1,2})'
        rr_match = re.search(rr_pattern, text, re.IGNORECASE)
        if rr_match:
            vital_signs['respiratory_rate'] = int(rr_match.group(1))
        
        # Oxygen saturation pattern
        o2_pattern = r'(?:o2 sat|oxygen saturation|spo2)[\s:]*(\d{2,3})%?'
        o2_match = re.search(o2_pattern, text, re.IGNORECASE)
        if o2_match:
            vital_signs['oxygen_saturation'] = int(o2_match.group(1))
        
        return vital_signs
    
    def classify_urgency(self, text: str) -> Dict[str, Any]:
        """
        Classify the urgency level of medical text.
        
        Args:
            text: Input medical text
            
        Returns:
            Dict[str, Any]: Urgency classification
        """
        text_lower = text.lower()
        
        # Critical keywords
        critical_keywords = [
            "emergency", "urgent", "critical", "severe", "acute",
            "chest pain", "difficulty breathing", "unconscious",
            "bleeding", "stroke", "heart attack", "seizure"
        ]
        
        # High priority keywords
        high_keywords = [
            "pain", "fever", "infection", "shortness of breath",
            "nausea", "vomiting", "dizziness", "weakness"
        ]
        
        # Low priority keywords
        low_keywords = [
            "routine", "follow-up", "check-up", "mild", "chronic"
        ]
        
        critical_count = sum(1 for keyword in critical_keywords if keyword in text_lower)
        high_count = sum(1 for keyword in high_keywords if keyword in text_lower)
        low_count = sum(1 for keyword in low_keywords if keyword in text_lower)
        
        if critical_count > 0:
            urgency_level = "critical"
            priority_score = 0.9 + (critical_count * 0.02)
        elif high_count > low_count:
            urgency_level = "high"
            priority_score = 0.6 + (high_count * 0.05)
        elif low_count > 0:
            urgency_level = "low"
            priority_score = 0.2 + (low_count * 0.02)
        else:
            urgency_level = "medium"
            priority_score = 0.5
        
        return {
            "urgency_level": urgency_level,
            "priority_score": min(priority_score, 1.0),
            "critical_indicators": critical_count,
            "high_priority_indicators": high_count,
            "low_priority_indicators": low_count
        }


# Global medical NLP instance
medical_nlp = MedicalNLP()


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from text (simplified interface).
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dict[str, List[str]]: Categorized medical entities as strings
    """
    entities = medical_nlp.extract_medical_entities(text)
    
    # Convert to simple string lists
    result = {}
    for category, entity_list in entities.items():
        result[category] = [entity.text for entity in entity_list]
    
    return result


def normalize_medical_terms(text: str) -> str:
    """
    Normalize medical terminology in text (simplified interface).
    
    Args:
        text: Input text to normalize
        
    Returns:
        str: Normalized text
    """
    return medical_nlp.normalize_medical_terms(text)


def extract_vital_signs(text: str) -> Dict[str, Any]:
    """
    Extract vital signs from text (simplified interface).
    
    Args:
        text: Input text containing vital signs
        
    Returns:
        Dict[str, Any]: Extracted vital signs
    """
    return medical_nlp.extract_vital_signs(text)


def classify_medical_urgency(text: str) -> Dict[str, Any]:
    """
    Classify the urgency level of medical text (simplified interface).
    
    Args:
        text: Input medical text
        
    Returns:
        Dict[str, Any]: Urgency classification
    """
    return medical_nlp.classify_urgency(text)
