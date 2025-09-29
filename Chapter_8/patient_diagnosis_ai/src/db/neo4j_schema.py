
"""
Neo4j schema and initialization for medical knowledge graph.

This module defines the Neo4j graph schema for medical ontologies,
disease-symptom relationships, drug interactions, and clinical pathways.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from neo4j import GraphDatabase, Session
from src.config.settings import get_settings
from src.db.engines import get_database_engines

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class NodeType:
    """Define node types in the medical knowledge graph."""
    label: str
    properties: Dict[str, str]
    indexes: List[str]
    constraints: List[str]


@dataclass
class RelationshipType:
    """Define relationship types in the medical knowledge graph."""
    type: str
    properties: Dict[str, str]
    from_node: str
    to_node: str


class MedicalKnowledgeGraphSchema:
    """
    Medical knowledge graph schema manager for Neo4j.
    
    This class manages the creation and maintenance of the medical
    knowledge graph schema including nodes, relationships, indexes,
    and constraints.
    """
    
    def __init__(self):
        """Initialize the schema manager."""
        self.engines = get_database_engines()
        
        # Define node types
        self.node_types = {
            "Patient": NodeType(
                label="Patient",
                properties={
                    "id": "string",
                    "age": "integer",
                    "gender": "string",
                    "race": "string",
                    "ethnicity": "string",
                    "created_at": "datetime",
                    "updated_at": "datetime"
                },
                indexes=["id", "age", "gender"],
                constraints=["id"]
            ),
            "Disease": NodeType(
                label="Disease",
                properties={
                    "id": "string",
                    "name": "string",
                    "icd10_code": "string",
                    "snomed_code": "string",
                    "description": "string",
                    "severity": "string",
                    "prevalence": "float",
                    "mortality_rate": "float"
                },
                indexes=["name", "icd10_code", "snomed_code", "severity"],
                constraints=["id", "icd10_code", "snomed_code"]
            ),
            "Symptom": NodeType(
                label="Symptom",
                properties={
                    "id": "string",
                    "name": "string",
                    "snomed_code": "string",
                    "description": "string",
                    "body_system": "string",
                    "severity_scale": "string"
                },
                indexes=["name", "snomed_code", "body_system"],
                constraints=["id", "snomed_code"]
            ),
            "Drug": NodeType(
                label="Drug",
                properties={
                    "id": "string",
                    "name": "string",
                    "generic_name": "string",
                    "brand_names": "list",
                    "rxnorm_code": "string",
                    "drug_class": "string",
                    "mechanism_of_action": "string",
                    "dosage_forms": "list",
                    "route_of_administration": "list"
                },
                indexes=["name", "generic_name", "rxnorm_code", "drug_class"],
                constraints=["id", "rxnorm_code"]
            ),
            "Procedure": NodeType(
                label="Procedure",
                properties={
                    "id": "string",
                    "name": "string",
                    "cpt_code": "string",
                    "snomed_code": "string",
                    "description": "string",
                    "category": "string",
                    "invasiveness": "string",
                    "duration_minutes": "integer"
                },
                indexes=["name", "cpt_code", "snomed_code", "category"],
                constraints=["id", "cpt_code"]
            ),
            "BodySystem": NodeType(
                label="BodySystem",
                properties={
                    "id": "string",
                    "name": "string",
                    "description": "string",
                    "organs": "list"
                },
                indexes=["name"],
                constraints=["id"]
            ),
            "Encounter": NodeType(
                label="Encounter",
                properties={
                    "id": "string",
                    "encounter_type": "string",
                    "date": "datetime",
                    "duration_minutes": "integer",
                    "chief_complaint": "string",
                    "diagnosis_codes": "list",
                    "provider_id": "string"
                },
                indexes=["date", "encounter_type", "provider_id"],
                constraints=["id"]
            ),
            "Provider": NodeType(
                label="Provider",
                properties={
                    "id": "string",
                    "name": "string",
                    "specialty": "string",
                    "license_number": "string",
                    "institution": "string"
                },
                indexes=["specialty", "institution"],
                constraints=["id", "license_number"]
            ),
            "ClinicalGuideline": NodeType(
                label="ClinicalGuideline",
                properties={
                    "id": "string",
                    "title": "string",
                    "organization": "string",
                    "publication_date": "datetime",
                    "evidence_level": "string",
                    "recommendations": "list",
                    "url": "string"
                },
                indexes=["organization", "publication_date", "evidence_level"],
                constraints=["id"]
            )
        }
        
        # Define relationship types
        self.relationship_types = {
            "HAS_SYMPTOM": RelationshipType(
                type="HAS_SYMPTOM",
                properties={
                    "severity": "string",
                    "frequency": "string",
                    "duration": "string",
                    "onset": "string"
                },
                from_node="Disease",
                to_node="Symptom"
            ),
            "PRESENTS_WITH": RelationshipType(
                type="PRESENTS_WITH",
                properties={
                    "severity": "string",
                    "onset_date": "datetime",
                    "resolution_date": "datetime"
                },
                from_node="Patient",
                to_node="Symptom"
            ),
            "DIAGNOSED_WITH": RelationshipType(
                type="DIAGNOSED_WITH",
                properties={
                    "diagnosis_date": "datetime",
                    "confidence": "float",
                    "primary": "boolean",
                    "status": "string"
                },
                from_node="Patient",
                to_node="Disease"
            ),
            "PRESCRIBED": RelationshipType(
                type="PRESCRIBED",
                properties={
                    "dosage": "string",
                    "frequency": "string",
                    "start_date": "datetime",
                    "end_date": "datetime",
                    "indication": "string"
                },
                from_node="Patient",
                to_node="Drug"
            ),
            "TREATS": RelationshipType(
                type="TREATS",
                properties={
                    "efficacy": "float",
                    "evidence_level": "string",
                    "line_of_therapy": "integer"
                },
                from_node="Drug",
                to_node="Disease"
            ),
            "INTERACTS_WITH": RelationshipType(
                type="INTERACTS_WITH",
                properties={
                    "interaction_type": "string",
                    "severity": "string",
                    "mechanism": "string",
                    "clinical_significance": "string"
                },
                from_node="Drug",
                to_node="Drug"
            ),
            "CONTRAINDICATED_IN": RelationshipType(
                type="CONTRAINDICATED_IN",
                properties={
                    "contraindication_type": "string",
                    "severity": "string",
                    "reason": "string"
                },
                from_node="Drug",
                to_node="Disease"
            ),
            "AFFECTS": RelationshipType(
                type="AFFECTS",
                properties={
                    "effect_type": "string",
                    "mechanism": "string"
                },
                from_node="Disease",
                to_node="BodySystem"
            ),
            "PART_OF": RelationshipType(
                type="PART_OF",
                properties={},
                from_node="Symptom",
                to_node="BodySystem"
            ),
            "HAS_ENCOUNTER": RelationshipType(
                type="HAS_ENCOUNTER",
                properties={},
                from_node="Patient",
                to_node="Encounter"
            ),
            "PERFORMED_BY": RelationshipType(
                type="PERFORMED_BY",
                properties={},
                from_node="Encounter",
                to_node="Provider"
            ),
            "INCLUDES_PROCEDURE": RelationshipType(
                type="INCLUDES_PROCEDURE",
                properties={
                    "order": "integer",
                    "status": "string"
                },
                from_node="Encounter",
                to_node="Procedure"
            ),
            "RECOMMENDS": RelationshipType(
                type="RECOMMENDS",
                properties={
                    "strength": "string",
                    "evidence_level": "string"
                },
                from_node="ClinicalGuideline",
                to_node="Drug"
            ),
            "NEXT": RelationshipType(
                type="NEXT",
                properties={
                    "time_interval": "integer"
                },
                from_node="Encounter",
                to_node="Encounter"
            )
        }
    
    def create_constraints(self, session: Session) -> None:
        """Create uniqueness constraints for the graph."""
        logger.info("Creating Neo4j constraints...")
        
        for node_type in self.node_types.values():
            for constraint in node_type.constraints:
                query = f"""
                CREATE CONSTRAINT {node_type.label.lower()}_{constraint}_unique 
                IF NOT EXISTS 
                FOR (n:{node_type.label}) 
                REQUIRE n.{constraint} IS UNIQUE
                """
                try:
                    session.run(query)
                    logger.info(f"Created constraint for {node_type.label}.{constraint}")
                except Exception as e:
                    logger.warning(f"Constraint creation failed for {node_type.label}.{constraint}: {e}")
    
    def create_indexes(self, session: Session) -> None:
        """Create indexes for better query performance."""
        logger.info("Creating Neo4j indexes...")
        
        for node_type in self.node_types.values():
            for index_property in node_type.indexes:
                query = f"""
                CREATE INDEX {node_type.label.lower()}_{index_property}_index 
                IF NOT EXISTS 
                FOR (n:{node_type.label}) 
                ON (n.{index_property})
                """
                try:
                    session.run(query)
                    logger.info(f"Created index for {node_type.label}.{index_property}")
                except Exception as e:
                    logger.warning(f"Index creation failed for {node_type.label}.{index_property}: {e}")
    
    def create_composite_indexes(self, session: Session) -> None:
        """Create composite indexes for complex queries."""
        logger.info("Creating composite indexes...")
        
        composite_indexes = [
            ("Disease", ["name", "icd10_code"]),
            ("Drug", ["name", "drug_class"]),
            ("Symptom", ["name", "body_system"]),
            ("Encounter", ["date", "encounter_type"]),
            ("Patient", ["age", "gender"])
        ]
        
        for label, properties in composite_indexes:
            prop_list = ", ".join([f"n.{prop}" for prop in properties])
            index_name = f"{label.lower()}_{'_'.join(properties)}_composite"
            
            query = f"""
            CREATE INDEX {index_name} 
            IF NOT EXISTS 
            FOR (n:{label}) 
            ON ({prop_list})
            """
            try:
                session.run(query)
                logger.info(f"Created composite index for {label}: {properties}")
            except Exception as e:
                logger.warning(f"Composite index creation failed for {label}: {e}")
    
    def initialize_schema(self) -> None:
        """Initialize the complete Neo4j schema."""
        logger.info("Initializing Neo4j medical knowledge graph schema...")
        
        with self.engines.neo4j_session() as session:
            # Create constraints first (they also create indexes)
            self.create_constraints(session)
            
            # Create additional indexes
            self.create_indexes(session)
            
            # Create composite indexes
            self.create_composite_indexes(session)
            
            logger.info("Neo4j schema initialization completed")
    
    def load_sample_data(self) -> None:
        """Load sample medical data for testing and development."""
        logger.info("Loading sample medical data...")
        
        with self.engines.neo4j_session() as session:
            # Sample diseases
            diseases_data = [
                {
                    "id": "disease_001",
                    "name": "Hypertension",
                    "icd10_code": "I10",
                    "snomed_code": "38341003",
                    "description": "High blood pressure",
                    "severity": "moderate",
                    "prevalence": 0.45
                },
                {
                    "id": "disease_002",
                    "name": "Type 2 Diabetes Mellitus",
                    "icd10_code": "E11",
                    "snomed_code": "44054006",
                    "description": "Non-insulin dependent diabetes",
                    "severity": "moderate",
                    "prevalence": 0.11
                },
                {
                    "id": "disease_003",
                    "name": "Myocardial Infarction",
                    "icd10_code": "I21",
                    "snomed_code": "22298006",
                    "description": "Heart attack",
                    "severity": "severe",
                    "prevalence": 0.007
                }
            ]
            
            # Sample symptoms
            symptoms_data = [
                {
                    "id": "symptom_001",
                    "name": "Chest Pain",
                    "snomed_code": "29857009",
                    "description": "Pain in the chest area",
                    "body_system": "cardiovascular"
                },
                {
                    "id": "symptom_002",
                    "name": "Shortness of Breath",
                    "snomed_code": "267036007",
                    "description": "Difficulty breathing",
                    "body_system": "respiratory"
                },
                {
                    "id": "symptom_003",
                    "name": "Excessive Thirst",
                    "snomed_code": "17173007",
                    "description": "Polydipsia",
                    "body_system": "endocrine"
                }
            ]
            
            # Sample drugs
            drugs_data = [
                {
                    "id": "drug_001",
                    "name": "Lisinopril",
                    "generic_name": "Lisinopril",
                    "rxnorm_code": "29046",
                    "drug_class": "ACE Inhibitor",
                    "mechanism_of_action": "Inhibits angiotensin-converting enzyme"
                },
                {
                    "id": "drug_002",
                    "name": "Metformin",
                    "generic_name": "Metformin",
                    "rxnorm_code": "6809",
                    "drug_class": "Biguanide",
                    "mechanism_of_action": "Decreases hepatic glucose production"
                }
            ]
            
            # Create nodes
            for disease in diseases_data:
                session.run(
                    "MERGE (d:Disease {id: $id}) SET d += $props",
                    id=disease["id"], props=disease
                )
            
            for symptom in symptoms_data:
                session.run(
                    "MERGE (s:Symptom {id: $id}) SET s += $props",
                    id=symptom["id"], props=symptom
                )
            
            for drug in drugs_data:
                session.run(
                    "MERGE (dr:Drug {id: $id}) SET dr += $props",
                    id=drug["id"], props=drug
                )
            
            # Create relationships
            relationships = [
                # Disease-Symptom relationships
                ("disease_001", "HAS_SYMPTOM", "symptom_001", {"severity": "mild"}),
                ("disease_002", "HAS_SYMPTOM", "symptom_003", {"severity": "moderate"}),
                ("disease_003", "HAS_SYMPTOM", "symptom_001", {"severity": "severe"}),
                ("disease_003", "HAS_SYMPTOM", "symptom_002", {"severity": "moderate"}),
                
                # Drug-Disease relationships
                ("drug_001", "TREATS", "disease_001", {"efficacy": 0.85, "evidence_level": "A"}),
                ("drug_002", "TREATS", "disease_002", {"efficacy": 0.78, "evidence_level": "A"}),
            ]
            
            for from_id, rel_type, to_id, props in relationships:
                if rel_type == "HAS_SYMPTOM":
                    session.run(
                        """
                        MATCH (d:Disease {id: $from_id}), (s:Symptom {id: $to_id})
                        MERGE (d)-[r:HAS_SYMPTOM]->(s)
                        SET r += $props
                        """,
                        from_id=from_id, to_id=to_id, props=props
                    )
                elif rel_type == "TREATS":
                    session.run(
                        """
                        MATCH (dr:Drug {id: $from_id}), (d:Disease {id: $to_id})
                        MERGE (dr)-[r:TREATS]->(d)
                        SET r += $props
                        """,
                        from_id=from_id, to_id=to_id, props=props
                    )
            
            logger.info("Sample medical data loaded successfully")
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the current schema."""
        with self.engines.neo4j_session() as session:
            # Get node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]
            
            # Get relationship types
            rel_types_result = session.run("CALL db.relationshipTypes()")
            relationship_types = [record["relationshipType"] for record in rel_types_result]
            
            # Get indexes
            indexes_result = session.run("SHOW INDEXES")
            indexes = [dict(record) for record in indexes_result]
            
            # Get constraints
            constraints_result = session.run("SHOW CONSTRAINTS")
            constraints = [dict(record) for record in constraints_result]
            
            return {
                "node_labels": labels,
                "relationship_types": relationship_types,
                "indexes": indexes,
                "constraints": constraints,
                "node_count": self._get_node_count(session),
                "relationship_count": self._get_relationship_count(session)
            }
    
    def _get_node_count(self, session: Session) -> int:
        """Get total number of nodes."""
        result = session.run("MATCH (n) RETURN count(n) as count")
        return result.single()["count"]
    
    def _get_relationship_count(self, session: Session) -> int:
        """Get total number of relationships."""
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        return result.single()["count"]


def initialize_neo4j_schema():
    """Initialize the Neo4j schema for the medical knowledge graph."""
    schema_manager = MedicalKnowledgeGraphSchema()
    schema_manager.initialize_schema()
    schema_manager.load_sample_data()
    return schema_manager


if __name__ == "__main__":
    # Initialize schema when run directly
    initialize_neo4j_schema()
