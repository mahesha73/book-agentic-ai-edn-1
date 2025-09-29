#!/usr/bin/env python3
"""
Quick start script for the Patient Diagnosis AI system.
This script demonstrates how to run the system and make a sample diagnosis request.
"""

import asyncio
import json
from datetime import datetime
from src.orchestrator.supervisor import DiagnosisOrchestrator
from src.orchestrator.state_management import StateManager, PatientContext, Priority
from src.config.settings import get_settings

# Mock LLM for demonstration
class MockLLM:
    def invoke(self, prompt):
        return "Based on the patient's symptoms of chest pain and shortness of breath, along with their medical history of hypertension, this could indicate several conditions including angina, myocardial infarction, or pulmonary embolism. Immediate clinical evaluation is recommended."

async def main():
    """Demonstrate the Patient Diagnosis AI system."""
    print("🏥 Patient Diagnosis AI System - Demo")
    print("=" * 50)
    
    # Initialize system components
    print("Initializing system...")
    settings = get_settings()
    llm = MockLLM()
    state_manager = StateManager()
    orchestrator = DiagnosisOrchestrator(llm, state_manager)
    
    # Create sample patient context
    patient_context = PatientContext(
        patient_id="DEMO_001",
        age=65,
        gender="male",
        chief_complaint="chest pain and shortness of breath",
        symptoms=["chest pain", "shortness of breath", "sweating"],
        medical_history=[
            {"condition": "hypertension", "onset_date": "2020-01-01", "status": "active"},
            {"condition": "diabetes mellitus type 2", "onset_date": "2018-06-15", "status": "active"}
        ],
        current_medications=[
            {"name": "lisinopril", "dosage": "10mg daily", "start_date": "2020-01-01"},
            {"name": "metformin", "dosage": "500mg twice daily", "start_date": "2018-06-15"}
        ],
        allergies=["penicillin"],
        vital_signs={
            "blood_pressure": {"systolic": 150, "diastolic": 95},
            "heart_rate": 88,
            "temperature": 98.6,
            "respiratory_rate": 18,
            "oxygen_saturation": 96
        }
    )
    
    clinical_question = "Evaluate chest pain and shortness of breath in elderly male with hypertension and diabetes"
    
    print(f"\n📋 Processing diagnosis for patient: {patient_context.patient_id}")
    print(f"Chief complaint: {patient_context.chief_complaint}")
    print(f"Clinical question: {clinical_question}")
    
    # Process diagnosis request
    try:
        final_state = await orchestrator.process_diagnosis_request(
            patient_context=patient_context,
            clinical_question=clinical_question,
            user_id="demo_user",
            priority=Priority.HIGH
        )
        
        print(f"\n✅ Diagnosis completed!")
        print(f"Session ID: {final_state.session_id}")
        print(f"Status: {final_state.status.value}")
        print(f"Overall confidence: {final_state.overall_confidence:.2f}")
        print(f"Requires escalation: {final_state.requires_escalation}")
        print(f"Requires human review: {final_state.requires_human_review}")
        
        print(f"\n🔍 Differential Diagnosis:")
        for i, diagnosis in enumerate(final_state.differential_diagnosis, 1):
            print(f"  {i}. {diagnosis}")
        
        print(f"\n💡 Recommendations:")
        for i, recommendation in enumerate(final_state.recommendations, 1):
            print(f"  {i}. {recommendation}")
        
        print(f"\n⚠️  Safety Alerts:")
        for i, alert in enumerate(final_state.safety_alerts, 1):
            print(f"  {i}. {alert}")
        
        print(f"\n📊 Agent Execution Summary:")
        execution_summary = final_state.get_execution_summary()
        print(f"  - Total agents executed: {execution_summary['total_agents_executed']}")
        print(f"  - Completed agents: {execution_summary['completed_agents']}")
        print(f"  - Failed agents: {execution_summary['failed_agents']}")
        print(f"  - Total execution time: {execution_summary['total_execution_time']:.2f}s")
        print(f"  - Average confidence: {execution_summary['average_confidence']:.2f}")
        
        # Generate comprehensive report
        report = await orchestrator.generate_diagnosis_report(final_state.session_id)
        if report:
            print(f"\n📄 Full report generated with {len(str(report))} characters")
        
        print(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during diagnosis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
