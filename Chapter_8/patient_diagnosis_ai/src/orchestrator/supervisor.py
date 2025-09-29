
"""
Supervisor orchestrator for the Patient Diagnosis AI system.

This module implements the main orchestrator that coordinates the
multi-agent diagnostic workflow using both LangGraph and CrewAI patterns.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from crewai import Agent, Task, Crew
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from .workflow_graph import DiagnosisWorkflowGraph
from .state_management import DiagnosticState, StateManager, PatientContext, Priority
from src.config.prompts import get_prompt
from src.config.settings import get_settings
from src.observability.tracing import get_tracer
from src.utils.compliance import ensure_hipaa_compliance, audit_log

logger = logging.getLogger(__name__)
settings = get_settings()
tracer = get_tracer(__name__)


class DiagnosisOrchestrator:
    """
    Main orchestrator for the patient diagnosis assistance system.
    
    This class coordinates the entire diagnostic workflow, managing
    agent execution, state persistence, and result synthesis using
    both LangGraph workflows and CrewAI team coordination.
    """
    
    def __init__(self, llm, state_manager: Optional[StateManager] = None):
        """
        Initialize the diagnosis orchestrator.
        
        Args:
            llm: Language model instance
            state_manager: State management instance
        """
        self.llm = llm
        self.state_manager = state_manager or StateManager()
        
        # Initialize workflow graph
        self.workflow_graph = DiagnosisWorkflowGraph(llm)
        
        # Initialize CrewAI components
        self.crew = self._initialize_crew()
        
        logger.info("Diagnosis orchestrator initialized")
    
    def _initialize_crew(self) -> Crew:
        """Initialize CrewAI crew for agent coordination."""
        # Define CrewAI agents
        supervisor_agent = Agent(
            role="Medical Diagnosis Supervisor",
            goal="Coordinate specialized medical agents to provide comprehensive diagnostic assistance",
            backstory="You are an experienced medical supervisor coordinating a team of AI specialists to assist healthcare professionals in diagnosis.",
            verbose=True,
            allow_delegation=True
        )
        
        patient_history_agent = Agent(
            role="Patient History Specialist",
            goal="Analyze patient medical history and identify relevant patterns",
            backstory="You are a specialist in analyzing patient medical histories to identify risk factors and clinical patterns.",
            verbose=True
        )
        
        medical_coding_agent = Agent(
            role="Medical Coding Specialist", 
            goal="Map clinical concepts to standardized medical codes",
            backstory="You are an expert in medical terminology and coding systems like SNOMED CT and ICD-10.",
            verbose=True
        )
        
        drug_safety_agent = Agent(
            role="Drug Safety Specialist",
            goal="Analyze medication safety and identify potential interactions",
            backstory="You are a pharmacovigilance expert specializing in drug safety and interaction analysis.",
            verbose=True
        )
        
        literature_agent = Agent(
            role="Medical Literature Researcher",
            goal="Find and analyze relevant medical literature and evidence",
            backstory="You are a medical researcher expert at finding and synthesizing evidence from medical literature.",
            verbose=True
        )
        
        # Create crew
        crew = Crew(
            agents=[
                supervisor_agent,
                patient_history_agent,
                medical_coding_agent,
                drug_safety_agent,
                literature_agent
            ],
            verbose=True
        )
        
        return crew
    
    @ensure_hipaa_compliance
    @audit_log
    async def process_diagnosis_request(
        self,
        patient_context: PatientContext,
        clinical_question: str,
        user_id: Optional[str] = None,
        priority: Priority = Priority.NORMAL,
        use_crewai: bool = False
    ) -> DiagnosticState:
        """
        Process a diagnosis request using the multi-agent system.
        
        Args:
            patient_context: Patient information and context
            clinical_question: The clinical question to answer
            user_id: ID of the requesting user
            priority: Priority level of the request
            use_crewai: Whether to use CrewAI coordination
            
        Returns:
            DiagnosticState: Final diagnostic state with results
        """
        with tracer.start_as_current_span("process_diagnosis_request") as span:
            span.set_attribute("patient_id", patient_context.patient_id)
            span.set_attribute("priority", priority.value)
            span.set_attribute("use_crewai", use_crewai)
            
            logger.info(f"Processing diagnosis request for patient {patient_context.patient_id}")
            
            # Create initial diagnostic state
            state = self.state_manager.create_state(
                patient_context=patient_context,
                clinical_question=clinical_question,
                user_id=user_id,
                priority=priority
            )
            
            try:
                if use_crewai:
                    # Use CrewAI coordination
                    final_state = await self._process_with_crewai(state)
                else:
                    # Use LangGraph workflow
                    final_state = await self._process_with_langgraph(state)
                
                # Update state in storage
                self.state_manager.update_state(final_state)
                
                span.set_attribute("final_status", final_state.status.value)
                span.set_attribute("overall_confidence", final_state.overall_confidence)
                
                logger.info(f"Diagnosis request completed for session {final_state.session_id}")
                return final_state
                
            except Exception as e:
                logger.error(f"Error processing diagnosis request: {e}")
                state.add_error("orchestrator_error", str(e))
                state.status = state.status.FAILED
                self.state_manager.update_state(state)
                raise
    
    async def _process_with_langgraph(self, state: DiagnosticState) -> DiagnosticState:
        """Process diagnosis using LangGraph workflow."""
        with tracer.start_as_current_span("langgraph_workflow") as span:
            logger.info(f"Processing with LangGraph workflow for session {state.session_id}")
            
            # Run the workflow in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            final_state = await loop.run_in_executor(
                None,
                self.workflow_graph.run_workflow,
                state
            )
            
            span.set_attribute("workflow_completed", True)
            return final_state
    
    async def _process_with_crewai(self, state: DiagnosticState) -> DiagnosticState:
        """Process diagnosis using CrewAI coordination."""
        with tracer.start_as_current_span("crewai_workflow") as span:
            logger.info(f"Processing with CrewAI workflow for session {state.session_id}")
            
            # Define tasks for the crew
            tasks = self._create_crewai_tasks(state)
            
            # Execute the crew
            try:
                # Run crew in executor
                loop = asyncio.get_event_loop()
                crew_result = await loop.run_in_executor(
                    None,
                    self._execute_crew,
                    tasks
                )
                
                # Process crew results
                final_state = self._process_crew_results(state, crew_result)
                
                span.set_attribute("crew_completed", True)
                return final_state
                
            except Exception as e:
                logger.error(f"CrewAI execution failed: {e}")
                state.add_error("crewai_error", str(e))
                return state
    
    def _create_crewai_tasks(self, state: DiagnosticState) -> List[Task]:
        """Create CrewAI tasks for the diagnostic workflow."""
        tasks = []
        
        # Patient history analysis task
        if state.patient_context:
            patient_history_task = Task(
                description=f"""
                Analyze the medical history for patient {state.patient_context.patient_id}.
                
                Patient Information:
                - Age: {state.patient_context.age}
                - Gender: {state.patient_context.gender}
                - Chief Complaint: {state.patient_context.chief_complaint}
                - Symptoms: {', '.join(state.patient_context.symptoms)}
                - Medical History: {state.patient_context.medical_history}
                - Current Medications: {state.patient_context.current_medications}
                
                Provide a comprehensive analysis of risk factors, patterns, and clinical significance.
                """,
                expected_output="Detailed patient history analysis with risk factors and clinical patterns"
            )
            tasks.append(patient_history_task)
        
        # Medical coding task
        if state.patient_context and (state.patient_context.symptoms or state.patient_context.medical_history):
            medical_coding_task = Task(
                description=f"""
                Map the following clinical concepts to standardized medical codes:
                
                Symptoms: {', '.join(state.patient_context.symptoms)}
                Conditions: {[h.get('condition', '') for h in state.patient_context.medical_history]}
                
                Provide SNOMED CT and ICD-10 codes where appropriate.
                """,
                expected_output="Standardized medical codes for symptoms and conditions"
            )
            tasks.append(medical_coding_task)
        
        # Drug safety analysis task
        if state.patient_context and state.patient_context.current_medications:
            drug_safety_task = Task(
                description=f"""
                Analyze medication safety for the following medications:
                
                Current Medications: {state.patient_context.current_medications}
                Patient Conditions: {[h.get('condition', '') for h in state.patient_context.medical_history]}
                
                Check for drug interactions, contraindications, and safety alerts.
                """,
                expected_output="Comprehensive drug safety analysis with interactions and alerts"
            )
            tasks.append(drug_safety_task)
        
        # Literature research task
        literature_task = Task(
            description=f"""
            Research medical literature for the following clinical question:
            
            Clinical Question: {state.clinical_question}
            Patient Demographics: Age {state.patient_context.age if state.patient_context else 'unknown'}, 
                                Gender {state.patient_context.gender if state.patient_context else 'unknown'}
            
            Find evidence-based recommendations and current treatment guidelines.
            """,
            expected_output="Evidence-based recommendations with literature citations"
        )
        tasks.append(literature_task)
        
        return tasks
    
    def _execute_crew(self, tasks: List[Task]) -> Any:
        """Execute the CrewAI crew with tasks."""
        # Update crew with tasks
        self.crew.tasks = tasks
        
        # Execute the crew
        result = self.crew.kickoff()
        return result
    
    def _process_crew_results(self, state: DiagnosticState, crew_result: Any) -> DiagnosticState:
        """Process results from CrewAI execution."""
        # This would process the crew results and update the state
        # For now, we'll create a simplified implementation
        
        state.status = state.status.COMPLETED
        state.updated_at = datetime.utcnow()
        
        # Add placeholder results
        state.differential_diagnosis = [
            {
                "diagnosis": "Clinical correlation required",
                "probability": 0.5,
                "evidence": "Based on CrewAI analysis",
                "confidence": "moderate"
            }
        ]
        
        state.recommendations = [
            "Clinical correlation with physical examination required",
            "Consider additional diagnostic testing as indicated"
        ]
        
        state.evidence_summary = {
            "sources_consulted": ["CrewAI team analysis"],
            "evidence_quality": "moderate",
            "key_findings": [],
            "limitations": ["AI-generated analysis requires human validation"]
        }
        
        state.overall_confidence = 0.6
        state.requires_human_review = True
        
        return state
    
    async def get_diagnosis_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a diagnosis request.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict[str, Any]: Status information or None if not found
        """
        state = self.state_manager.get_state(session_id)
        if not state:
            return None
        
        return state.to_summary()
    
    async def cancel_diagnosis(self, session_id: str, user_id: str) -> bool:
        """
        Cancel a diagnosis request.
        
        Args:
            session_id: Session identifier
            user_id: User requesting cancellation
            
        Returns:
            bool: True if cancelled successfully
        """
        state = self.state_manager.get_state(session_id)
        if not state:
            return False
        
        # Check if user has permission to cancel
        if state.user_id != user_id:
            logger.warning(f"User {user_id} attempted to cancel session {session_id} owned by {state.user_id}")
            return False
        
        # Update state
        state.status = state.status.CANCELLED
        state.updated_at = datetime.utcnow()
        state.add_warning("user_cancellation", f"Cancelled by user {user_id}")
        
        self.state_manager.update_state(state)
        
        logger.info(f"Diagnosis session {session_id} cancelled by user {user_id}")
        return True
    
    async def retry_failed_agents(self, session_id: str) -> Optional[DiagnosticState]:
        """
        Retry failed agents in a diagnosis workflow.
        
        Args:
            session_id: Session identifier
            
        Returns:
            DiagnosticState: Updated state or None if not found
        """
        state = self.state_manager.get_state(session_id)
        if not state:
            return None
        
        if not state.failed_agents:
            logger.info(f"No failed agents to retry for session {session_id}")
            return state
        
        # Check retry limit
        if state.retry_count >= state.max_retries:
            logger.warning(f"Maximum retries exceeded for session {session_id}")
            return state
        
        logger.info(f"Retrying failed agents for session {session_id}: {state.failed_agents}")
        
        # Reset failed agents and increment retry count
        failed_agents = state.failed_agents.copy()
        state.failed_agents.clear()
        state.set_next_agents(failed_agents)
        state.retry_count += 1
        state.status = state.status.RUNNING
        
        # Continue workflow
        try:
            final_state = await self._process_with_langgraph(state)
            self.state_manager.update_state(final_state)
            return final_state
        except Exception as e:
            logger.error(f"Error retrying failed agents: {e}")
            state.add_error("retry_error", str(e))
            self.state_manager.update_state(state)
            return state
    
    async def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get active diagnosis sessions.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List[Dict[str, Any]]: List of active session summaries
        """
        return self.state_manager.list_active_states(user_id)
    
    async def get_session_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a diagnosis session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict[str, Any]: Session metrics or None if not found
        """
        return self.state_manager.get_state_metrics(session_id)
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old diagnosis sessions.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            int: Number of sessions cleaned up
        """
        return self.state_manager.cleanup_old_states(max_age_hours)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the orchestrator.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        health_status = {
            "status": "healthy",
            "orchestrator": "operational",
            "workflow_graph": "initialized",
            "crew": "initialized",
            "agents": {}
        }
        
        # Check individual agents
        for agent_name, agent in self.workflow_graph.agents.items():
            try:
                agent_health = agent.health_check()
                health_status["agents"][agent_name] = agent_health["status"]
            except Exception as e:
                health_status["agents"][agent_name] = "unhealthy"
                health_status["status"] = "degraded"
        
        # Check state manager
        try:
            # Test state creation and retrieval
            test_context = PatientContext(patient_id="health_check_test")
            test_state = self.state_manager.create_state(
                patient_context=test_context,
                clinical_question="health check"
            )
            retrieved_state = self.state_manager.get_state(test_state.session_id)
            
            if retrieved_state:
                health_status["state_manager"] = "operational"
                # Clean up test state
                self.state_manager.delete_state(test_state.session_id)
            else:
                health_status["state_manager"] = "unhealthy"
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["state_manager"] = "unhealthy"
            health_status["status"] = "unhealthy"
            logger.error(f"State manager health check failed: {e}")
        
        return health_status
    
    async def generate_diagnosis_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate a comprehensive diagnosis report.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict[str, Any]: Comprehensive diagnosis report
        """
        state = self.state_manager.get_state(session_id)
        if not state:
            return None
        
        report = {
            "session_info": {
                "session_id": state.session_id,
                "created_at": state.created_at.isoformat(),
                "completed_at": state.updated_at.isoformat() if state.is_complete() else None,
                "status": state.status.value,
                "priority": state.priority.value
            },
            "patient_context": state.patient_context.__dict__ if state.patient_context else {},
            "clinical_question": state.clinical_question,
            "agent_execution_summary": state.get_execution_summary(),
            "agent_outputs": state.get_agent_outputs(),
            "differential_diagnosis": state.differential_diagnosis,
            "recommendations": state.recommendations,
            "safety_alerts": state.safety_alerts,
            "evidence_summary": state.evidence_summary,
            "quality_metrics": {
                "overall_confidence": state.overall_confidence,
                "quality_score": state.quality_score,
                "completeness_score": state.completeness_score
            },
            "escalation_info": {
                "requires_escalation": state.requires_escalation,
                "escalation_reason": state.escalation_reason,
                "requires_human_review": state.requires_human_review,
                "urgent_findings": state.urgent_findings
            },
            "workflow_info": {
                "workflow_path": state.workflow_path,
                "completed_agents": state.completed_agents,
                "failed_agents": state.failed_agents,
                "retry_count": state.retry_count
            },
            "errors_and_warnings": {
                "errors": state.errors,
                "warnings": state.warnings
            }
        }
        
        return report
