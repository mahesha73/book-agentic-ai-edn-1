
"""
Workflow graph definition for the Patient Diagnosis AI system.

This module defines the LangGraph workflow that orchestrates the
multi-agent diagnostic process using a graph-based approach.
"""

import logging
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from .state_management import DiagnosticState, WorkflowStatus, AgentExecution
from src.agents import (
    PatientHistoryAgent,
    MedicalCodingAgent, 
    DrugSafetyAgent,
    LiteratureResearchAgent,
    ImageAnalysisAgent
)
from src.config.settings import get_settings
from src.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
settings = get_settings()
tracer = get_tracer(__name__)


class DiagnosisWorkflowGraph:
    """
    LangGraph-based workflow for multi-agent diagnosis.
    
    This class defines and manages the graph-based workflow that
    coordinates multiple specialized agents to provide comprehensive
    diagnostic assistance.
    """
    
    def __init__(self, llm):
        """
        Initialize the diagnosis workflow graph.
        
        Args:
            llm: Language model instance for agents
        """
        self.llm = llm
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Build the workflow graph
        self.graph = self._build_workflow_graph()
        
        logger.info("Diagnosis workflow graph initialized")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all specialized agents."""
        agents = {}
        
        if settings.agents.patient_history_agent_enabled:
            agents["patient_history"] = PatientHistoryAgent(self.llm)
        
        if settings.agents.medical_coding_agent_enabled:
            agents["medical_coding"] = MedicalCodingAgent(self.llm)
        
        if settings.agents.drug_safety_agent_enabled:
            agents["drug_safety"] = DrugSafetyAgent(self.llm)
        
        if settings.agents.literature_research_agent_enabled:
            agents["literature_research"] = LiteratureResearchAgent(self.llm)
        
        if settings.agents.image_analysis_agent_enabled:
            agents["image_analysis"] = ImageAnalysisAgent(self.llm)
        
        logger.info(f"Initialized {len(agents)} agents: {list(agents.keys())}")
        return agents
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(DiagnosticState)
        
        # Add nodes for each agent
        workflow.add_node("start", self._start_workflow)
        workflow.add_node("route_initial", self._route_initial_agents)
        
        if "patient_history" in self.agents:
            workflow.add_node("patient_history", self._execute_patient_history_agent)
        
        if "medical_coding" in self.agents:
            workflow.add_node("medical_coding", self._execute_medical_coding_agent)
        
        if "drug_safety" in self.agents:
            workflow.add_node("drug_safety", self._execute_drug_safety_agent)
        
        if "literature_research" in self.agents:
            workflow.add_node("literature_research", self._execute_literature_research_agent)
        
        if "image_analysis" in self.agents:
            workflow.add_node("image_analysis", self._execute_image_analysis_agent)
        
        workflow.add_node("synthesize", self._synthesize_results)
        workflow.add_node("finalize", self._finalize_diagnosis)
        
        # Define edges
        workflow.set_entry_point("start")
        workflow.add_edge("start", "route_initial")
        
        # Conditional routing from initial router
        workflow.add_conditional_edges(
            "route_initial",
            self._route_next_agent,
            {
                "patient_history": "patient_history",
                "medical_coding": "medical_coding", 
                "drug_safety": "drug_safety",
                "literature_research": "literature_research",
                "image_analysis": "image_analysis",
                "synthesize": "synthesize",
                "end": END
            }
        )
        
        # Add conditional edges from each agent
        for agent_name in self.agents.keys():
            workflow.add_conditional_edges(
                agent_name,
                self._route_next_agent,
                {
                    "patient_history": "patient_history",
                    "medical_coding": "medical_coding",
                    "drug_safety": "drug_safety", 
                    "literature_research": "literature_research",
                    "image_analysis": "image_analysis",
                    "synthesize": "synthesize",
                    "end": END
                }
            )
        
        workflow.add_edge("synthesize", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _start_workflow(self, state: DiagnosticState) -> DiagnosticState:
        """Start the diagnostic workflow."""
        with tracer.start_as_current_span("workflow_start") as span:
            span.set_attribute("session_id", state.session_id)
            span.set_attribute("patient_id", state.patient_context.patient_id if state.patient_context else "unknown")
            
            logger.info(f"Starting diagnostic workflow for session {state.session_id}")
            
            state.status = WorkflowStatus.RUNNING
            state.workflow_path.append("start")
            state.updated_at = datetime.utcnow()
            
            # Initialize trace information
            span_context = span.get_span_context()
            state.trace_id = format(span_context.trace_id, '032x')
            state.span_id = format(span_context.span_id, '016x')
            
            return state
    
    def _route_initial_agents(self, state: DiagnosticState) -> DiagnosticState:
        """Route to initial agents based on available data."""
        with tracer.start_as_current_span("route_initial") as span:
            logger.info(f"Routing initial agents for session {state.session_id}")
            
            # Determine which agents to run based on available data
            next_agents = []
            
            # Always start with patient history if available
            if "patient_history" in self.agents and state.patient_context:
                next_agents.append("patient_history")
            
            # Add medical coding if we have symptoms or conditions
            if "medical_coding" in self.agents and (
                state.patient_context and (
                    state.patient_context.symptoms or 
                    state.patient_context.medical_history
                )
            ):
                next_agents.append("medical_coding")
            
            # Add drug safety if we have medications
            if "drug_safety" in self.agents and (
                state.patient_context and state.patient_context.current_medications
            ):
                next_agents.append("drug_safety")
            
            # Add literature research for evidence-based recommendations
            if "literature_research" in self.agents and state.clinical_question:
                next_agents.append("literature_research")
            
            # Add image analysis if we have imaging studies
            if "image_analysis" in self.agents and (
                state.patient_context and state.patient_context.imaging_studies
            ):
                next_agents.append("image_analysis")
            
            state.set_next_agents(next_agents)
            state.workflow_path.append("route_initial")
            
            span.set_attribute("next_agents", ",".join(next_agents))
            logger.info(f"Routed to agents: {next_agents}")
            
            return state
    
    def _route_next_agent(self, state: DiagnosticState) -> str:
        """Determine the next agent to execute."""
        # Check if we have any agents left to execute
        next_agent = state.get_next_agent()
        
        if next_agent:
            logger.info(f"Routing to next agent: {next_agent}")
            return next_agent
        
        # Check if all required agents have completed
        required_agents = self._get_required_agents(state)
        completed_agents = set(state.completed_agents)
        
        if required_agents.issubset(completed_agents):
            logger.info("All required agents completed, moving to synthesis")
            return "synthesize"
        
        # Check if we should continue with optional agents
        optional_agents = self._get_optional_agents(state)
        remaining_optional = optional_agents - completed_agents
        
        if remaining_optional and len(state.failed_agents) < 2:  # Don't continue if too many failures
            # Add remaining optional agents
            state.set_next_agents(list(remaining_optional))
            return state.get_next_agent()
        
        logger.info("No more agents to execute, moving to synthesis")
        return "synthesize"
    
    def _get_required_agents(self, state: DiagnosticState) -> set:
        """Get the set of required agents based on available data."""
        required = set()
        
        if state.patient_context:
            if state.patient_context.symptoms or state.patient_context.medical_history:
                required.add("patient_history")
                required.add("medical_coding")
            
            if state.patient_context.current_medications:
                required.add("drug_safety")
        
        if state.clinical_question:
            required.add("literature_research")
        
        return required
    
    def _get_optional_agents(self, state: DiagnosticState) -> set:
        """Get the set of optional agents."""
        optional = set()
        
        if state.patient_context and state.patient_context.imaging_studies:
            optional.add("image_analysis")
        
        return optional
    
    def _execute_patient_history_agent(self, state: DiagnosticState) -> DiagnosticState:
        """Execute the patient history agent."""
        return self._execute_agent("patient_history", state)
    
    def _execute_medical_coding_agent(self, state: DiagnosticState) -> DiagnosticState:
        """Execute the medical coding agent."""
        return self._execute_agent("medical_coding", state)
    
    def _execute_drug_safety_agent(self, state: DiagnosticState) -> DiagnosticState:
        """Execute the drug safety agent."""
        return self._execute_agent("drug_safety", state)
    
    def _execute_literature_research_agent(self, state: DiagnosticState) -> DiagnosticState:
        """Execute the literature research agent."""
        return self._execute_agent("literature_research", state)
    
    def _execute_image_analysis_agent(self, state: DiagnosticState) -> DiagnosticState:
        """Execute the image analysis agent."""
        return self._execute_agent("image_analysis", state)
    
    def _execute_agent(self, agent_type: str, state: DiagnosticState) -> DiagnosticState:
        """Execute a specific agent."""
        with tracer.start_as_current_span(f"execute_{agent_type}") as span:
            span.set_attribute("agent_type", agent_type)
            span.set_attribute("session_id", state.session_id)
            
            logger.info(f"Executing {agent_type} agent for session {state.session_id}")
            
            start_time = datetime.utcnow()
            state.current_agent = agent_type
            state.status = WorkflowStatus.AGENT_EXECUTING
            
            # Create agent execution record
            execution = AgentExecution(
                agent_name=self.agents[agent_type].name,
                agent_type=agent_type,
                status="running",
                start_time=start_time
            )
            
            try:
                # Prepare agent input
                agent_input = self._prepare_agent_input(agent_type, state)
                execution.input_data = agent_input
                
                # Execute the agent
                agent = self.agents[agent_type]
                response = agent.process_request(
                    request=agent_input,
                    session_id=state.session_id,
                    user_id=state.user_id
                )
                
                # Update execution record
                execution.end_time = datetime.utcnow()
                execution.execution_time_seconds = (execution.end_time - start_time).total_seconds()
                execution.status = "completed"
                execution.output_data = response.to_dict()
                execution.confidence_score = self._convert_confidence_to_float(response.confidence)
                
                # Update state with agent output
                state.update_agent_output(agent_type, response.primary_output)
                state.mark_agent_completed(agent_type)
                
                # Check for escalation requirements
                if response.escalation_required:
                    state.requires_escalation = True
                    state.escalation_reason = f"Agent {agent_type} requires escalation"
                
                # Add urgent findings
                if hasattr(response, 'urgent_findings') and response.urgent_findings:
                    state.urgent_findings.extend(response.urgent_findings)
                
                # Add safety alerts
                if hasattr(response, 'safety_alerts') and response.safety_alerts:
                    state.safety_alerts.extend(response.safety_alerts)
                
                span.set_attribute("execution_status", "success")
                span.set_attribute("confidence_score", execution.confidence_score)
                
                logger.info(f"Successfully executed {agent_type} agent in {execution.execution_time_seconds:.2f}s")
                
            except Exception as e:
                # Handle agent execution failure
                execution.end_time = datetime.utcnow()
                execution.execution_time_seconds = (execution.end_time - start_time).total_seconds()
                execution.status = "failed"
                execution.error_message = str(e)
                
                state.mark_agent_failed(agent_type, str(e))
                
                span.set_attribute("execution_status", "failed")
                span.set_attribute("error_message", str(e))
                
                logger.error(f"Failed to execute {agent_type} agent: {e}")
            
            finally:
                state.add_agent_execution(execution)
                state.workflow_path.append(agent_type)
                state.current_agent = None
                state.status = WorkflowStatus.RUNNING
            
            return state
    
    def _prepare_agent_input(self, agent_type: str, state: DiagnosticState) -> Dict[str, Any]:
        """Prepare input for a specific agent."""
        base_input = {
            "session_id": state.session_id,
            "clinical_question": state.clinical_question,
            "patient_context": state.patient_context.__dict__ if state.patient_context else {}
        }
        
        if agent_type == "patient_history":
            base_input.update({
                "patient_id": state.patient_context.patient_id if state.patient_context else None,
                "query": f"Analyze medical history for patient with chief complaint: {state.patient_context.chief_complaint if state.patient_context else 'Unknown'}"
            })
        
        elif agent_type == "medical_coding":
            symptoms = state.patient_context.symptoms if state.patient_context else []
            conditions = [h.get("condition", "") for h in (state.patient_context.medical_history if state.patient_context else [])]
            base_input.update({
                "clinical_text": f"Symptoms: {', '.join(symptoms)}. Conditions: {', '.join(conditions)}",
                "concepts": symptoms + conditions
            })
        
        elif agent_type == "drug_safety":
            medications = state.patient_context.current_medications if state.patient_context else []
            base_input.update({
                "medications": [med.get("name", "") for med in medications],
                "patient_conditions": [h.get("condition", "") for h in (state.patient_context.medical_history if state.patient_context else [])]
            })
        
        elif agent_type == "literature_research":
            base_input.update({
                "research_query": state.clinical_question or "diagnostic evaluation",
                "patient_demographics": {
                    "age": state.patient_context.age if state.patient_context else None,
                    "gender": state.patient_context.gender if state.patient_context else None
                }
            })
        
        elif agent_type == "image_analysis":
            imaging_studies = state.patient_context.imaging_studies if state.patient_context else []
            if imaging_studies:
                base_input.update({
                    "image_data": imaging_studies[0].get("image_data"),
                    "image_type": imaging_studies[0].get("modality"),
                    "clinical_indication": state.patient_context.chief_complaint if state.patient_context else ""
                })
        
        return base_input
    
    def _synthesize_results(self, state: DiagnosticState) -> DiagnosticState:
        """Synthesize results from all agents."""
        with tracer.start_as_current_span("synthesize_results") as span:
            logger.info(f"Synthesizing results for session {state.session_id}")
            
            # Collect all agent outputs
            agent_outputs = state.get_agent_outputs()
            
            # Generate differential diagnosis
            state.differential_diagnosis = self._generate_differential_diagnosis(agent_outputs)
            
            # Generate recommendations
            state.recommendations = self._generate_recommendations(agent_outputs)
            
            # Create evidence summary
            state.evidence_summary = self._create_evidence_summary(agent_outputs)
            
            # Calculate overall confidence
            state.calculate_overall_confidence()
            
            # Assess quality and completeness
            state.quality_score = self._assess_quality_score(agent_outputs)
            state.completeness_score = self._assess_completeness_score(agent_outputs)
            
            state.workflow_path.append("synthesize")
            
            span.set_attribute("differential_diagnosis_count", len(state.differential_diagnosis))
            span.set_attribute("recommendations_count", len(state.recommendations))
            span.set_attribute("overall_confidence", state.overall_confidence)
            
            logger.info(f"Synthesis completed with {len(state.differential_diagnosis)} differential diagnoses")
            
            return state
    
    def _finalize_diagnosis(self, state: DiagnosticState) -> DiagnosticState:
        """Finalize the diagnostic workflow."""
        with tracer.start_as_current_span("finalize_diagnosis") as span:
            logger.info(f"Finalizing diagnosis for session {state.session_id}")
            
            state.status = WorkflowStatus.COMPLETED
            state.workflow_path.append("finalize")
            state.updated_at = datetime.utcnow()
            
            # Final quality checks
            if state.overall_confidence < 0.5:
                state.add_warning("low_confidence", "Overall confidence is below 50%")
            
            if len(state.errors) > 0:
                state.add_warning("execution_errors", f"Workflow completed with {len(state.errors)} errors")
            
            # Ensure human review is required for medical decisions
            state.requires_human_review = True
            
            span.set_attribute("final_status", state.status.value)
            span.set_attribute("requires_escalation", state.requires_escalation)
            span.set_attribute("requires_human_review", state.requires_human_review)
            
            total_time = (state.updated_at - state.created_at).total_seconds()
            logger.info(f"Diagnostic workflow completed in {total_time:.2f} seconds")
            
            return state
    
    # Helper methods
    
    def _convert_confidence_to_float(self, confidence) -> float:
        """Convert confidence enum to float."""
        confidence_map = {
            "very_low": 0.1,
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "very_high": 0.9
        }
        
        if hasattr(confidence, 'value'):
            return confidence_map.get(confidence.value, 0.5)
        return confidence_map.get(str(confidence).lower(), 0.5)
    
    def _generate_differential_diagnosis(self, agent_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate differential diagnosis from agent outputs."""
        # This would be more sophisticated in practice
        differential = []
        
        # Extract potential diagnoses from various agents
        if agent_outputs.get("patient_history"):
            # Extract conditions from patient history
            pass
        
        if agent_outputs.get("medical_coding"):
            # Extract coded conditions
            pass
        
        if agent_outputs.get("literature_research"):
            # Extract evidence-based diagnoses
            pass
        
        # Placeholder differential diagnosis
        differential.append({
            "diagnosis": "Requires clinical correlation",
            "probability": 0.5,
            "evidence": "Based on available data",
            "confidence": "moderate"
        })
        
        return differential
    
    def _generate_recommendations(self, agent_outputs: Dict[str, Any]) -> List[str]:
        """Generate recommendations from agent outputs."""
        recommendations = []
        
        # Always recommend clinical correlation
        recommendations.append("Clinical correlation with physical examination required")
        
        # Add agent-specific recommendations
        for agent_type, output in agent_outputs.items():
            if output and isinstance(output, dict):
                agent_recommendations = output.get("recommendations", [])
                recommendations.extend(agent_recommendations)
        
        return list(set(recommendations))  # Remove duplicates
    
    def _create_evidence_summary(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create evidence summary from agent outputs."""
        return {
            "sources_consulted": list(agent_outputs.keys()),
            "evidence_quality": "moderate",
            "key_findings": [],
            "limitations": ["AI-generated analysis requires human validation"]
        }
    
    def _assess_quality_score(self, agent_outputs: Dict[str, Any]) -> float:
        """Assess overall quality score."""
        # Simple quality assessment based on number of successful agents
        successful_agents = len([output for output in agent_outputs.values() if output])
        total_possible_agents = len(self.agents)
        
        return successful_agents / total_possible_agents if total_possible_agents > 0 else 0.0
    
    def _assess_completeness_score(self, agent_outputs: Dict[str, Any]) -> float:
        """Assess completeness score."""
        # Assess based on data availability and agent execution
        completeness_factors = []
        
        # Check if we have patient history
        if agent_outputs.get("patient_history"):
            completeness_factors.append(0.3)
        
        # Check if we have medical coding
        if agent_outputs.get("medical_coding"):
            completeness_factors.append(0.2)
        
        # Check if we have drug safety analysis
        if agent_outputs.get("drug_safety"):
            completeness_factors.append(0.2)
        
        # Check if we have literature research
        if agent_outputs.get("literature_research"):
            completeness_factors.append(0.2)
        
        # Check if we have image analysis (if applicable)
        if agent_outputs.get("image_analysis"):
            completeness_factors.append(0.1)
        
        return sum(completeness_factors)
    
    def run_workflow(self, state: DiagnosticState) -> DiagnosticState:
        """
        Run the complete diagnostic workflow.
        
        Args:
            state: Initial diagnostic state
            
        Returns:
            DiagnosticState: Final state after workflow completion
        """
        try:
            logger.info(f"Starting workflow execution for session {state.session_id}")
            
            # Execute the workflow graph
            final_state = self.graph.invoke(state)
            
            logger.info(f"Workflow execution completed for session {state.session_id}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed for session {state.session_id}: {e}")
            state.status = WorkflowStatus.FAILED
            state.add_error("workflow_execution", str(e))
            return state
