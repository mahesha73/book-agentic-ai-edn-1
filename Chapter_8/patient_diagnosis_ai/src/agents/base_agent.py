
"""
Base agent class for the Patient Diagnosis AI system.

This module provides the foundational BaseAgent class that all specialized
medical agents inherit from. It includes common functionality for HIPAA
compliance, error handling, observability, and agent lifecycle management.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.utils.compliance import ensure_hipaa_compliance, audit_log
from src.utils.monitoring import log_agent_action, track_performance
from src.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
settings = get_settings()
tracer = get_tracer(__name__)


class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent outputs."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AgentResponse:
    """Standardized response format for all agents."""
    
    agent_name: str
    agent_type: str
    status: AgentStatus
    confidence: ConfidenceLevel
    
    # Core response data
    primary_output: Dict[str, Any]
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    execution_time_seconds: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    
    # Evidence and citations
    evidence_sources: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Compliance
    hipaa_compliant: bool = True
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations for next steps
    next_actions: List[str] = field(default_factory=list)
    escalation_required: bool = False
    human_review_required: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "primary_output": self.primary_output,
            "supporting_data": self.supporting_data,
            "execution_time_seconds": self.execution_time_seconds,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "evidence_sources": self.evidence_sources,
            "citations": self.citations,
            "errors": self.errors,
            "warnings": self.warnings,
            "hipaa_compliant": self.hipaa_compliant,
            "audit_trail": self.audit_trail,
            "next_actions": self.next_actions,
            "escalation_required": self.escalation_required,
            "human_review_required": self.human_review_required,
            "created_at": self.created_at.isoformat()
        }


class AgentError(Exception):
    """Custom exception for agent errors."""
    
    def __init__(self, message: str, agent_name: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.agent_name = agent_name
        self.error_code = error_code or "AGENT_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for agent observability."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.start_time = None
        self.tokens_used = 0
        self.cost_usd = 0.0
    
    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        logger.debug(f"Agent {self.agent_name} taking action: {action.tool}")
    
    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes."""
        logger.debug(f"Agent {self.agent_name} finished with output: {finish.return_values}")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts."""
        self.start_time = time.time()
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends."""
        if hasattr(response, 'llm_output') and response.llm_output:
            self.tokens_used += response.llm_output.get('token_usage', {}).get('total_tokens', 0)


class BaseAgent(ABC):
    """
    Base class for all medical diagnosis agents.
    
    This abstract base class provides common functionality for all specialized
    agents including HIPAA compliance, error handling, observability, and
    standardized response formats.
    """
    
    def __init__(
        self,
        name: str,
        agent_type: str,
        description: str,
        tools: List[BaseTool],
        llm,
        max_iterations: int = 10,
        timeout_seconds: int = 300
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Human-readable name for the agent
            agent_type: Type identifier for the agent
            description: Description of agent capabilities
            tools: List of tools available to the agent
            llm: Language model instance
            max_iterations: Maximum number of iterations for agent execution
            timeout_seconds: Timeout for agent execution
        """
        self.name = name
        self.agent_type = agent_type
        self.description = description
        self.tools = tools
        self.llm = llm
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        
        # Initialize memory and callback handler
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=settings.agents.agent_memory_max_tokens
        )
        self.callback_handler = AgentCallbackHandler(self.name)
        
        # Create agent executor
        self.agent_executor = self._create_agent_executor()
        
        # Agent state
        self.status = AgentStatus.IDLE
        self.current_session_id = None
        
        logger.info(f"Initialized {self.agent_type} agent: {self.name}")
    
    @abstractmethod
    def _create_agent_executor(self) -> AgentExecutor:
        """
        Create the LangChain agent executor for this agent.
        
        This method must be implemented by each specialized agent to define
        its specific behavior, prompts, and tool usage patterns.
        
        Returns:
            AgentExecutor: Configured agent executor
        """
        pass
    
    @abstractmethod
    def _validate_input(self, request: Dict[str, Any]) -> None:
        """
        Validate agent-specific input requirements.
        
        Args:
            request: Input request to validate
            
        Raises:
            AgentError: If input validation fails
        """
        pass
    
    @abstractmethod
    def _process_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and format agent-specific output.
        
        Args:
            raw_output: Raw output from agent execution
            
        Returns:
            Dict[str, Any]: Processed and formatted output
        """
        pass
    
    @ensure_hipaa_compliance
    @audit_log
    @log_agent_action
    @track_performance
    def process_request(
        self,
        request: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> AgentResponse:
        """
        Process a request and return a standardized response.
        
        This is the main entry point for agent execution. It handles the
        complete lifecycle including validation, execution, error handling,
        and response formatting.
        
        Args:
            request: The input request containing query and context
            session_id: Optional session identifier for tracking
            user_id: Optional user identifier for audit trails
            
        Returns:
            AgentResponse: Standardized agent response
            
        Raises:
            AgentError: If agent execution fails
        """
        start_time = time.time()
        self.current_session_id = session_id or str(uuid.uuid4())
        
        # Initialize response
        response = AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            status=AgentStatus.RUNNING,
            confidence=ConfidenceLevel.MEDIUM,
            primary_output={}
        )
        
        try:
            with tracer.start_as_current_span(f"agent_{self.agent_type}_execution") as span:
                span.set_attribute("agent.name", self.name)
                span.set_attribute("agent.type", self.agent_type)
                span.set_attribute("session.id", self.current_session_id)
                
                # Update agent status
                self.status = AgentStatus.RUNNING
                
                # Validate input
                self._validate_input(request)
                span.add_event("input_validated")
                
                # Execute agent
                raw_output = self._execute_agent(request)
                span.add_event("agent_executed")
                
                # Process output
                processed_output = self._process_output(raw_output)
                span.add_event("output_processed")
                
                # Update response
                response.primary_output = processed_output
                response.status = AgentStatus.COMPLETED
                response.confidence = self._calculate_confidence(processed_output)
                response.execution_time_seconds = time.time() - start_time
                response.tokens_used = self.callback_handler.tokens_used
                response.cost_usd = self._calculate_cost()
                
                # Add evidence and citations
                response.evidence_sources = self._extract_evidence_sources(raw_output)
                response.citations = self._extract_citations(raw_output)
                
                # Determine next actions
                response.next_actions = self._suggest_next_actions(processed_output)
                response.escalation_required = self._requires_escalation(processed_output)
                response.human_review_required = self._requires_human_review(processed_output)
                
                self.status = AgentStatus.COMPLETED
                logger.info(f"Agent {self.name} completed successfully in {response.execution_time_seconds:.2f}s")
                
        except Exception as e:
            self.status = AgentStatus.FAILED
            response.status = AgentStatus.FAILED
            response.errors.append(str(e))
            response.execution_time_seconds = time.time() - start_time
            response.escalation_required = True
            response.human_review_required = True
            
            logger.error(f"Agent {self.name} failed: {e}", exc_info=True)
            
            if not isinstance(e, AgentError):
                raise AgentError(
                    message=f"Agent execution failed: {str(e)}",
                    agent_name=self.name,
                    error_code="EXECUTION_ERROR",
                    details={"original_error": str(e)}
                )
            raise
        
        return response
    
    def _execute_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent with the given request.
        
        Args:
            request: Input request
            
        Returns:
            Dict[str, Any]: Raw agent output
        """
        try:
            # Prepare input for agent
            agent_input = self._prepare_agent_input(request)
            
            # Execute agent with timeout
            result = self.agent_executor.invoke(
                agent_input,
                callbacks=[self.callback_handler],
                config={
                    "max_iterations": self.max_iterations,
                    "timeout": self.timeout_seconds
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise AgentError(
                message=f"Agent execution failed: {str(e)}",
                agent_name=self.name,
                error_code="EXECUTION_ERROR"
            )
    
    def _prepare_agent_input(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input for agent execution.
        
        Args:
            request: Original request
            
        Returns:
            Dict[str, Any]: Formatted input for agent
        """
        return {
            "input": request.get("query", ""),
            "context": request.get("context", {}),
            "session_id": self.current_session_id
        }
    
    def _calculate_confidence(self, output: Dict[str, Any]) -> ConfidenceLevel:
        """
        Calculate confidence level based on output quality.
        
        Args:
            output: Processed agent output
            
        Returns:
            ConfidenceLevel: Calculated confidence level
        """
        # Default implementation - can be overridden by specialized agents
        if not output:
            return ConfidenceLevel.VERY_LOW
        
        # Simple heuristic based on output completeness
        completeness_score = len(output) / 10  # Adjust based on expected output size
        
        if completeness_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif completeness_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif completeness_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif completeness_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_cost(self) -> float:
        """
        Calculate the cost of agent execution.
        
        Returns:
            float: Estimated cost in USD
        """
        # Simple cost calculation based on tokens used
        # This should be updated with actual pricing models
        cost_per_token = 0.00002  # Example rate
        return self.callback_handler.tokens_used * cost_per_token
    
    def _extract_evidence_sources(self, raw_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract evidence sources from agent output.
        
        Args:
            raw_output: Raw agent output
            
        Returns:
            List[Dict[str, Any]]: Evidence sources
        """
        # Default implementation - can be overridden
        return []
    
    def _extract_citations(self, raw_output: Dict[str, Any]) -> List[str]:
        """
        Extract citations from agent output.
        
        Args:
            raw_output: Raw agent output
            
        Returns:
            List[str]: Citations
        """
        # Default implementation - can be overridden
        return []
    
    def _suggest_next_actions(self, output: Dict[str, Any]) -> List[str]:
        """
        Suggest next actions based on agent output.
        
        Args:
            output: Processed agent output
            
        Returns:
            List[str]: Suggested next actions
        """
        # Default implementation - can be overridden
        return []
    
    def _requires_escalation(self, output: Dict[str, Any]) -> bool:
        """
        Determine if the case requires escalation.
        
        Args:
            output: Processed agent output
            
        Returns:
            bool: True if escalation is required
        """
        # Default implementation - can be overridden
        return False
    
    def _requires_human_review(self, output: Dict[str, Any]) -> bool:
        """
        Determine if the case requires human review.
        
        Args:
            output: Processed agent output
            
        Returns:
            bool: True if human review is required
        """
        # Default implementation - always require human review for medical decisions
        return True
    
    def reset_memory(self) -> None:
        """Reset agent memory."""
        self.memory.clear()
        logger.info(f"Reset memory for agent {self.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Dict[str, Any]: Agent status information
        """
        return {
            "name": self.name,
            "type": self.agent_type,
            "status": self.status.value,
            "description": self.description,
            "tools_count": len(self.tools),
            "memory_size": len(self.memory.chat_memory.messages),
            "current_session_id": self.current_session_id
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the agent.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Test basic agent functionality
            test_request = {"query": "health check", "context": {}}
            self._validate_input(test_request)
            
            return {
                "status": "healthy",
                "agent_name": self.name,
                "agent_type": self.agent_type,
                "tools_available": len(self.tools),
                "memory_functional": self.memory is not None,
                "llm_available": self.llm is not None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent_name": self.name,
                "agent_type": self.agent_type,
                "error": str(e)
            }
