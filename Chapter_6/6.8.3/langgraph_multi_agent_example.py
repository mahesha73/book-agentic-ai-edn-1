# langgraph_multi_agent_example.py
# Section 6.8.3
# Page 163

"""
Multi-Agent Research Paper Analysis System using LangGraph

This example demonstrates a comprehensive multi-agent system that analyzes research papers
through a coordinated workflow involving multiple specialized agents. The system showcases:

1. State management and sharing between agents
2. Agent handoffs and control flow
3. Error handling and logging
4. Practical multi-agent coordination

Use Case: Research Paper Analysis Workflow
- SearchAgent: Finds and retrieves research papers
- SummaryAgent: Creates structured summaries
- CritiqueAgent: Provides critical analysis and evaluation
- CoordinatorAgent: Manages workflow and final synthesis

Author: Multi-Agent Systems Tutorial
Date: June 2025
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass, field
from datetime import datetime
import traceback

# Core LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports for LLM integration
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Using mock implementations.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_agent_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class ResearchState(TypedDict):
    """
    Shared state structure for the multi-agent research workflow.

    This state is passed between all agents and maintains the complete
    context of the research analysis process.
    """
    # Input and configuration
    research_topic: str
    max_papers: int

    # Research data
    papers_found: List[Dict]
    current_paper: Optional[Dict]

    # Analysis results
    summaries: List[Dict]
    critiques: List[Dict]
    final_report: Optional[str]

    # Workflow control
    current_agent: str
    next_agent: Optional[str]
    workflow_step: int
    errors: List[str]

    # Metadata
    start_time: str
    last_updated: str
    status: Literal["initializing", "searching", "summarizing", "critiquing", "coordinating", "completed", "error"]

def create_initial_state(research_topic: str, max_papers: int = 3) -> ResearchState:
    """Create initial state for the research workflow."""
    return ResearchState(
        research_topic=research_topic,
        max_papers=max_papers,
        papers_found=[],
        current_paper=None,
        summaries=[],
        critiques=[],
        final_report=None,
        current_agent="search_agent",
        next_agent=None,
        workflow_step=1,
        errors=[],
        start_time=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
        status="initializing"
    )

# ============================================================================
# MOCK IMPLEMENTATIONS (for when LangChain is not available)
# ============================================================================

class MockLLM:
    """Mock LLM for demonstration when OpenAI is not available."""

    def __init__(self, model_name: str = "mock-gpt-4"):
        self.model_name = model_name

    def invoke(self, messages):
        """Mock LLM response based on the last message content."""
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            content = str(messages)

        # Generate mock responses based on content
        if "search" in content.lower() or "find papers" in content.lower():
            return AIMessage(content=json.dumps([
                {
                    "title": "Attention Is All You Need",
                    "authors": ["Vaswani et al."],
                    "year": 2017,
                    "abstract": "We propose the Transformer, a novel neural network architecture based solely on attention mechanisms...",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "citations": 50000
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": ["Devlin et al."],
                    "year": 2018,
                    "abstract": "We introduce BERT, a new language representation model which stands for Bidirectional Encoder Representations from Transformers...",
                    "url": "https://arxiv.org/abs/1810.04805",
                    "citations": 40000
                }
            ]))
        elif "summarize" in content.lower():
            return AIMessage(content="This paper introduces a groundbreaking approach to natural language processing. Key contributions include: 1) Novel architecture design, 2) Improved performance metrics, 3) Practical applications. The methodology is sound and results are significant.")
        elif "critique" in content.lower():
            return AIMessage(content="Strengths: Strong empirical results, clear methodology, significant impact. Weaknesses: Limited theoretical analysis, narrow evaluation scope. Overall assessment: High-quality work with practical implications.")
        else:
            return AIMessage(content="This is a mock response for demonstration purposes.")

# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class SearchAgent:
    """
    Agent responsible for finding and retrieving research papers.

    This agent searches for papers based on the research topic and
    populates the shared state with relevant papers.
    """

    def __init__(self, llm=None):
        self.name = "SearchAgent"
        self.llm = llm or MockLLM()
        logger.info(f"Initialized {self.name}")

    def __call__(self, state: ResearchState) -> ResearchState:
        """Execute the search agent workflow."""
        logger.info(f"{self.name}: Starting paper search for topic: {state['research_topic']}")

        try:
            # Update state
            state["current_agent"] = self.name
            state["status"] = "searching"
            state["last_updated"] = datetime.now().isoformat()

            # Create search prompt
            search_prompt = f"""
            You are a research assistant tasked with finding academic papers.

            Topic: {state['research_topic']}
            Maximum papers needed: {state['max_papers']}

            Please find and return a JSON list of relevant research papers with the following structure:
            [
                {{
                    "title": "Paper Title",
                    "authors": ["Author 1", "Author 2"],
                    "year": 2023,
                    "abstract": "Paper abstract...",
                    "url": "https://arxiv.org/abs/...",
                    "citations": 1000
                }}
            ]

            Focus on high-impact, recent papers that are most relevant to the topic.
            """

            # Get LLM response
            messages = [SystemMessage(content="You are a research paper search assistant."),
                       HumanMessage(content=search_prompt)]
            response = self.llm.invoke(messages)

            # Parse response
            try:
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)

                # Try to extract JSON from the response
                if content.startswith('[') and content.endswith(']'):
                    papers = json.loads(content)
                else:
                    # If not pure JSON, try to find JSON within the text
                    import re
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        papers = json.loads(json_match.group())
                    else:
                        raise ValueError("No valid JSON found in response")

                # Limit to max_papers
                papers = papers[:state['max_papers']]

                # Update state with found papers
                state["papers_found"] = papers
                state["next_agent"] = "summary_agent"
                state["workflow_step"] += 1

                logger.info(f"{self.name}: Found {len(papers)} papers")

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f"Failed to parse search results: {str(e)}"
                logger.error(f"{self.name}: {error_msg}")
                state["errors"].append(error_msg)
                state["status"] = "error"

        except Exception as e:
            error_msg = f"Search agent error: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            logger.error(traceback.format_exc())
            state["errors"].append(error_msg)
            state["status"] = "error"

        return state

class SummaryAgent:
    """
    Agent responsible for creating structured summaries of research papers.

    This agent processes each paper found by the SearchAgent and creates
    detailed summaries with key insights and findings.
    """

    def __init__(self, llm=None):
        self.name = "SummaryAgent"
        self.llm = llm or MockLLM()
        logger.info(f"Initialized {self.name}")

    def __call__(self, state: ResearchState) -> ResearchState:
        """Execute the summary agent workflow."""
        logger.info(f"{self.name}: Starting paper summarization")

        try:
            # Update state
            state["current_agent"] = self.name
            state["status"] = "summarizing"
            state["last_updated"] = datetime.now().isoformat()

            summaries = []

            for i, paper in enumerate(state["papers_found"]):
                logger.info(f"{self.name}: Summarizing paper {i+1}/{len(state['papers_found'])}: {paper['title']}")

                # Set current paper
                state["current_paper"] = paper

                # Create summary prompt
                summary_prompt = f"""
                You are an expert research analyst. Please provide a comprehensive summary of this research paper:

                Title: {paper['title']}
                Authors: {', '.join(paper['authors'])}
                Year: {paper['year']}
                Abstract: {paper['abstract']}

                Please provide a structured summary including:
                1. Main Research Question/Problem
                2. Key Methodology
                3. Primary Findings
                4. Significance and Impact
                5. Limitations
                6. Future Research Directions

                Keep the summary concise but comprehensive (200-300 words).
                """

                # Get LLM response
                messages = [SystemMessage(content="You are an expert research analyst."),
                           HumanMessage(content=summary_prompt)]
                response = self.llm.invoke(messages)

                # Create summary object
                summary = {
                    "paper_title": paper["title"],
                    "paper_authors": paper["authors"],
                    "paper_year": paper["year"],
                    "summary_text": response.content if hasattr(response, 'content') else str(response),
                    "summary_timestamp": datetime.now().isoformat()
                }

                summaries.append(summary)
                logger.info(f"{self.name}: Completed summary for '{paper['title']}'")

            # Update state
            state["summaries"] = summaries
            state["current_paper"] = None
            state["next_agent"] = "critique_agent"
            state["workflow_step"] += 1

            logger.info(f"{self.name}: Completed {len(summaries)} summaries")

        except Exception as e:
            error_msg = f"Summary agent error: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            logger.error(traceback.format_exc())
            state["errors"].append(error_msg)
            state["status"] = "error"

        return state

class CritiqueAgent:
    """
    Agent responsible for providing critical analysis and evaluation.

    This agent reviews the summaries and provides critical insights,
    identifies gaps, and suggests areas for further investigation.
    """

    def __init__(self, llm=None):
        self.name = "CritiqueAgent"
        self.llm = llm or MockLLM()
        logger.info(f"Initialized {self.name}")

    def __call__(self, state: ResearchState) -> ResearchState:
        """Execute the critique agent workflow."""
        logger.info(f"{self.name}: Starting critical analysis")

        try:
            # Update state
            state["current_agent"] = self.name
            state["status"] = "critiquing"
            state["last_updated"] = datetime.now().isoformat()

            critiques = []

            for i, summary in enumerate(state["summaries"]):
                logger.info(f"{self.name}: Critiquing paper {i+1}/{len(state['summaries'])}: {summary['paper_title']}")

                # Create critique prompt
                critique_prompt = f"""
                You are a senior research critic with expertise in evaluating academic work.
                Please provide a critical analysis of this research paper based on its summary:

                Paper: {summary['paper_title']} ({summary['paper_year']})
                Authors: {', '.join(summary['paper_authors'])}

                Summary:
                {summary['summary_text']}

                Please provide a critical evaluation covering:
                1. Methodological Strengths and Weaknesses
                2. Novelty and Originality Assessment
                3. Evidence Quality and Reliability
                4. Practical Implications and Applications
                5. Theoretical Contributions
                6. Areas for Improvement
                7. Overall Quality Rating (1-10 with justification)

                Be constructive but thorough in your critique.
                """

                # Get LLM response
                messages = [SystemMessage(content="You are a senior research critic and evaluator."),
                           HumanMessage(content=critique_prompt)]
                response = self.llm.invoke(messages)

                # Create critique object
                critique = {
                    "paper_title": summary["paper_title"],
                    "paper_authors": summary["paper_authors"],
                    "paper_year": summary["paper_year"],
                    "critique_text": response.content if hasattr(response, 'content') else str(response),
                    "critique_timestamp": datetime.now().isoformat()
                }

                critiques.append(critique)
                logger.info(f"{self.name}: Completed critique for '{summary['paper_title']}'")

            # Update state
            state["critiques"] = critiques
            state["next_agent"] = "coordinator_agent"
            state["workflow_step"] += 1

            logger.info(f"{self.name}: Completed {len(critiques)} critiques")

        except Exception as e:
            error_msg = f"Critique agent error: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            logger.error(traceback.format_exc())
            state["errors"].append(error_msg)
            state["status"] = "error"

        return state

class CoordinatorAgent:
    """
    Agent responsible for coordinating the workflow and creating final synthesis.

    This agent manages the overall workflow, synthesizes results from other agents,
    and produces the final research report.
    """

    def __init__(self, llm=None):
        self.name = "CoordinatorAgent"
        self.llm = llm or MockLLM()
        logger.info(f"Initialized {self.name}")

    def __call__(self, state: ResearchState) -> ResearchState:
        """Execute the coordinator agent workflow."""
        logger.info(f"{self.name}: Starting final coordination and synthesis")

        try:
            # Update state
            state["current_agent"] = self.name
            state["status"] = "coordinating"
            state["last_updated"] = datetime.now().isoformat()

            # Prepare synthesis data
            papers_info = []
            for i, paper in enumerate(state["papers_found"]):
                paper_data = {
                    "paper": paper,
                    "summary": state["summaries"][i] if i < len(state["summaries"]) else None,
                    "critique": state["critiques"][i] if i < len(state["critiques"]) else None
                }
                papers_info.append(paper_data)

            # Create final report prompt
            synthesis_prompt = f"""
            You are a senior research coordinator tasked with creating a comprehensive research report.

            Research Topic: {state['research_topic']}
            Number of Papers Analyzed: {len(state['papers_found'])}

            Based on the analysis of {len(state['papers_found'])} research papers, their summaries, and critical evaluations,
            please create a comprehensive research report that includes:

            1. Executive Summary
            2. Research Landscape Overview
            3. Key Findings and Trends
            4. Methodological Insights
            5. Critical Analysis Synthesis
            6. Research Gaps and Opportunities
            7. Recommendations for Future Work
            8. Conclusion

            Papers analyzed:
            """

            # Add paper information
            for i, paper_data in enumerate(papers_info):
                synthesis_prompt += f"\n\nPaper {i+1}: {paper_data['paper']['title']}\n"
                if paper_data['summary']:
                    synthesis_prompt += f"Summary: {paper_data['summary']['summary_text'][:200]}...\n"
                if paper_data['critique']:
                    synthesis_prompt += f"Critique: {paper_data['critique']['critique_text'][:200]}...\n"

            synthesis_prompt += "\n\nPlease provide a well-structured, comprehensive report (800-1200 words)."

            # Get LLM response
            messages = [SystemMessage(content="You are a senior research coordinator and report writer."),
                       HumanMessage(content=synthesis_prompt)]
            response = self.llm.invoke(messages)

            # Create final report
            final_report = response.content if hasattr(response, 'content') else str(response)

            # Update state
            state["final_report"] = final_report
            state["status"] = "completed"
            state["next_agent"] = None
            state["workflow_step"] += 1

            logger.info(f"{self.name}: Completed final synthesis report")

        except Exception as e:
            error_msg = f"Coordinator agent error: {str(e)}"
            logger.error(f"{self.name}: {error_msg}")
            logger.error(traceback.format_exc())
            state["errors"].append(error_msg)
            state["status"] = "error"

        return state

# ============================================================================
# AGENT HANDOFF TOOLS
# ============================================================================

def create_handoff_tool(target_agent: str, description: str = None):
    """
    Create a handoff tool for transferring control between agents.

    This implements the LangGraph Command pattern for agent handoffs,
    allowing seamless control transfer with state preservation.
    """
    tool_name = f"transfer_to_{target_agent}"
    tool_description = description or f"Transfer control to {target_agent}"

    @tool(tool_name, description=tool_description)
    def handoff_tool(state: ResearchState) -> Command:
        """Execute handoff to target agent."""
        logger.info(f"Handoff: Transferring control to {target_agent}")

        # Update state for handoff
        updated_state = state.copy()
        updated_state["current_agent"] = target_agent
        updated_state["last_updated"] = datetime.now().isoformat()

        return Command(
            goto=target_agent,
            update=updated_state,
            graph=Command.PARENT
        )

    return handoff_tool

# ============================================================================
# MULTI-AGENT SYSTEM ORCHESTRATION
# ============================================================================

class MultiAgentResearchSystem:
    """
    Main orchestrator for the multi-agent research system.

    This class manages the creation, configuration, and execution of the
    multi-agent workflow using LangGraph's StateGraph.
    """

    def __init__(self, llm=None, use_checkpointing=True):
        """Initialize the multi-agent system."""
        self.llm = llm or MockLLM()
        self.use_checkpointing = use_checkpointing

        # Initialize agents
        self.search_agent = SearchAgent(self.llm)
        self.summary_agent = SummaryAgent(self.llm)
        self.critique_agent = CritiqueAgent(self.llm)
        self.coordinator_agent = CoordinatorAgent(self.llm)

        # Create the workflow graph
        self.graph = self._create_workflow_graph()

        logger.info("MultiAgentResearchSystem initialized successfully")

    def _create_workflow_graph(self) -> StateGraph:
        """Create and configure the LangGraph workflow."""
        logger.info("Creating workflow graph...")

        # Create state graph
        workflow = StateGraph(ResearchState)

        # Add agent nodes
        workflow.add_node("search_agent", self.search_agent)
        workflow.add_node("summary_agent", self.summary_agent)
        workflow.add_node("critique_agent", self.critique_agent)
        workflow.add_node("coordinator_agent", self.coordinator_agent)

        # Define workflow edges
        workflow.add_edge(START, "search_agent")
        workflow.add_edge("search_agent", "summary_agent")
        workflow.add_edge("summary_agent", "critique_agent")
        workflow.add_edge("critique_agent", "coordinator_agent")
        workflow.add_edge("coordinator_agent", END)

        # Add conditional edges for error handling
        def should_continue(state: ResearchState) -> str:
            """Determine next step based on current state."""
            if state["status"] == "error":
                return END
            elif state["status"] == "completed":
                return END
            else:
                return state.get("next_agent", END)

        # Compile the graph
        if self.use_checkpointing:
            # Use memory saver for checkpointing
            memory = MemorySaver()
            compiled_graph = workflow.compile(checkpointer=memory)
        else:
            compiled_graph = workflow.compile()

        logger.info("Workflow graph created successfully")
        return compiled_graph

    def run_research_workflow(self, research_topic: str, max_papers: int = 3) -> ResearchState:
        """
        Execute the complete research workflow.

        Args:
            research_topic: The topic to research
            max_papers: Maximum number of papers to analyze

        Returns:
            Final state containing all results
        """
        logger.info(f"Starting research workflow for topic: '{research_topic}'")

        try:
            # Create initial state
            initial_state = create_initial_state(research_topic, max_papers)

            # Execute the workflow
            if self.use_checkpointing:
                # Use thread ID for checkpointing
                config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
                final_state = self.graph.invoke(initial_state, config=config)
            else:
                final_state = self.graph.invoke(initial_state)

            # Log completion
            if final_state["status"] == "completed":
                logger.info("Research workflow completed successfully")
            else:
                logger.warning(f"Research workflow ended with status: {final_state['status']}")

            return final_state

        except Exception as e:
            error_msg = f"Workflow execution error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            # Return error state
            error_state = create_initial_state(research_topic, max_papers)
            error_state["status"] = "error"
            error_state["errors"].append(error_msg)
            return error_state

    def get_workflow_visualization(self) -> str:
        """Get a text representation of the workflow graph."""
        return """
        Multi-Agent Research Workflow:

        START → SearchAgent → SummaryAgent → CritiqueAgent → CoordinatorAgent → END

        Agent Responsibilities:
        - SearchAgent: Find and retrieve research papers
        - SummaryAgent: Create structured summaries
        - CritiqueAgent: Provide critical analysis
        - CoordinatorAgent: Synthesize final report

        State Management:
        - Shared ResearchState passed between all agents
        - Persistent checkpointing available
        - Error handling and recovery
        """

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_research_results(state: ResearchState):
    """Print formatted research results."""
    print("\n" + "="*80)
    print("MULTI-AGENT RESEARCH SYSTEM RESULTS")
    print("="*80)

    print(f"\nResearch Topic: {state['research_topic']}")
    print(f"Status: {state['status']}")
    print(f"Workflow Steps Completed: {state['workflow_step']}")
    print(f"Start Time: {state['start_time']}")
    print(f"Last Updated: {state['last_updated']}")

    if state['errors']:
        print(f"\nErrors Encountered: {len(state['errors'])}")
        for i, error in enumerate(state['errors'], 1):
            print(f"  {i}. {error}")

    print(f"\nPapers Found: {len(state['papers_found'])}")
    for i, paper in enumerate(state['papers_found'], 1):
        print(f"  {i}. {paper['title']} ({paper['year']})")

    print(f"\nSummaries Created: {len(state['summaries'])}")
    print(f"Critiques Created: {len(state['critiques'])}")

    if state['final_report']:
        print("\n" + "-"*60)
        print("FINAL RESEARCH REPORT")
        print("-"*60)
        print(state['final_report'])

    print("\n" + "="*80)

def save_results_to_file(state: ResearchState, filename: str = None):
    """Save research results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_results_{timestamp}.json"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
        print(f"Results saved to: {filename}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        print(f"Error saving results: {str(e)}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function demonstrating the multi-agent system."""
    print("Multi-Agent Research System using LangGraph")
    print("=" * 50)

    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        print("\nNote: Running with mock implementations (LangChain not available)")
        print("Install langchain and langchain-openai for full functionality")

    # Initialize LLM (use mock if OpenAI not available)
    try:
        if LANGCHAIN_AVAILABLE:
            # Try to use OpenAI (requires API key)
            import os
            if os.getenv("OPENAI_API_KEY"):
                llm = ChatOpenAI(model="gpt-4", temperature=0.1)
                print("Using OpenAI GPT-4")
            else:
                llm = MockLLM()
                print("Using Mock LLM (set OPENAI_API_KEY for real LLM)")
        else:
            llm = MockLLM()
            print("Using Mock LLM")
    except Exception as e:
        llm = MockLLM()
        print(f"Using Mock LLM due to error: {str(e)}")

    # Create multi-agent system
    research_system = MultiAgentResearchSystem(llm=llm, use_checkpointing=True)

    # Print workflow visualization
    print(research_system.get_workflow_visualization())

    # Example research topics
    example_topics = [
        "Transformer architectures in natural language processing",
        "Multi-agent systems in artificial intelligence",
        "Reinforcement learning for robotics applications"
    ]

    # Run example workflow
    research_topic = example_topics[0]  # Use first topic
    max_papers = 2  # Limit for demo

    print(f"\nExecuting research workflow...")
    print(f"Topic: {research_topic}")
    print(f"Max papers: {max_papers}")
    print("\nStarting workflow execution...\n")

    # Execute the workflow
    final_state = research_system.run_research_workflow(
        research_topic=research_topic,
        max_papers=max_papers
    )

    # Display results
    print_research_results(final_state)

    # Save results
    save_results_to_file(final_state)

    print("\nWorkflow execution completed!")
    print("Check the log file 'multi_agent_system.log' for detailed execution logs.")

if __name__ == "__main__":
    main()
