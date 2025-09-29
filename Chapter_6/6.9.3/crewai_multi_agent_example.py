# crewai_multi_agent_example.py
# Section 6.9.3
# Page 166

"""
CrewAI Multi-Agent Research Paper Analysis System

This module implements a comprehensive multi-agent system using CrewAI framework
for analyzing research papers. The system includes specialized agents for different
aspects of paper analysis including content extraction, methodology review,
citation analysis, and report generation.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from langchain.llms import OpenAI
    from langchain.tools import tool
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install required packages: pip install crewai langchain openai")
    exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperAnalysisInput(BaseModel):
    """Input model for paper analysis"""
    paper_path: str = Field(description="Path to the research paper file")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    output_format: str = Field(default="detailed", description="Output format preference")


class PaperExtractionTool(BaseTool):
    """Tool for extracting content from research papers"""
    name: str = "paper_extraction_tool"
    description: str = "Extracts text content, metadata, and structure from research papers"

    def _run(self, paper_path: str) -> Dict[str, Any]:
        """Extract content from paper"""
        try:
            # Simulate paper content extraction
            # In real implementation, this would use PyPDF2, pdfplumber, or similar
            extracted_content = {
                "title": "Sample Research Paper Title",
                "authors": ["Dr. Jane Smith", "Dr. John Doe"],
                "abstract": "This paper presents a novel approach to...",
                "sections": {
                    "introduction": "Research in this field has shown...",
                    "methodology": "We employed a mixed-methods approach...",
                    "results": "Our findings indicate that...",
                    "conclusion": "In conclusion, this study demonstrates..."
                },
                "references": ["Smith, J. (2023). Previous work...", "Doe, J. (2022). Related research..."],
                "metadata": {
                    "publication_year": "2024",
                    "journal": "Journal of Advanced Research",
                    "doi": "10.1000/sample.doi"
                }
            }
            logger.info(f"Successfully extracted content from {paper_path}")
            return extracted_content
        except Exception as e:
            logger.error(f"Error extracting paper content: {str(e)}")
            raise


class CitationAnalysisTool(BaseTool):
    """Tool for analyzing citations and references"""
    name: str = "citation_analysis_tool"
    description: str = "Analyzes citation patterns, reference quality, and academic impact"

    def _run(self, references: List[str]) -> Dict[str, Any]:
        """Analyze citations"""
        try:
            analysis = {
                "total_references": len(references),
                "citation_types": {
                    "journal_articles": len([r for r in references if "journal" in r.lower()]),
                    "conference_papers": len([r for r in references if "conference" in r.lower()]),
                    "books": len([r for r in references if "book" in r.lower()]),
                    "other": 0
                },
                "recency_analysis": {
                    "recent_citations": len([r for r in references if "2023" in r or "2024" in r]),
                    "older_citations": len([r for r in references if "2020" in r or "2021" in r or "2022" in r])
                },
                "quality_score": 8.5  # Simulated quality score
            }
            logger.info("Citation analysis completed")
            return analysis
        except Exception as e:
            logger.error(f"Error in citation analysis: {str(e)}")
            raise


class MethodologyEvaluationTool(BaseTool):
    """Tool for evaluating research methodology"""
    name: str = "methodology_evaluation_tool"
    description: str = "Evaluates research methodology, experimental design, and statistical approaches"

    def _run(self, methodology_text: str) -> Dict[str, Any]:
        """Evaluate methodology"""
        try:
            evaluation = {
                "methodology_type": "Mixed Methods",
                "strengths": [
                    "Clear experimental design",
                    "Appropriate statistical methods",
                    "Well-defined control groups"
                ],
                "weaknesses": [
                    "Limited sample size",
                    "Potential selection bias"
                ],
                "rigor_score": 7.8,
                "reproducibility": "High",
                "statistical_validity": "Good"
            }
            logger.info("Methodology evaluation completed")
            return evaluation
        except Exception as e:
            logger.error(f"Error in methodology evaluation: {str(e)}")
            raise


class ResearchPaperAnalysisSystem:
    """Main system orchestrating the multi-agent research paper analysis"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the analysis system"""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Using mock LLM for demonstration.")
            self.llm = None
        else:
            self.llm = OpenAI(openai_api_key=self.api_key, temperature=0.1)

        self.setup_agents()
        self.setup_tools()

    def setup_tools(self):
        """Initialize analysis tools"""
        self.paper_extraction_tool = PaperExtractionTool()
        self.citation_analysis_tool = CitationAnalysisTool()
        self.methodology_evaluation_tool = MethodologyEvaluationTool()

    def setup_agents(self):
        """Setup specialized agents for different analysis tasks"""

        # Content Extraction Agent
        self.content_extractor = Agent(
            role='Research Paper Content Extractor',
            goal='Extract and structure content from research papers including text, metadata, and references',
            backstory="""You are an expert in document processing and content extraction.
            You specialize in parsing academic papers and extracting structured information
            including abstracts, methodologies, results, and citations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.paper_extraction_tool]
        )

        # Methodology Analyst Agent
        self.methodology_analyst = Agent(
            role='Research Methodology Analyst',
            goal='Evaluate research methodologies, experimental designs, and statistical approaches',
            backstory="""You are a senior research methodologist with expertise in evaluating
            the quality and rigor of research designs. You can identify strengths and weaknesses
            in experimental setups and statistical analyses.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.methodology_evaluation_tool]
        )

        # Citation Analyst Agent
        self.citation_analyst = Agent(
            role='Citation and Reference Analyst',
            goal='Analyze citation patterns, reference quality, and academic impact indicators',
            backstory="""You are an expert in bibliometrics and citation analysis. You evaluate
            the quality and relevance of references, analyze citation patterns, and assess
            the academic impact of research work.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.citation_analysis_tool]
        )

        # Quality Assessor Agent
        self.quality_assessor = Agent(
            role='Research Quality Assessor',
            goal='Provide comprehensive quality assessment and recommendations for research papers',
            backstory="""You are a senior academic reviewer with extensive experience in
            peer review processes. You synthesize findings from various analyses to provide
            comprehensive quality assessments and actionable recommendations.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

        # Report Generator Agent
        self.report_generator = Agent(
            role='Analysis Report Generator',
            goal='Generate comprehensive, well-structured analysis reports',
            backstory="""You are an expert technical writer specializing in research analysis
            reports. You excel at synthesizing complex analytical findings into clear,
            actionable reports for academic and research audiences.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def create_analysis_tasks(self, paper_input: PaperAnalysisInput) -> List[Task]:
        """Create analysis tasks for the crew"""

        # Task 1: Content Extraction
        content_extraction_task = Task(
            description=f"""Extract comprehensive content from the research paper at {paper_input.paper_path}.
            Your analysis should include:
            1. Paper metadata (title, authors, publication details)
            2. Abstract and key sections
            3. Methodology description
            4. Results and findings
            5. References and citations
            6. Document structure analysis

            Provide structured output that can be used by other agents for further analysis.""",
            agent=self.content_extractor,
            expected_output="Structured dictionary containing extracted paper content, metadata, and references"
        )

        # Task 2: Methodology Analysis
        methodology_analysis_task = Task(
            description="""Analyze the research methodology extracted from the paper.
            Your analysis should cover:
            1. Research design evaluation
            2. Statistical methods assessment
            3. Experimental setup review
            4. Data collection and analysis procedures
            5. Identification of methodological strengths and weaknesses
            6. Reproducibility assessment

            Provide detailed methodology evaluation with scores and recommendations.""",
            agent=self.methodology_analyst,
            expected_output="Comprehensive methodology evaluation with scores, strengths, weaknesses, and recommendations"
        )

        # Task 3: Citation Analysis
        citation_analysis_task = Task(
            description="""Perform comprehensive citation and reference analysis.
            Your analysis should include:
            1. Citation count and patterns
            2. Reference quality assessment
            3. Recency of citations
            4. Diversity of sources
            5. Academic impact indicators
            6. Citation network analysis

            Provide detailed citation analysis with quality metrics.""",
            agent=self.citation_analyst,
            expected_output="Detailed citation analysis with quality metrics, patterns, and impact assessment"
        )

        # Task 4: Quality Assessment
        quality_assessment_task = Task(
            description="""Synthesize all previous analyses to provide comprehensive quality assessment.
            Your assessment should include:
            1. Overall paper quality score
            2. Strengths and weaknesses summary
            3. Contribution to field assessment
            4. Methodological rigor evaluation
            5. Citation quality impact
            6. Recommendations for improvement

            Provide actionable quality assessment with specific recommendations.""",
            agent=self.quality_assessor,
            expected_output="Comprehensive quality assessment with overall scores and specific recommendations"
        )

        # Task 5: Report Generation
        report_generation_task = Task(
            description=f"""Generate a comprehensive analysis report in {paper_input.output_format} format.
            The report should include:
            1. Executive summary
            2. Detailed findings from all analyses
            3. Quality assessment results
            4. Methodology evaluation
            5. Citation analysis results
            6. Recommendations and conclusions
            7. Supporting data and metrics

            Create a professional, well-structured report suitable for academic review.""",
            agent=self.report_generator,
            expected_output="Professional analysis report with executive summary, detailed findings, and recommendations"
        )

        return [
            content_extraction_task,
            methodology_analysis_task,
            citation_analysis_task,
            quality_assessment_task,
            report_generation_task
        ]

    def analyze_paper(self, paper_input: PaperAnalysisInput) -> Dict[str, Any]:
        """Execute the complete paper analysis workflow"""
        try:
            logger.info(f"Starting analysis of paper: {paper_input.paper_path}")

            # Create tasks
            tasks = self.create_analysis_tasks(paper_input)

            # Create and configure crew
            crew = Crew(
                agents=[
                    self.content_extractor,
                    self.methodology_analyst,
                    self.citation_analyst,
                    self.quality_assessor,
                    self.report_generator
                ],
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )

            # Execute the crew
            logger.info("Executing analysis crew...")
            result = crew.kickoff()

            # Prepare final output
            analysis_result = {
                "paper_path": paper_input.paper_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_type": paper_input.analysis_type,
                "crew_result": result,
                "status": "completed"
            }

            logger.info("Analysis completed successfully")
            return analysis_result

        except Exception as e:
            logger.error(f"Error during paper analysis: {str(e)}")
            return {
                "paper_path": paper_input.paper_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            }

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise


def main():
    """Main execution function with example usage"""
    try:
        # Initialize the analysis system
        print("Initializing CrewAI Research Paper Analysis System...")
        analysis_system = ResearchPaperAnalysisSystem()

        # Example paper analysis
        paper_input = PaperAnalysisInput(
            paper_path="/path/to/research_paper.pdf",
            analysis_type="comprehensive",
            output_format="detailed"
        )

        print(f"Analyzing paper: {paper_input.paper_path}")

        # Execute analysis
        results = analysis_system.analyze_paper(paper_input)

        # Save results
        output_path = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analysis_system.save_results(results, output_path)

        print(f"Analysis completed. Results saved to: {output_path}")
        print("\nAnalysis Summary:")
        print(f"Status: {results['status']}")
        print(f"Timestamp: {results['analysis_timestamp']}")

        if results['status'] == 'completed':
            print("✅ Analysis completed successfully!")
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
