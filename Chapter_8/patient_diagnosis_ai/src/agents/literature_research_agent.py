
"""
Literature Research Agent for the Patient Diagnosis AI system.

This agent specializes in searching and analyzing medical literature,
clinical guidelines, and evidence-based recommendations from PubMed
and other medical databases.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentResponse, AgentError, ConfidenceLevel
from src.config.prompts import get_prompt
from src.config.settings import get_settings
from src.knowledge.pubmed_client import PubMedClient

logger = logging.getLogger(__name__)
settings = get_settings()


class LiteratureResearchAgent(BaseAgent):
    """
    Specialized agent for medical literature research and evidence synthesis.
    
    This agent focuses on:
    - PubMed and medical database searching
    - Clinical guideline retrieval and analysis
    - Evidence quality assessment and grading
    - Systematic review and meta-analysis interpretation
    - Clinical practice recommendation synthesis
    """
    
    def __init__(self, llm, pubmed_client: Optional[PubMedClient] = None):
        """
        Initialize the Literature Research Agent.
        
        Args:
            llm: Language model instance
            pubmed_client: PubMed API client
        """
        self.pubmed_client = pubmed_client or PubMedClient()
        
        # Initialize tools
        tools = self._create_tools()
        
        super().__init__(
            name="Literature Research Specialist",
            agent_type="literature_research",
            description="Searches and analyzes medical literature for evidence-based recommendations",
            tools=tools,
            llm=llm,
            max_iterations=settings.agents.agent_retry_attempts,
            timeout_seconds=settings.agents.agent_timeout
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for literature research."""
        return [
            Tool(
                name="search_pubmed",
                description="Search PubMed for medical literature",
                func=self._search_pubmed
            ),
            Tool(
                name="get_clinical_guidelines",
                description="Retrieve clinical practice guidelines",
                func=self._get_clinical_guidelines
            ),
            Tool(
                name="analyze_study_quality",
                description="Analyze the quality and evidence level of studies",
                func=self._analyze_study_quality
            ),
            Tool(
                name="synthesize_evidence",
                description="Synthesize evidence from multiple studies",
                func=self._synthesize_evidence
            ),
            Tool(
                name="grade_recommendations",
                description="Grade strength of clinical recommendations",
                func=self._grade_recommendations
            ),
            Tool(
                name="search_systematic_reviews",
                description="Search for systematic reviews and meta-analyses",
                func=self._search_systematic_reviews
            ),
            Tool(
                name="get_treatment_guidelines",
                description="Get evidence-based treatment guidelines",
                func=self._get_treatment_guidelines
            ),
            Tool(
                name="analyze_publication_bias",
                description="Analyze potential publication bias in literature",
                func=self._analyze_publication_bias
            ),
            Tool(
                name="extract_key_findings",
                description="Extract key findings from research papers",
                func=self._extract_key_findings
            ),
            Tool(
                name="assess_clinical_relevance",
                description="Assess clinical relevance of research findings",
                func=self._assess_clinical_relevance
            )
        ]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with literature research prompt."""
        prompt = get_prompt("literature_research")
        
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
        """Validate input for literature research."""
        if not request:
            raise AgentError(
                "Empty request provided",
                agent_name=self.name,
                error_code="INVALID_INPUT"
            )
        
        # Check for research query
        query = request.get("research_query", "")
        if not query:
            raise AgentError(
                "No research query provided",
                agent_name=self.name,
                error_code="NO_RESEARCH_QUERY"
            )
    
    def _process_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format literature research output."""
        try:
            output_text = raw_output.get("output", "")
            intermediate_steps = raw_output.get("intermediate_steps", [])
            
            processed_output = {
                "research_summary": self._extract_research_summary(output_text),
                "evidence_synthesis": self._extract_evidence_synthesis(output_text),
                "clinical_guidelines": self._extract_clinical_guidelines(output_text),
                "study_analysis": self._extract_study_analysis(output_text),
                "recommendations": self._extract_recommendations(output_text),
                "evidence_quality": self._extract_evidence_quality(output_text),
                "key_findings": self._extract_key_findings_from_output(output_text),
                "clinical_implications": self._extract_clinical_implications(output_text),
                "research_gaps": self._extract_research_gaps(output_text)
            }
            
            # Add metadata
            processed_output["sources_searched"] = self._identify_sources_searched(intermediate_steps)
            processed_output["search_strategy"] = self._extract_search_strategy(intermediate_steps)
            processed_output["evidence_level"] = self._assess_overall_evidence_level(processed_output)
            
            return processed_output
            
        except Exception as e:
            logger.error(f"Error processing literature research output: {e}")
            raise AgentError(
                f"Failed to process research output: {str(e)}",
                agent_name=self.name,
                error_code="OUTPUT_PROCESSING_ERROR"
            )
    
    # Tool Implementation Methods
    
    def _search_pubmed(self, query: str) -> str:
        """Search PubMed for medical literature."""
        try:
            # Perform PubMed search
            search_results = self.pubmed_client.search(
                query=query,
                max_results=20,
                sort="relevance"
            )
            
            if not search_results:
                return f"No results found for query: {query}"
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "pmid": result.get("pmid"),
                    "title": result.get("title"),
                    "authors": result.get("authors", []),
                    "journal": result.get("journal"),
                    "publication_date": result.get("publication_date"),
                    "abstract": result.get("abstract", "")[:500] + "..." if len(result.get("abstract", "")) > 500 else result.get("abstract", ""),
                    "doi": result.get("doi"),
                    "study_type": self._identify_study_type(result.get("abstract", "")),
                    "relevance_score": result.get("relevance_score", 0.5)
                })
            
            return f"PubMed search results for '{query}': Found {len(search_results)} articles. Top results: {formatted_results}"
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return f"Error searching PubMed: {str(e)}"
    
    def _get_clinical_guidelines(self, condition: str) -> str:
        """Retrieve clinical practice guidelines."""
        try:
            # Search for clinical guidelines
            guideline_query = f"{condition} clinical practice guidelines"
            guidelines = self.pubmed_client.search(
                query=guideline_query,
                max_results=10,
                filters=["practice guideline", "consensus development conference"]
            )
            
            if not guidelines:
                return f"No clinical guidelines found for: {condition}"
            
            # Format guidelines
            formatted_guidelines = []
            for guideline in guidelines:
                formatted_guidelines.append({
                    "title": guideline.get("title"),
                    "organization": self._extract_organization(guideline),
                    "publication_date": guideline.get("publication_date"),
                    "pmid": guideline.get("pmid"),
                    "key_recommendations": self._extract_recommendations_from_abstract(guideline.get("abstract", "")),
                    "evidence_level": self._assess_guideline_evidence_level(guideline)
                })
            
            return f"Clinical guidelines for {condition}: {formatted_guidelines}"
            
        except Exception as e:
            logger.error(f"Error retrieving clinical guidelines: {e}")
            return f"Error retrieving clinical guidelines: {str(e)}"
    
    def _analyze_study_quality(self, study_data: str) -> str:
        """Analyze the quality and evidence level of studies."""
        try:
            # Parse study information
            studies = self._parse_study_data(study_data)
            
            quality_analysis = []
            for study in studies:
                analysis = {
                    "pmid": study.get("pmid"),
                    "study_type": study.get("study_type"),
                    "evidence_level": self._determine_evidence_level(study),
                    "sample_size": self._extract_sample_size(study),
                    "study_design_quality": self._assess_study_design(study),
                    "bias_risk": self._assess_bias_risk(study),
                    "clinical_relevance": self._assess_study_relevance(study),
                    "overall_quality": "high"  # Placeholder
                }
                quality_analysis.append(analysis)
            
            return f"Study quality analysis: {quality_analysis}"
            
        except Exception as e:
            logger.error(f"Error analyzing study quality: {e}")
            return f"Error analyzing study quality: {str(e)}"
    
    def _synthesize_evidence(self, studies: str) -> str:
        """Synthesize evidence from multiple studies."""
        try:
            # Parse studies
            study_list = self._parse_study_data(studies)
            
            synthesis = {
                "total_studies": len(study_list),
                "study_types": self._categorize_study_types(study_list),
                "consistent_findings": [],
                "conflicting_findings": [],
                "overall_conclusion": "",
                "strength_of_evidence": "moderate",
                "clinical_significance": "significant"
            }
            
            # Analyze consistency of findings
            findings = [self._extract_main_finding(study) for study in study_list]
            synthesis["consistent_findings"] = self._identify_consistent_findings(findings)
            synthesis["conflicting_findings"] = self._identify_conflicting_findings(findings)
            
            # Generate overall conclusion
            synthesis["overall_conclusion"] = self._generate_evidence_conclusion(synthesis)
            
            return f"Evidence synthesis: {synthesis}"
            
        except Exception as e:
            logger.error(f"Error synthesizing evidence: {e}")
            return f"Error synthesizing evidence: {str(e)}"
    
    def _grade_recommendations(self, evidence_data: str) -> str:
        """Grade strength of clinical recommendations."""
        try:
            # Parse evidence data
            evidence = self._parse_evidence_data(evidence_data)
            
            grading = {
                "recommendation_strength": "strong",  # strong, weak, conditional
                "evidence_quality": "high",  # high, moderate, low, very low
                "grade": "A",  # A, B, C, D
                "rationale": "",
                "considerations": []
            }
            
            # Apply GRADE methodology
            grading = self._apply_grade_methodology(evidence, grading)
            
            return f"Recommendation grading: {grading}"
            
        except Exception as e:
            logger.error(f"Error grading recommendations: {e}")
            return f"Error grading recommendations: {str(e)}"
    
    def _search_systematic_reviews(self, topic: str) -> str:
        """Search for systematic reviews and meta-analyses."""
        try:
            # Search for systematic reviews
            review_query = f"{topic} systematic review OR meta-analysis"
            reviews = self.pubmed_client.search(
                query=review_query,
                max_results=15,
                filters=["systematic review", "meta-analysis"]
            )
            
            if not reviews:
                return f"No systematic reviews found for: {topic}"
            
            # Format reviews
            formatted_reviews = []
            for review in reviews:
                formatted_reviews.append({
                    "title": review.get("title"),
                    "pmid": review.get("pmid"),
                    "publication_date": review.get("publication_date"),
                    "journal": review.get("journal"),
                    "review_type": self._classify_review_type(review),
                    "included_studies": self._extract_included_studies_count(review),
                    "main_conclusion": self._extract_main_conclusion(review),
                    "quality_assessment": self._assess_review_quality(review)
                })
            
            return f"Systematic reviews for {topic}: {formatted_reviews}"
            
        except Exception as e:
            logger.error(f"Error searching systematic reviews: {e}")
            return f"Error searching systematic reviews: {str(e)}"
    
    def _get_treatment_guidelines(self, condition: str) -> str:
        """Get evidence-based treatment guidelines."""
        try:
            # Search for treatment guidelines
            treatment_query = f"{condition} treatment guidelines therapy"
            guidelines = self.pubmed_client.search(
                query=treatment_query,
                max_results=10,
                filters=["practice guideline", "review"]
            )
            
            treatment_guidelines = []
            for guideline in guidelines:
                treatment_guidelines.append({
                    "condition": condition,
                    "guideline_title": guideline.get("title"),
                    "organization": self._extract_organization(guideline),
                    "first_line_treatment": self._extract_first_line_treatment(guideline),
                    "alternative_treatments": self._extract_alternative_treatments(guideline),
                    "contraindications": self._extract_contraindications_from_text(guideline.get("abstract", "")),
                    "monitoring_requirements": self._extract_monitoring_from_text(guideline.get("abstract", "")),
                    "evidence_level": self._assess_guideline_evidence_level(guideline)
                })
            
            return f"Treatment guidelines for {condition}: {treatment_guidelines}"
            
        except Exception as e:
            logger.error(f"Error getting treatment guidelines: {e}")
            return f"Error getting treatment guidelines: {str(e)}"
    
    def _analyze_publication_bias(self, studies: str) -> str:
        """Analyze potential publication bias in literature."""
        try:
            study_list = self._parse_study_data(studies)
            
            bias_analysis = {
                "total_studies": len(study_list),
                "publication_bias_indicators": [],
                "funnel_plot_assessment": "not_available",
                "small_study_effects": False,
                "language_bias": False,
                "time_lag_bias": False,
                "overall_bias_risk": "low"
            }
            
            # Analyze for bias indicators
            if len(study_list) < 10:
                bias_analysis["publication_bias_indicators"].append("Small number of studies")
            
            # Check for small study effects
            small_studies = [s for s in study_list if self._extract_sample_size(s) < 100]
            if len(small_studies) > len(study_list) * 0.5:
                bias_analysis["small_study_effects"] = True
            
            return f"Publication bias analysis: {bias_analysis}"
            
        except Exception as e:
            logger.error(f"Error analyzing publication bias: {e}")
            return f"Error analyzing publication bias: {str(e)}"
    
    def _extract_key_findings(self, paper_data: str) -> str:
        """Extract key findings from research papers."""
        try:
            # Parse paper data
            papers = self._parse_paper_data(paper_data)
            
            key_findings = []
            for paper in papers:
                findings = {
                    "pmid": paper.get("pmid"),
                    "title": paper.get("title"),
                    "primary_outcome": self._extract_primary_outcome(paper),
                    "secondary_outcomes": self._extract_secondary_outcomes(paper),
                    "statistical_significance": self._extract_statistical_significance(paper),
                    "clinical_significance": self._assess_clinical_significance_of_findings(paper),
                    "limitations": self._extract_study_limitations(paper),
                    "implications": self._extract_clinical_implications_from_paper(paper)
                }
                key_findings.append(findings)
            
            return f"Key findings extracted: {key_findings}"
            
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return f"Error extracting key findings: {str(e)}"
    
    def _assess_clinical_relevance(self, research_data: str) -> str:
        """Assess clinical relevance of research findings."""
        try:
            # Parse research data
            research = self._parse_research_data(research_data)
            
            relevance_assessment = {
                "clinical_applicability": "high",
                "patient_population_relevance": "applicable",
                "intervention_feasibility": "feasible",
                "outcome_importance": "important",
                "practice_changing_potential": "moderate",
                "implementation_barriers": [],
                "overall_relevance": "high"
            }
            
            # Assess various aspects of clinical relevance
            relevance_assessment = self._perform_relevance_assessment(research, relevance_assessment)
            
            return f"Clinical relevance assessment: {relevance_assessment}"
            
        except Exception as e:
            logger.error(f"Error assessing clinical relevance: {e}")
            return f"Error assessing clinical relevance: {str(e)}"
    
    # Helper Methods (simplified implementations)
    
    def _identify_study_type(self, abstract: str) -> str:
        """Identify study type from abstract."""
        abstract_lower = abstract.lower()
        if "randomized controlled trial" in abstract_lower or "rct" in abstract_lower:
            return "randomized_controlled_trial"
        elif "systematic review" in abstract_lower:
            return "systematic_review"
        elif "meta-analysis" in abstract_lower:
            return "meta_analysis"
        elif "cohort" in abstract_lower:
            return "cohort_study"
        elif "case-control" in abstract_lower:
            return "case_control_study"
        else:
            return "other"
    
    def _extract_organization(self, guideline: Dict[str, Any]) -> str:
        """Extract organization from guideline."""
        # Simplified implementation
        return "Professional Medical Organization"
    
    def _extract_recommendations_from_abstract(self, abstract: str) -> List[str]:
        """Extract recommendations from abstract."""
        # Simplified implementation
        return ["Recommendation 1", "Recommendation 2"]
    
    def _assess_guideline_evidence_level(self, guideline: Dict[str, Any]) -> str:
        """Assess evidence level of guideline."""
        # Simplified implementation
        return "Level A"
    
    # Additional helper methods would be implemented here...
    
    def _calculate_confidence(self, output: Dict[str, Any]) -> ConfidenceLevel:
        """Calculate confidence level for literature research."""
        evidence_level = output.get("evidence_level", "moderate")
        
        if evidence_level == "high":
            return ConfidenceLevel.VERY_HIGH
        elif evidence_level == "moderate":
            return ConfidenceLevel.HIGH
        elif evidence_level == "low":
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _requires_escalation(self, output: Dict[str, Any]) -> bool:
        """Determine if research findings require escalation."""
        # Escalate if conflicting evidence or low quality
        evidence_synthesis = output.get("evidence_synthesis", {})
        conflicting_findings = evidence_synthesis.get("conflicting_findings", [])
        
        return len(conflicting_findings) > 3
    
    def _suggest_next_actions(self, output: Dict[str, Any]) -> List[str]:
        """Suggest next actions based on research findings."""
        actions = ["Review evidence with clinical context"]
        
        research_gaps = output.get("research_gaps", [])
        if research_gaps:
            actions.append("Consider additional research to fill evidence gaps")
        
        return actions
    
    # Placeholder methods for complex functionality
    def _parse_study_data(self, data): return []
    def _determine_evidence_level(self, study): return "moderate"
    def _extract_sample_size(self, study): return 100
    def _assess_study_design(self, study): return "good"
    def _assess_bias_risk(self, study): return "low"
    def _assess_study_relevance(self, study): return "high"
    def _categorize_study_types(self, studies): return {}
    def _extract_main_finding(self, study): return ""
    def _identify_consistent_findings(self, findings): return []
    def _identify_conflicting_findings(self, findings): return []
    def _generate_evidence_conclusion(self, synthesis): return ""
    def _parse_evidence_data(self, data): return {}
    def _apply_grade_methodology(self, evidence, grading): return grading
    def _classify_review_type(self, review): return "systematic_review"
    def _extract_included_studies_count(self, review): return 10
    def _extract_main_conclusion(self, review): return ""
    def _assess_review_quality(self, review): return "high"
    def _extract_first_line_treatment(self, guideline): return ""
    def _extract_alternative_treatments(self, guideline): return []
    def _extract_contraindications_from_text(self, text): return []
    def _extract_monitoring_from_text(self, text): return []
    def _parse_paper_data(self, data): return []
    def _extract_primary_outcome(self, paper): return ""
    def _extract_secondary_outcomes(self, paper): return []
    def _extract_statistical_significance(self, paper): return ""
    def _assess_clinical_significance_of_findings(self, paper): return ""
    def _extract_study_limitations(self, paper): return []
    def _extract_clinical_implications_from_paper(self, paper): return []
    def _parse_research_data(self, data): return {}
    def _perform_relevance_assessment(self, research, assessment): return assessment
    
    # Output extraction methods
    def _extract_research_summary(self, text): return {}
    def _extract_evidence_synthesis(self, text): return {}
    def _extract_clinical_guidelines(self, text): return []
    def _extract_study_analysis(self, text): return {}
    def _extract_recommendations(self, text): return []
    def _extract_evidence_quality(self, text): return {}
    def _extract_key_findings_from_output(self, text): return []
    def _extract_clinical_implications(self, text): return []
    def _extract_research_gaps(self, text): return []
    def _identify_sources_searched(self, steps): return []
    def _extract_search_strategy(self, steps): return {}
    def _assess_overall_evidence_level(self, output): return "moderate"
