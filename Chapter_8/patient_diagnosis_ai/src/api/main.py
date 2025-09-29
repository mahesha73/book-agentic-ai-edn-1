
"""
Main FastAPI application for the Patient Diagnosis AI system.
"""

import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from src.config.settings import get_settings
from src.orchestrator.supervisor import DiagnosisOrchestrator
from src.orchestrator.state_management import StateManager, PatientContext, Priority
from src.db.engines import get_database_engines
from src.api.models import DiagnosisRequest, DiagnosisResponse
from src.utils.compliance import get_compliance_status
from src.utils.monitoring import get_monitoring_status, health_checker

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Patient Diagnosis AI system")
    
    # Initialize database engines
    db_engines = get_database_engines()
    
    # Initialize orchestrator
    from langchain.llms.fake import FakeListLLM
    llm = FakeListLLM(responses=["This is a medical analysis response."])
    
    state_manager = StateManager()
    orchestrator = DiagnosisOrchestrator(llm, state_manager)
    
    # Store in app state
    app.state.orchestrator = orchestrator
    app.state.db_engines = db_engines
    
    yield
    
    # Shutdown
    logger.info("Shutting down Patient Diagnosis AI system")
    db_engines.close_all()


# Create FastAPI app
app = FastAPI(
    title=settings.app.app_name,
    description=settings.app.app_description,
    version=settings.app.app_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.cors_origins,
    allow_credentials=True,
    allow_methods=settings.app.cors_methods,
    allow_headers=settings.app.cors_headers,
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Patient Diagnosis AI System",
        "version": settings.app.app_version,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Run health checks
        health_results = health_checker.run_checks()
        
        # Check orchestrator health
        orchestrator_health = app.state.orchestrator.health_check()
        health_results["orchestrator"] = orchestrator_health
        
        # Check database health
        db_health = await app.state.db_engines.check_all_health()
        health_results["databases"] = db_health
        
        return health_results
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/api/v1/diagnose", response_model=DiagnosisResponse)
async def create_diagnosis(
    request: DiagnosisRequest,
    background_tasks: BackgroundTasks
):
    """Create a new diagnosis request."""
    try:
        # Create patient context
        patient_context = PatientContext(
            patient_id=request.patient_id,
            age=request.patient_age,
            gender=request.patient_gender,
            chief_complaint=request.chief_complaint,
            symptoms=request.symptoms,
            medical_history=request.medical_history or [],
            current_medications=request.current_medications or [],
            allergies=request.allergies or [],
            vital_signs=request.vital_signs or {},
            lab_results=request.lab_results or [],
            imaging_studies=request.imaging_studies or []
        )
        
        # Process diagnosis request
        final_state = await app.state.orchestrator.process_diagnosis_request(
            patient_context=patient_context,
            clinical_question=request.clinical_question,
            user_id=request.user_id,
            priority=Priority(request.priority) if request.priority else Priority.NORMAL
        )
        
        # Return response
        return DiagnosisResponse(
            session_id=final_state.session_id,
            status=final_state.status.value,
            differential_diagnosis=final_state.differential_diagnosis,
            recommendations=final_state.recommendations,
            safety_alerts=final_state.safety_alerts,
            evidence_summary=final_state.evidence_summary,
            overall_confidence=final_state.overall_confidence,
            requires_escalation=final_state.requires_escalation,
            requires_human_review=final_state.requires_human_review,
            created_at=final_state.created_at,
            updated_at=final_state.updated_at
        )
        
    except Exception as e:
        logger.error(f"Error processing diagnosis request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/diagnose/{session_id}")
async def get_diagnosis_status(session_id: str):
    """Get diagnosis status."""
    try:
        status = await app.state.orchestrator.get_diagnosis_status(session_id)
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        return status
    except Exception as e:
        logger.error(f"Error getting diagnosis status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/compliance")
async def get_compliance():
    """Get HIPAA compliance status."""
    return get_compliance_status()


@app.get("/api/v1/monitoring")
async def get_monitoring():
    """Get monitoring status."""
    return get_monitoring_status()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.app.api_host,
        port=settings.app.api_port,
        reload=settings.app.api_reload,
        workers=settings.app.api_workers if not settings.app.api_reload else 1
    )
