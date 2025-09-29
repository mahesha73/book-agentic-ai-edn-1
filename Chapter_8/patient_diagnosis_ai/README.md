
# Patient Diagnosis AI - Multi-Agent Healthcare Assistant

A comprehensive, production-ready Patient Diagnosis Assistance System built with agentic AI principles using LangChain, LangGraph, and CrewAI. This system demonstrates advanced multi-agent architectures for healthcare applications while maintaining strict HIPAA compliance and observability.

## 🏥 Overview

This system implements a sophisticated multi-agent architecture where specialized AI agents collaborate to assist healthcare professionals in the diagnostic process. Each agent focuses on a specific domain (patient history, medical coding, drug safety, literature research, etc.) and works together through a coordinated workflow.

## 🎯 Key Features

- **Multi-Agent Architecture**: Hierarchical agent system using LangChain, LangGraph, and CrewAI
- **Agentic RAG System**: Advanced retrieval-augmented generation with medical knowledge bases
- **Healthcare API Integration**: Real-time access to FDA, UMLS, SNOMED CT, and PubMed APIs
- **Comprehensive Observability**: Monitoring with LangSmith and LangFuse
- **Multi-Database Architecture**: PostgreSQL, Neo4j, Redis, and MongoDB for different data needs
- **HIPAA Compliance**: Built-in security, encryption, and audit trails
- **Production Ready**: Docker deployment, comprehensive testing, and monitoring

## 🏗️ Architecture

### Agent Hierarchy
```
┌─────────────────────────────────────────────────────────────┐
│                Agent Orchestrator                           │
│           (LangGraph Supervisor Agent)                      │
└─────┬─────────┬─────────┬─────────┬─────────┬──────────────┘
      │         │         │         │         │
┌─────▼───┐ ┌──▼────┐ ┌──▼────┐ ┌──▼────┐ ┌──▼──────────────┐
│Patient  │ │Medical│ │Drug   │ │Image  │ │Literature       │
│History  │ │Coding │ │Safety │ │Analysis│ │Research         │
│Agent    │ │Agent  │ │Agent  │ │Agent  │ │Agent            │
└─────────┘ └───────┘ └───────┘ └───────┘ └─────────────────┘
```

### Technology Stack
- **Frameworks**: LangChain, LangGraph, CrewAI, FastAPI
- **Databases**: PostgreSQL, Neo4j, Redis, MongoDB
- **Observability**: LangSmith, LangFuse
- **APIs**: FDA OpenFDA, UMLS, SNOMED CT, PubMed
- **Deployment**: Docker, Docker Compose

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.10+
- API keys for LLM providers (OpenAI, Anthropic)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd patient_diagnosis_ai
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Run database migrations**
   ```bash
   docker-compose exec app alembic upgrade head
   ```

5. **Access the application**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - LangFuse Dashboard: http://localhost:3000
   - Neo4j Browser: http://localhost:7474

### Development Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📊 Database Schema

### PostgreSQL (Primary Database)
- Patient records and clinical data
- User management and authentication
- Audit logs and compliance tracking

### Neo4j (Knowledge Graph)
- Medical ontologies (SNOMED CT, ICD-10)
- Disease-symptom relationships
- Drug interaction networks
- Clinical pathway modeling

### Redis (Cache & Sessions)
- Session management
- API response caching
- Rate limiting
- Real-time agent communication

### MongoDB (Document Store)
- Unstructured medical documents
- Literature research results
- Agent conversation histories
- Configuration and prompt templates

## 🤖 Agent System

### Core Agents

1. **Patient History Agent**
   - Retrieves and analyzes patient medical history
   - Integrates with FHIR-compliant EHR systems
   - Identifies relevant patterns and risk factors

2. **Medical Coding Agent**
   - Maps symptoms and conditions to standard codes
   - Uses SNOMED CT, ICD-10, and UMLS terminologies
   - Ensures consistent medical language

3. **Drug Safety Agent**
   - Analyzes medication interactions and contraindications
   - Monitors FDA safety alerts and recalls
   - Provides dosage and administration guidance

4. **Literature Research Agent**
   - Searches PubMed and medical databases
   - Synthesizes evidence-based recommendations
   - Tracks latest clinical guidelines

5. **Image Analysis Agent**
   - Processes medical imaging data
   - Integrates with radiology systems
   - Provides preliminary analysis and annotations

### Agent Orchestrator
The supervisor agent coordinates the workflow using LangGraph:
- Routes queries to appropriate specialist agents
- Manages state and context across agents
- Synthesizes results into coherent recommendations
- Handles error recovery and escalation

## 🔒 Security & Compliance

### HIPAA Compliance
- End-to-end encryption for all patient data
- Comprehensive audit logging
- Access controls and authentication
- Data retention and deletion policies
- Business Associate Agreement (BAA) ready

### Security Features
- JWT-based authentication
- Role-based access control (RBAC)
- API rate limiting
- Input validation and sanitization
- Secure configuration management

## 📈 Observability & Monitoring

### LangSmith Integration
- Real-time trace collection
- Performance metrics and analytics
- Error tracking and debugging
- A/B testing capabilities

### LangFuse Integration
- Open-source observability
- Custom metrics and dashboards
- Cost tracking and optimization
- Model performance monitoring

### Health Checks
- Application health endpoints
- Database connectivity monitoring
- External API availability checks
- Resource utilization tracking

## 🧪 Testing

### Test Categories
- Unit tests for individual agents
- Integration tests for multi-agent workflows
- API endpoint testing
- Database integration testing
- Security and compliance testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/api/
```

## 📚 API Documentation

### Core Endpoints

#### Diagnosis Assistance
```http
POST /api/v1/diagnose
Content-Type: application/json

{
  "patient_id": "12345",
  "symptoms": ["chest pain", "shortness of breath"],
  "medical_history": {...},
  "current_medications": [...]
}
```

#### Agent Status
```http
GET /api/v1/agents/status
```

#### Health Check
```http
GET /health
```

### Authentication
All API endpoints require authentication via JWT tokens:
```http
Authorization: Bearer <your-jwt-token>
```

## 🔧 Configuration

### Environment Variables
Key configuration options in `.env`:

- **LLM Configuration**: API keys for OpenAI, Anthropic
- **Database URLs**: Connection strings for all databases
- **API Keys**: Healthcare APIs (UMLS, SNOMED CT)
- **Observability**: LangSmith and LangFuse configuration
- **Security**: Encryption keys and JWT secrets

### Agent Configuration
Agent behavior can be customized through:
- Prompt templates in `src/config/prompts.py`
- Model parameters in `src/config/settings.py`
- Workflow definitions in `src/orchestrator/workflow_graph.py`

## 🚀 Deployment

### Production Deployment
1. **Environment Setup**
   - Configure production environment variables
   - Set up SSL certificates
   - Configure load balancers

2. **Database Setup**
   - Set up managed database services
   - Configure backups and replication
   - Apply security configurations

3. **Monitoring Setup**
   - Configure log aggregation
   - Set up alerting and notifications
   - Deploy monitoring dashboards

### Scaling Considerations
- Horizontal scaling with multiple app instances
- Database read replicas for performance
- Redis clustering for high availability
- CDN for static assets

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

### Code Quality
- Follow PEP 8 style guidelines
- Maintain test coverage above 80%
- Document all public APIs
- Use type hints throughout

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is designed for educational and research purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Review the documentation

---

**Built with ❤️ for advancing healthcare through responsible AI**
