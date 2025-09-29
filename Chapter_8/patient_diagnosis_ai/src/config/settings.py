
"""
Configuration settings for the Patient Diagnosis AI system.

This module handles all application configuration including database connections,
API keys, security settings, and agent parameters. It uses Pydantic for
validation and type safety.
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL (Primary Database)
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="patient_diagnosis_ai", env="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    
    # Neo4j (Knowledge Graph)
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="neo4j", env="NEO4J_PASSWORD")
    
    # Redis (Cache and Sessions)
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # MongoDB (Document Store)
    mongodb_uri: str = Field(default="mongodb://localhost:27017", env="MONGODB_URI")
    mongodb_db: str = Field(default="patient_diagnosis_ai", env="MONGODB_DB")
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class LLMSettings(BaseSettings):
    """LLM provider configuration settings."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    anthropic_temperature: float = Field(default=0.1, env="ANTHROPIC_TEMPERATURE")
    anthropic_max_tokens: int = Field(default=4000, env="ANTHROPIC_MAX_TOKENS")
    
    # Default LLM Provider
    default_llm_provider: str = Field(default="openai", env="DEFAULT_LLM_PROVIDER")
    
    @validator("default_llm_provider")
    def validate_llm_provider(cls, v):
        """Validate LLM provider choice."""
        if v not in ["openai", "anthropic"]:
            raise ValueError("LLM provider must be 'openai' or 'anthropic'")
        return v


class ObservabilitySettings(BaseSettings):
    """Observability and monitoring configuration."""
    
    # LangSmith Configuration
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="patient-diagnosis-ai", env="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, env="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")
    
    # LangFuse Configuration
    langfuse_secret_key: Optional[str] = Field(default=None, env="LANGFUSE_SECRET_KEY")
    langfuse_public_key: Optional[str] = Field(default=None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", env="LANGFUSE_HOST")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")


class HealthcareAPISettings(BaseSettings):
    """Healthcare API configuration settings."""
    
    # FDA OpenFDA API
    fda_api_base_url: str = Field(default="https://api.fda.gov", env="FDA_API_BASE_URL")
    
    # UMLS API (requires license)
    umls_api_key: Optional[str] = Field(default=None, env="UMLS_API_KEY")
    umls_api_base_url: str = Field(default="https://uts-ws.nlm.nih.gov/rest", env="UMLS_API_BASE_URL")
    
    # SNOMED CT API (requires license)
    snomed_api_key: Optional[str] = Field(default=None, env="SNOMED_API_KEY")
    snomed_api_base_url: str = Field(default="https://snowstorm.ihtsdotools.org", env="SNOMED_API_BASE_URL")
    
    # PubMed API
    pubmed_api_base_url: str = Field(default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils", env="PUBMED_API_BASE_URL")
    pubmed_email: Optional[str] = Field(default=None, env="PUBMED_EMAIL")
    pubmed_tool: str = Field(default="patient-diagnosis-ai", env="PUBMED_TOOL")


class SecuritySettings(BaseSettings):
    """Security and compliance configuration."""
    
    # JWT Configuration
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Encryption Configuration
    encryption_key: str = Field(default="change-me-32-bytes-key-for-prod", env="ENCRYPTION_KEY")
    
    # HIPAA Compliance
    audit_log_enabled: bool = Field(default=True, env="AUDIT_LOG_ENABLED")
    data_retention_days: int = Field(default=2555, env="DATA_RETENTION_DAYS")  # 7 years
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=100, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_burst: int = Field(default=20, env="RATE_LIMIT_BURST")
    
    @validator("encryption_key")
    def validate_encryption_key(cls, v):
        """Validate encryption key length."""
        if len(v.encode()) != 32:
            raise ValueError("Encryption key must be exactly 32 bytes")
        return v


class AgentSettings(BaseSettings):
    """Agent-specific configuration settings."""
    
    # Agent Timeouts (in seconds)
    agent_timeout: int = Field(default=300, env="AGENT_TIMEOUT")
    agent_retry_attempts: int = Field(default=3, env="AGENT_RETRY_ATTEMPTS")
    agent_retry_delay: float = Field(default=1.0, env="AGENT_RETRY_DELAY")
    
    # Agent Concurrency
    max_concurrent_agents: int = Field(default=5, env="MAX_CONCURRENT_AGENTS")
    agent_queue_size: int = Field(default=100, env="AGENT_QUEUE_SIZE")
    
    # Agent Memory Configuration
    agent_memory_max_tokens: int = Field(default=8000, env="AGENT_MEMORY_MAX_TOKENS")
    agent_memory_window: int = Field(default=10, env="AGENT_MEMORY_WINDOW")
    
    # Specialized Agent Settings
    patient_history_agent_enabled: bool = Field(default=True, env="PATIENT_HISTORY_AGENT_ENABLED")
    medical_coding_agent_enabled: bool = Field(default=True, env="MEDICAL_CODING_AGENT_ENABLED")
    drug_safety_agent_enabled: bool = Field(default=True, env="DRUG_SAFETY_AGENT_ENABLED")
    literature_research_agent_enabled: bool = Field(default=True, env="LITERATURE_RESEARCH_AGENT_ENABLED")
    image_analysis_agent_enabled: bool = Field(default=False, env="IMAGE_ANALYSIS_AGENT_ENABLED")


class ApplicationSettings(BaseSettings):
    """Main application configuration."""
    
    # Application Metadata
    app_name: str = Field(default="Patient Diagnosis AI", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    app_description: str = Field(default="Multi-Agent Patient Diagnosis Assistance System", env="APP_DESCRIPTION")
    
    # Environment
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    
    # CORS Configuration
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections."""
    
    # Configuration sections
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    healthcare_apis: HealthcareAPISettings = HealthcareAPISettings()
    security: SecuritySettings = SecuritySettings()
    agents: AgentSettings = AgentSettings()
    app: ApplicationSettings = ApplicationSettings()
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def model_dump_safe(self) -> Dict[str, Any]:
        """Return configuration without sensitive information."""
        config = self.dict()
        
        # Remove sensitive keys
        sensitive_keys = [
            "postgres_password", "neo4j_password", "redis_password",
            "openai_api_key", "anthropic_api_key", "langsmith_api_key",
            "langfuse_secret_key", "umls_api_key", "snomed_api_key",
            "secret_key", "encryption_key"
        ]
        
        def remove_sensitive(obj, keys):
            if isinstance(obj, dict):
                return {k: remove_sensitive(v, keys) if k not in keys else "***" 
                       for k, v in obj.items()}
            return obj
        
        return remove_sensitive(config, sensitive_keys)


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.
    
    This function creates and caches the settings instance to avoid
    repeated environment variable parsing.
    
    Returns:
        Settings: The application settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()
