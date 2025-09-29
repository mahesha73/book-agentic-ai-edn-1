
"""
Database engines and connection management for the Patient Diagnosis AI system.

This module provides database engines for PostgreSQL, Neo4j, Redis, and MongoDB,
with proper connection pooling, health checks, and error handling.
"""

import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager, contextmanager
import asyncio
from functools import lru_cache

# Database drivers
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import redis.asyncio as redis
import redis as sync_redis
from neo4j import GraphDatabase, AsyncGraphDatabase
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DatabaseEngines:
    """
    Database engines manager for all database connections.
    
    This class manages connections to PostgreSQL, Neo4j, Redis, and MongoDB
    with proper connection pooling, health checks, and lifecycle management.
    """
    
    def __init__(self):
        """Initialize database engines."""
        self._postgres_engine = None
        self._postgres_async_engine = None
        self._neo4j_driver = None
        self._neo4j_async_driver = None
        self._redis_client = None
        self._redis_async_client = None
        self._mongodb_client = None
        self._mongodb_async_client = None
        
        # Session makers
        self._postgres_session_maker = None
        self._postgres_async_session_maker = None
    
    # PostgreSQL Engine Management
    @property
    def postgres_engine(self):
        """Get PostgreSQL synchronous engine."""
        if self._postgres_engine is None:
            self._postgres_engine = create_engine(
                settings.database.postgres_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.app.debug,
            )
            logger.info("PostgreSQL synchronous engine created")
        return self._postgres_engine
    
    @property
    def postgres_async_engine(self):
        """Get PostgreSQL asynchronous engine."""
        if self._postgres_async_engine is None:
            # Convert sync URL to async URL
            async_url = settings.database.postgres_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            self._postgres_async_engine = create_async_engine(
                async_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.app.debug,
            )
            logger.info("PostgreSQL asynchronous engine created")
        return self._postgres_async_engine
    
    @property
    def postgres_session_maker(self):
        """Get PostgreSQL session maker."""
        if self._postgres_session_maker is None:
            self._postgres_session_maker = sessionmaker(
                bind=self.postgres_engine,
                class_=Session,
                expire_on_commit=False,
            )
        return self._postgres_session_maker
    
    @property
    def postgres_async_session_maker(self):
        """Get PostgreSQL async session maker."""
        if self._postgres_async_session_maker is None:
            self._postgres_async_session_maker = sessionmaker(
                bind=self.postgres_async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._postgres_async_session_maker
    
    # Neo4j Driver Management
    @property
    def neo4j_driver(self):
        """Get Neo4j synchronous driver."""
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                settings.database.neo4j_uri,
                auth=(settings.database.neo4j_user, settings.database.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
            )
            logger.info("Neo4j synchronous driver created")
        return self._neo4j_driver
    
    @property
    def neo4j_async_driver(self):
        """Get Neo4j asynchronous driver."""
        if self._neo4j_async_driver is None:
            self._neo4j_async_driver = AsyncGraphDatabase.driver(
                settings.database.neo4j_uri,
                auth=(settings.database.neo4j_user, settings.database.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
            )
            logger.info("Neo4j asynchronous driver created")
        return self._neo4j_async_driver
    
    # Redis Client Management
    @property
    def redis_client(self):
        """Get Redis synchronous client."""
        if self._redis_client is None:
            self._redis_client = sync_redis.from_url(
                settings.database.redis_url,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            logger.info("Redis synchronous client created")
        return self._redis_client
    
    @property
    def redis_async_client(self):
        """Get Redis asynchronous client."""
        if self._redis_async_client is None:
            self._redis_async_client = redis.from_url(
                settings.database.redis_url,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            logger.info("Redis asynchronous client created")
        return self._redis_async_client
    
    # MongoDB Client Management
    @property
    def mongodb_client(self):
        """Get MongoDB synchronous client."""
        if self._mongodb_client is None:
            self._mongodb_client = pymongo.MongoClient(
                settings.database.mongodb_uri,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
            )
            logger.info("MongoDB synchronous client created")
        return self._mongodb_client
    
    @property
    def mongodb_async_client(self):
        """Get MongoDB asynchronous client."""
        if self._mongodb_async_client is None:
            self._mongodb_async_client = AsyncIOMotorClient(
                settings.database.mongodb_uri,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
            )
            logger.info("MongoDB asynchronous client created")
        return self._mongodb_async_client
    
    # Database Access Methods
    def get_postgres_db(self):
        """Get PostgreSQL database instance."""
        return self.postgres_session_maker()
    
    def get_mongodb_db(self):
        """Get MongoDB database instance."""
        return self.mongodb_client[settings.database.mongodb_db]
    
    async def get_mongodb_async_db(self):
        """Get MongoDB async database instance."""
        return self.mongodb_async_client[settings.database.mongodb_db]
    
    # Context Managers
    @contextmanager
    def postgres_session(self):
        """Context manager for PostgreSQL session."""
        session = self.get_postgres_db()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def postgres_async_session(self):
        """Context manager for PostgreSQL async session."""
        async with self.postgres_async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    @contextmanager
    def neo4j_session(self):
        """Context manager for Neo4j session."""
        with self.neo4j_driver.session() as session:
            yield session
    
    @asynccontextmanager
    async def neo4j_async_session(self):
        """Context manager for Neo4j async session."""
        async with self.neo4j_async_driver.session() as session:
            yield session
    
    # Health Check Methods
    async def check_postgres_health(self) -> bool:
        """Check PostgreSQL connection health."""
        try:
            async with self.postgres_async_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
    
    async def check_neo4j_health(self) -> bool:
        """Check Neo4j connection health."""
        try:
            async with self.neo4j_async_session() as session:
                result = await session.run("RETURN 1 as health")
                record = await result.single()
                return record["health"] == 1
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False
    
    async def check_redis_health(self) -> bool:
        """Check Redis connection health."""
        try:
            await self.redis_async_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def check_mongodb_health(self) -> bool:
        """Check MongoDB connection health."""
        try:
            db = await self.get_mongodb_async_db()
            await db.command("ping")
            return True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return False
    
    async def check_all_health(self) -> Dict[str, bool]:
        """Check health of all database connections."""
        health_checks = await asyncio.gather(
            self.check_postgres_health(),
            self.check_neo4j_health(),
            self.check_redis_health(),
            self.check_mongodb_health(),
            return_exceptions=True
        )
        
        return {
            "postgres": health_checks[0] if not isinstance(health_checks[0], Exception) else False,
            "neo4j": health_checks[1] if not isinstance(health_checks[1], Exception) else False,
            "redis": health_checks[2] if not isinstance(health_checks[2], Exception) else False,
            "mongodb": health_checks[3] if not isinstance(health_checks[3], Exception) else False,
        }
    
    # Cleanup Methods
    def close_all(self):
        """Close all database connections."""
        if self._postgres_engine:
            self._postgres_engine.dispose()
            logger.info("PostgreSQL engine disposed")
        
        if self._postgres_async_engine:
            asyncio.create_task(self._postgres_async_engine.dispose())
            logger.info("PostgreSQL async engine disposed")
        
        if self._neo4j_driver:
            self._neo4j_driver.close()
            logger.info("Neo4j driver closed")
        
        if self._neo4j_async_driver:
            asyncio.create_task(self._neo4j_async_driver.close())
            logger.info("Neo4j async driver closed")
        
        if self._redis_client:
            self._redis_client.close()
            logger.info("Redis client closed")
        
        if self._redis_async_client:
            asyncio.create_task(self._redis_async_client.close())
            logger.info("Redis async client closed")
        
        if self._mongodb_client:
            self._mongodb_client.close()
            logger.info("MongoDB client closed")
        
        if self._mongodb_async_client:
            self._mongodb_async_client.close()
            logger.info("MongoDB async client closed")


# Global database engines instance
@lru_cache()
def get_database_engines() -> DatabaseEngines:
    """Get the global database engines instance."""
    return DatabaseEngines()


# Convenience functions for dependency injection
def get_postgres_db():
    """Dependency for PostgreSQL database session."""
    engines = get_database_engines()
    with engines.postgres_session() as session:
        yield session


async def get_postgres_async_db():
    """Dependency for PostgreSQL async database session."""
    engines = get_database_engines()
    async with engines.postgres_async_session() as session:
        yield session


def get_neo4j_session():
    """Dependency for Neo4j session."""
    engines = get_database_engines()
    with engines.neo4j_session() as session:
        yield session


async def get_neo4j_async_session():
    """Dependency for Neo4j async session."""
    engines = get_database_engines()
    async with engines.neo4j_async_session() as session:
        yield session


def get_redis_client():
    """Dependency for Redis client."""
    engines = get_database_engines()
    return engines.redis_client


async def get_redis_async_client():
    """Dependency for Redis async client."""
    engines = get_database_engines()
    return engines.redis_async_client


def get_mongodb_db():
    """Dependency for MongoDB database."""
    engines = get_database_engines()
    return engines.get_mongodb_db()


async def get_mongodb_async_db():
    """Dependency for MongoDB async database."""
    engines = get_database_engines()
    return await engines.get_mongodb_async_db()
