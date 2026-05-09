# =============================================================================
# MEGA AI — Database Connection Manager
# =============================================================================
# Async PostgreSQL connection pool with pgvector support.
# =============================================================================

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import (
    DATABASE_CONFIG,
    DATABASE_URL,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Globals
# =============================================================================

_engine = None

_session_maker = None


# =============================================================================
# Engine
# =============================================================================

def get_engine():
    """
    Get or create async DB engine.
    """

    global _engine

    if _engine is None:

        _engine = create_async_engine(
            DATABASE_URL,

            pool_size=int(
                DATABASE_CONFIG.get(
                    "pool_size",
                    10,
                )
            ),

            max_overflow=int(
                DATABASE_CONFIG.get(
                    "max_overflow",
                    20,
                )
            ),

            pool_timeout=int(
                DATABASE_CONFIG.get(
                    "pool_timeout",
                    30,
                )
            ),

            echo=False,

            future=True,
        )

        logger.info(
            "Database engine initialized"
        )

    return _engine


# =============================================================================
# Session Maker
# =============================================================================

def get_session_maker() -> async_sessionmaker:
    """
    Get or create async session maker.
    """

    global _session_maker

    if _session_maker is None:

        _session_maker = async_sessionmaker(
            get_engine(),

            class_=AsyncSession,

            expire_on_commit=False,

            autoflush=False,
        )

    return _session_maker


# =============================================================================
# Session Context
# =============================================================================

@asynccontextmanager
async def get_session() -> AsyncGenerator[
    AsyncSession,
    None,
]:
    """
    Async DB session context manager.
    """

    session_maker = get_session_maker()

    session = session_maker()

    try:

        yield session

        await session.commit()

    except Exception:

        await session.rollback()

        raise

    finally:

        await session.close()


# =============================================================================
# DB Init
# =============================================================================

async def init_db() -> None:
    """
    Initialize database tables.
    """

    from app.db.models import Base

    engine = get_engine()

    async with engine.begin() as conn:

        await conn.run_sync(
            Base.metadata.create_all
        )

    logger.info(
        "Database tables initialized"
    )


# =============================================================================
# Shutdown
# =============================================================================

async def close_db() -> None:
    """
    Close database connections.
    """

    global _engine

    if _engine:

        await _engine.dispose()

        _engine = None

        logger.info(
            "Database connections closed"
        )


# =============================================================================
# Health Check
# =============================================================================

async def health_check() -> Dict[str, Any]:
    """
    Check database connectivity.
    """

    import time
    from sqlalchemy import text

    start = time.time()

    try:

        engine = get_engine()

        async with engine.connect() as conn:

            result = await conn.execute(
                text("SELECT 1")
            )

            row = result.scalar()

        latency_ms = int(
            (
                time.time()
                - start
            ) * 1000
        )

        return {
            "status": "healthy",

            "latency_ms": latency_ms,

            "database": DATABASE_CONFIG.get(
                "database"
            ),

            "connected": row == 1,
        }

    except Exception as e:

        latency_ms = int(
            (
                time.time()
                - start
            ) * 1000
        )

        logger.error(
            f"Database health check failed: {e}"
        )

        return {
            "status": "unhealthy",

            "latency_ms": latency_ms,

            "error": str(e),
        }


# =============================================================================
# pgvector Helpers
# =============================================================================

async def insert_document_chunk(
    content: str,
    embedding: List[float],
    source: str = "",
    chunk_index: int = 0,
) -> str:
    """
    Insert RAG document chunk.
    """

    from sqlalchemy import text

    async with get_session() as session:

        embedding_str = (
            f"[{','.join(str(x) for x in embedding)}]"
        )

        query = text("""
            INSERT INTO rag_documents
            (
                content,
                embedding,
                source,
                chunk_index
            )
            VALUES
            (
                :content,
                :embedding::vector,
                :source,
                :chunk_index
            )
            RETURNING id
        """)

        result = await session.execute(
            query,
            {
                "content": content,

                "embedding": embedding_str,

                "source": source,

                "chunk_index": chunk_index,
            },
        )

        doc_id = result.scalar()

        await session.commit()

        logger.info(
            f"Inserted document chunk "
            f"{chunk_index} from {source}"
        )

        return str(doc_id)


# =============================================================================
# Similarity Search
# =============================================================================

async def similarity_search(
    query_embedding: List[float],
    top_k: int = 5,
    threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Search vector DB using cosine similarity.
    """

    from sqlalchemy import text

    engine = get_engine()

    embedding_str = (
        f"[{','.join(str(x) for x in query_embedding)}]"
    )

    async with engine.connect() as conn:

        query = text("""
            SELECT
                id,
                content,
                source,
                chunk_index,
                1 - (embedding <=> :embedding::vector)
                    as similarity
            FROM rag_documents
            WHERE
                1 - (
                    embedding <=> :embedding::vector
                ) > :threshold
            ORDER BY embedding <=> :embedding::vector
            LIMIT :top_k
        """)

        result = await conn.execute(
            query,
            {
                "embedding": embedding_str,

                "top_k": top_k,

                "threshold": threshold,
            },
        )

        rows = []

        for row in result.mappings():

            rows.append({
                "id": str(row["id"]),

                "content": row["content"],

                "source": row["source"],

                "chunk_index": row[
                    "chunk_index"
                ],

                "similarity": float(
                    row["similarity"]
                ),
            })

        return rows