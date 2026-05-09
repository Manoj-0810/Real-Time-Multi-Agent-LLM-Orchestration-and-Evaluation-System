# =============================================================================
# MEGA AI — Log UI (Lightweight FastAPI Interface)
# =============================================================================
# Simple API for querying structured logs. Runs on port 8001.
# =============================================================================

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, LOG_UI_API_KEY
from app.db.connection import get_session

logger = logging.getLogger(__name__)

# =============================================================================
# Create FastAPI App
# =============================================================================

log_ui_app = FastAPI(
    title="MEGA AI — Log Query Interface",
    description="Lightweight interface for querying structured agent logs",
    version="1.0.0",
)

# =============================================================================
# CORS
# =============================================================================

log_ui_app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(request: Request):
    """
    Verify API key for log UI access.
    """

    if not LOG_UI_API_KEY:
        return

    api_key = request.headers.get("X-API-Key", "")

    if api_key != LOG_UI_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

# =============================================================================
# Routes
# =============================================================================

@log_ui_app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "service": "log_ui",
        "timestamp": datetime.utcnow().isoformat(),
    }


@log_ui_app.get("/logs", dependencies=[Depends(verify_api_key)])
async def query_logs(
    job_id: Optional[str] = Query(None, description="Filter by job ID"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Query structured logs.
    """

    from sqlalchemy import and_, select
    from app.db.models import AgentLog

    async with get_session() as session:

        query = select(AgentLog)

        filters = []

        if job_id:
            filters.append(AgentLog.job_id == job_id)

        if agent_id:
            filters.append(AgentLog.agent_id == agent_id)

        if event_type:
            filters.append(AgentLog.event_type == event_type)

        if start_time:
            filters.append(AgentLog.timestamp >= start_time)

        if end_time:
            filters.append(AgentLog.timestamp <= end_time)

        if filters:
            query = query.where(and_(*filters))

        query = (
            query
            .order_by(AgentLog.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )

        result = await session.execute(query)
        logs = result.scalars().all()

        return {
            "total": len(logs),
            "offset": offset,
            "limit": limit,
            "logs": [
                {
                    "id": str(log.id),
                    "job_id": str(log.job_id),
                    "agent_id": log.agent_id,
                    "event_type": log.event_type,
                    "input_hash": log.input_hash,
                    "output_hash": log.output_hash,
                    "input_payload": log.input_payload,
                    "output_payload": log.output_payload,
                    "latency_ms": log.latency_ms,
                    "token_count": log.token_count,
                    "policy_violations": log.policy_violations,
                    "timestamp": (
                        log.timestamp.isoformat()
                        if log.timestamp else None
                    ),
                }
                for log in logs
            ],
        }


@log_ui_app.get("/logs/agents", dependencies=[Depends(verify_api_key)])
async def get_agent_summary(
    hours: int = Query(24, ge=1, le=168),
):
    """
    Get per-agent statistics.
    """

    from sqlalchemy import func, select
    from app.db.models import AgentLog

    since = datetime.utcnow() - timedelta(hours=hours)

    async with get_session() as session:

        query = (
            select(
                AgentLog.agent_id,
                func.count().label("event_count"),
                func.avg(AgentLog.latency_ms).label("avg_latency"),
                func.sum(AgentLog.token_count).label("total_tokens"),
            )
            .where(AgentLog.timestamp >= since)
            .group_by(AgentLog.agent_id)
            .order_by(func.count().desc())
        )

        result = await session.execute(query)
        rows = result.all()

        return {
            "period_hours": hours,
            "agents": [
                {
                    "agent_id": row.agent_id,
                    "event_count": row.event_count,
                    "avg_latency_ms": round(row.avg_latency or 0, 2),
                    "total_tokens": row.total_tokens or 0,
                }
                for row in rows
            ],
        }


@log_ui_app.get("/logs/violations", dependencies=[Depends(verify_api_key)])
async def get_violations(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Get policy violations.
    """

    from sqlalchemy import select
    from app.db.models import AgentLog

    since = datetime.utcnow() - timedelta(hours=hours)

    async with get_session() as session:

        query = (
            select(AgentLog)
            .where(AgentLog.timestamp >= since)
            .where(AgentLog.policy_violations != [])
            .order_by(AgentLog.timestamp.desc())
            .limit(limit)
        )

        result = await session.execute(query)
        logs = result.scalars().all()

        violations = []

        for log in logs:
            for violation in (log.policy_violations or []):

                violations.append({
                    "job_id": str(log.job_id),
                    "agent_id": log.agent_id,
                    "event_type": log.event_type,
                    "violation": violation,
                    "timestamp": (
                        log.timestamp.isoformat()
                        if log.timestamp else None
                    ),
                })

        return {
            "total": len(violations),
            "period_hours": hours,
            "violations": violations,
        }


@log_ui_app.get("/logs/tools", dependencies=[Depends(verify_api_key)])
async def get_tool_usage(
    hours: int = Query(24, ge=1, le=168),
):
    """
    Get tool usage statistics.
    """

    from sqlalchemy import func, select
    from app.db.models import ToolCall

    since = datetime.utcnow() - timedelta(hours=hours)

    async with get_session() as session:

        query = (
            select(
                ToolCall.tool_name,
                func.count().label("call_count"),
                func.avg(ToolCall.latency_ms).label("avg_latency"),
                func.sum(
                    func.case(
                        (ToolCall.accepted == True, 1),
                        else_=0,
                    )
                ).label("accepted_count"),
            )
            .where(ToolCall.timestamp >= since)
            .group_by(ToolCall.tool_name)
            .order_by(func.count().desc())
        )

        result = await session.execute(query)
        rows = result.all()

        return {
            "period_hours": hours,
            "tools": [
                {
                    "tool_name": row.tool_name,
                    "call_count": row.call_count,
                    "avg_latency_ms": round(row.avg_latency or 0, 2),
                    "acceptance_rate": (
                        round(
                            (row.accepted_count or 0) / row.call_count,
                            4,
                        )
                        if row.call_count > 0 else 0
                    ),
                }
                for row in rows
            ],
        }