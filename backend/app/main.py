# =============================================================================
# MEGA AI — Main FastAPI Application
# =============================================================================
# Exactly 5 endpoints as specified.
# SSE streaming, job tracking, evaluation, and prompt rewrite management.
# =============================================================================

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.agents.base import BaseAgent
from app.agents.compression import CompressionAgent
from app.agents.critique import CritiqueAgent
from app.agents.decomposition import DecompositionAgent
from app.agents.meta import MetaAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.rag import RAGAgent
from app.agents.synthesis import SynthesisAgent

from app.celery_app import celery_app

from app.config import (
    APP_CONFIG,
    CORS_ORIGINS,
    ENABLED_AGENTS,
)

from app.context import ContextBudgetManager

from app.db.connection import (
    close_db,
    health_check as db_health_check,
    init_db,
    create_job,
    update_job_status,
)

from app.evaluation.pipeline import EvaluationPipeline
from app.evaluation.scorer import EvalScorer
from app.evaluation.test_cases import get_test_cases

from app.llm_gateway import LLMGateway

from app.schemas import (
    BudgetStatus,
    ContextObject,
    EvalRunSummary,
    JobStatus,
    JobTraceResponse,
    PromptRewriteProposal,
    QueryRequest,
    QueryResponse,
    RewriteReviewRequest,
)

from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info("MEGA AI starting up...")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Initialize LLM gateway
    app.state.llm_gateway = LLMGateway()
    app.state.tool_registry = ToolRegistry()
    
    logger.info("MEGA AI startup complete")
    
    yield
    
    # Shutdown
    logger.info("MEGA AI shutting down...")
    await close_db()
    logger.info("MEGA AI shutdown complete")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="MEGA AI",
    description="Multi-Agent Evaluation & Generation Architecture",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Helper Functions
# =============================================================================

def get_llm_gateway(request: Request) -> LLMGateway:
    """Get LLM gateway from app state.
    
    Args:
        request: FastAPI request.
        
    Returns:
        LLMGateway instance.
    """
    return request.app.state.llm_gateway


def get_tool_registry(request: Request) -> ToolRegistry:
    """Get tool registry from app state.
    
    Args:
        request: FastAPI request.
        
    Returns:
        ToolRegistry instance.
    """
    return request.app.state.tool_registry


def create_agents(
    llm: LLMGateway,
    tools: ToolRegistry,
    budget: Optional[ContextBudgetManager] = None,
) -> Dict[str, BaseAgent]:
    """Create all agent instances.
    
    Args:
        llm: LLM gateway.
        tools: Tool registry.
        budget: Optional budget manager.
        
    Returns:
        Dict mapping agent_id -> agent instance.
    """
    agents: Dict[str, BaseAgent] = {}
    
    agent_classes = {
        "orchestrator": OrchestratorAgent,
        "decomposition": DecompositionAgent,
        "rag": RAGAgent,
        "critique": CritiqueAgent,
        "synthesis": SynthesisAgent,
        "compression": CompressionAgent,
        "meta": MetaAgent,
    }
    
    for agent_id, agent_class in agent_classes.items():
        if agent_id in ENABLED_AGENTS:
            try:
                agents[agent_id] = agent_class(
                    llm_gateway=llm,
                    tool_registry=tools,
                    budget_manager=budget,
                )
                logger.debug(f"Created agent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to create agent {agent_id}: {e}")
    
    return agents


# =============================================================================
# Endpoint 1: POST /api/v1/query — SSE Streaming
# =============================================================================

@app.post("/api/v1/query")
async def submit_query(
    request: QueryRequest,
    http_request: Request,
):
    """Submit a query and receive streaming SSE response.
    
    Events:
    - job_start: {job_id, timestamp}
    - agent_start: {agent_id, timestamp}
    - agent_token: {agent_id, token}
    - tool_call: {agent_id, tool_name, status, retry_number}
    - budget_update: {agent_id, remaining_tokens}
    - job_complete: {job_id, summary}
    
    Args:
        request: Query request.
        http_request: HTTP request.
        
    Returns:
        StreamingResponse with SSE events.
    """
    job_id = str(uuid.uuid4())
    query = request.query
    context = request.context or {}
    
    logger.info(f"Query submitted: {job_id}", extra={"query": query[:50]})
    
    # Create context and budget manager
    ctx = ContextObject(
        job_id=job_id,
        query=query,
        metadata=context,
    )
    
    budget = ContextBudgetManager(
        job_id=job_id,
        default_budget=APP_CONFIG.get("default_context_budget", 8192),
    )
    
    # Get shared resources
    llm = get_llm_gateway(http_request)
    tools = get_tool_registry(http_request)
    
    # Create agents
    agents = create_agents(llm, tools, budget)
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        start_time = datetime.utcnow()
        
        # 1. Create job record in DB
        await create_job(job_id, query)
        
        # Send job start
        yield f"event: job_start\ndata: {json.dumps({'job_id': job_id, 'timestamp': start_time.isoformat()})}\n\n"
        
        streamed_tools = set()
        
        try:
            # Execute orchestrator
            orchestrator = agents.get("orchestrator")
            if orchestrator:
                yield f"event: agent_start\ndata: {json.dumps({'agent_id': 'orchestrator', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                
                output = await orchestrator.execute(ctx)
                ctx.agent_outputs["orchestrator"] = output
                
                # Stream orchestrator tokens (simulated real-time token streaming)
                if output.content:
                    words = output.content.split(' ')
                    for i, word in enumerate(words):
                        token = word + ' ' if i < len(words) - 1 else word
                        yield f"event: agent_token\ndata: {json.dumps({'agent_id': 'orchestrator', 'token': token})}\n\n"
                        await asyncio.sleep(0.005)
                
                yield f"event: agent_end\ndata: {json.dumps({'agent_id': 'orchestrator', 'status': 'completed'})}\n\n"
            
            # Execute agent chain
            execution_order = _get_execution_order(ctx)
            
            for agent_id in execution_order:
                agent = agents.get(agent_id)
                if not agent:
                    continue
                
                # Send agent start
                yield f"event: agent_start\ndata: {json.dumps({'agent_id': agent_id, 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                
                try:
                    # Check budget
                    remaining = budget.check_remaining(agent_id)
                    yield f"event: budget_update\ndata: {json.dumps({'agent_id': agent_id, 'remaining_tokens': remaining})}\n\n"
                    
                    # Trigger compression if needed
                    if budget.should_trigger_compression(agent_id) and APP_CONFIG.get("enable_compression"):
                        compression = agents.get("compression")
                        if compression:
                            yield f"event: compression_triggered\ndata: {json.dumps({'agent_id': agent_id, 'reason': 'budget_threshold'})}\n\n"
                            await compression.execute(ctx)
                    
                    # Execute agent
                    output = await agent.execute(ctx)
                    ctx.agent_outputs[agent_id] = output
                    
                    # Stream tokens (simulated real-time token streaming)
                    if output.content:
                        words = output.content.split(' ')
                        for i, word in enumerate(words):
                            token = word + ' ' if i < len(words) - 1 else word
                            yield f"event: agent_token\ndata: {json.dumps({'agent_id': agent_id, 'token': token})}\n\n"
                            await asyncio.sleep(0.005)
                    
                    # Yield tool calls that occurred during this agent execution
                    for key, tr in list(ctx.tool_results.items()):
                        if tr.tool_name not in streamed_tools:
                            yield f"event: tool_call\ndata: {json.dumps({'agent_id': agent_id, 'tool_name': tr.tool_name, 'status': 'completed' if tr.accepted else 'failed', 'retry_number': tr.retry_number})}\n\n"
                            streamed_tools.add(tr.tool_name)
                    
                    yield f"event: agent_end\ndata: {json.dumps({'agent_id': agent_id, 'status': 'completed'})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}", extra={"job_id": job_id})
                    yield f"event: agent_end\ndata: {json.dumps({'agent_id': agent_id, 'status': 'failed', 'error': str(e)})}\n\n"
            
            # Job complete
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Get final answer
            synthesis = ctx.agent_outputs.get("synthesis", {})
            final_answer = ""
            if synthesis and hasattr(synthesis, 'content'):
                try:
                    data = json.loads(synthesis.content)
                    final_answer = data.get("answer", synthesis.content)
                except (json.JSONDecodeError, TypeError):
                    final_answer = synthesis.content
            
            summary = {
                "job_id": job_id,
                "status": "completed",
                "duration_ms": duration_ms,
                "agents_invoked": list(ctx.agent_outputs.keys()),
                "answer_preview": final_answer[:200] if final_answer else "",
                "tokens_used": budget.get_total_usage().get("total_consumed", 0),
            }
            
            # Update job status in database to completed
            await update_job_status(job_id, "completed")
            
            yield f"event: job_complete\ndata: {json.dumps(summary)}\n\n"
            
        except Exception as e:
            logger.error(f"Job failed: {e}", extra={"job_id": job_id})
            # Update job status in database to failed
            await update_job_status(job_id, "failed")
            yield f"event: error\ndata: {json.dumps({'job_id': job_id, 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _get_execution_order(ctx: ContextObject) -> List[str]:
    """Get agent execution order from orchestrator output.
    
    Args:
        ctx: Context object.
        
    Returns:
        List of agent IDs.
    """
    orchestrator_output = ctx.agent_outputs.get("orchestrator", {})
    
    if orchestrator_output and hasattr(orchestrator_output, "content"):
        try:
            data = json.loads(orchestrator_output.content)
            order = data.get("execution_order", [])
            if order:
                return [a for a in order if a != "orchestrator"]
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Fallback order
    return ["decomposition", "rag", "critique", "synthesis"]


# =============================================================================
# Endpoint 2: GET /api/v1/jobs/{job_id}/trace
# =============================================================================

@app.get("/api/v1/jobs/{job_id}/trace")
async def get_job_trace(
    job_id: str = Path(..., description="Job ID"),
):
    """Get full execution trace for a job.
    
    Args:
        job_id: The job ID.
        
    Returns:
        JobTraceResponse with execution details.
    """
    
    from sqlalchemy import select
    from app.db.connection import get_session
    from app.db.models import AgentLog, ToolCall
    
    async with get_session() as session:
        # Get agent logs
        log_query = (
            select(AgentLog)
            .where(AgentLog.job_id == job_id)
            .order_by(AgentLog.timestamp)
        )
        log_result = await session.execute(log_query)
        logs = log_result.scalars().all()
        
        # Get tool calls
        tool_query = (
            select(ToolCall)
            .where(ToolCall.job_id == job_id)
            .order_by(ToolCall.timestamp)
        )
        tool_result = await session.execute(tool_query)
        tools = tool_result.scalars().all()
        
        # Build trace
        events = []
        agent_sequence = []
        tool_calls = []
        budget_history = []
        
        for log in logs:
            events.append({
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "agent_id": log.agent_id,
                "event_type": log.event_type,
                "input_hash": log.input_hash,
                "output_hash": log.output_hash,
                "latency_ms": log.latency_ms,
                "token_count": log.token_count,
                "violations": log.policy_violations,
            })
            
            if log.agent_id not in agent_sequence:
                agent_sequence.append(log.agent_id)
        
        for tool in tools:
            tool_calls.append({
                "timestamp": tool.timestamp.isoformat() if tool.timestamp else None,
                "agent_id": tool.agent_id,
                "tool_name": tool.tool_name,
                "input": tool.input,
                "output": tool.output,
                "latency_ms": tool.latency_ms,
                "accepted": tool.accepted,
                "retry_number": tool.retry_number,
            })
        
        return {
            "job_id": job_id,
            "query": "",  # Would be populated from jobs table
            "status": "completed" if logs else "unknown",
            "events": events,
            "agent_sequence": agent_sequence,
            "tool_calls": tool_calls,
            "budget_history": budget_history,
        }


# =============================================================================
# Endpoint 3: GET /api/v1/eval/latest
# =============================================================================

@app.get("/api/v1/eval/latest")
async def get_latest_eval():
    """Get latest evaluation summary.
    
    Returns:
        EvalRunSummary with scores by category and dimension.
    """
    from sqlalchemy import select
    from app.db.connection import get_session
    from app.db.models import EvalRun
    async with get_session() as session:
        query = (
            select(EvalRun)
            .order_by(EvalRun.run_timestamp.desc())
            .limit(1)
        )
        
        result = await session.execute(query)
        eval_run = result.scalar_one_or_none()
        
        if not eval_run:
            return {
                "status": "no_eval_runs",
                "message": "No evaluation runs found. Run evaluation first.",
            }
        
        return {
            "run_id": str(eval_run.id),
            "timestamp": eval_run.run_timestamp.isoformat() if eval_run.run_timestamp else None,
            "test_cases": eval_run.test_cases,
            "summary": eval_run.summary,
            "diff_from_previous": eval_run.diff_from_previous,
        }


# =============================================================================
# Endpoint 4: POST /api/v1/eval/rewrites/{rewrite_id}/review
# =============================================================================

@app.post("/api/v1/eval/rewrites/{rewrite_id}/review")
async def review_rewrite(
    request: RewriteReviewRequest,
    rewrite_id: str = Path(..., description="Rewrite proposal ID"),
):
    """Approve or reject a prompt rewrite proposal.
    
    This is the human-in-the-loop approval mechanism.
    The Meta Agent proposes but NEVER auto-applies.
    
    Args:
        request: Review action and reason.
        rewrite_id: The rewrite ID.
        
    Returns:
        Updated proposal status.
    """
    from sqlalchemy import select, update
    from app.db.connection import get_session
    from app.db.models import PromptRewrite
    
    async with get_session() as session:
        # Check if exists
        query = select(PromptRewrite).where(PromptRewrite.id == rewrite_id)
        result = await session.execute(query)
        rewrite = result.scalar_one_or_none()
        
        if not rewrite:
            raise HTTPException(status_code=404, detail="Rewrite proposal not found")
        
        if rewrite.status != "pending":
            raise HTTPException(
                status_code=400,
                detail=f"Rewrite already {rewrite.status}",
            )
        
        # Update status
        new_status = "approved" if request.action == "approve" else "rejected"
        
        await session.execute(
            update(PromptRewrite)
            .where(PromptRewrite.id == rewrite_id)
            .values(
                status=new_status,
                reviewed_at=datetime.utcnow(),
            )
        )
        
        logger.info(
            f"Rewrite {rewrite_id} {new_status}: {request.reason}",
            extra={"rewrite_id": rewrite_id, "action": request.action},
        )
        
        return {
            "rewrite_id": rewrite_id,
            "status": new_status,
            "reason": request.reason,
            "reviewed_at": datetime.utcnow().isoformat(),
        }


# =============================================================================
# Celery Task Wrapper & Endpoint 5: POST /api/v1/eval/rerun-failed
# =============================================================================

async def _get_failed_case_ids(eval_run_id: Optional[str]) -> Optional[List[str]]:
    """Helper to query database for failed case IDs from an evaluation run."""
    from sqlalchemy import select
    from app.db.connection import get_session
    from app.db.models import EvalRun
    import uuid
    
    try:
        async with get_session() as session:
            if eval_run_id:
                query = select(EvalRun).where(EvalRun.id == uuid.UUID(eval_run_id))
            else:
                # Latest eval run
                query = select(EvalRun).order_by(EvalRun.run_timestamp.desc()).limit(1)
                
            result = await session.execute(query)
            eval_run = result.scalar_one_or_none()
            
            if not eval_run or not eval_run.test_cases:
                return None
                
            # Filter test cases where score in any dimension is below threshold
            thresholds = {
                "answer_correctness": 0.6,
                "citation_accuracy": 0.5,
                "contradiction_resolution": 0.6,
                "tool_selection_efficiency": 0.7,
                "context_budget_compliance": 0.7,
                "critique_agreement_rate": 0.6,
            }
            
            failed_ids = []
            for case in eval_run.test_cases:
                case_id = case.get("test_case_id")
                scores = case.get("scores", {})
                is_failed = False
                for dim, score_data in scores.items():
                    score = 0.0
                    if isinstance(score_data, dict):
                        score = score_data.get("score", 0.0)
                    elif isinstance(score_data, (int, float)):
                        score = float(score_data)
                    if score < thresholds.get(dim, 0.6):
                        is_failed = True
                        break
                if is_failed and case_id:
                    failed_ids.append(case_id)
            
            return failed_ids if failed_ids else None
    except Exception as e:
        logger.error(f"Failed to retrieve failed case IDs: {e}")
        return None


@celery_app.task(name="app.main.run_evaluation")
def run_evaluation_task(eval_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Celery task to run evaluation on failed cases using latest approved prompts."""
    import asyncio
    
    async def run_pipeline():
        llm = LLMGateway()
        tools = ToolRegistry()
        
        # Instantiate pipeline
        pipeline = EvaluationPipeline(
            agent_factory=lambda: create_agents(llm, tools),
        )
        
        case_ids_filter = await _get_failed_case_ids(eval_run_id)
        logger.info(f"Running Celery evaluation background task with case filter: {case_ids_filter}")
        result = await pipeline.run_evaluation(case_ids_filter=case_ids_filter)
        return result

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    return loop.run_until_complete(run_pipeline())


@app.post("/api/v1/eval/rerun-failed")
async def rerun_failed_eval(
    eval_run_id: Optional[str] = None,
):
    """Trigger re-evaluation on failed cases via Celery.
    
    Re-runs evaluation using latest approved prompts.
    
    Args:
        eval_run_id: Optional specific eval run ID to re-run.
        
    Returns:
        Job status for Celery background re-evaluation.
    """
    task = run_evaluation_task.delay(eval_run_id)
    
    logger.info(f"Re-evaluation started via Celery: {task.id}", extra={"eval_run_id": eval_run_id})
    
    return {
        "job_id": task.id,
        "status": "running",
        "message": "Re-evaluation started in Celery background task",
        "check_status": f"/api/v1/eval/latest",
    }


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status for all services.
    """
    # Check database
    db_status = await db_health_check()
    
    # Check LLM gateway
    llm_stats = app.state.llm_gateway.get_call_stats() if hasattr(app.state, "llm_gateway") else {}
    
    return {
        "status": "healthy" if db_status.get("connected") else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0",
        "services": {
            "api": "healthy",
            "database": db_status,
            "llm_gateway": {
                "status": "configured",
                "total_calls": llm_stats.get("total_calls", 0),
            },
        },
    }


# =============================================================================
# Additional Utility Endpoints
# =============================================================================

@app.get("/api/v1/agents")
async def list_agents():
    """List all available agents.
    
    Returns:
        List of agent descriptions.
    """
    from app.agents.orchestrator import OrchestratorAgent
    
    agents_info = []
    for agent_id in ENABLED_AGENTS:
        info = {
            "agent_id": agent_id,
            "enabled": True,
            "description": _get_agent_description(agent_id),
        }
        agents_info.append(info)
    
    return {"agents": agents_info, "total": len(agents_info)}


def _get_agent_description(agent_id: str) -> str:
    """Get description for an agent.
    
    Args:
        agent_id: Agent identifier.
        
    Returns:
        Description string.
    """
    descriptions = {
        "orchestrator": "Routes queries to appropriate sub-agents with structured reasoning",
        "decomposition": "Breaks ambiguous queries into typed sub-tasks with dependency graph",
        "rag": "Multi-hop retrieval and generation with citation",
        "critique": "Reviews outputs of all other agents with per-claim scoring",
        "synthesis": "Merges all sub-agent outputs and resolves contradictions",
        "compression": "Compresses context when budget is exceeded",
        "meta": "Proposes prompt improvements from evaluation failures",
    }
    return descriptions.get(agent_id, "Unknown agent")


@app.get("/api/v1/tools")
async def list_tools(
    http_request: Request,
):
    """List all available tools.
    
    Returns:
        List of tool descriptions.
    """
    tools = get_tool_registry(http_request)
    return tools.list_tools()


# =============================================================================
# Import log_ui as sub-application
# =============================================================================

from app.log_ui import log_ui_app
app.mount("/logs", log_ui_app)
