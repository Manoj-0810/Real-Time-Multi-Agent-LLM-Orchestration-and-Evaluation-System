# =============================================================================
# MEGA AI — Pydantic Schemas
# =============================================================================
# All input/output schemas defined in one place for consistency.
# =============================================================================

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class JobStatus(str, Enum):
    """Valid job statuses throughout the pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(str, Enum):
    """Valid structured logging event types."""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_CALL = "tool_call"
    BUDGET_CHECK = "budget_check"
    POLICY_VIOLATION = "policy_violation"
    ROUTING_DECISION = "routing_decision"
    COMPRESSION_TRIGGERED = "compression_triggered"
    CRITIQUE_REVIEW = "critique_review"
    SYNTHESIS_MERGE = "synthesis_merge"
    META_PROPOSAL = "meta_proposal"


class RewriteStatus(str, Enum):
    """Status of a prompt rewrite proposal."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# =============================================================================
# Context Object — Shared Inter-Agent Communication Schema
# =============================================================================

class ContextObject(BaseModel):
    """Shared context passed between ALL agents. Agents NEVER call each other 
    directly — all communication flows through this schema.
    
    This is the backbone of the Chassis architecture. Every agent reads from
    and writes to this object. The orchestrator manages its lifecycle.
    """
    job_id: str = Field(..., description="Unique job identifier")
    query: str = Field(..., description="Original user query")
    sub_tasks: List[SubTask] = Field(default_factory=list)
    agent_outputs: Dict[str, AgentOutput] = Field(default_factory=dict)
    tool_results: Dict[str, ToolResult] = Field(default_factory=dict)
    critique_results: List[CritiqueResult] = Field(default_factory=list)
    synthesis: Optional[SynthesisResult] = None
    provenance_map: Dict[str, ProvenanceEntry] = Field(default_factory=dict)
    compressed: bool = Field(default=False, description="Whether context was compressed")
    compression_ratio: float = Field(default=0.0)
    budget_status: BudgetStatus = Field(default_factory=lambda: BudgetStatus())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubTask(BaseModel):
    """A single decomposed sub-task with dependency tracking."""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type: retrieval, computation, reasoning, etc.")
    description: str = Field(..., description="What this task should accomplish")
    depends_on: List[str] = Field(default_factory=list, description="task_ids that must complete first")
    status: str = Field(default="pending", description="pending|running|completed|failed")
    result: Optional[str] = None


class AgentOutput(BaseModel):
    """Output produced by a single agent."""
    agent_id: str = Field(..., description="Agent that produced this output")
    content: str = Field(..., description="The actual output text")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: int = Field(default=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ToolResult(BaseModel):
    """Result from a tool invocation."""
    tool_name: str = Field(..., description="Name of the tool that was called")
    input_params: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(default=0)
    accepted: bool = Field(default=True)
    rejection_reason: Optional[str] = None
    retry_number: int = Field(default=0, ge=0, le=2)


class CritiqueResult(BaseModel):
    """Structured critique of a single claim."""
    claim: str = Field(..., description="The claim being evaluated")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this claim")
    flagged_span: Optional[str] = Field(None, description="Specific text span flagged")
    reason: str = Field(..., description="Why this confidence score was given")
    agent_source: str = Field(..., description="Which agent produced the original claim")


class SynthesisResult(BaseModel):
    """Final synthesized answer."""
    answer: str = Field(..., description="The final merged answer")
    contradictions_resolved: int = Field(default=0, description="How many contradictions were fixed")
    sources_used: List[str] = Field(default_factory=list)


class ProvenanceEntry(BaseModel):
    """Tracks where each sentence in the answer came from."""
    sentence: str = Field(..., description="The sentence text")
    source_agent: str = Field(..., description="Which agent produced it")
    source_chunk: Optional[str] = Field(None, description="Which RAG chunk (if applicable)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class BudgetStatus(BaseModel):
    """Current context budget status for a job."""
    total_budget: int = Field(default=8192)
    tokens_used: int = Field(default=0)
    tokens_remaining: int = Field(default=8192)
    compression_triggered: bool = Field(default=False)
    violations: List[str] = Field(default_factory=list)


# =============================================================================
# API Request/Response Schemas
# =============================================================================

class QueryRequest(BaseModel):
    """POST /api/v1/query request body."""
    query: str = Field(..., min_length=1, max_length=10000, description="User query")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional additional context")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class QueryResponse(BaseModel):
    """Final response after query processing completes."""
    job_id: str
    status: JobStatus
    answer: Optional[str] = None
    provenance_map: Dict[str, ProvenanceEntry] = Field(default_factory=dict)
    agent_chain: List[str] = Field(default_factory=list)
    total_tokens_used: int = 0
    processing_time_ms: int = 0


class JobTraceResponse(BaseModel):
    """GET /api/v1/jobs/{job_id}/trace response."""
    job_id: str
    query: str
    status: JobStatus
    events: List[TraceEvent]
    agent_sequence: List[str]
    tool_calls: List[ToolCallRecord]
    budget_history: List[BudgetSnapshot]


class TraceEvent(BaseModel):
    """Single event in the execution trace."""
    timestamp: datetime
    agent_id: str
    event_type: EventType
    description: str
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None


class ToolCallRecord(BaseModel):
    """Recorded tool invocation in the trace."""
    timestamp: datetime
    agent_id: str
    tool_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    latency_ms: int
    accepted: bool
    retry_number: int


class BudgetSnapshot(BaseModel):
    """Budget state at a point in time."""
    timestamp: datetime
    agent_id: str
    tokens_used: int
    tokens_remaining: int
    violation: bool = False


# =============================================================================
# Evaluation Schemas
# =============================================================================

class EvalScore(BaseModel):
    """Score for a single dimension with justification."""
    score: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field(..., min_length=1)


class EvalResult(BaseModel):
    """Result for a single test case."""
    test_case_id: str
    category: str
    query: str
    scores: Dict[str, EvalScore]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    execution_trace: List[str] = Field(default_factory=list)


class EvalRunSummary(BaseModel):
    """Summary of a complete evaluation run."""
    run_id: str
    timestamp: datetime
    total_cases: int
    passed_cases: int
    failed_cases: int
    category_scores: Dict[str, float]
    dimension_scores: Dict[str, float]
    regression_detected: bool = False
    diff_from_previous: Optional[Dict[str, Any]] = None


class PromptRewriteProposal(BaseModel):
    """A proposed prompt rewrite from the Meta Agent."""
    id: str
    eval_run_id: str
    agent_id: str
    dimension: str
    original_prompt: str
    proposed_prompt: str
    diff: str
    justification: str
    status: RewriteStatus
    performance_delta: Optional[Dict[str, Any]] = None
    created_at: datetime
    reviewed_at: Optional[datetime] = None


class RewriteReviewRequest(BaseModel):
    """POST /api/v1/eval/rewrites/{rewrite_id}/review request body."""
    action: str = Field(..., pattern="^(approve|reject)$")
    reason: str = Field(..., min_length=1)


# =============================================================================
# Streaming Event Schemas (SSE)
# =============================================================================

class SSEEvent(BaseModel):
    """Base schema for SSE streaming events."""
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)


class AgentStartEvent(SSEEvent):
    """Sent when an agent begins processing."""
    event_type: str = "agent_start"
    agent_id: str


class AgentTokenEvent(SSEEvent):
    """Real-time token streaming from an agent."""
    event_type: str = "agent_token"
    agent_id: str
    token: str


class ToolCallEvent(SSEEvent):
    """Sent when a tool is invoked."""
    event_type: str = "tool_call"
    agent_id: str
    tool_name: str
    status: str


class BudgetUpdateEvent(SSEEvent):
    """Sent when budget status changes."""
    event_type: str = "budget_update"
    agent_id: str
    remaining_tokens: int


class JobCompleteEvent(SSEEvent):
    """Sent when the entire job finishes."""
    event_type: str = "job_complete"
    job_id: str
    summary: str


# =============================================================================
# Structured Logging Schema
# =============================================================================

class StructuredLogEntry(BaseModel):
    """Every agent boundary must emit this exact schema."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    job_id: str
    event_type: EventType
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    latency_ms: int = 0
    token_count: int = 0
    policy_violations: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging frameworks."""
        return self.model_dump(mode="json")
