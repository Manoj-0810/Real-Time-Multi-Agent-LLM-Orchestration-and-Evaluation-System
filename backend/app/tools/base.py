# =============================================================================
# MEGA AI — Tool Base Classes
# =============================================================================
# All tools must inherit from BaseTool and implement the failure contracts.
# =============================================================================

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Base exception for tool errors."""
    pass


class ToolTimeoutError(ToolError):
    """Raised when tool execution exceeds timeout."""
    pass


class ToolInputError(ToolError):
    """Raised when tool input is invalid."""
    pass


class ToolSandboxError(ToolError):
    """Raised when sandbox security violation detected."""
    pass


class ToolResultStatus(str, Enum):
    """Status codes for tool execution."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    SANDBOX_VIOLATION = "sandbox_violation"
    NO_RESULTS = "no_results"
    INVALID_INPUT = "invalid_input"


@dataclass
class ToolCallRecord:
    """MANDATORY logging record for every tool call."""
    tool_name: str
    agent_id: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    latency_ms: int
    accepted: bool
    rejection_reason: str
    retry_number: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolOutput:
    """Standardized output from any tool."""
    status: ToolResultStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: int = 0
    fallback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging and API responses."""
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "fallback": self.fallback,
        }


class BaseTool(ABC):
    """Abstract base class that ALL tools must inherit from.
    
    Provides:
    - Standardized input validation
    - Failure contract enforcement
    - Structured logging of every call
    - Retry counting
    - Latency tracking
    
    Subclasses must implement:
    - name: Tool identifier
    - description: What the tool does
    - _execute(): The actual tool logic
    
    Attributes:
        max_retries: Maximum retry attempts (0-2).
        timeout_ms: Maximum execution time in milliseconds.
    """
    
    # Override in subclass
    name: str = "base_tool"
    description: str = "Base tool — do not use directly"
    max_retries: int = 2
    timeout_ms: int = 10000
    
    def __init__(self) -> None:
        self.call_history: List[ToolCallRecord] = []
    
    @abstractmethod
    async def _execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Implement the actual tool logic here.
        
        Args:
            **kwargs: Tool-specific parameters.
            
        Returns:
            Dict with the tool output data.
            
        Raises:
            ToolError: On failure — will be converted to structured output.
        """
        raise NotImplementedError
    
    async def execute(
        self,
        agent_id: str,
        retry_number: int = 0,
        **kwargs: Any,
    ) -> ToolOutput:
        """Execute the tool with full logging and failure handling.
        
        This method must NOT be overridden. All tool logic goes in _execute().
        
        Args:
            agent_id: The agent invoking this tool.
            retry_number: Current retry attempt (0 = first call).
            **kwargs: Tool-specific parameters.
            
        Returns:
            ToolOutput with status, data, and optional fallback instruction.
        """
        start_time = time.time()
        
        try:
            # Validate retry count
            if retry_number > self.max_retries:
                logger.warning(
                    f"Tool {self.name} exceeded max retries ({self.max_retries})",
                    extra={"agent_id": agent_id},
                )
                return ToolOutput(
                    status=ToolResultStatus.ERROR,
                    error=f"Max retries ({self.max_retries}) exceeded",
                    latency_ms=int((time.time() - start_time) * 1000),
                )
            
            logger.info(
                f"Tool execution: {self.name} (attempt {retry_number})",
                extra={
                    "agent_id": agent_id,
                    "tool": self.name,
                    "retry": retry_number,
                    "input": kwargs,
                },
            )
            
            # Execute the tool logic
            data = await self._execute(**kwargs)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Log successful call
            record = ToolCallRecord(
                tool_name=self.name,
                agent_id=agent_id,
                input=kwargs,
                output=data,
                latency_ms=latency_ms,
                accepted=True,
                rejection_reason="",
                retry_number=retry_number,
            )
            self.call_history.append(record)
            
            logger.info(
                f"Tool success: {self.name} in {latency_ms}ms",
                extra={"agent_id": agent_id, "tool": self.name},
            )
            
            return ToolOutput(
                status=ToolResultStatus.SUCCESS,
                data=data,
                latency_ms=latency_ms,
            )
            
        except ToolTimeoutError as e:
            return self._handle_error(
                error=e,
                agent_id=agent_id,
                retry_number=retry_number,
                start_time=start_time,
                kwargs=kwargs,
                status=ToolResultStatus.TIMEOUT,
            )
        except ToolSandboxError as e:
            return self._handle_error(
                error=e,
                agent_id=agent_id,
                retry_number=retry_number,
                start_time=start_time,
                kwargs=kwargs,
                status=ToolResultStatus.SANDBOX_VIOLATION,
            )
        except ToolInputError as e:
            return self._handle_error(
                error=e,
                agent_id=agent_id,
                retry_number=retry_number,
                start_time=start_time,
                kwargs=kwargs,
                status=ToolResultStatus.INVALID_INPUT,
            )
        except Exception as e:
            return self._handle_error(
                error=e,
                agent_id=agent_id,
                retry_number=retry_number,
                start_time=start_time,
                kwargs=kwargs,
                status=ToolResultStatus.ERROR,
            )
    
    def _handle_error(
        self,
        error: Exception,
        agent_id: str,
        retry_number: int,
        start_time: float,
        kwargs: Dict[str, Any],
        status: ToolResultStatus,
        fallback: Optional[str] = None,
    ) -> ToolOutput:
        """Handle tool errors with structured logging.
        
        Args:
            error: The exception that occurred.
            agent_id: The agent that invoked the tool.
            retry_number: Current retry attempt.
            start_time: When execution started.
            kwargs: The input parameters.
            status: The ToolResultStatus to return.
            fallback: Optional fallback instruction.
            
        Returns:
            ToolOutput with error details.
        """
        latency_ms = int((time.time() - start_time) * 1000)
        error_msg = str(error)
        
        # Auto-generate fallback if not provided
        if fallback is None:
            fallback = self._get_fallback(status, error_msg)
        
        # Log failed call
        record = ToolCallRecord(
            tool_name=self.name,
            agent_id=agent_id,
            input=kwargs,
            output={"error": error_msg, "status": status.value},
            latency_ms=latency_ms,
            accepted=False,
            rejection_reason=error_msg,
            retry_number=retry_number,
        )
        self.call_history.append(record)
        
        logger.error(
            f"Tool failed: {self.name} — {error_msg}",
            extra={
                "agent_id": agent_id,
                "tool": self.name,
                "status": status.value,
                "latency_ms": latency_ms,
            },
        )
        
        return ToolOutput(
            status=status,
            error=error_msg,
            latency_ms=latency_ms,
            fallback=fallback,
        )
    
    def _get_fallback(
        self,
        status: ToolResultStatus,
        error_msg: str,
    ) -> Optional[str]:
        """Generate fallback instruction based on error type.
        
        Override in subclass for tool-specific fallbacks.
        
        Args:
            status: The error status.
            error_msg: The error message.
            
        Returns:
            Fallback instruction string or None.
        """
        fallbacks = {
            ToolResultStatus.TIMEOUT: "use_knowledge_base",
            ToolResultStatus.NO_RESULTS: "reformulate_query",
            ToolResultStatus.INVALID_INPUT: "decompose_query",
            ToolResultStatus.SANDBOX_VIOLATION: "use_safe_alternative",
            ToolResultStatus.ERROR: "retry_with_fallback",
        }
        return fallbacks.get(status)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics.
        
        Returns:
            Dict with total_calls, successful, failed, avg_latency.
        """
        total = len(self.call_history)
        successful = sum(1 for r in self.call_history if r.accepted)
        failed = total - successful
        latencies = [r.latency_ms for r in self.call_history]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "tool_name": self.name,
            "total_calls": total,
            "successful": successful,
            "failed": failed,
            "avg_latency_ms": round(avg_latency, 2),
            "total_retries": sum(r.retry_number for r in self.call_history),
        }
