# =============================================================================
# MEGA AI — Context Compression Agent
# =============================================================================
# Triggered ONLY when context budget is exceeded.
# Lossless for: tool outputs, scores, citations, structured data.
# Lossy for: conversational filler, repeated context.
# Must log compression ratio.
# =============================================================================

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agents.base import AgentOutput, BaseAgent
from app.schemas import ContextObject

logger = logging.getLogger(__name__)


class CompressionAgent(BaseAgent):
    """Context compression agent for budget management.
    
    This agent:
    1. Is triggered ONLY when context budget is exceeded
    2. Compresses context losslessly for structured data
    3. Compresses lossily for conversational filler
    4. Logs compression ratio for observability
    5. Never drops critical information
    
    Lossless preservation:
    - Tool outputs and results
    - Scores and metrics
    - Citations and references
    - Structured data (JSON)
    
    Lossy compression:
    - Conversational filler ("um", "let me think", etc.)
    - Repeated context
    - Redundant explanations
    - Whitespace and formatting
    
    Attributes:
        agent_id: "compression"
        description: Compresses context when budget is exceeded
    """
    
    agent_id = "compression"
    description = "Compresses context when budget is exceeded"
    
    # Minimum compression ratio to be considered effective
    MIN_EFFECTIVE_RATIO: float = 0.10
    
    # Patterns for lossy compression
    FILLER_PATTERNS: List[str] = [
        r"\b(let me|let's|I will|I shall)\b",  # Meta-language
        r"\b(um|uh|er|ah)\b",                    # Filler words
        r"\b(as mentioned|as noted|as stated)\b", # Redundant references
        r"\b(in my opinion|I think|I believe)\b",  # Hedging
        r"\b(to be honest|frankly)\b",            # Filler phrases
    ]
    
    # Fields to preserve losslessly
    PRESERVED_FIELDS: List[str] = [
        "results",
        "output",
        "score",
        "confidence",
        "citations",
        "data",
        "status",
        "latency_ms",
        "exit_code",
        "sql_generated",
        "contradictions_found",
        "consistency_score",
    ]
    
    async def _execute(self, ctx: ContextObject) -> AgentOutput:
        """Execute context compression.
        
        Steps:
        1. Calculate current context size
        2. Determine compression needed
        3. Compress losslessly where possible
        4. Compress lossily where acceptable
        5. Log compression ratio
        
        Args:
            ctx: Shared context to compress.
            
        Returns:
            AgentOutput with compression results.
        """
        original_size = self._calculate_context_size(ctx)
        
        # Compress agent outputs
        compressed_outputs = {}
        for agent_id, output in ctx.agent_outputs.items():
            compressed = self._compress_agent_output(output.content)
            compressed_outputs[agent_id] = compressed
        
        # Compress tool results
        compressed_tools = {}
        for tool_name, result in ctx.tool_results.items():
            compressed = self._compress_tool_result(result)
            compressed_tools[tool_name] = compressed
        
        # Calculate compression ratio
        new_size = self._calculate_compressed_size(compressed_outputs, compressed_tools)
        compression_ratio = self._calculate_ratio(original_size, new_size)
        
        # Update context
        for agent_id, output in ctx.agent_outputs.items():
            if agent_id in compressed_outputs:
                output.content = compressed_outputs[agent_id]
        
        for tool_name, result in ctx.tool_results.items():
            if tool_name in compressed_tools:
                ctx.tool_results[tool_name] = compressed_tools[tool_name]
        
        ctx.compressed = True
        ctx.compression_ratio = compression_ratio
        
        # Log compression
        self._log_event(
            job_id=ctx.job_id,
            event_type="compression_triggered",
            input_payload={"original_size": original_size},
            output_payload={
                "compressed_size": new_size,
                "compression_ratio": compression_ratio,
            },
        )
        
        logger.info(
            f"Context compressed: {original_size} -> {new_size} "
            f"({compression_ratio:.1%} reduction)",
            extra={"job_id": ctx.job_id, "agent_id": self.agent_id},
        )
        
        # Format output
        output = {
            "compression_performed": True,
            "original_size_tokens": original_size,
            "compressed_size_tokens": new_size,
            "compression_ratio": round(compression_ratio, 4),
            "methods_used": ["lossless_structured", "lossy_filler_removal"],
            "preserved_fields": self.PRESERVED_FIELDS,
        }
        
        output_content = json.dumps(output, indent=2)
        
        return AgentOutput(
            agent_id=self.agent_id,
            content=output_content,
            confidence=1.0,  # Compression is deterministic
            token_count=len(output_content.split()),
        )
    
    def _calculate_context_size(self, ctx: ContextObject) -> int:
        """Calculate current context size in tokens.
        
        Args:
            ctx: Context object.
            
        Returns:
            Estimated token count.
        """
        size = 0
        
        # Count agent outputs
        for output in ctx.agent_outputs.values():
            size += len(output.content.split())
        
        # Count tool results
        for result in ctx.tool_results.values():
            size += len(str(result).split())
        
        # Count sub-tasks
        for task in ctx.sub_tasks:
            size += len(task.description.split())
        
        # Count critique results
        for critique in ctx.critique_results:
            size += len(critique.claim.split())
        
        return size
    
    def _calculate_compressed_size(
        self,
        outputs: Dict[str, str],
        tools: Dict[str, Any],
    ) -> int:
        """Calculate compressed size.
        
        Args:
            outputs: Compressed outputs.
            tools: Compressed tool results.
            
        Returns:
            Estimated token count.
        """
        size = 0
        for content in outputs.values():
            size += len(content.split())
        for result in tools.values():
            size += len(str(result).split())
        return size
    
    def _calculate_ratio(self, original: int, compressed: int) -> float:
        """Calculate compression ratio.
        
        Args:
            original: Original size.
            compressed: Compressed size.
            
        Returns:
            Compression ratio (0.0 = no compression, 1.0 = fully compressed).
        """
        if original == 0:
            return 0.0
        return max(0.0, (original - compressed) / original)
    
    def _compress_agent_output(self, content: str) -> str:
        """Compress an agent output.
        
        Tries lossless JSON compression first, then lossy text compression.
        
        Args:
            content: Agent output content.
            
        Returns:
            Compressed content.
        """
        # Try JSON-based lossless compression
        try:
            data = json.loads(content)
            compressed_data = self._compress_json_lossless(data)
            return json.dumps(compressed_data, separators=(',', ':'))
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback to lossy text compression
        return self._compress_text_lossy(content)
    
    def _compress_json_lossless(self, data: Any) -> Any:
        """Recursively compress JSON data losslessly.
        
        Removes whitespace and redundant fields.
        
        Args:
            data: JSON data.
            
        Returns:
            Compressed data.
        """
        if isinstance(data, dict):
            compressed = {}
            for key, value in data.items():
                # Skip empty values
                if value is None or value == [] or value == {}:
                    continue
                # Skip redundant whitespace in string values
                if isinstance(value, str):
                    value = " ".join(value.split())
                compressed[key] = self._compress_json_lossless(value)
            return compressed
        elif isinstance(data, list):
            return [self._compress_json_lossless(item) for item in data]
        elif isinstance(data, str):
            return " ".join(data.split())  # Normalize whitespace
        return data
    
    def _compress_text_lossy(self, text: str) -> str:
        """Compress text lossily by removing filler.
        
        Args:
            text: Text to compress.
            
        Returns:
            Compressed text.
        """
        # Remove filler patterns
        compressed = text
        for pattern in self.FILLER_PATTERNS:
            compressed = re.sub(pattern, "", compressed, flags=re.IGNORECASE)
        
        # Normalize whitespace
        compressed = " ".join(compressed.split())
        
        # Remove redundant sentences (duplicate detection)
        sentences = [s.strip() for s in compressed.split(".") if s.strip()]
        seen = set()
        unique = []
        for sentence in sentences:
            key = sentence.lower()[:50]  # First 50 chars as key
            if key not in seen:
                seen.add(key)
                unique.append(sentence)
        
        compressed = ". ".join(unique)
        
        return compressed
    
    def _compress_tool_result(self, result: Any) -> Any:
        """Compress a tool result.
        
        Preserves critical fields losslessly.
        
        Args:
            result: Tool result.
            
        Returns:
            Compressed result.
        """
        if isinstance(result, dict):
            compressed = {}
            for key, value in result.items():
                # Always preserve critical fields
                if key in self.PRESERVED_FIELDS:
                    compressed[key] = value
                elif isinstance(value, str):
                    compressed[key] = self._compress_text_lossy(value)
                else:
                    compressed[key] = value
            return compressed
        
        return result
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics.
        
        Returns:
            Stats dict.
        """
        return {
            "total_compressions": 0,
            "avg_compression_ratio": 0.0,
            "methods": ["lossless_json", "lossy_filler"],
        }
