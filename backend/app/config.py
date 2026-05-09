# =============================================================================
# MEGA AI — Configuration Module
# =============================================================================
# THE ONLY PLACE WHERE LLM MODELS AND SYSTEM CONSTANTS ARE CONFIGURED.
# No hardcoded strings anywhere else in the codebase.
# =============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final, Set


# =============================================================================
# LLM Configuration — The Chassis Rule
# =============================================================================
# Model is swapped via .env ONLY — NEVER in code.
# The system is architected to instantly support any provider.
# =============================================================================

# Provider display names for logging
PROVIDER_DISPLAY_NAMES: Final[dict[str, str]] = {
    "gemini": "Google Gemini",
    "groq": "Groq",
    "deepseek": "DeepSeek",
    "anthropic": "Anthropic",
}

# Tier mapping for cost tracking and benchmarking
MODEL_TIERS: Final[dict[str, dict[str, str | float]]] = {
    "gemini-2.5-flash": {
        "tier": "free",
        "provider": "gemini",
        "cost_per_1k_tokens": 0.0,
        "typical_latency_ms": 800,
    },
    "llama-3.3-70b-versatile": {
        "tier": "free",
        "provider": "groq",
        "cost_per_1k_tokens": 0.0,
        "typical_latency_ms": 300,
    },
    "deepseek-v4-pro": {
        "tier": "better",
        "provider": "deepseek",
        "cost_per_1k_tokens": 0.5,
        "typical_latency_ms": 1200,
    },
    "claude-opus-4-7": {
        "tier": "best",
        "provider": "anthropic",
        "cost_per_1k_tokens": 15.0,
        "typical_latency_ms": 2000,
    },
}

# LLM configuration loaded from environment
LLM_CONFIG: Final[dict[str, str]] = {
    "provider": os.getenv("LLM_PROVIDER", "gemini"),
    "model": os.getenv("LLM_MODEL", "gemini-2.5-flash"),
    "api_key": os.getenv("LLM_API_KEY", ""),
    "fallback_provider": os.getenv("FALLBACK_PROVIDER", "groq"),
    "fallback_model": os.getenv("FALLBACK_MODEL", "llama-3.3-70b-versatile"),
    "fallback_api_key": os.getenv("FALLBACK_API_KEY", ""),
    "tertiary_provider": os.getenv("TERTIARY_PROVIDER", "deepseek"),
    "tertiary_model": os.getenv("TERTIARY_MODEL", "deepseek-v4-pro"),
    "tertiary_api_key": os.getenv("TERTIARY_API_KEY", ""),
}

# =============================================================================
# Database Configuration
# =============================================================================

DATABASE_CONFIG: Final[dict[str, str | int]] = {
    "user": os.getenv("POSTGRES_USER", "megaai"),
    "password": os.getenv("POSTGRES_PASSWORD", "megaai"),
    "host": os.getenv("POSTGRES_HOST", "db"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "megaai"),
    "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
    "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
}

# Build connection string
DATABASE_URL: Final[str] = (
    f"postgresql+asyncpg://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
)

# =============================================================================
# Redis Configuration
# =============================================================================

REDIS_CONFIG: Final[dict[str, str | int]] = {
    "host": os.getenv("REDIS_HOST", "redis"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "password": os.getenv("REDIS_PASSWORD", ""),
}

# Build Redis URL
if REDIS_CONFIG["password"]:
    REDIS_URL: Final[str] = (
        f"redis://:{REDIS_CONFIG['password']}@"
        f"{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}/0"
    )
else:
    REDIS_URL: Final[str] = (
        f"redis://{REDIS_CONFIG['host']}:"
        f"{REDIS_CONFIG['port']}/0"
    )

# =============================================================================
# Application Configuration
# =============================================================================

APP_CONFIG: Final[dict[str, str | int | bool]] = {
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "default_context_budget": int(os.getenv("DEFAULT_CONTEXT_BUDGET", "8192")),
    "eval_max_retries": int(os.getenv("EVAL_MAX_RETRIES", "2")),
    "eval_timeout_seconds": int(os.getenv("EVAL_TIMEOUT_SECONDS", "300")),
    "enable_meta_agent": os.getenv("ENABLE_META_AGENT", "true").lower() == "true",
    "enable_adversarial_tests": os.getenv("ENABLE_ADVERSARIAL_TESTS", "true").lower() == "true",
    "enable_compression": os.getenv("ENABLE_COMPRESSION", "true").lower() == "true",
    "enable_sse_streaming": os.getenv("ENABLE_SSE_STREAMING", "true").lower() == "true",
}

# Parse enabled agents
ENABLED_AGENTS: Final[Set[str]] = set(
    os.getenv("ENABLED_AGENTS", "orchestrator,decomposition,rag,critique,synthesis,compression,meta").split(",")
)

# =============================================================================
# CORS Configuration
# =============================================================================

CORS_ORIGINS: Final[list[str]] = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
]

# =============================================================================
# Context Budget Manager Configuration
# =============================================================================

BUDGET_CONFIG: Final[dict[str, float | int]] = {
    "default_budget": int(os.getenv("DEFAULT_CONTEXT_BUDGET", "8192")),
    "compression_trigger_threshold": 0.20,  # Trigger when remaining < 20%
    "max_compression_ratio": 0.70,  # Max 70% of content can be compressed
    "log_violations": True,  # Always log budget violations
}

# =============================================================================
# Tool Configuration
# =============================================================================

TOOL_CONFIG: Final[dict[str, dict[str, int | float | str]]] = {
    "web_search": {
        "timeout_ms": 10000,
        "max_results": 5,
        "rate_limit_per_minute": 30,
    },
    "code_execution": {
        "timeout_ms": 5000,
        "max_memory_mb": 256,
        "sandboxed": True,
    },
    "nl2sql": {
        "timeout_ms": 15000,
        "max_rows": 100,
        "read_only": True,
    },
    "self_reflection": {
        "max_history_turns": 10,
        "consistency_threshold": 0.7,
    },
}

# =============================================================================
# Evaluation Configuration
# =============================================================================

EVAL_CONFIG: Final[dict[str, int | float]] = {
    "baseline_cases": 5,
    "ambiguous_cases": 5,
    "adversarial_cases": 5,
    "min_answer_correctness": 0.6,
    "min_citation_accuracy": 0.5,
    "min_tool_selection_efficiency": 0.7,
    "contradiction_resolution_threshold": 0.6,
}

# =============================================================================
# Event Types for Structured Logging
# =============================================================================

VALID_EVENT_TYPES: Final[Set[str]] = {
    "agent_start",
    "agent_end",
    "tool_call",
    "budget_check",
    "policy_violation",
    "routing_decision",
    "compression_triggered",
    "critique_review",
    "synthesis_merge",
    "meta_proposal",
}

# =============================================================================
# Security
# =============================================================================

LOG_UI_API_KEY: Final[str] = os.getenv("LOG_UI_API_KEY", "")

# =============================================================================
# Helper Functions
# =============================================================================


def get_llm_config_with_fallback() -> list[dict[str, str]]:
    """Return ordered list of LLM configs to try (primary, fallback, tertiary).
    
    Returns:
        List of dicts with provider, model, and api_key for each tier.
    """
    configs = [
        {
            "provider": LLM_CONFIG["provider"],
            "model": LLM_CONFIG["model"],
            "api_key": LLM_CONFIG["api_key"],
            "tier": "primary",
        },
        {
            "provider": LLM_CONFIG["fallback_provider"],
            "model": LLM_CONFIG["fallback_model"],
            "api_key": LLM_CONFIG["fallback_api_key"],
            "tier": "fallback",
        },
    ]
    if LLM_CONFIG.get("tertiary_api_key"):
        configs.append({
            "provider": LLM_CONFIG["tertiary_provider"],
            "model": LLM_CONFIG["tertiary_model"],
            "api_key": LLM_CONFIG["tertiary_api_key"],
            "tier": "tertiary",
        })
    return configs


def get_model_tier(model_name: str) -> str:
    """Return the tier (free/better/best) for a given model name.
    
    Args:
        model_name: The model identifier string.
        
    Returns:
        Tier string: 'free', 'better', 'best', or 'unknown'.
    """
    info = MODEL_TIERS.get(model_name, {})
    return info.get("tier", "unknown")  # type: ignore


def validate_config() -> list[str]:
    """Validate that all required configuration is present.
    
    Returns:
        List of validation error messages. Empty list = all valid.
    """
    errors = []
    
    if not LLM_CONFIG.get("api_key"):
        errors.append("LLM_API_KEY is not set in environment")
    
    if not LLM_CONFIG.get("fallback_api_key"):
        errors.append("FALLBACK_API_KEY is not set — fallback will not work")
    
    if not DATABASE_CONFIG.get("password"):
        errors.append("POSTGRES_PASSWORD is not set")
    
    if not REDIS_CONFIG.get("password"):
        errors.append("REDIS_PASSWORD is not set")
    
    return errors
