# =============================================================================
# MEGA AI — Celery Configuration
# =============================================================================
# Background task processing for evaluation and long-running jobs.
# =============================================================================

from __future__ import annotations

import logging

from celery import Celery

from app.config import REDIS_URL

logger = logging.getLogger(__name__)

# =============================================================================
# Create Celery App
# =============================================================================

celery_app = Celery(
    "megaai",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.main"],
)

# =============================================================================
# Celery Configuration
# =============================================================================

celery_app.conf.update(

    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Results
    result_expires=3600,
    result_backend=REDIS_URL,

    # Worker behavior
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,

    # Execution
    task_always_eager=False,
    task_store_eager_result=True,

    # Retry behavior
    task_default_retry_delay=60,
    task_max_retries=3,

    # Rate limiting
    task_annotations={
        "*": {
            "rate_limit": "100/s",
        },
    },
)

# =============================================================================
# Queue Routing
# =============================================================================

celery_app.conf.task_routes = {
    "app.main.process_query": {"queue": "query"},
    "app.main.run_evaluation": {"queue": "eval"},
}

logger.info(
    "Celery app initialized",
    extra={"broker": REDIS_URL[:20] + "..."},
)

# =============================================================================
# Helper
# =============================================================================

def get_celery_app():
    """
    Return Celery application instance.
    """
    return celery_app