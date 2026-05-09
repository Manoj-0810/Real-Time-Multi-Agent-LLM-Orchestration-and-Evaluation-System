# MEGA AI — Setup & Configuration Guide

This guide walks you through configuring, running, and testing the MEGA AI system.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [API Key Setup](#api-key-setup)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)
7. [Code Files Requiring API Keys](#code-files-requiring-api-keys)

---

## Prerequisites

- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **Git**
- Free API keys from Google AI Studio and Groq Cloud

---

## API Key Setup

### Step 1: Get Gemini API Key (Free)

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key" in the top right
4. Create a new API key
5. Copy the key

### Step 2: Get Groq API Key (Free)

1. Visit [Groq Cloud Console](https://console.groq.com/)
2. Sign up for an account
3. Navigate to "API Keys"
4. Create a new API key
5. Copy the key

### Step 3: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env with your keys
# Replace the placeholder values:

LLM_API_KEY=your_actual_gemini_api_key_here
FALLBACK_API_KEY=your_actual_groq_api_key_here

# Optional: Add tertiary key for better tier
# TERTIARY_API_KEY=your_deepseek_key_here

# Set database password
POSTGRES_PASSWORD=your_secure_password_here

# Set Redis password
REDIS_PASSWORD=your_redis_password_here
```

---

## Configuration

### Where to Put API Keys

| File | What to Change | Required |
|------|---------------|----------|
| `.env` | `LLM_API_KEY` | **YES** — Gemini key |
| `.env` | `FALLBACK_API_KEY` | **YES** — Groq key |
| `.env` | `POSTGRES_PASSWORD` | **YES** — DB password |
| `.env` | `REDIS_PASSWORD` | **YES** — Redis password |

### Files That Read API Keys

These files automatically read from `.env` — **do not edit them**:

- `backend/app/config.py` — Reads all LLM and DB config
- `backend/app/llm_gateway.py` — Uses config.py values
- `docker-compose.yml` — Injects env vars into containers

### Changing the LLM Model

**Only edit `.env`:**

```bash
# Switch to DeepSeek (Better tier)
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-pro
LLM_API_KEY=your_deepseek_key

# Or switch to Anthropic (Best tier)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-4-7
LLM_API_KEY=your_anthropic_key
```

**Never edit code files to change the model.**

---

## Running the System

### Option 1: Full Docker Compose (Recommended)

```bash
# Start all services
docker compose up --build

# Or run in background
docker compose up -d --build

# View logs
docker compose logs -f api

# View specific service
docker compose logs -f worker
```

### Services Started

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | Main FastAPI server |
| `log_ui` | 8001 | Log query interface |
| `db` | 5432 | PostgreSQL + pgvector |
| `redis` | 6379 | Celery broker |
| `worker` | — | Background task processor |

### Option 2: Local Development (without Docker)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies
cd backend
pip install -r requirements.txt

# 3. Start PostgreSQL and Redis locally
# (Install PostgreSQL with pgvector extension)

# 4. Run the API
uvicorn app.main:app --reload --port 8000

# 5. In another terminal, run Celery worker
celery -A app.celery_app worker --loglevel=info

# 6. In another terminal, run Log UI
uvicorn app.log_ui:app --reload --port 8001
```

---

## Testing

### 1. Health Check

```bash
# Check API health
curl http://localhost:8000/health

# Check Log UI health
curl http://localhost:8001/health
```

### 2. Submit a Query

```bash
# Submit query (returns SSE stream)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'

# Submit ambiguous query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about the impact"}'
```

### 3. Get Job Trace

```bash
# Replace {job_id} with actual ID from query response
curl http://localhost:8000/api/v1/jobs/{job_id}/trace
```

### 4. List Agents

```bash
curl http://localhost:8000/api/v1/agents
```

### 5. List Tools

```bash
curl http://localhost:8000/api/v1/tools
```

### 6. Run Evaluation

```bash
# Start evaluation (runs in background)
curl -X POST http://localhost:8000/api/v1/eval/rerun-failed

# Check latest results
curl http://localhost:8000/api/v1/eval/latest
```

### 7. Query Logs

```bash
# Get recent logs (no auth if LOG_UI_API_KEY not set)
curl http://localhost:8001/logs

# Filter by agent
curl "http://localhost:8001/logs?agent_id=rag"

# Filter by event type
curl "http://localhost:8001/logs?event_type=tool_call"

# Get agent summary
curl "http://localhost:8001/logs/agents?hours=24"

# Get tool usage
curl "http://localhost:8001/logs/tools?hours=24"

# Get violations
curl "http://localhost:8001/logs/violations?hours=24"
```

---

## Troubleshooting

### Issue: `docker compose up` fails

**Symptom:** Build errors or service startup failures.

**Solutions:**
```bash
# Clean build
docker compose down -v
docker compose up --build

# Check logs
docker compose logs api
docker compose logs db

# Verify ports are free
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows
```

### Issue: Database connection errors

**Symptom:** API logs show PostgreSQL connection failures.

**Solutions:**
```bash
# Check DB is healthy
docker compose ps

# Verify pgvector extension
docker compose exec db psql -U megaai -d megaai -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Reset database (WARNING: destroys data)
docker compose down -v
docker compose up -d db
docker compose logs db
```

### Issue: LLM API errors

**Symptom:** "LLMProviderError" or timeout errors.

**Solutions:**
```bash
# Verify API keys are set
cat .env | grep API_KEY

# Test with health endpoint
curl http://localhost:8000/health

# Check LLM gateway stats
curl http://localhost:8000/health | python -m json.tool
```

### Issue: Slow responses

**Symptom:** Queries take >30 seconds.

**Cause:** Free tier API rate limiting.

**Solutions:**
- This is expected on free tier
- Add delays between requests
- Upgrade to DeepSeek or Anthropic for better latency

### Issue: Port already in use

**Symptom:** `bind: address already in use`.

**Solutions:**
```bash
# Find and kill process on port
lsof -ti:8000 | xargs kill -9  # Mac/Linux
# Or change ports in docker-compose.yml
```

---

## Code Files Requiring API Keys

### ⚠️ NEVER Commit These Files

| File | Contains | Action |
|------|----------|--------|
| `.env` | ALL API keys and passwords | **NEVER commit** — already in `.gitignore` |

### ✅ Safe to Commit

These files read from `.env` — they contain no hardcoded secrets:

| File | Reads From | Description |
|------|-----------|-------------|
| `backend/app/config.py` | `.env` | All configuration |
| `docker-compose.yml` | `.env` | Service configuration |
| `backend/app/llm_gateway.py` | `config.py` | LLM provider initialization |

### Quick Reference: Where to Put Your Keys

```
.env file:
├── LLM_API_KEY=your_gemini_key          ← Google AI Studio
├── FALLBACK_API_KEY=your_groq_key       ← Groq Cloud
├── TERTIARY_API_KEY=your_deepseek_key   ← DeepSeek (optional)
├── POSTGRES_PASSWORD=your_db_password   ← Any secure password
└── REDIS_PASSWORD=your_redis_password   ← Any secure password
```

---

## Verification Checklist

After setup, verify everything works:

- [ ] `docker compose up --build` starts without errors
- [ ] `curl http://localhost:8000/health` returns `{"status": "healthy"}`
- [ ] `curl http://localhost:8001/health` returns `{"status": "healthy"}`
- [ ] Query submission returns SSE stream
- [ ] Job trace endpoint returns execution details
- [ ] Agent list shows 7 agents
- [ ] Tool list shows 4 tools
- [ ] Evaluation runs and produces scores
- [ ] Log UI shows agent events
- [ ] No API keys are in committed code

---

## Support

For issues or questions, check:
1. Docker logs: `docker compose logs -f`
2. Health endpoints: `/health`
3. This troubleshooting guide
4. The README architecture section
