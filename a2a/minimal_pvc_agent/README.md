# Minimal PVC Agent

Minimal A2A agent demonstrating Kagenti persistence (PVC + Postgres).

## What it does

- One-node LangGraph chat agent (LLM-backed when `LLM_API_KEY` is real;
  echo mode otherwise — useful for offline smoke tests).
- Writes one JSONL line per turn to
  `<CONTEXT_DIR>/<sanitized-context_id>.jsonl` (default
  `/shared/minimal-pvc-agent/`). This is the see-with-your-own-eyes
  proof that the PVC is mounted and writable.
- Uses LangGraph's checkpointer for conversation state:
  - `MemorySaver` by default.
  - `AsyncPostgresSaver` when `CHECKPOINT_DB_URL` is set, so state
    survives pod restarts.
- Exposes two read-only HTTP endpoints to verify persistence without
  `kubectl exec`:
  - `GET /history?context_id=<id>` — the JSONL turn log.
  - `GET /checkpoint?context_id=<id>` — the LangGraph state.

## Local run

```sh
cd a2a/minimal_pvc_agent
uv sync

# Minimal (echo mode — no LLM needed):
CONTEXT_DIR=/tmp/minimal-pvc-agent uv run server

# With a local Ollama-style LLM:
LLM_API_BASE=http://localhost:11434/v1 \
LLM_MODEL=qwen3:4b \
CONTEXT_DIR=/tmp/minimal-pvc-agent \
uv run server
```

The server listens on `0.0.0.0:8000`. Send turns through any A2A client
(e.g. `kagenti_chat`), then:

```sh
curl 'http://localhost:8000/history?context_id=<ctx>'
curl 'http://localhost:8000/checkpoint?context_id=<ctx>'
```

## Deploy on Kagenti

Build and push the container, then deploy as a `StatefulSet` with a PVC:

```sh
kagenti agent deploy \
  --workload-type statefulset \
  --persistent-storage \
  --persistent-storage-size 1Gi \
  --name minimal-pvc-agent \
  --container-image <registry>/minimal-pvc-agent:<tag>
```

For the Postgres path (LangGraph checkpoints + future A2A task store),
also pass:

```sh
  --env TASK_STORE_DB_URL=postgresql://user:pw@pg-host:5432/a2a \
  --env CHECKPOINT_DB_URL=postgresql://user:pw@pg-host:5432/langgraph
```

When `CHECKPOINT_DB_URL` is unset the agent uses `MemorySaver`, which is
fine for the PVC-only demo (the JSONL log on `/shared` is the
durable artifact).

## Verify persistence

```sh
# Send a turn through your A2A client, capture context_id, then:
ROUTE=https://minimal-pvc-agent.<cluster-domain>

curl "$ROUTE/history?context_id=<ctx>"
curl "$ROUTE/checkpoint?context_id=<ctx>"

# Force a restart and re-check:
kubectl delete pod -l app=minimal-pvc-agent
kubectl rollout status statefulset/minimal-pvc-agent

curl "$ROUTE/history?context_id=<ctx>"        # JSONL turns survive (PVC)
curl "$ROUTE/checkpoint?context_id=<ctx>"     # LangGraph state survives only
                                              # when CHECKPOINT_DB_URL is set
```
