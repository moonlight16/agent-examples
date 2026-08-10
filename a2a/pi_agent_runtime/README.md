# Pi Agent Runtime

A generic, instruction-free [Pi](https://pi.dev/) coding agent exposed through
the A2A protocol. It starts a fresh Pi turn for each request and operates in
`/shared`, which Rosso can back with ephemeral storage or a PVC.

The image pins the same `kagenti/pi` fork used by Serverless Harness, but does
not include Knative, Redis, workload queues, or Serverless Harness routing.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PI_MODEL_BASE_URL` | required | OpenAI- or Anthropic-compatible model endpoint |
| `PI_MODEL` | `meta-llama/Llama-3.3-70B-Instruct` | Model identifier served by the endpoint |
| `PI_MODEL_PROVIDER` | `llmd` | Pi custom-provider name |
| `PI_MODEL_API` | `anthropic-messages` | Pi wire protocol |
| `PI_API_KEY` | `unused` | Endpoint credential or placeholder |
| `PI_WORKSPACE` | `/shared` | Agent working directory |

No `AGENTS.md`, `CLAUDE.md`, repository, or initial instructions are included.

## Local test

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
```

The first cluster test will build this Dockerfile with Shipwright, import the
result through `rossoctl agents import from-image`, and send an A2A prompt that
writes a file into a PVC-backed `/shared` workspace.
