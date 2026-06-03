# Kagenti Chat

A general-purpose chat agent exposed via the [A2A protocol](https://github.com/a2a-protocol).
Runs as a small Starlette server, calls an OpenAI-compatible LLM endpoint (designed for
[llm-d](https://github.com/llm-d/llm-d) on Kagenti, but works against Ollama/OpenAI/etc.),
and streams progress + final text back to A2A clients.

## What it does

- Exposes a single `kagenti_chat` skill on the A2A AgentCard.
- One LLM call per request — stateless, no tool use, no MCP.
- Uses [AG2](https://pypi.org/project/ag2/) `ConversableAgent` under the hood (kept as the
  abstraction so we can later add MCP tools/RAG without re-plumbing everything).

## Layout

```
src/kagenti_chat/
├── main.py                  # Entrypoint — `uv run server`
├── config/settings.py       # Pydantic settings from env vars / .env
├── a2a_server/server.py     # A2A AgentCard, executor, Starlette app
└── agent/
    ├── chat_agent.py        # AG2 ConversableAgent wrapper
    └── prompts.py           # System prompt
```

## Configuration

| Variable          | Description                                                         | Default                                    |
| ----------------- | ------------------------------------------------------------------- | ------------------------------------------ |
| `LLM_MODEL`       | Model ID the LLM endpoint advertises (e.g. via `/v1/models`)        | `meta-llama/Llama-3.3-70B-Instruct`        |
| `LLM_API_BASE`    | OpenAI-compatible base URL (also accepts `LLM_BASE_URL`)            | _(required)_                               |
| `LLM_API_KEY`     | API key — vLLM/llm-d typically accept any non-empty value           | `dummy`                                    |
| `LLM_TEMPERATURE` | Sampling temperature                                                | `0.2`                                      |
| `EXTRA_HEADERS`   | JSON object of extra HTTP headers for the LLM call                  | `{}`                                       |
| `A2A_HOST`        | Bind address                                                        | `0.0.0.0`                                  |
| `A2A_PORT`        | Bind port                                                           | `8000`                                     |
| `A2A_PUBLIC_URL`  | Publicly routable URL advertised in the AgentCard (set in cluster)  | _(unset)_                                  |
| `LOG_LEVEL`       | `DEBUG` / `INFO` / `WARNING` / `ERROR`                              | `INFO`                                     |

## Running locally against llm-d

In one terminal, port-forward the cluster's llm-d gateway:

```bash
kubectl -n llm-d port-forward svc/infra-inference-scheduling-inference-gateway-istio 8080:80
```

In another:

```bash
cd a2a/kagenti_chat
cp .env.llm-d .env       # default: Llama 3.3 70B
# or: cp .env.llm-d-small .env   # Llama 3.1 8B for faster iteration
uv sync
uv run server
```

Smoke test:

```bash
curl -s http://localhost:8000/.well-known/agent.json | jq .name
```

## Available llm-d models

Five models are pre-staged on the cluster. Switch by setting `LLM_MODEL` and the path
prefix in `LLM_API_BASE` together — both must change.

| `LLM_MODEL`                            | Path prefix                              | Context | Notes                          |
| -------------------------------------- | ---------------------------------------- | ------- | ------------------------------ |
| `meta-llama/Llama-3.3-70B-Instruct`    | `/meta-llama-llama-3-3-70b-instruct/v1`  | 131K    | **Default — best quality**     |
| `meta-llama/Llama-3.1-70B-Instruct`    | `/meta-llama-llama-3-1-70b-instruct/v1`  | 131K    | Older sibling of 3.3           |
| `meta-llama/Llama-3.1-8B-Instruct`     | `/meta-llama-llama-3-1-8b-instruct/v1`   | 131K    | Fast, lighter for dev/tests    |
| `openai/gpt-oss-120b`                  | `/openai-gpt-oss-120b/v1`                | 32K     | Capable but small context      |
| `openai/gpt-oss-20b`                   | `/openai-gpt-oss-20b/v1`                 | 131K    | Mid-size                       |

## Deploying to Kagenti

### Quick Deploy (CLI)

The `deploy-to-kagenti.sh` script handles the full build + deploy pipeline:

```bash
cd a2a/kagenti_chat
./deploy-to-kagenti.sh [namespace]   # defaults to team1
```

This script:

1. Labels the namespace for shared gateway access
2. Creates a Shipwright `Build` using `buildah-insecure-direct` strategy
3. Triggers a `BuildRun` and waits for completion (typically 2-5 min)
4. Deploys `Deployment`, `Service`, and `HTTPRoute`
5. Waits for the pod to be ready

**Prerequisites:**

- Kagenti cluster with Shipwright installed
- `buildah-insecure-direct` `ClusterBuildStrategy` available cluster-wide
- Internal registry at `registry.cr-system` (ClusterIP `10.43.28.116:5000`)
- k3s nodes configured with the registry as insecure
- llm-d deployed with `Llama-3.3-70B-Instruct` model

### Verifying the deployment

```bash
# Check the agent card
curl -sk https://kagenti-chat.163-75-85-180.sslip.io/.well-known/agent-card.json | jq .

# Send a chat message
curl -sk -X POST https://kagenti-chat.163-75-85-180.sslip.io/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Hello!"}],
        "messageId": "test-1"
      }
    }
  }' | jq -r '.result.artifacts[0].parts[0].text'
```

### Architecture notes

The deployment uses these specific configurations to work around Kagenti cluster constraints:

- **Image reference uses ClusterIP** (`10.43.28.116:5000`) instead of the registry's DNS name —
  kubelet on the nodes can't resolve cluster-internal DNS names like
  `registry.cr-system.svc.cluster.local`.
- **HTTPRoute attaches to `kagenti-system/http` Gateway** rather than the llm-d gateway —
  this is the gateway with both HTTP and HTTPS listeners exposed externally.
- **Direct llm-d service URL** in `LLM_API_BASE` (not the gateway path) —
  `http://ms-meta-llama-llama-3-3-70b-instruct-svc.llm-d.svc.cluster.local:8000/v1` —
  avoids HTTP→HTTPS redirect issues from the gateway.
- **`kagenti.io/inject: disabled`** label on pods skips the authbridge sidecar injection,
  which would otherwise require a Keycloak client secret.

### Manifest reference

See `k8s/kagenti-deploy.yaml` for the full Deployment + Service + HTTPRoute manifest.

### Alternative: Kagenti UI

You can also deploy via the Kagenti UI under "Import New Agent". Select the `llm-d` env
block from `sample-environments.yaml` and customize `LLM_MODEL` + `LLM_API_BASE` for the
model you want this agent to use. Set `A2A_PUBLIC_URL` to the externally-reachable URL.

## CLI client

A Claude/Codex-style CLI lets you (and remote colleagues) chat with the deployed agent
from any laptop with Python 3.12+.

### Install and run

```bash
git clone https://github.com/moonlight16/agent-examples.git
cd agent-examples/a2a/kagenti_chat
uv sync
uv run kagenti-chat
```

The default endpoint is the deployed cluster URL. Override with `--url` or
`KAGENTI_CHAT_URL`:

```bash
uv run kagenti-chat --url https://kagenti-chat.163-75-85-180.sslip.io
KAGENTI_CHAT_URL=http://localhost:8000 uv run kagenti-chat
```

### Features

- **Streaming responses** — agent output appears as it's generated.
- **Markdown rendering** — code blocks, lists, headers all rendered with Rich.
- **Conversation history** — chat history persisted at `~/.kagenti-chat/history.jsonl`,
  input history at `~/.kagenti-chat/input_history`.
- **Slash commands** — `/help`, `/clear`, `/save`, `/info`, `/exit`.

### Common flags

| Flag             | Description                                     |
| ---------------- | ----------------------------------------------- |
| `--url URL`      | Agent endpoint (default: deployed cluster URL)  |
| `--insecure`/`-k`| Skip TLS cert verification (self-signed certs)  |
| `--no-stream`    | Disable streaming responses                     |

## Tracing

If `OTEL_EXPORTER_OTLP_ENDPOINT` is set, AG2 LLM/agent spans go there.
For local debugging, set `OTEL_CONSOLE_TRACING=true` to print spans to stdout.
