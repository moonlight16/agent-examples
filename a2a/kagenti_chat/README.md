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

1. Build/push an image (the existing repo CI / Kagenti UI handles this).
2. In the Kagenti UI under "Import New Agent", select the `llm-d` env block from
   `sample-environments.yaml` and customize `LLM_MODEL` + `LLM_API_BASE` for the model
   you want this agent to use.
3. Set `A2A_PUBLIC_URL` to the externally-reachable URL once you know the route.

## Tracing

If `OTEL_EXPORTER_OTLP_ENDPOINT` is set, AG2 LLM/agent spans go there.
For local debugging, set `OTEL_CONSOLE_TRACING=true` to print spans to stdout.
