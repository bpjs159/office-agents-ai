# Office Agents AI

Multi-agent office orchestrator in TypeScript.

This project runs a small autonomous "office" where multiple agents collaborate through standups, chat delegation, persistent memory (ChromaDB), and isolated workspaces.

## What it does

- Loads agent definitions from JSON files in `agents/`.
- Spawns one Node worker process per agent.
- Runs periodic standups with a configured leader.
- Lets agents chat with each other and delegate tasks.
- Stores/retrieves agent memory in ChromaDB.
- Gives each agent its own workspace directory.
- Performs startup health checks:
  - ChromaDB connection
  - LLM connectivity
- Exposes a CLI for interactive operation.

## Project structure

- `office.ts` — main orchestrator (process management, routing, standups, CLI)
- `agents/agent-worker.ts` — worker runtime for each agent
- `llm.ts` — LLM client (Groq/Ollama modes)
- `office.config.json` — office-level config (standups, leader, target, MCP servers)
- `agents/*.json` — per-agent profiles (name, prompt, MCP access, workspace)
- `agent-workspaces/` — isolated file space per agent
- `chroma-db/` — local Chroma persistence path (when running local server)

## Requirements

- Node.js 18+
- npm
- Chroma CLI (installed through npm package `chromadb`, already in dependencies)
- One LLM backend:
  - **Groq** (`MODEL_MODE=groq` or `grow` + `GROQ_API_KEY`)
  - **Ollama** (`MODEL_MODE=ollama` + running Ollama server)

## Installation

```bash
npm install
```

## Environment variables

Create a `.env` file at project root (optional but recommended):

```env
# Chroma
CHROMA_URL=http://localhost:8000

# LLM mode: groq | grow | ollama
MODEL_MODE=ollama

# Groq (required for groq/grow mode)
GROQ_API_KEY=your_key_here

# Ollama (used in ollama mode)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

# Optional trace switch (default is on)
# AGENT_TRACE=1

# Optional LLM/internal-memory trace switch (default is off)
# AGENT_LLM_TRACE=0
```

## Run

### Recommended (auto-start/reuse Chroma)

```bash
npm run start
```

This script:
- Reuses existing Chroma server on port `8000` if present.
- Otherwise starts local Chroma (`npx chroma run --path ./chroma-db`).
- Waits for Chroma heartbeat.
- Starts `office.ts`.

### Run orchestrator only (no Chroma auto-start)

```bash
npm run start:with-no-chroma
```

### Run only Chroma server

```bash
npm run chroma:run
```

### Build

```bash
npm run build
```

## CLI commands (inside `office>` prompt)

- `agents` — list agents, MCP access, workspace paths
- `mcp` — show MCP server status
- `trace <on|off>` — toggle runtime trace logs
- `llm-trace <on|off>` — toggle LLM/internal-memory trace logs (default off)
- `standup-now` — trigger standup immediately
- `chat <fromAgent> <toAgent> <message>` — delegate message between agents
- `ask <agent> <message>` — send direct user message to an agent
- `memory <agent> <query>` — query agent memory
- `exit` — stop office and worker processes

## Current team configuration

- `pedro-redactor` (leader)
- `juan-searcher-1`
- `lucas-searcher-2`

Standup leader and schedule are configured in `office.config.json`.

## Agent configuration format

Each agent file in `agents/*.json` uses this shape:

```json
{
  "name": "agent-name",
  "systemPrompt": "Agent behavior instructions",
  "mcpAccess": ["internet", "files"],
  "workspace": {
    "root": "./agent-workspaces/agent-name"
  }
}
```

## Office configuration format

`office.config.json`:

```json
{
  "standupIntervalMinutes": 30,
  "standupLeader": "pedro-redactor",
  "mainTarget": "High-level objective of the office",
  "mcpServers": {
    "terminal": { "enabled": true, "description": "..." },
    "files": { "enabled": true, "description": "..." },
    "internet": { "enabled": true, "description": "..." }
  }
}
```

## Tracing and monitoring

Trace is enabled by default.

LLM/internal-memory trace is disabled by default.

Example trace line:

```text
[2026-03-20 14:35:10] [trace:juan-searcher-1] chat.received {"from":"SYSTEM","preview":"..."}
```

Example LLM trace line:

```text
[2026-03-20 14:35:11] [llm-trace:pedro-redactor] llm.response {"length":883}
```

To persist logs to file:

```bash
npm run start 2>&1 | tee .office.log
```

Filter one agent:

```bash
tail -f .office.log | grep --line-buffered '\[trace:lucas-searcher-2\]\|\[chat\] lucas-searcher-2'
```

Filter only LLM traces for one agent:

```bash
tail -f .office.log | grep --line-buffered '\[llm-trace:lucas-searcher-2\]'
```

## Behavior notes

- Startup fails fast if Chroma or LLM is unavailable.
- Standup is triggered on boot and then by interval.
- Leader-driven workflow is proactive by design.
- Agent responses to `USER` and `SYSTEM` are printed in terminal.

## Troubleshooting

### `Address localhost:8000 is not available`

Port `8000` is already in use. The default start script now reuses the existing server if available.

### `ChromaConnectionError`

- Verify Chroma is running.
- Check `CHROMA_URL`.
- Confirm heartbeat endpoint:

```bash
curl -sf http://localhost:8000/api/v2/heartbeat
```

### `GROQ_API_KEY is not set ...`

Set `GROQ_API_KEY` or switch to Ollama mode:

```env
MODEL_MODE=ollama
```

### No visible activity after startup

- Run `trace on`
- Trigger `standup-now`
- Use `ask <agent> <message>` to generate immediate work

## License

ISC (see `LICENSE`).
