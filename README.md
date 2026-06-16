# Goal

Hands-on exercises following the LangGraph/LangChain frameworks

## Practices

| Folder | Title | Key idea |
|--------|-------|----------|
| `01-base` | Base AI Workflow | Minimal `prompt \| llm` LCEL chain — no agents, no tools. Baseline only. |
| `02-react` | ReAct Agent | `create_agent` with tool use and structured output (Groq workaround + Ollama `ProviderStrategy`). |
| `03-react-loop-under-the-hood` | ReAct Loop Under the Hood | Manually reimplements `create_agent`'s loop. Educational — everything it shows is available in `02` via config. |
| `04-react-loop-raw-function-calling` | Raw Function Calling | No LangChain: Ollama called directly, tools as raw JSON schemas, LangSmith for tracing only. |
| `05-raw-react-prompts` | Prompt-Driven ReAct | No native tool-calling: `Thought / Action / Observation` prompt drives the loop; a `stop` token prevents the model from hallucinating observations. |
| `06-rag` | RAG | Ollama embeddings + Pinecone. Three side-by-side implementations: raw LLM, manual chain, and LCEL chain. |
| `07-agentic-rag` | Agentic RAG | Upgrades `06` — the LLM decides when to retrieve. Ingestion crawls live LangChain docs via `TavilyCrawl`; retrieval tool uses `response_format="content_and_artifact"` to surface both context and raw `Document` objects. |
| `08-langgraph-react` | LangGraph ReAct | Builds a ReAct loop as an explicit `StateGraph`: `agent_reason` node → conditional edge (`should_continue`) → `act` node (`ToolNode`) → back to `agent_reason`. Exports `flow.png` via `draw_mermaid_png`. Shows the graph wiring that `create_agent` hides. |

## Key Concepts

- **ReAct pattern** — reason → act (call a tool) → observe (tool result) → repeat until no more tool calls
- **Tool use** — `@tool`-decorated functions bound to the LLM via `bind_tools` or passed to `create_agent`
- **Structured output** — Pydantic schemas as response format; Groq requires a separate `.with_structured_output()` call, Ollama supports `ProviderStrategy` natively
- **Iteration limit** — `recursion_limit` in the invoke config caps how many graph steps `create_agent` will run (default: 25; ~2 steps per iteration)

## Setup

```bash
uv sync
cp .env.sample .env    # fill in the values — see .env.sample for all required keys
```

Run any lesson:

```bash
uv run python 01-base/main.py
uv run python 02-react/main.py
uv run python 03-react-loop-under-the-hood/main.py
uv run python 04-react-loop-raw-function-calling/main.py
uv run python 05-raw-react-prompts/main.py
uv run python 06-rag/ingestion.py       # one-time: embed & upsert docs into Pinecone
uv run python 06-rag/main.py
uv run python 07-agentic-rag/ingestion.py   # one-time: crawl & upsert LangChain docs via Tavily
uv run python 07-agentic-rag/backend/core.py
uv run python 08-langgraph-react/main.py
```
