# Goal

Hands-on exercises following the LangGraph/LangChain frameworks

## Practices

| Folder | Title | Key idea |
|--------|-------|----------|
| `01-base` | Base AI Workflow | Minimal LCEL chain — no agents, no tools. Baseline only. |
| `02-react` | ReAct Agent | Agent with tool use and structured output, covering Groq and Ollama provider differences. |
| `03-react-loop-under-the-hood` | ReAct Loop Under the Hood | Manually reimplements the agent loop to expose what the framework hides. |
| `04-react-loop-raw-function-calling` | Raw Function Calling | Calls the model directly with no LangChain abstractions — tools as plain JSON schemas. |
| `05-raw-react-prompts` | Prompt-Driven ReAct | ReAct loop driven purely by prompt engineering — no native tool-calling API. |
| `06-rag` | RAG | Retrieval-Augmented Generation with Ollama embeddings and Pinecone, shown in three variants. |
| `07-agentic-rag` | Agentic RAG | Upgrades `06` — the LLM decides when to retrieve rather than always retrieving. |
| `08-langgraph-react` | LangGraph ReAct | ReAct loop built as an explicit state graph, showing the wiring that `create_agent` abstracts away. |
| `09-langgraph-reflection` | LangGraph Reflection | Self-reflection loop: a generation node and a critique node alternate for a fixed number of rounds. |
| `10-langgraph-reflexion` | LangGraph Reflexion | [Reflexion](https://arxiv.org/abs/2303.11366) architecture: the agent drafts an answer, searches the web to gather evidence, then revises with citations — looping until a max iteration limit. |
| `11-langgraph-corrective-rag` | LangGraph Advanced RAG | Corrective RAG graph: retrieves docs, grades their relevance, falls back to web search when needed, then generates — routing decisions are made at each node. |
| `12-langgraph-self-rag` | LangGraph Self-RAG | Extends `11` with Self-RAG: after generating, the graph grades the output for hallucinations (grounded in docs?) and answer relevance (addresses the question?). Returns `useful` → end, `not useful` → web search, `not supported` → regenerate. |
| `13-langgraph-adaptative-rag` | LangGraph Adaptive RAG | Extends `12` with Adaptive RAG: a router node inspects the question first and picks between vectorstore retrieval and web search before any docs are fetched. Combines up-front routing with the full self-RAG pipeline (doc grading → generate → hallucination check → answer relevance check). |
| `14-mcp-server` | MCP Server | Exposes tools via the [Model Context Protocol](https://modelcontextprotocol.io/) using `FastMCP`: a `math` server over stdio transport and a `weather` server over SSE transport, consumed through `langchain-mcp-adapters`. |

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
uv run python 09-langgraph-reflection/main.py
uv run python 10-langgraph-reflexion/main.py
uv run python 11-langgraph-corrective-rag/ingestion.py   # one-time: embed & upsert docs into Pinecone
uv run python 11-langgraph-corrective-rag/main.py
uv run python 12-langgraph-self-rag/ingestion.py         # one-time: embed & upsert docs into Chroma
uv run python 12-langgraph-self-rag/main.py
uv run python 13-langgraph-adaptative-rag/ingestion.py   # one-time: embed & upsert docs into Chroma
uv run python 13-langgraph-adaptative-rag/main.py
uv run python 14-mcp-server/servers/weather_server.py     # start first, in a separate terminal: SSE server
uv run python 14-mcp-server/main.py
```
