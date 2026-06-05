# LangGraph — Udemy Course (Eden Marco)

Hands-on exercises following the LangGraph/LangChain course by Eden Marco.

## Lessons

| Folder | Title | Purpose |
|--------|-------|---------|
| `01-base` | Base AI Workflow | Minimal LCEL chain (`prompt \| llm`). No agents, no tools — establishes the baseline before introducing agents. |
| `02-react` | ReAct Agent | Full ReAct agent built with `create_agent`. The reason → act → observe loop is hidden inside the framework. Covers tool use, structured output (Groq workaround and Ollama `ProviderStrategy`). |
| `03-react-loop-under-the-hood` | ReAct Loop Under the Hood | Manually implements what `create_agent` does internally: explicit iteration, `tool_calls` inspection, and `ToolMessage` dispatch. Educational only — everything here can be achieved with `02`'s approach via config (`recursion_limit`) and callbacks. |

### `02` vs `03` — same thing, different abstraction level

`03` is not a separate technique — it is a teaching exercise. Every capability it exposes manually is available in `02`:

| `03` explicit | `02` equivalent |
|--------------|-----------------|
| `for iteration in range(MAX_ITERATIONS)` | `recursion_limit` in invoke config |
| `print` per iteration | LangSmith tracing or a custom callback |
| Custom `SystemMessage` | Pass it in the `messages` list on `invoke` |
| Manual `ToolMessage` dispatch | Handled internally by `create_agent` |

Once you understand the loop `03` shows, use `02`'s approach for all real code.

## Key Concepts

- **ReAct pattern** — reason → act (call a tool) → observe (tool result) → repeat until no more tool calls
- **Tool use** — `@tool`-decorated functions bound to the LLM via `bind_tools` or passed to `create_agent`
- **Structured output** — Pydantic schemas as response format; Groq requires a separate `.with_structured_output()` call, Ollama supports `ProviderStrategy` natively
- **Iteration limit** — `recursion_limit` in the invoke config caps how many graph steps `create_agent` will run (default: 25; ~2 steps per iteration)

## Setup

```bash
uv sync
cp .env.example .env   # set GROQ_MODEL, OLLAMA_MODEL, TAVILY_API_KEY
```

Run any lesson:

```bash
uv run python 01-base/main.py
uv run python 02-react/main.py
uv run python 03-react-loop-under-the-hood/main.py
```
