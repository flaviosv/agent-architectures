# LangGraph — Udemy Course (Eden Marco)

Hands-on exercises following the LangGraph/LangChain frameworks

## Lessons

| Folder | Title | Purpose |
|--------|-------|---------|
| `01-base` | Base AI Workflow | Minimal LCEL chain (`prompt \| llm`). No agents, no tools — establishes the baseline before introducing agents. |
| `02-react` | ReAct Agent | Full ReAct agent built with `create_agent`. The reason → act → observe loop is hidden inside the framework. Covers tool use, structured output (Groq workaround and Ollama `ProviderStrategy`). |
| `03-react-loop-under-the-hood` | ReAct Loop Under the Hood | Manually implements what `create_agent` does internally: explicit iteration, `tool_calls` inspection, and `ToolMessage` dispatch. Educational only — everything here can be achieved with `02`'s approach via config (`recursion_limit`) and callbacks. |
| `04-react-loop-raw-function-calling` | ReAct Loop — Raw Function Calling | Agent workflow built without LangChain: calls Ollama directly, uses LangSmith for tracing only. Tool definitions are written manually as JSON schemas (OpenAI function-calling format), though Google-style docstrings on the functions would achieve the same result. Shows what the framework abstracts away at the protocol level. |
| `05-raw-react-prompts` | ReAct Loop — Prompt-Driven Reasoning | Same stack as `04` (Ollama + LangSmith, no LangChain) but removes native tool-calling entirely. Instead, the classic ReAct prompt format (`Thought / Action / Action Input / Observation / Final Answer`) is injected into the system prompt, and tool descriptions are generated at runtime from function signatures and docstrings via `inspect`. The LLM reasons in plain text and decides which tool to call; the loop parses that text to dispatch the right function. There is no system prompt — the question is embedded directly into the ReAct prompt, making the entire conversation a single unified user-level prompt. A `scratchpad` string accumulates `Thought/Action/Observation` turns and is appended to the prompt each iteration, since there is no message history. A `stop` token (`"\nObservation"`) halts LLM generation at the observation boundary, preventing the model from hallucinating its own tool results. |
| `06-rag` | RAG — Retrieval-Augmented Generation | Introduces RAG using Ollama embeddings and Pinecone as the vector store. `ingestion.py` loads a local text file, splits it into chunks (`CharacterTextSplitter`, 1 000-char chunks, 8-char overlap), embeds them with Ollama, and upserts into a Pinecone index. `main.py` demonstrates three implementations against the same query: **(0)** raw LLM with no retrieval context; **(1)** a manual retrieval chain — retrieve → format → build prompt → invoke LLM — without LCEL, to expose the individual steps; **(2)** the same pipeline rewritten as a composable LCEL chain (`RunnablePassthrough.assign | prompt | llm | StrOutputParser`), which gains streaming, async, and batch support for free. The side-by-side comparison makes the verbosity vs. composability trade-off concrete. |
| `07-agentic-rag` | Agentic RAG | Upgrades `06` by making retrieval agent-driven: the LLM decides when to call the retrieval tool rather than retrieval always running unconditionally. `ingestion.py` crawls `https://docs.langchain.com/oss/python/langchain/overview` via `TavilyCrawl` (`max_depth=5`), splits with `RecursiveCharacterTextSplitter` (4 000-char chunks, 200-char overlap), and upserts into Pinecone asynchronously in parallel batches of 500 via `asyncio.gather`. `backend/core.py` wraps Pinecone retrieval as a `@tool(response_format="content_and_artifact")` — the tool returns both a serialized string (with source URLs) for the LLM context and the raw `Document` objects as an artifact for downstream use; `main()` collects these artifacts from `ToolMessage.artifact` to return alongside the answer. The agent is given a system prompt requiring it to cite sources and admit when retrieved docs don't cover the question. Two key contrasts with `06`: ingestion pulls live docs from the web (not a static file), and the retrieval step only fires when the agent judges it necessary. |

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
```
