# Architecture

## Overview

Single-repo collection of standalone Python scripts, each in its own numbered module. Each module is a self-contained lesson: it imports LangChain primitives, builds a chain or agent, and runs it from `main()`. No shared code between modules — intentional for course clarity.

## Layers

| Layer | Responsibility | Location |
|-------|---------------|---------|
| Entry point | `main()` runner, strategy selection | `<module>/main.py` |
| Agent / Chain | LangChain graph or chain construction | `<module>/main.py` |
| Tools | `@tool`-decorated callables wrapping external APIs | `<module>/main.py` |
| Schemas | Pydantic `BaseModel` for structured LLM output | `<module>/main.py` |
| Config | `.env` loaded via `python-dotenv` | project root `.env` |

## Module Progression

```
01-base  →  PromptTemplate | LLM chain  (ChatGroq / ChatOllama)
02-react →  ReAct agent | @tool | structured output (Groq 2-step / Ollama native)
03-react-under-the-hood  →  (WIP)
```

## Request / Data Flow — 02-react (ReAct Agent)

```
main()
  └─ run_groq_strategy() / run_ollama_strategy()
        └─ create_agent(llm, tools, [response_format])
              └─ agent.invoke({"messages": [HumanMessage]})
                    ├─ LLM decides to call @tool
                    │     └─ search(query) → TavilyClient.search()
                    └─ LLM produces final answer
                          └─ [Groq] structured_llm.invoke(last_message)
                          └─ [Ollama] result["structured_response"]
                                └─ AgentResponse (Pydantic)
```

## Key Components

| Component | Role |
|-----------|------|
| `ChatGroq` | Remote LLM via Groq cloud |
| `ChatOllama` | Local LLM via Ollama server |
| `create_agent` | Builds a LangGraph ReAct agent graph |
| `@tool search` | Wraps Tavily web search as a LangChain tool |
| `AgentResponse` | Pydantic schema: `answer: str` + `sources: list[Source]` |
| `PromptTemplate` | `01-base` simple string prompt with variable interpolation |

## Structured Output Strategies

Two approaches demonstrated in `02-react` for getting a typed `AgentResponse`:

| Strategy | Mechanism | When |
|----------|-----------|------|
| **Groq** | Agent runs → separate `llm.with_structured_output()` call on final message | Groq doesn't support tools + structured output simultaneously |
| **Ollama** | `create_agent(..., response_format=AgentResponse)` → `result["structured_response"]` | Ollama supports native combined tool-use + structured output |

## External Dependencies

| Service | How Used | Protocol |
|---------|---------|---------|
| Groq API | `ChatGroq` for cloud inference | REST via `langchain-groq` SDK |
| LangSmith | Automatic tracing when env vars set | REST via LangChain callbacks |
| Ollama (local) | `ChatOllama` for local inference | HTTP via `langchain-ollama` SDK |
| Tavily | `TavilyClient.search()` inside `@tool` | REST via `tavily-python` SDK |

## Testing Strategy

No tests present. Course exercises are run and verified manually.