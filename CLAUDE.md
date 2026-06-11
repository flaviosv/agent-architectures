# Project Guidelines

## Learning instructions

- This project has the goal to study Agent architectures using LangGraph / LangChain, use the most recent techniques available
- Prefer showing / teaching approaches over doing code yourself

## LangChain 1.x API

| Do | Don't |
|----|-------|
| `from langchain.agents import create_agent` | `from langgraph.prebuilt import create_react_agent` |
| `from langchain_core.tools import tool` | `from langchain.tools import tool` |
| `from langchain_core.messages import HumanMessage` | `from langchain.schema import HumanMessage` |

## Agents & Tools

- Build agents: `create_agent(llm, tools)` — optionally with `response_format=...` for structured output.
- Tools: `@tool` from `langchain_core.tools`, always with a docstring.

## Structured Output

- **Groq** — no simultaneous tools + structured output: run agent first, then `llm.with_structured_output(Schema).invoke(last_message)`.
- **Ollama** — use `response_format=ProviderStrategy(Schema)`; extract from `result["structured_response"]`.
