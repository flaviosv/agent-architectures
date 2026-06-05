# Project Guidelines

## LangChain — Modern API Rules

Always use the **LangChain 1.x APIs**. `create_react_agent` from `langgraph.prebuilt` is outdated.

| Do | Don't |
|----|-------|
| `from langchain.agents import create_agent` | `from langgraph.prebuilt import create_react_agent` |
| `from langchain_core.tools import tool` | `from langchain.tools import tool` |
| `from langchain_core.messages import HumanMessage` | `from langchain.schema import HumanMessage` |

### Agents

- Build agents with `langchain.agents.create_agent(llm, tools)`.
- Pass `response_format=MySchema` to get structured output natively.
- Extract structured output from `result["structured_response"]` (not `result["messages"][-1]`).

### Tools

- Decorate tools with `@tool` from `langchain_core.tools`.
- Always include a docstring — it is used as the tool description for the LLM.

### Structured output

- Define schemas as `pydantic.BaseModel` subclasses.
- For models that don't support simultaneous tool-use + structured output (e.g. Groq), run the agent first, then call `llm.with_structured_output(Schema).invoke(last_message)` separately.
- For models that support it natively (e.g. Ollama), use `response_format=ProviderStrategy(Schema)` — **not** `response_format=Schema` directly.
  - Passing `Schema` directly uses `ToolStrategy`: the LLM must call the schema as a tool; unreliable after multiple tool calls.
  - `ProviderStrategy` enforces JSON output at the provider level, guaranteeing `result["structured_response"]` is always populated.
