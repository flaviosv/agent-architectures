# Project Details

## Overview

Hands-on course exercises following the Udemy LangGraph course by Eden Marco. Each numbered module demonstrates a progressively more advanced LangChain/LangGraph pattern, from basic prompt chains to ReAct agents with structured output.

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python >= 3.14 |
| LLM Framework | LangChain / LangGraph |
| Local LLM | Ollama |
| Remote LLM | Groq |
| Package Manager | uv |
| Schema Validation | Pydantic v2 |
| Code Formatter | black, isort |

## Key Libraries

| Library | Purpose                                          |
|---------|--------------------------------------------------|
| `langchain` | Corehe chains, agents, prompt templates          |
| `langchain-groq` | ChatGroq LLM integration                         |
| `langchain-ollama` | ChatOllama local LLM integration                 |
| `langchain-tavily` | Tavily tool integration for LangChain            |
| `pydantic` | Structured output schemas (`BaseModel`, `Field`) |
| `python-dotenv` | Load `.env` at runtime                           |
| `tavily-python` | Tavily web search client                         |

## Project Structure

```
.
├── 01-base/           # Basic prompt chain with PromptTemplate
├── 02-react/          # ReAct agent — Groq and Ollama strategies, structured output
├── 03-react-under-the-hood/  # (WIP) ReAct internals
├── docs/              # Architecture context files
├── pyproject.toml     # Project metadata and dependencies
└── uv.lock            # Locked dependency tree
```

## Commands

| Task | Command |
|------|---------|
| Install dependencies | `uv sync` |
| Run a module | `uv run python <module>/main.py` |
| Format code | `uv run black .` |
| Sort imports | `uv run isort .` |

## Environment Configuration

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `GROQ_MODEL` | Groq model name (e.g. `llama3-8b-8192`) |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing |
| `LANGSMITH_ENDPOINT` | LangSmith endpoint URL |
| `LANGSMITH_PROJECT` | LangSmith project name |
| `LANGSMITH_TRACING` | Enable LangSmith tracing (`true`/`false`) |
| `OLLAMA_MODEL` | Ollama model name (e.g. `llama3.2`) |
| `TAVILY_API_KEY` | Tavily web search API key |

`.env` is gitignored. Copy variables from the table above and fill in values before running.

## External Integrations

| Service | Type | Purpose | Direction | Protocol |
|---------|------|---------|-----------|---------|
| Groq | Remote LLM API | Cloud inference | Outbound | REST (SDK) |
| LangSmith | Observability | Trace agent runs | Outbound | REST (SDK) |
| Ollama | Local LLM server | Local inference | Outbound | HTTP (SDK) |
| Tavily | Search API | Web search tool | Outbound | REST (SDK) |
