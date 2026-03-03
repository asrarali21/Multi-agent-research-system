# Multi-Agent Research System

A research automation backend that takes a user query, decomposes it into subtasks, dispatches parallel sub-agents to gather information from the web, and synthesizes a structured final report.

Built with **LangGraph** for orchestration, **Llama 3.3 70B on Groq** for reasoning, and **Firecrawl** for web search/scraping.

## How It Works

```
User Query
    │
    ▼
1. Generate Research Plan  ─── Llama 3.3 70B produces step-by-step research instructions
    │
    ▼
2. Split into Subtasks     ─── Llama 3.3 70B decomposes plan into 3-8 independent subtasks
    │
    ▼
3. Spawn Sub-Agents        ─── LangGraph Send() fans out one ReAct agent per subtask
    │                           Each agent searches/scrapes the web via Firecrawl
    ▼
4. Synthesize Report       ─── Llama 3.3 70B merges all sub-reports into a structured final report
```

Sub-agents run in parallel using LangGraph's `Send()` API. Each is a `create_react_agent` with access to web search and page scraping tools. Results are collected via a reducer (`operator.add`) and passed to the synthesis step.

## Project Structure

```
python-backennd/app/
├── agents/
│   ├── research_plan_agent.py   # Node 1: generates research instructions
│   ├── sub_task_agent.py        # Node 2: splits plan into subtasks (structured output)
│   └── coordinator_agent.py     # Nodes 3-4: sub-agent execution + report synthesis
├── graphs/
│   └── research_graph.py        # LangGraph StateGraph definition
├── tools/
│   └── fire_crawl.py            # Firecrawl search & scrape tools (@tool decorated)
├── api/
│   └── user_query.py            # FastAPI POST /ask endpoint
└── main.py                      # FastAPI app entry point
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Orchestration | LangGraph (StateGraph + Send) |
| LLM | Llama 3.3 70B via Groq |
| Agent framework | LangGraph `create_react_agent` |
| Web search/scrape | Firecrawl |
| API | FastAPI |
| Structured output | Pydantic + `with_structured_output` |

## Key Design Decisions

**Why LangGraph over raw asyncio?**  
`Send()` handles dynamic fan-out/fan-in with state management. We don't know the number of subtasks upfront (varies 3-8 per query), and we need each sub-agent's output merged into a shared state before synthesis.

**Why `create_react_agent` over `deepagents`?**  
Sub-agents only need search + scrape tools. `deepagents` adds file I/O, shell access, and planning middleware — unnecessary overhead for research tasks.



