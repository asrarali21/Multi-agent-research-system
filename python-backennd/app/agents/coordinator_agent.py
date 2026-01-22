from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal
import os
import json

GOOGLE_APIKEY = os.getenv("GOOGLE_APIKEY")


class SubTask(BaseModel):
    """Represents a single research sub-task assigned to an agent."""
    task_id: int = Field(..., description="Unique identifier for the task")
    agent: Literal["Web Search Agent", "Academic Agent", "Data Analysis Agent", "Synthesis Agent"] = Field(
        ..., description="The specialist agent assigned to this task"
    )
    objective: str = Field(..., description="Clear, specific research objective for this task")
    focus_areas: List[str] = Field(..., description="Specific topics or aspects to focus on")
    priority: Literal["high", "medium", "low"] = Field(..., description="Task priority level")


class CoordinationPlan(BaseModel):
    """Complete coordination plan for the research task."""
    query_understanding: str = Field(..., description="Brief explanation of what the user is asking for")
    research_scope: str = Field(..., description="Define the boundaries and depth of research needed")
    sub_tasks: List[SubTask] = Field(..., description="List of sub-tasks assigned to various agents")
    expected_deliverables: List[str] = Field(..., description="What the final output should include")
    estimated_complexity: Literal["low", "medium", "high"] = Field(..., description="Overall complexity assessment")

class CoordinatorAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GOOGLE_APIKEY,
            temperature=0.3
        )

        # 🔐 Schema-enforced LLM
        self.structured_llm = self.llm.with_structured_output(
            CoordinationPlan
        )

    def build_prompt(self, user_query: str) -> str:
        return f"""
You are an expert Research Coordinator Agent responsible for orchestrating multi-agent research tasks.

## YOUR ROLE
You are the project manager of a research team. You DO NOT conduct research yourself.

## AVAILABLE SPECIALIST AGENTS (Use exact names)
- Web Search Agent
- Academic Agent
- Data Analysis Agent
- Synthesis Agent

## USER QUERY
{user_query}

## REQUIREMENTS
- Minimum 4 sub-tasks (one per agent)
- Maximum 8 sub-tasks
- Each sub-task must have at least 2 focus areas
- task_id must start at 1 and increment by 1
- Use priorities: high / medium / low
- Estimated complexity: low / medium / high

Respond ONLY with a valid CoordinationPlan object.
"""

    async def coordinate(self, user_query: str) -> CoordinationPlan:
        return await self.structured_llm.ainvoke(
            self.build_prompt(user_query)
        )
