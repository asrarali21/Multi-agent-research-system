from pydantic import BaseModel, Field
from typing import List
import os
from langchain_groq import ChatGroq

class Subtask(BaseModel):
    id: str = Field(
        ...,
        description="short identifier for sub task(e.g. 'A','History' , 'drivers').",
    )
    title: str = Field(
        ...,
        description="Short descriptive title for each sub task"
    )
    description: str = Field(
        ...,
        description="Clear , defined Instructions for the sub agent that will research this sub task"
    )

class SubTaskList(BaseModel):
    subtasks: List[Subtask] = Field(
        ...,
        description="List of subtasks that together cover the whole research plan"
    )

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class SubTaskAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.3
        )
        self.structured_llm = self.llm.with_structured_output(SubTaskList)

    def build_prompt(self, research_plan: str):
        return f"""
You will be given a set of research instructions (a research plan).
Your job is to break this plan into a set of coherent, non-overlapping subtasks that can be researched independently by separate agents.
Requirements:
- 3 to 8 subtasks is usually a good range. Use your judgment.
- Each subtask should have:
- an 'id' (short string),
- a 'title' (short descriptive title),
- a 'description' (clear, detailed instructions for the sub-agent).
- Subtasks should collectively cover the full scope of the original plan without unnecessary duplication.
- Prefer grouping by dimensions: time periods, regions, actors, themes, causal mechanisms, etc., depending on the topic.
- Each description should be very clear and detailed about everything that the agent needs to research to cover that topic.
- Do not include a final task that will put everything together.
This will be done later in another step.
{research_plan}
"""
    
    async def sub_task(self, research_plan: str) -> SubTaskList:
        response = await self.structured_llm.ainvoke(self.build_prompt(research_plan=research_plan))
        return response