from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Annotated
from pydantic import BaseModel, Field
import operator
import os
import asyncio

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ✅ Define State Schema
class ResearchState(TypedDict):
    """State that flows through the research workflow"""
    user_query: str
    research_plan: str
    subtasks: List[Dict[str, str]]  # List of {id, title, description}
    sub_reports: Annotated[List[Dict[str, str]], operator.add]  # Accumulate reports
    final_report: str
    error: str | None


# ✅ Pydantic Models for Structured Output
class SubAgentReport(BaseModel):
    subtask_id: str
    subtask_title: str
    report: str


class FinalReport(BaseModel):
    executive_summary: str = Field(description="Brief overview of all findings")
    detailed_findings: str = Field(description="Comprehensive analysis with sections")
    key_insights: List[str] = Field(description="Main takeaways (5-10 points)")
    open_questions: List[str] = Field(description="Areas needing further research")
    bibliography: List[str] = Field(description="All sources used, deduplicated")


class CoordinatorAgent:
    """
    LangGraph-based Coordinator Agent that orchestrates research workflow:
    1. Takes research plan and subtasks
    2. Spawns sub-agents for each subtask (parallel execution)
    3. Collects all reports
    4. Synthesizes final comprehensive report
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
        
        self.structured_llm = self.llm.with_structured_output(FinalReport)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()


    async def run_sub_agents_node(self, state: ResearchState) -> ResearchState:
        """
        Spawns and runs all sub-agents in parallel.
        """
        print(f"🎯 Coordinator: Running {len(state['subtasks'])} sub-agents...")
        
        # Create tasks for all sub-agents
        tasks = [
            self._run_single_sub_agent(
                user_query=state["user_query"],
                research_plan=state["research_plan"],
                subtask=subtask
            )
            for subtask in state["subtasks"]
        ]
        
        # Execute all sub-agents concurrently
        sub_reports = await asyncio.gather(*tasks)
        
        print(f"✅ All {len(sub_reports)} sub-agents completed!")
        
        return {
            **state,
            "sub_reports": sub_reports
        }

    async def _run_single_sub_agent(
        self,
        user_query: str,
        research_plan: str,
        subtask: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Executes a single sub-agent for one subtask.
        """
        
        prompt = f"""
You are a specialized research sub-agent.

## CONTEXT
Global user query: {user_query}
Overall research plan: {research_plan}

## YOUR SPECIFIC SUBTASK
ID: {subtask['id']}
Title: {subtask['title']}
Instructions: {subtask['description']}

## YOUR JOB
- Focus ONLY on this subtask
- Search for up-to-date, high-quality sources
- Prioritize primary and official sources
- Be explicit about uncertainties and gaps

## OUTPUT FORMAT (Markdown)
# {subtask['title']}

## Summary
Brief overview (2-3 paragraphs)

## Detailed Analysis
Comprehensive explanation with subsections

## Key Points
- Important finding 1
- Important finding 2

## Data & Statistics
Relevant numbers, trends

## Sources
- [Source 1 Title](url) - Why relevant
- [Source 2 Title](url) - Why relevant

Return ONLY the markdown report.
"""
        
        response = await self.llm.ainvoke(prompt)
        
        return {
            "subtask_id": subtask["id"],
            "subtask_title": subtask["title"],
            "report": response.content
        }


    async def synthesize_node(self, state: ResearchState) -> ResearchState:
        """
        Synthesizes all sub-agent reports into final comprehensive report.
        """
        print("🔄 Coordinator: Synthesizing final report...")
        
        # Combine all sub-reports
        all_reports = "\n\n---\n\n".join([
            f"# SUBTASK: {report['subtask_title']}\n\n{report['report']}"
            for report in state["sub_reports"]
        ])
        
        prompt = f"""
You are the LEAD RESEARCH COORDINATOR synthesizing multiple research reports.

## ORIGINAL USER QUERY
{state['user_query']}

## RESEARCH PLAN
{state['research_plan']}

## ALL SUB-AGENT REPORTS
{all_reports}

## YOUR TASK
Synthesize ALL findings into ONE comprehensive, cohesive research report.

Requirements:
1. **Executive Summary**: Brief overview (3-4 paragraphs)
2. **Detailed Findings**: Comprehensive analysis organized by themes (not by subtask)
3. **Key Insights**: 5-10 bullet points
4. **Open Questions**: Areas requiring further research
5. **Bibliography**: Deduplicated list of all sources

Guidelines:
- Integrate findings across all subtasks
- Avoid redundancy
- Highlight patterns and connections
- Use clear headings
- DO NOT mention "subtask A", "subtask B" in final report

Respond with a FinalReport object.
"""
        
        final_report = await self.structured_llm.ainvoke(prompt)
        
        print("✅ Final report ready!")
        
        return {
            **state,
            "final_report": final_report.model_dump_json(indent=2)
        }


    def _build_workflow(self) -> StateGraph:
        """
        Builds the LangGraph workflow for coordination.
        
        Graph structure:
        START → run_sub_agents → synthesize → END
        """
        
        # Create workflow graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("run_sub_agents", self.run_sub_agents_node)
        workflow.add_node("synthesize", self.synthesize_node)
        
        # Define edges (flow)
        workflow.set_entry_point("run_sub_agents")
        workflow.add_edge("run_sub_agents", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()


    async def coordinate(
        self,
        user_query: str,
        research_plan: str,
        subtasks_list
    ) -> str:
        """
        Main coordination method.
        
        Args:
            user_query: Original user query
            research_plan: Research plan from ResearchPlan agent
            subtasks_list: SubTaskList from SubTaskAgent
            
        Returns:
            Final report as JSON string
        """
        
        # Prepare initial state
        initial_state: ResearchState = {
            "user_query": user_query,
            "research_plan": research_plan,
            "subtasks": [
                {
                    "id": subtask.id,
                    "title": subtask.title,
                    "description": subtask.description
                }
                for subtask in subtasks_list.subtasks
            ],
            "sub_reports": [],
            "final_report": "",
            "error": None
        }
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        return final_state["final_report"]