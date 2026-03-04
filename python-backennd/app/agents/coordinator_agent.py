from pydantic import BaseModel, Field
from typing import TypedDict, Dict, List, Annotated
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
import asyncio
import operator
import os
import json
from app.agents.research_plan_agent import ResearchPlan
from app.agents.sub_task_agent import SubTaskAgent
from app.tools.fire_crawl import research_tools

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


#Structured output model for the final report 
class FinalReport(BaseModel):
    executive_summary: str = Field(description="Brief overview of all findings")
    detailed_findings: str = Field(description="Comprehensive analysis with sections")
    key_insights: List[str] = Field(description="Main takeaways (5-10 points)")
    open_questions: List[str] = Field(description="Areas needing further research")
    bibliography: List[str] = Field(description="All sources used, deduplicated")


#LangGraph State definitions 

class ResearchState(TypedDict):
    """State that flows through the entire workflow."""
    user_query: str
    research_plan: str
    subtasks: List[dict]                                    
    sub_reports: Annotated[List[dict], operator.add]        
    final_report: dict | None
    errors: Annotated[List[str], operator.add]


class SubAgentInput(TypedDict):
    """Input state for a single sub-agent (received via Send())."""
    user_query: str
    research_plan: str
    subtask: dict
    stagger_index: int  
    sub_reports: Annotated[List[dict], operator.add]
    errors: Annotated[List[str], operator.add]




class CoordinatorAgent:

    def __init__(self):
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.3,
        )
        self.research_planner = ResearchPlan()
        self.subtask_splitter = SubTaskAgent()

    

    async def generate_plan(self, state: ResearchState) -> dict:
        """Generate a research plan from the user query."""
        print(f"\nGenerating research plan for: {state['user_query'][:80]}...")

        try:
            plan = await self.research_planner.research(state["user_query"])
            print(f"Research plan generated ({len(plan)} chars)")
            return {"research_plan": plan}
        except Exception as e:
            print(f"Error generating plan: {e}")
            return {
                "research_plan": "",
                "errors": [f"Plan generation failed: {str(e)}"],
            }

    

    async def split_subtasks(self, state: ResearchState) -> dict:
        """Split the research plan into independent subtasks."""
        print(f"\n Splitting research plan into subtasks...")

        try:
            subtask_list = await self.subtask_splitter.sub_task(state["research_plan"])
            subtasks = [st.model_dump() for st in subtask_list.subtasks]
            print(f"Created {len(subtasks)} subtasks:")
            for st in subtasks:
                print(f"   • [{st['id']}] {st['title']}")
            return {"subtasks": subtasks}
        except Exception as e:
            print(f" Error splitting subtasks: {e}")
            return {
                "subtasks": [],
                "errors": [f"Subtask splitting failed: {str(e)}"],
            }



    async def run_sub_agent(self, state: SubAgentInput) -> dict:
        """Run a single research sub-agent for one subtask with retry on rate limits."""
        subtask = state["subtask"]
        subtask_id = subtask["id"]
        subtask_title = subtask["title"]

        print(f"\nSub-agent [{subtask_id}]: Starting → {subtask_title} (Groq/Llama 3.3 70B)")

        sub_agent_prompt = f"""You are a specialized research sub-agent.

Global user query:
{state['user_query']}

Overall research plan:
{state['research_plan']}

Your specific subtask (ID: {subtask_id}, Title: {subtask_title}) is:

\"\"\"{subtask['description']}\"\"\"

Instructions:
- Focus ONLY on this subtask, but keep the global query in mind for context.
- Use the available tools to search for up-to-date, high-quality sources.
- Prioritize primary and official sources when possible.
- Be explicit about uncertainties, disagreements in the literature, and gaps.
- Return your results as a MARKDOWN report with this structure:

# [{subtask_id}] {subtask_title}

## Summary
Short overview of the main findings.

## Detailed Analysis
Well-structured explanation with subsections as needed.

## Key Points
- Bullet point list of main findings

## Sources
- [Title](url) - short comment on why this source is relevant

Now perform the research and return ONLY the markdown report.
"""

        max_retries = 3
        base_delay = 25  

        for attempt in range(max_retries + 1):
            try:
               
                agent = create_react_agent(
                    model=self.llm,
                    tools=research_tools,
                    prompt="You are a research sub-agent. Use the search and scrape tools to gather information. Be thorough but focused.",
                )

                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": sub_agent_prompt}]}
                )

                final_message = result["messages"][-1]
                report_content = final_message.content if hasattr(final_message, "content") else str(final_message)

                print(f"Sub-agent [{subtask_id}]: Done ({len(report_content)} chars)")

                return {
                    "sub_reports": [{
                        "subtask_id": subtask_id,
                        "subtask_title": subtask_title,
                        "report": report_content,
                    }]
                }

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "RESOURCE_EXHAUSTED" in error_str or "429" in error_str

                if is_rate_limit and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 25s, 50s, 100s
                    print(f"⏳ Sub-agent [{subtask_id}]: Rate limited (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue

                error_msg = f"Sub-agent [{subtask_id}] failed: {error_str}"
                print(f"{error_msg}")
                return {
                    "sub_reports": [{
                        "subtask_id": subtask_id,
                        "subtask_title": subtask_title,
                        "report": f"**Error**: {error_str}",
                    }],
                    "errors": [error_msg],
                }



    async def synthesize_report(self, state: ResearchState) -> dict:
        """Combine all sub-agent reports into a single coherent research report."""
        print(f"\nSynthesizing {len(state['sub_reports'])} sub-reports into final report...")


        combined_reports = ""
        for report in state["sub_reports"]:
            combined_reports += f"\n\n{'='*60}\n"
            combined_reports += f"SUB-REPORT: [{report['subtask_id']}] {report['subtask_title']}\n"
            combined_reports += f"{'='*60}\n"
            combined_reports += report["report"]

        synthesis_prompt = f"""You are the LEAD RESEARCH COORDINATOR.

The user originally asked:
\"\"\"{state['user_query']}\"\"\"

The research plan was:
\"\"\"{state['research_plan']}\"\"\"

Multiple sub-agents have independently researched different aspects. Here are ALL their reports:

{combined_reports}

Your job is to SYNTHESIZE all sub-agent findings into a SINGLE, coherent, deeply researched final report.

Requirements:
- Integrate all sub-agent findings; avoid redundancy
- Make the structure clear with headings and subheadings
- Highlight: key drivers, historical evolution, geographic/thematic patterns, open questions
- Include a final "Open Questions and Further Research" section
- Include a "Bibliography / Sources" section: merge and deduplicate sources from all sub-agents
- Do NOT expose internal mechanics (sub-agents, tool calls, etc.) to the user
- Your final answer should be a polished, professional markdown report

Return your response as structured JSON with these fields:
- executive_summary: Brief overview of all findings
- detailed_findings: Comprehensive analysis with sections (in markdown)
- key_insights: List of 5-10 main takeaways
- open_questions: List of areas needing further research
- bibliography: List of all sources used, deduplicated
"""

        try:
            structured_llm = self.llm.with_structured_output(FinalReport)
            result = await structured_llm.ainvoke(synthesis_prompt)
            final = result.model_dump()

            print(f" Final report synthesized!")
            print(f"    {len(final['key_insights'])} key insights")
            print(f"    {len(final['open_questions'])} open questions")
            print(f"  {len(final['bibliography'])} sources")

            return {"final_report": final}

        except Exception as e:
            error_msg = f"Report synthesis failed: {str(e)}"
            print(f" {error_msg}")

            return {
                "final_report": {
                    "executive_summary": "Report synthesis encountered an error. Raw sub-reports are included below.",
                    "detailed_findings": combined_reports,
                    "key_insights": ["Synthesis failed — see detailed_findings for raw sub-agent reports"],
                    "open_questions": [str(e)],
                    "bibliography": [],
                },
                "errors": [error_msg],
            }
