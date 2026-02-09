from pydantic import BaseModel , Field
from typing import TypedDict , Dict , List , Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
import operator
import os
from deepagents import create_deep_agent
from app.tools.fire_crawl import firecrawl_scrape_tool
from app.tools.fire_crawl import firecrawl_search_tool
from app.tools.fire_crawl import firecrawl_agent_tool
from langchain.agents import create_agent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class Finalreport(BaseModel):
    executive_summary: str = Field(description="Brief overview of all findings")
    detailed_findings: str = Field(description="Comprehensive analysis with sections")
    key_insights: List[str] = Field(description="Main takeaways (5-10 points)")
    open_questions: List[str] = Field(description="Areas needing further research")
    bibliography: List[str] = Field(description="All sources used, deduplicated")
     
class ResearchState(TypedDict):
    """State that flows through the entire workflow"""
    user_query: str
    research_plan: str
    subtasks: List[dict]  # List of {id, title, description}
    sub_reports: Annotated[List[dict], operator.add]  # Accumulate reports
    final_report: dict | None
    errors: List[str]


class SubAgentReport(BaseModel):
    """Report from a single sub-agent"""
    subtask_id: str
    subtask_title: str
    report: str



class CoordinatorAgent:
    def __init__(self):
         self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.3
         )

         self.structured_llm = self.llm.with_structured_output(Finalreport)


         self.tools = [
              firecrawl_agent_tool(),
              firecrawl_scrape_tool(),
              firecrawl_search_tool()
         ]

    async def _run_single_sub_agent(self , user_query:str , research_plan:str  , subtask:Dict)->Dict:
         
         print(f"  🤖 Sub-agent [{subtask['id']}]: {subtask['title']}")

         agent = create_agent(   
            model=self.llm,
            tools=self.tools, 
        )
         
         sub_agent_prompt = f"""You are a specialized research sub-agent.

Global user query:
{user_query}

Overall research plan:
{research_plan}

Your specific subtask (ID: {subtask['id']}, Title: {subtask['title']}) is:

\"\"\"{subtask['description']}\"\"\"



Instructions:
- Focus ONLY on this subtask, but keep the global query in mind for context.
- Use the available tools to search for up-to-date, high-quality sources.
- Prioritize primary and official sources when possible.
- Be explicit about uncertainties, disagreements in the literature, and gaps.
- Return your results as a MARKDOWN report with this structure:

# [Subtask ID] [Subtask Title]

## Summary
Short overview of the main findings.

## Detailed Analysis
Well-structured explanation with subsections as needed.

## Key Points
- Bullet point
- Bullet point

## Sources
- [Title](url) - short comment on why this source is relevant

Now perform the research and return ONLY the markdown report.
"""
         agent.ainvoke({
              "messages": [{"role": "user", "content": sub_agent_prompt}]
         })

    def build_promt(self , user_query ,research_plan ,subtasks_json):
         return f"""
You are the LEAD RESEARCH COORDINATOR AGENT.

The user has asked:
\"\"\"{user_query}\"\"\"

A detailed research plan has already been created:

\"\"\"{research_plan}\"\"\"

This plan has been split into the following subtasks (JSON):

```json
{subtasks_json}
```
Each element has the shape:
{{
“id”: “timeframe_confirmation”,
“title”: “Confirm Research Scope Parameters”,
“description”: “Analyze the scope parameters…”
}}

You have access to a tool called:
• initialize_subagent(subtask_id: str, subtask_title: str, subtask_description: str)

Your job:
1. For EACH subtask in the JSON array, call initialize_subagent exactly once
with:
• subtask_id       = subtask[“id”]
• subtask_title    = subtask[“title”]
• subtask_description = subtask[“description”]
2. Wait for all sub-agent reports to come back. Each tool call returns a
markdown report for that subtask.
3. After you have results for ALL subtasks, synthesize them into a SINGLE,
coherent, deeply researched report addressing the original user query
("{user_query}").

Final report requirements:
• Integrate all sub-agent findings; avoid redundancy.
• Make the structure clear with headings and subheadings.
• Highlight:
• key drivers and mechanisms of insecurity,
• historical and temporal evolution,
• geographic and thematic patterns,
• state capacity, public perception, and socioeconomic correlates,
• open questions and uncertainties.
• Include final sections:
• Open Questions and Further Research
• Bibliography / Sources: merge and deduplicate the key sources from all sub-agents.

Important:
• DO NOT expose internal tool-call mechanics to the user.
• Your final answer to the user should be a polished markdown report.

"""
    
    def coordinate(self ,user_query ,subtasks_json ,research_plan):
         self.structured_llm.ainvoke(self.build_promt(user_query ,subtasks_json , research_plan ))
         
