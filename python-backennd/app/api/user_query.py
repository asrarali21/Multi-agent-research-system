from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.graphs.research_graph import research_workflow


class QueryRequest(BaseModel):
    query: str


router = APIRouter()


@router.post("/ask")
async def user_query(request: QueryRequest):
    """
    Run the full multi-agent research pipeline:
    1. Generate research plan
    2. Split into subtasks
    3. Fan-out: spawn sub-agents in parallel
    4. Synthesize final report
    """
    try:
        print(f"\n{'='*60}")
        print(f"🔬 New research query: {request.query[:100]}...")
        print(f"{'='*60}")

        result = await research_workflow.ainvoke({
            "user_query": request.query,
            "research_plan": "",
            "subtasks": [],
            "sub_reports": [],
            "final_report": None,
            "errors": [],
        })

        return {
            "status": "success",
            "report": result.get("final_report"),
            "errors": result.get("errors", []),
        }

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))