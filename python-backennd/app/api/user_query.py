from fastapi import APIRouter , HTTPException
from pydantic import BaseModel
from app.agents.coordinator_agent import CoordinatorAgent
from dotenv import load_dotenv 

from app.agents.research_plan_agent import ResearchPlan


load_dotenv()

class QueryRequest(BaseModel):
    query : str



router = APIRouter()

  
              
@router.post("/ask")
async def user_query(request : QueryRequest):


    coordinator = CoordinatorAgent()
    researcher=ResearchPlan()

    research_plan = await researcher.research(request.query)

    return {
        "status" : "success",
        "research_plan":research_plan
    }