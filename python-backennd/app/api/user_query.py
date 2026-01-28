from fastapi import APIRouter , HTTPException
from pydantic import BaseModel
from app.agents.coordinator_agent import CoordinatorAgent
from dotenv import load_dotenv 

from app.agents.research_plan_agent import ResearchPlan
from app.agents.sub_task_agent import SubTaskAgent


load_dotenv()

class QueryRequest(BaseModel):
    query : str



router = APIRouter()

  
              
@router.post("/ask")
async def user_query(request : QueryRequest):


    coordinator = CoordinatorAgent()
    researcher=ResearchPlan()
    sub_task = SubTaskAgent()


    research_plan = await researcher.research(request.query)
    
    sub_tasks_list =await sub_task.sub_task(research_plan=research_plan)

    return {
        "status" : "success",
        "research_plan":sub_tasks_list
    }