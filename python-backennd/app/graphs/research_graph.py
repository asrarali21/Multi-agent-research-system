"""
LangGraph workflow for the multi-agent research system.

Pipeline:
  User Query → generate_plan → split_subtasks → fan_out (Send) → run_sub_agent (x N) → synthesize_report

The fan-out uses LangGraph's Send() API to dynamically spawn one run_sub_agent
node per subtask, running them in parallel. Results accumulate via the
Annotated[List[dict], operator.add] reducer in ResearchState, and once all
sub-agents finish, the synthesize_report node fires.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from app.agents.coordinator_agent import (
    CoordinatorAgent,
    ResearchState,
)


def build_research_graph() -> StateGraph:
    """Build and compile the research workflow graph."""

    coordinator = CoordinatorAgent()

    # Fan-out function: creates one Send() per subtask

    def fan_out_subtasks(state: ResearchState) -> list[Send]:
        """
        Conditional edge that uses Send() to spawn one run_sub_agent
        node per subtask. Each Send() gets its own copy of the input.
        """
        if not state.get("subtasks"):
            return [Send("synthesize_report", state)]

        sends = []
        for idx, subtask in enumerate(state["subtasks"]):
            sends.append(
                Send("run_sub_agent", {
                    "user_query": state["user_query"],
                    "research_plan": state["research_plan"],
                    "subtask": subtask,
                    "stagger_index": idx,
                    "sub_reports": [],
                    "errors": [],
                })
            )
        return sends

    # Build the graph 

    graph = StateGraph(ResearchState)

    graph.add_node("generate_plan", coordinator.generate_plan)
    graph.add_node("split_subtasks", coordinator.split_subtasks)
    graph.add_node("run_sub_agent", coordinator.run_sub_agent)
    graph.add_node("synthesize_report", coordinator.synthesize_report)


    graph.add_edge(START, "generate_plan")
    graph.add_edge("generate_plan", "split_subtasks")


    graph.add_conditional_edges("split_subtasks", fan_out_subtasks, ["run_sub_agent", "synthesize_report"])

    # Fan-in: all run_sub_agent results 
    graph.add_edge("run_sub_agent", "synthesize_report")


    graph.add_edge("synthesize_report", END)

    return graph.compile()



research_workflow = build_research_graph()
