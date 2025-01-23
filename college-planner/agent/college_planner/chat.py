import json
from college_planner.state import AgentState
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from college_planner.suql_tool import search_for_colleges
#from college_planner.search import search_for_places
from college_planner.trips import add_trips, update_trips, delete_trips
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage
from typing import cast
from langchain_core.tools import tool

@tool
def select_trip(trip_id: str):
    """Select a trip"""
    return f"Selected trip {trip_id}"

llm = ChatOpenAI(model="gpt-4o")
tools = [search_for_colleges]#, select_trip]

async def chat_node(state: AgentState, config: RunnableConfig):
    """Handle chat operations"""
    llm_with_tools = llm.bind_tools(
        [
            *tools,
            add_trips,
            update_trips,
            delete_trips,
            select_trip,
        ],
        parallel_tool_calls=False,
    )

    system_message = f"""
    You are an agent that helps the user with planning for college.
    
    If the user asks you to search for colleges, call the search_for_colleges tool which will return a list of colleges.
    
    Unless the users prompt specifies otherwise, only use the first 10 results from the search_for_colleges tool to start planning for college.

    If an operation is cancelled by the user, DO NOT try to perform the operation again. Just ask what the user would like to do now
    instead.

    If the user asks you to search for colleges, call the search_for_colleges tool which will return a list of colleges.

    Current colleges: {json.dumps(state.get('colleges', []))}
    """

    # calling ainvoke instead of invoke is essential to get streaming to work properly on tool calls.
    response = await llm_with_tools.ainvoke(
        [
            SystemMessage(content=system_message),
            *state["messages"]
        ],
        config=config,
    )

    ai_message = cast(AIMessage, response)

    if ai_message.tool_calls:
        if ai_message.tool_calls[0]["name"] == "select_trip":
            return {
                "selected_trip_id": ai_message.tool_calls[0]["args"].get("trip_id", ""),
                "messages": [ai_message, ToolMessage(
                    tool_call_id=ai_message.tool_calls[0]["id"],
                    content="Trip selected."
                )]
            }

    return {
        "messages": [response],
        "selected_trip_id": state.get("selected_trip_id", None),
        "trips": state.get("trips", [])
    }
