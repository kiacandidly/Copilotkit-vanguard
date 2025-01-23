from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage, ToolMessage
from langchain.tools import tool
from college_planner.suql.sql_free_text_support.execute_free_text_sql import suql_execute
from langchain_core.runnables import RunnableConfig
from typing import cast
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from college_planner.state import AgentState
import json


# System prompt with schema and guidelines
system_prompt = """You are a semantic parser that converts natural language queries into SUQL queries for a college database. 
You should generate queries that return all columns except reviews_text.

Note that SQL queries are fully compatible with the SUQL syntax, so you can use SQL queries as well.

Schema:
CREATE TABLE schools (
    school_id INT PRIMARY KEY,
    school_name TEXT,
    school_city TEXT,
    school_zip TEXT,
    school_url TEXT,
    latitude FLOAT,
    longitude FLOAT,
    school_state TEXT,
    is_public BOOLEAN,
    hbcu BOOLEAN,
    women_only BOOLEAN,
    religious_affiliation TEXT,
    enrollment_size INTEGER,
    admission_rate_pct FLOAT,
    in_state_tuition_and_fees INTEGER,
    out_of_state_tuition_and_fees INTEGER,
    net_price_0_30k INTEGER,
    net_price_30k_48k INTEGER,
    net_price_48k_75k INTEGER,
    net_price_75k_110k INTEGER,
    net_price_110k_and_up INTEGER,
    mean_earnings_6_yrs_after_entry INTEGER,
    reviews_text TEXT[]
);

Guidelines:
1. Always SELECT all columns except reviews_text
2. Use 2-letter state codes (e.g., 'CA' for California)
3. For text analysis of reviews, use the answer() function
4. Include LIMIT when a specific number of results is requested
5. For boolean fields, use TRUE/FALSE (not 't'/'f')"""

# Few-shot examples showing various query patterns
few_shots = """Examples:

Input: Show me 3 schools in Texas
Output: SELECT school_id, school_name, school_city, school_state, school_zip, school_url, latitude, longitude, is_public, hbcu, women_only, religious_affiliation, enrollment_size, admission_rate_pct, in_state_tuition_and_fees, out_of_state_tuition_and_fees, net_price_0_30k, net_price_30k_48k, net_price_48k_75k, net_price_75k_110k, net_price_110k_and_up, mean_earnings_6_yrs_after_entry FROM schools WHERE school_state = 'TX' LIMIT 3;

Input: Find schools in California with good campus food
Output: SELECT school_id, school_name, school_city, school_state, school_zip, school_url, latitude, longitude, is_public, hbcu, women_only, religious_affiliation, enrollment_size, admission_rate_pct, in_state_tuition_and_fees, out_of_state_tuition_and_fees, net_price_0_30k, net_price_30k_48k, net_price_48k_75k, net_price_75k_110k, net_price_110k_and_up, mean_earnings_6_yrs_after_entry FROM schools WHERE school_state = 'CA' AND answer(reviews_text, 'does this school have good campus food?') = 'Yes';

Input: Find public universities in Florida with strong engineering programs
Output: SELECT school_id, school_name, school_city, school_state, school_zip, school_url, latitude, longitude, is_public, hbcu, women_only, religious_affiliation, enrollment_size, admission_rate_pct, in_state_tuition_and_fees, out_of_state_tuition_and_fees, net_price_0_30k, net_price_30k_48k, net_price_48k_75k, net_price_75k_110k, net_price_110k_and_up, mean_earnings_6_yrs_after_entry FROM schools WHERE school_state = 'FL' AND is_public = TRUE AND answer(reviews_text, 'does this school have a strong engineering program?') = 'Yes';

Input: Show me women's colleges with tuition under 30000
Output: SELECT school_id, school_name, school_city, school_state, school_zip, school_url, latitude, longitude, is_public, hbcu, women_only, religious_affiliation, enrollment_size, admission_rate_pct, in_state_tuition_and_fees, out_of_state_tuition_and_fees, net_price_0_30k, net_price_30k_48k, net_price_48k_75k, net_price_75k_110k, net_price_110k_and_up, mean_earnings_6_yrs_after_entry FROM schools WHERE women_only = TRUE AND in_state_tuition_and_fees < 30000;

Input: Find HBCUs with good sports programs
Output: SELECT school_id, school_name, school_city, school_state, school_zip, school_url, latitude, longitude, is_public, hbcu, women_only, religious_affiliation, enrollment_size, admission_rate_pct, in_state_tuition_and_fees, out_of_state_tuition_and_fees, net_price_0_30k, net_price_30k_48k, net_price_48k_75k, net_price_75k_110k, net_price_110k_and_up, mean_earnings_6_yrs_after_entry FROM schools WHERE hbcu = TRUE AND answer(reviews_text, 'does this school have good sports programs?') = 'Yes';

Input: Find 5 schools with the best party scene in New York
Output: SELECT school_id, school_name, school_city, school_state, school_zip, school_url, latitude, longitude, is_public, hbcu, women_only, religious_affiliation, enrollment_size, admission_rate_pct, in_state_tuition_and_fees, out_of_state_tuition_and_fees, net_price_0_30k, net_price_30k_48k, net_price_48k_75k, net_price_75k_110k, net_price_110k_and_up, mean_earnings_6_yrs_after_entry FROM schools WHERE school_state = 'NY' AND answer(reviews_text, 'does this school have a good party scene?') = 'Yes' LIMIT 5;

Input: Find all the private schools in Los Angeles
Output: SELECT school_id, school_name, school_city, school_state, school_zip, school_url, latitude, longitude, is_public, hbcu, women_only, religious_affiliation, enrollment_size, admission_rate_pct, in_state_tuition_and_fees, out_of_state_tuition_and_fees, net_price_0_30k, net_price_30k_48k, net_price_48k_75k, net_price_75k_110k, net_price_110k_and_up, mean_earnings_6_yrs_after_entry FROM schools WHERE school_city = 'Los Angeles' AND is_public = FALSE;
"""

# If chain.invoke is synchronous, keep this as a normal function. 
# If chain.invoke is truly async, mark it "async def" and do: await chain.invoke(...)
def get_suql_query(input_text: str) -> str:
    response = chain.invoke({"query": input_text})  # synchronous call
    return response.content

# Same pattern: if suql_execute is synchronous, keep it sync. If it's async, 
# mark this "async def" and do: `results, column_names, _ = await suql_execute(...)`
def search_schools(suql_query: str):
    results, column_names, _ = suql_execute(
        suql_query,
        {
            "schools": "school_id",
        },
        "college_planning",
        embedding_server_address="http://127.0.0.1:8501",
        llm_model_name="gpt-4o"
    )
    return column_names, results

# ---------------------------------------------------------------------
# 3. The tool definition
# ---------------------------------------------------------------------
@tool
def search_for_colleges(question: str) -> list[dict]:
    """
    Search for colleges based on natural language queries.
    Returns colleges including name, address, coordinates, and ratings.
    """
    # This is just the docstring for your tool. 
    # Typically the actual logic is handled by the node below.
    return []

# ---------------------------------------------------------------------
# 4. The "college_search_node" function
# ---------------------------------------------------------------------
async def college_search_node(state: AgentState, config: RunnableConfig):
    """
    The college search node is responsible for searching for colleges.
    Adjust 'async' vs. 'await' usage depending on whether your subcalls are actually async.
    """

    # Possibly customize the config so it doesn't emit intermediate steps
    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=False,
        emit_messages=False
    )

    print(f"[DEBUG] College search node called with state: {state}")

    # 1. Extract the tool call arguments from the last AIMessage
    ai_message = state["messages"][-1]  # type: AIMessage
    tool_call_data = ai_message.tool_calls[0]  # e.g. { "id": "...", "name": "search_for_colleges", "args": {...} }
    query = tool_call_data["args"]["question"]
    print(f"[DEBUG] Extracted query: {query}. State: {state}")

    # 2. Get SUQL query (sync or async)
    suql_query = await get_suql_query(query)  # If you want this async, do: suql_query = await get_suql_query(query)

    # 3. Execute the SUQL query (sync or async)
    column_names, results = await search_schools(suql_query)  # If needed, await search_schools(suql_query)
    print(f"[DEBUG] SUQL results: {results}. State: {state}")

    # 4. Convert DB rows => list of dictionaries
    colleges = []
    for row in results:
        college_dict = {}
        for col_name, value in zip(column_names, row):
            college_dict[col_name] = value
        colleges.append(college_dict)

    print(f"[DEBUG] Final list of college dicts: {colleges}. State: {state}")

    # 5. Build a ToolMessage with those results
    tool_message = ToolMessage(
        tool_call_id=tool_call_data["id"],
        content=f"Added the following search results: {json.dumps(colleges)}"
    )

    # 6. Append that ToolMessage to the conversation state (so the chain can see it),
    #    or in some designs, you might just return it.
    if "messages" not in state: 
        state["messages"] = []
    state["messages"].append(tool_message)

    print(f"[DEBUG] Appended ToolMessage to state: {tool_message}")

    # 7. Return the updated state
    return { "messages": state["messages"]+[tool_message] }

# Create the chat model
chat = ChatOpenAI(temperature=0)

# Create the prompt template    
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", few_shots + "\n\nInput: {query}\nOutput:"),
])

# Create the chain
chain = prompt_template | chat

def get_suql_query(input_text):
    response = chain.invoke({"query": input_text})
    return response.content

def search_schools(suql_query):
    results, column_names, _ = suql_execute(
        suql_query,
        {
            "schools": "school_id",
        },
        "college_planning",
        embedding_server_address="http://127.0.0.1:8501",
        llm_model_name="gpt-4o"
    )
    return column_names, results

@tool
def search_for_colleges(question: str) -> list[dict]:
    """Search for coslleges based on a natural language queries, returns colleges including their name, address, and coordinates, and ratings with type signature of College class"""

async def college_search_node(state: AgentState, config: RunnableConfig):
    """
    The college search node is responsible for searching the for colleges.
    """

    config = copilotkit_customize_config(config, emit_tool_calls=False, emit_intermediate_state=False, emit_messages=False)
    print(f"[DEBUG] College search node called with state: {state}")
    ai_message = cast(AIMessage, state["messages"][-1])
    query = ai_message.tool_calls[0]["args"]["question"]
    print(f"[DEBUG] College search node called extracted query: {query}. State: {state}")
    # Get colleges from database
    suql_query = get_suql_query(query)
    column_names, results = search_schools(suql_query)
    print(f"[DEBUG] College search node called extracted results: {results}. State: {state}")
    # Convert results to list of college dictionaries
    colleges = []
    for row in results:
        college_dict = {}
        for col_name, value in zip(column_names, row):
            college_dict[col_name] = value
        colleges.append(college_dict)

    print(f"[DEBUG] College search node called extracted results: {colleges}. State: {state}")
    
    ToolMessage(
        tool_call_id=ai_message.tool_calls[0]["id"],
        content=f"Added the following search results: {json.dumps(colleges)}")

    print(f"[DEBUG] College search node called extracted results: {colleges}. added ToolMessage to state messg history: {state}")

    return state

# Keep the main function for direct testing
def main(query: str):
    suql = get_suql_query(query)
    res = search_schools(suql)
    
    print(res[0])
if __name__ == "__main__":
    print(main("I'm interested in schools in california"))
