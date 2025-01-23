"""Server"""

import os
from dotenv import load_dotenv
load_dotenv() # pylint: disable=wrong-import-position

from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from college_planner.agent import graph


app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="college_planner",
            description="Manages a user's college planning.",
            agent=graph,
        )
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "college_planner.demo:app",
        host="localhost",
        port=port,
        reload=True,
        reload_dirs=(
            ["."] +
            (["../../../sdk-python/copilotkit"]
             if os.path.exists("../../../sdk-python/copilotkit")
             else []
             )
        )
    )
