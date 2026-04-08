import uvicorn
from openenv.core.env_server import create_app
from .models import Action, Observation
from server.scheme_env_environment import SchemeEnvEnvironment

# create_app wires up the /reset and /step HTTP endpoints using the openenv-core
# framework, binding SchemeEnvEnvironment as the active environment instance and
# Action/Observation as the request/response Pydantic models.
app = create_app(
    SchemeEnvEnvironment,
    Action,
    Observation,
    env_name="scheme_env",
)

@app.get("/health")
async def health():
    """Liveness probe used by Docker and load balancers to confirm the server is up."""
    return {"status": "ok"}

def main():
    # reload=False is intentional for production — hot-reload would reset shared
    # episode state between file-change events, corrupting in-flight episodes.
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()