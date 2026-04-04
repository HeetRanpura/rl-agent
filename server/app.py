import uvicorn
from openenv.core.env_server import create_app
from models import Action, Observation
from server.scheme_env_environment import SchemeEnvEnvironment

app = create_app(
    SchemeEnvEnvironment,
    Action,
    Observation,
    env_name="scheme_env",
)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()