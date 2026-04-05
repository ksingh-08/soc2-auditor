"""
FastAPI application for the SOC 2 Evidence Auditor Environment.

Exposes the SOC2Environment over HTTP and WebSocket endpoints.

Endpoints:
    POST /reset   — Start a new audit episode (optionally pass task_id in body)
    POST /step    — Execute an INSPECT_FILE or SUBMIT_DECISION action
    GET  /state   — Get current episode state (episode_id, step_count)
    GET  /schema  — JSON Schema for SOC2Action / SOC2Observation
    WS   /ws      — WebSocket for persistent low-latency sessions
    GET  /health  — Container health check
    GET  /web     — Interactive web UI

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Via uv:
    uv run server

    # Via Docker:
    docker build -t soc2-auditor:latest .
    docker run -p 8000:8000 soc2-auditor:latest
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install with: uv sync"
    ) from e

try:
    from models import SOC2Action, SOC2Observation
    from .soc2_environment import SOC2Environment
except ModuleNotFoundError:
    from models import SOC2Action, SOC2Observation
    from server.soc2_environment import SOC2Environment


app = create_app(
    SOC2Environment,
    SOC2Action,
    SOC2Observation,
    env_name="soc2_auditor",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)  # calls main()
