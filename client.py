"""SOC 2 Evidence Auditor — Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SOC2Action, SOC2Observation


class SOC2Env(EnvClient[SOC2Action, SOC2Observation, State]):
    """
    Client for the SOC 2 Evidence Auditor environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step audit interactions.

    Example (Docker):
        >>> env = await SOC2Env.from_docker_image("soc2-auditor:latest")
        >>> result = await env.reset(task_id="pr_approval_check")
        >>> result = await env.step(SOC2Action(
        ...     type="INSPECT_FILE", file_name="pull_request_log.json"
        ... ))
        >>> result = await env.step(SOC2Action(
        ...     type="SUBMIT_DECISION", decision="REJECT", reason="MISSING_APPROVAL"
        ... ))
        >>> await env.close()

    Example (SEARCH_LOGS for large log files):
        >>> env = await SOC2Env.from_docker_image("soc2-auditor:latest")
        >>> result = await env.reset(task_id="cloudtrail_privileged_access_audit")
        >>> result = await env.step(SOC2Action(
        ...     type="INSPECT_FILE", file_name="hr_terminations.json"
        ... ))
        >>> result = await env.step(SOC2Action(
        ...     type="SEARCH_LOGS",
        ...     file_name="aws_cloudtrail_full_log.json",
        ...     query_field="username",
        ...     query_value="alice_dev",
        ... ))
        >>> result = await env.step(SOC2Action(
        ...     type="SUBMIT_DECISION", decision="REJECT", reason="INCOMPLETE_REVOCATION"
        ... ))
        >>> await env.close()

    Example (running server):
        >>> async with SOC2Env(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_id="access_revocation_sla")
        ...     obs = result.observation
        ...     print(obs.control_requirement)
        ...     print(obs.available_files)
    """

    def _step_payload(self, action: SOC2Action) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": action.type}
        if action.file_name is not None:
            payload["file_name"] = action.file_name
        if action.decision is not None:
            payload["decision"] = action.decision
        if action.reason is not None:
            payload["reason"] = action.reason
        if action.query_field is not None:
            payload["query_field"] = action.query_field
        if action.query_value is not None:
            payload["query_value"] = action.query_value
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SOC2Observation]:
        obs_data = payload.get("observation", {})
        observation = SOC2Observation(
            task_id=obs_data.get("task_id", ""),
            control_requirement=obs_data.get("control_requirement", ""),
            available_files=obs_data.get("available_files", []),
            inspected_files=obs_data.get("inspected_files", {}),
            audit_status=obs_data.get("audit_status", "IN_PROGRESS"),
            step_reward=obs_data.get("step_reward", 0.0),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            message=obs_data.get("message", ""),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
