"""
Data models for the SOC 2 Evidence Auditor Environment.

The agent acts as a SOC 2 auditor: it inspects evidence files and makes
a deterministic APPROVE/REJECT decision against a stated security control.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SOC2Action(Action):
    """
    Action for the SOC 2 Evidence Auditor environment.

    Three action types:

    INSPECT_FILE:
        Reads the content of a named evidence file.
        Requires: file_name (str)
        Reward: +0.1 for first inspection of relevant file (capped at 0.3 total),
                -0.05 for distractor file, 0.0 for re-inspection, -0.1 for invalid file.
        NOTE: Large log files (e.g. *_full_log.json) cannot be directly inspected
              — use SEARCH_LOGS instead.

    SEARCH_LOGS:
        Queries a large log file by filtering on a specific field value.
        Requires: file_name (str), query_field (str), query_value (str)
        Reward: +0.1 if query returns results and file is relevant, -0.05 otherwise.
        Use this for log files that are too large to inspect in full.

    SUBMIT_DECISION:
        Renders the final audit verdict. Ends the episode (done=True).
        Requires: decision ("APPROVE" | "REJECT"), reason (reason code)

    Example actions:
        {"type": "INSPECT_FILE", "file_name": "pull_request_log.json"}
        {"type": "SEARCH_LOGS", "file_name": "aws_cloudtrail_full_log.json", "query_field": "username", "query_value": "alice_dev"}
        {"type": "SUBMIT_DECISION", "decision": "REJECT", "reason": "MISSING_APPROVAL"}
        {"type": "SUBMIT_DECISION", "decision": "APPROVE", "reason": "NONE"}
    """

    type: Literal["INSPECT_FILE", "SUBMIT_DECISION", "SEARCH_LOGS"] = Field(
        ...,
        description=(
            "Action type: INSPECT_FILE to read evidence, "
            "SEARCH_LOGS to query a large log file by field value, "
            "SUBMIT_DECISION to render final audit verdict."
        ),
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Evidence file to inspect or search. Required for INSPECT_FILE and SEARCH_LOGS.",
    )
    decision: Optional[Literal["APPROVE", "REJECT"]] = Field(
        default=None,
        description="Audit verdict. Required when type='SUBMIT_DECISION'.",
    )
    reason: Optional[Literal["MISSING_TIMESTAMP", "SLA_VIOLATION", "MISSING_APPROVAL", "INCOMPLETE_REVOCATION", "POLICY_VIOLATION", "NONE"]] = Field(
        default=None,
        description=(
            "Reason code for the decision. Required when type='SUBMIT_DECISION'. "
            "MISSING_APPROVAL: required sign-off, approval, or authorization is absent. "
            "SLA_VIOLATION: a time-based deadline or requirement was exceeded. "
            "MISSING_TIMESTAMP: a system-generated timestamp is required but missing. "
            "INCOMPLETE_REVOCATION: access not fully revoked across all required systems or credentials. "
            "POLICY_VIOLATION: a mandatory security control or policy setting was not enforced. "
            "NONE: use only when approving (no violation found)."
        ),
    )
    query_field: Optional[str] = Field(
        default=None,
        description="JSON field name to filter on. Required when type='SEARCH_LOGS'.",
    )
    query_value: Optional[str] = Field(
        default=None,
        description="Value to match against query_field. Required when type='SEARCH_LOGS'.",
    )


class SOC2Observation(Observation):
    """
    Observation from the SOC 2 Evidence Auditor environment.

    After reset():
        - task_id, control_requirement, available_files are populated.
        - inspected_files is empty — the agent must use INSPECT_FILE or SEARCH_LOGS actions.

    After INSPECT_FILE:
        - inspected_files grows with the file content (keyed by file_name).
        - step_reward is +0.1 (relevant, first time) or -0.05 (distractor) or 0.0 (re-inspect).

    After SEARCH_LOGS:
        - inspected_files grows with search results (keyed by "file?field=value").
        - step_reward is +0.1 (productive search of relevant file) or -0.05 (empty/distractor).

    After SUBMIT_DECISION:
        - audit_status is set to "APPROVED" or "REJECTED".
        - done=True — episode ends.
    """

    task_id: str = Field(
        default="",
        description="Unique identifier for the current audit task.",
    )
    control_requirement: str = Field(
        default="",
        description="The security control rule the agent must enforce.",
    )
    available_files: List[str] = Field(
        default_factory=list,
        description="Names of evidence files available for inspection.",
    )
    inspected_files: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Contents of all files inspected so far this episode. "
            "Keys are file_name for INSPECT_FILE results, or 'file?field=value' for SEARCH_LOGS results."
        ),
    )
    audit_status: str = Field(
        default="IN_PROGRESS",
        description="Current audit status: IN_PROGRESS | APPROVED | REJECTED.",
    )
    step_reward: float = Field(
        default=0.0,
        description="Reward earned in the most recent step.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated this episode.",
    )
    message: str = Field(
        default="",
        description="Human-readable feedback about the outcome of the last action.",
    )
