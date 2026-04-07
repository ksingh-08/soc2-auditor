---
title: SOC 2 Evidence Auditor
emoji: 🔐
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# SOC 2 Evidence Auditor

An OpenEnv reinforcement learning environment where an LLM agent acts as a **SOC 2 security auditor**.

The agent inspects mock evidence files (JSON logs from AWS, GitHub, HR systems) and makes deterministic **APPROVE** or **REJECT** decisions against stated security control requirements — exactly as a real compliance auditor would.

## Why This Environment Matters

SOC 2 compliance is mandatory for any SaaS company handling customer data. Auditors review hundreds of evidence items per audit cycle, each requiring careful cross-referencing of logs, timestamps, and policy documents. This is tedious, error-prone work that AI agents could meaningfully assist with — but requires precise, rule-following behavior with zero hallucination tolerance.

## Quick Start

```python
import asyncio
from soc2 import SOC2Env, SOC2Action

async def main():
    env = await SOC2Env.from_docker_image("soc2-auditor:latest")
    try:
        # Reset to a specific task
        result = await env.reset(task_id="pr_approval_check")
        obs = result.observation
        print(f"Task: {obs.task_id}")
        print(f"Control: {obs.control_requirement}")
        print(f"Files: {obs.available_files}")

        # Inspect evidence
        result = await env.step(SOC2Action(
            type="INSPECT_FILE",
            file_name="pull_request_log.json"
        ))
        print(f"Reward: {result.reward}")  # +0.1

        # Submit decision
        result = await env.step(SOC2Action(
            type="SUBMIT_DECISION",
            decision="REJECT",
            reason="MISSING_APPROVAL"
        ))
        print(f"Final reward: {result.reward}")  # +0.9
        print(f"Done: {result.done}")            # True
    finally:
        await env.close()

asyncio.run(main())
```

## Action Space

The agent has exactly **3 action types**:

| Action | Fields | Description |
|--------|--------|-------------|
| `INSPECT_FILE` | `file_name: str` | Read an evidence file. Adds content to `inspected_files`. Cannot be used on large log files. |
| `SEARCH_LOGS` | `file_name: str`, `query_field: str`, `query_value: str` | Query a large log file by filtering events where `query_field == query_value`. |
| `SUBMIT_DECISION` | `decision: APPROVE\|REJECT`, `reason: str` | Render the audit verdict. Ends the episode. |

**Reason codes** for `SUBMIT_DECISION`:

| Code | When to use |
|------|------------|
| `MISSING_APPROVAL` | Required sign-off, approval, or authorization is absent |
| `SLA_VIOLATION` | A time-based deadline or requirement was exceeded |
| `MISSING_TIMESTAMP` | A system-generated timestamp is required but missing |
| `INCOMPLETE_REVOCATION` | Access not fully revoked across all required systems or credentials |
| `NONE` | Use only when approving (no violation found) |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task identifier |
| `control_requirement` | str | The security rule to enforce |
| `available_files` | List[str] | Evidence files available to inspect or search |
| `inspected_files` | Dict[str, Any] | Contents of files inspected (INSPECT_FILE) and search results (SEARCH_LOGS) |
| `audit_status` | str | `IN_PROGRESS` / `APPROVED` / `REJECTED` |
| `step_reward` | float | Reward earned in the last step (delta only) |
| `cumulative_reward` | float | Total episode reward so far |
| `message` | str | Feedback about the last action |
| `done` | bool | True after `SUBMIT_DECISION` |

## Reward Function

The reward function provides **partial progress signals** throughout the episode. Maximum per episode is **1.0**.

| Action | Condition | Reward |
|--------|-----------|--------|
| `INSPECT_FILE` | First inspection of a relevant file (max 3 = 0.3 total) | **+0.1** |
| `INSPECT_FILE` | Re-inspection of same file | 0.0 |
| `INSPECT_FILE` | Distractor file | **-0.05** |
| `INSPECT_FILE` | Large log file (use SEARCH_LOGS) | **-0.05** |
| `INSPECT_FILE` | File not in available_files | **-0.1** |
| `SEARCH_LOGS` | Productive search of relevant log file | **+0.1** (shares 0.3 cap) |
| `SEARCH_LOGS` | Empty results on relevant file | **-0.05** |
| `SEARCH_LOGS` | Distractor or non-large file | **-0.05** |
| `SEARCH_LOGS` | Repeated identical query | 0.0 |
| `SUBMIT_DECISION` | Correct decision + correct reason | **+(1.0 − inspect_reward_earned)** |
| `SUBMIT_DECISION` | Correct decision + wrong reason | **+0.2** |
| `SUBMIT_DECISION` | APPROVE on non-compliant evidence | **-0.5** |
| `SUBMIT_DECISION` | REJECT on compliant evidence | **-0.3** |

**Maximum per episode**: 0.3 (inspect cap) + 0.7 (correct submit) = **1.0**

The `+(1.0 − inspect_reward_earned)` formula guarantees the episode total is exactly 1.0 regardless of how many relevant files were inspected before submitting.

## The 3 Graded Tasks

### Task 1: PR Approval Check (Easy)
**Control:** All code changes to production branches require peer review approval before merging.

**Evidence:** `pull_request_log.json` — a GitHub PR with `approved_by: null` and `approvals_count: 0`.

**Correct answer:** `REJECT` / `MISSING_APPROVAL`

**Why it's easy:** Single file, obvious null field, direct rule match. Agent must ignore 5 distractors.

---

### Task 2: Access Revocation SLA (Medium)
**Control:** User access must be revoked within 24 hours of employee termination.

**Evidence:**
- `hr_termination_ticket.json` — termination recorded at `2024-10-01T09:00:00Z`
- `aws_iam_audit_log.json` — access removed at `2024-10-05T11:30:00Z` (98.5 hours later)

**Correct answer:** `REJECT` / `SLA_VIOLATION`

**Why it's medium:** Requires inspecting both files and computing the time delta (98.5h > 24h SLA).

---

### Task 3: Multi-System Access Revocation (Hard)
**Control:** Upon termination, access to ALL production systems — AWS, GitHub, and Production DB — must be revoked.

**Evidence:**
- `hr_terminations.json` — alice_dev terminated `2024-10-01T17:00:00Z`
- `aws_users.json` — alice_dev removed ✓
- `github_users.json` — alice_dev removed ✓
- `prod_db_users.json` — alice_dev **still active** ✗

**Correct answer:** `REJECT` / `INCOMPLETE_REVOCATION`

**Why it's hard:** Agent must inspect all 4 system files. AWS and GitHub show proper revocation — only the prod DB reveals the violation. Rushing to REJECT after 2 files would get the reason wrong; approving after seeing 2 good files would miss the DB failure.

---

## Extended Task Pool (11 Total Controls)

The environment includes 11 audit controls total. When `reset()` is called without a `task_id`, a random task is selected:

| Task ID | Control | Difficulty |
|---------|---------|-----------|
| `pr_approval_check` | Code review approval | Easy |
| `access_revocation_sla` | 24-hour access revocation | Medium |
| `multi_system_access_revocation` | Full multi-system revocation | Hard |
| `mfa_enforcement_check` | MFA on all console logins | Easy |
| `encryption_at_rest_check` | S3 encryption required | Easy |
| `incident_response_sla` | 4-hour escalation SLA | Medium |
| `vendor_security_review` | Vendor questionnaire on file | Medium |
| `access_review_timestamp` | System timestamp required (hallucination trap) | Hard |
| `access_review_compliant` | System timestamp present — APPROVE trap | Medium |
| `change_management_approval` | CAB approval for HIGH-risk changes | Medium |
| `password_policy_compliance` | 100% password policy compliance | Medium |
| `cloudtrail_privileged_access_audit` | API key revocation via SEARCH_LOGS | Hard |

### SEARCH_LOGS Task: CloudTrail Privileged Access Audit
**Control:** All AWS credentials (API keys + tokens) must be fully revoked within 24 hours of termination.

**Evidence:**
- `hr_terminations.json` — alice_dev terminated `2024-10-01T17:00:00Z`
- `aws_cloudtrail_full_log.json` — 28 API events (too large to INSPECT_FILE directly)

**Workflow:**
1. `INSPECT_FILE` → `hr_terminations.json` → extract username `alice_dev`
2. `SEARCH_LOGS` → `aws_cloudtrail_full_log.json`, `query_field=username`, `query_value=alice_dev`
3. Results show 3 API calls (CT-011, CT-014, CT-017) made 14–41 hours **after** termination
4. `SUBMIT_DECISION` → `REJECT` / `INCOMPLETE_REVOCATION`

**Why it's hard:** Multi-hop reasoning — must extract a value from one file and use it as a query parameter for another. Tests whether agents can chain observations across steps.

## Setup & Usage

### Build Docker Image

```bash
docker build -t soc2-auditor:latest .
```

### Run Server Locally

```bash
# Via uv
uv run server

# Via Docker
docker run -p 8000:8000 soc2-auditor:latest
```

### Run Baseline Inference

```bash
# Set environment variables
export API_KEY=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export IMAGE_NAME=soc2-auditor:latest

uv run python inference.py
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | Yes | — | HuggingFace API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `IMAGE_NAME` | No | `soc2-auditor:latest` | Docker image name |

### OpenEnv Validate

```bash
openenv validate
```

## Project Structure

```
soc2_auditor/
├── inference.py                        # Baseline inference script (entry point)
├── models.py                           # SOC2Action + SOC2Observation Pydantic models
├── client.py                           # SOC2Env client (WebSocket + Docker)
├── __init__.py                         # Package exports
├── openenv.yaml                        # OpenEnv spec manifest
├── pyproject.toml                      # Project metadata + dependencies
├── Dockerfile                          # Container definition
├── .env                                # Local credentials (not committed)
└── server/
    ├── app.py                          # FastAPI application
    ├── soc2_environment.py             # SOC2Environment core logic
    ├── tasks.py                        # Task definitions, evidence data, grader
    └── __init__.py
```

## Baseline Scores

Scores from `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API:

| Task | Difficulty | Expected Agent Score |
|------|-----------|---------------------|
| `pr_approval_check` | Easy | 0.85–1.0 |
| `access_revocation_sla` | Medium | 0.5–0.85 |
| `multi_system_access_revocation` | Hard | 0.2–0.5 |
| **Overall** | — | **~0.55** |

The hard task requires inspecting all 4 system files and choosing `INCOMPLETE_REVOCATION` over `MISSING_APPROVAL` — a distinction many models miss without careful reasoning.
