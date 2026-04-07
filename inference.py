"""
SOC 2 Evidence Auditor — Baseline Inference Script
====================================================

Runs an LLM agent against all 3 graded audit tasks sequentially.
The agent must inspect evidence files and make APPROVE/REJECT decisions.

STDOUT FORMAT (strictly enforced):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Environment variables:
    API_BASE_URL      LLM proxy endpoint (injected by validator)
    API_KEY           API key for the LLM proxy (injected by validator)
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    LOCAL_IMAGE_NAME  Docker image name for the environment (default: soc2-auditor:latest)
"""

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from soc2.client import SOC2Env
from soc2.models import SOC2Action, SOC2Observation

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.environ["API_KEY"]

BENCHMARK = "soc2_auditor"
TASK_NAME = "soc2_3tasks"

MAX_STEPS_PER_TASK = 12      # enough for inspect + search + submit with distractor attempts
TEMPERATURE = 0.1            # low temp for deterministic auditing
MAX_TOKENS = 256

# 3 tasks × max 1.0 reward each = 3.0
# Per task: max 0.3 (inspect) + 0.7 (correct submit) = 1.0
MAX_TOTAL_REWARD = 3.0
SUCCESS_SCORE_THRESHOLD = 0.6  # 60% = 2/3 tasks correct

GRADED_TASK_IDS = [
    "pr_approval_check",              # easy   — 1 relevant file, 5 distractors
    "access_revocation_sla",          # medium — 2 relevant files, 5 distractors
    "multi_system_access_revocation", # hard   — 4 relevant files, 4 distractors
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a strict SOC 2 security auditor. You inspect evidence files and make
    deterministic APPROVE or REJECT decisions against stated security control requirements.

    YOUR JOB:
    1. Read the control_requirement carefully.
    2. From available_files, identify which files are RELEVANT to the control (by file name).
       WARNING: The file list contains DISTRACTOR files (e.g. lunch_menu.json, slack_channel.json,
       sprint_planning.json) that are irrelevant. Inspecting distractors costs -0.05 each.
    3. Inspect ONLY the relevant files. Use file names to decide relevance before inspecting.
       NOTE: Files ending in *_full_log.json are large — use SEARCH_LOGS (not INSPECT_FILE).
    4. Apply the control requirement LITERALLY — do not assume compliance if a field is missing.
    5. Submit SUBMIT_DECISION only after inspecting all relevant files.

    ACTION SPACE — respond with ONLY a valid JSON object, nothing else:

    Inspect a file:
      {"type": "INSPECT_FILE", "file_name": "<exact_filename>"}

    Search a large log file by filtering on a field value:
      {"type": "SEARCH_LOGS", "file_name": "<log_filename>", "query_field": "<field>", "query_value": "<value>"}
      Use this for *_full_log.json files. First inspect the HR/termination file to find the
      username or identifier, then search the log file for that value.

    Submit verdict:
      {"type": "SUBMIT_DECISION", "decision": "APPROVE", "reason": "NONE"}
      {"type": "SUBMIT_DECISION", "decision": "REJECT", "reason": "MISSING_APPROVAL"}
      {"type": "SUBMIT_DECISION", "decision": "REJECT", "reason": "SLA_VIOLATION"}
      {"type": "SUBMIT_DECISION", "decision": "REJECT", "reason": "MISSING_TIMESTAMP"}
      {"type": "SUBMIT_DECISION", "decision": "REJECT", "reason": "INCOMPLETE_REVOCATION"}
      {"type": "SUBMIT_DECISION", "decision": "REJECT", "reason": "POLICY_VIOLATION"}

    REASON CODES:
      MISSING_APPROVAL      — required sign-off, approval, or authorization is absent
      SLA_VIOLATION         — a time-based deadline or requirement was exceeded
      MISSING_TIMESTAMP     — a system-generated timestamp is required but missing
      INCOMPLETE_REVOCATION — access not fully revoked across all required systems or credentials
      POLICY_VIOLATION      — a mandatory security control or configuration policy was not enforced (e.g. MFA disabled, encryption missing, compliance threshold not met)
      NONE                  — use only when approving (no violation found)

    AUDITING RULES:
    - REJECT if ANY required field is null, missing, or does not meet the stated threshold.
    - Do NOT infer or assume compliance — the evidence must explicitly prove it.
    - A manually written date string is NOT the same as a system_timestamp.
    - Partial access revocation (one system still active) = INCOMPLETE_REVOCATION, not MISSING_APPROVAL.
    - Post-termination API calls in CloudTrail = INCOMPLETE_REVOCATION (credentials not fully revoked).
    - Missing MFA, missing encryption, or below-threshold compliance = POLICY_VIOLATION.
    - If evidence is fully compliant, APPROVE with reason NONE — do not reflexively REJECT.
    - Always inspect ALL relevant files before submitting a decision.
    - Output exactly ONE JSON object per turn. No explanation, no markdown, no extra text.
""").strip()


def build_user_prompt(obs: SOC2Observation, step: int) -> str:
    inspected_str = (
        json.dumps(obs.inspected_files, indent=2)
        if obs.inspected_files
        else "None yet — use INSPECT_FILE or SEARCH_LOGS."
    )
    return textwrap.dedent(f"""
        STEP {step}
        ══════════════════════════════════════════
        TASK: {obs.task_id}
        CONTROL REQUIREMENT:
          {obs.control_requirement}

        AVAILABLE FILES: {obs.available_files}
        (Some files are distractors — use file names to judge relevance before inspecting.
         Files named *_full_log.json are large — use SEARCH_LOGS, not INSPECT_FILE.)

        FILES INSPECTED / SEARCHES PERFORMED SO FAR:
        {inspected_str}

        STATUS: {obs.audit_status}
        LAST STEP REWARD: {obs.step_reward:+.2f}
        CUMULATIVE REWARD: {obs.cumulative_reward:+.2f}
        FEEDBACK: {obs.message}
        ══════════════════════════════════════════

        Respond with ONE JSON action object.
    """).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


VALID_REASONS = {"MISSING_TIMESTAMP", "SLA_VIOLATION", "MISSING_APPROVAL", "INCOMPLETE_REVOCATION", "POLICY_VIOLATION", "NONE"}
VALID_DECISIONS = {"APPROVE", "REJECT"}


def _normalize_reason(reason: str) -> str:
    """Map any LLM-invented reason string to the closest valid reason code."""
    r = reason.upper()
    if r in VALID_REASONS:
        return r
    # Fuzzy mapping — check more specific patterns first
    if any(k in r for k in ("TIMESTAMP", "SYSTEM_DATE", "GENERAT")):
        return "MISSING_TIMESTAMP"
    # INCOMPLETE_REVOCATION before MISSING_APPROVAL (more specific revocation patterns)
    if "REVOC" in r and any(k in r for k in ("INCOMPLET", "PARTIAL")):
        return "INCOMPLETE_REVOCATION"
    if any(k in r for k in ("SLA", "LATE", "DELAY", "HOUR", "EXCEED", "BREACH")):
        return "SLA_VIOLATION"
    if "REVOC" in r:
        return "INCOMPLETE_REVOCATION"
    # POLICY_VIOLATION for control/enforcement failures
    if any(k in r for k in ("POLICY", "CONTROL", "ENFORCE", "MFA", "ENCRYPT", "THRESHOLD", "COMPLIAN")):
        return "POLICY_VIOLATION"
    if any(k in r for k in ("INCOMPLET", "PARTIAL", "MISS", "APPROV")):
        return "MISSING_APPROVAL"
    return "NONE"


def _normalize_decision(decision: str) -> str:
    d = decision.upper()
    return d if d in VALID_DECISIONS else "REJECT"


def parse_action(text: str) -> SOC2Action:
    """
    Parse LLM response text into a SOC2Action.

    Strategy:
    1. Strip markdown code fences.
    2. Try JSON parse — normalize any invalid field values before constructing.
    3. Regex fallback for SUBMIT_DECISION.
    4. Regex fallback for SEARCH_LOGS.
    5. Regex fallback for INSPECT_FILE.
    6. Safe default: SUBMIT_DECISION REJECT NONE.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*\n?(.*?)```", r"\1", text, flags=re.DOTALL).strip()
    # Extract first {...} block
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
        if data.get("type") == "SUBMIT_DECISION":
            data["decision"] = _normalize_decision(data.get("decision", "REJECT"))
            data["reason"] = _normalize_reason(data.get("reason", "NONE"))
        return SOC2Action(**data)
    except Exception:
        pass

    # Regex: SUBMIT_DECISION
    m = re.search(
        r'"type"\s*:\s*"SUBMIT_DECISION".*?"decision"\s*:\s*"(\w+)".*?"reason"\s*:\s*"(\w+)"',
        text,
        re.DOTALL,
    )
    if m:
        return SOC2Action(
            type="SUBMIT_DECISION",
            decision=_normalize_decision(m.group(1)),
            reason=_normalize_reason(m.group(2)),
        )

    # Regex: SEARCH_LOGS
    m = re.search(
        r'"type"\s*:\s*"SEARCH_LOGS".*?"file_name"\s*:\s*"([^"]+)".*?"query_field"\s*:\s*"([^"]+)".*?"query_value"\s*:\s*"([^"]+)"',
        text,
        re.DOTALL,
    )
    if m:
        return SOC2Action(
            type="SEARCH_LOGS",
            file_name=m.group(1),
            query_field=m.group(2),
            query_value=m.group(3),
        )

    # Regex: INSPECT_FILE
    m = re.search(
        r'"type"\s*:\s*"INSPECT_FILE".*?"file_name"\s*:\s*"([^"]+)"',
        text,
        re.DOTALL,
    )
    if m:
        return SOC2Action(type="INSPECT_FILE", file_name=m.group(1))

    # Safe default
    print(f"[DEBUG] Could not parse action from: {text!r}. Using safe default.", flush=True)
    return SOC2Action(type="SUBMIT_DECISION", decision="REJECT", reason="NONE")


def action_to_str(action: SOC2Action) -> str:
    if action.type == "INSPECT_FILE":
        return f"INSPECT_FILE({action.file_name!r})"
    if action.type == "SEARCH_LOGS":
        return f"SEARCH_LOGS({action.file_name!r}, {action.query_field!r}={action.query_value!r})"
    return f"SUBMIT_DECISION({action.decision!r}, {action.reason!r})"


def get_model_action(
    client: OpenAI,
    conversation: List[dict],
    user_content: str,
) -> Tuple[SOC2Action, str]:
    """
    Call the LLM with the current conversation + new user message.
    Returns a parsed SOC2Action and the raw response string.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation)
    messages.append({"role": "user", "content": user_content})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        raw = ""

    return parse_action(raw), raw


async def run_task(
    env: SOC2Env,
    client: OpenAI,
    task_id: str,
    global_step: int,
) -> Tuple[List[float], int]:
    """
    Run one full audit episode for a given task_id.

    Returns:
        (step_rewards, updated_global_step)
    """
    result = await env.reset(task_id=task_id)
    obs = result.observation
    task_rewards: List[float] = []
    conversation: List[dict] = []

    for local_step in range(1, MAX_STEPS_PER_TASK + 1):
        if result.done:
            break

        user_content = build_user_prompt(obs, local_step)
        action, raw_response = get_model_action(client, conversation, user_content)

        result = await env.step(action)
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done

        task_rewards.append(reward)
        global_step += 1

        log_step(
            step=global_step,
            action=action_to_str(action),
            reward=reward,
            done=done,
            error=None,
        )

        # Update conversation history for multi-turn context
        conversation.append({"role": "user", "content": user_content})
        conversation.append({"role": "assistant", "content": raw_response or action_to_str(action)})

        if done:
            break

    # Timeout penalty — episode exhausted without SUBMIT_DECISION
    if not result.done:
        timeout_reward = -0.1
        task_rewards.append(timeout_reward)
        global_step += 1
        log_step(
            step=global_step,
            action="TIMEOUT(max_steps_exceeded)",
            reward=timeout_reward,
            done=True,
            error="max_steps_exceeded",
        )

    return task_rewards, global_step


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = None

    all_rewards: List[float] = []
    global_step = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        try:
            env = await SOC2Env.from_docker_image(LOCAL_IMAGE_NAME)
        except Exception as e:
            print(f"[DEBUG] Environment startup failed: {e}", flush=True)
            return

        for task_id in GRADED_TASK_IDS:
            print(f"[DEBUG] Starting task: {task_id}", flush=True)
            try:
                task_rewards, global_step = await run_task(env, client, task_id, global_step)
                all_rewards.extend(task_rewards)
                task_score = sum(task_rewards)
                print(f"[DEBUG] Task {task_id} raw reward: {task_score:.3f}", flush=True)
            except Exception as e:
                print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
                all_rewards.append(0.0)

    finally:
        score = sum(all_rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        try:
            if env is not None:
                await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=global_step, score=score, rewards=all_rewards)


if __name__ == "__main__":
    asyncio.run(main())
