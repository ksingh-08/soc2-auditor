"""
SOC 2 Evidence Auditor Environment Implementation.

Episode flow:
    1. reset()          — loads a task, returns observation with all available files visible.
    2. INSPECT_FILE     — agent reads one file at a time (relevant: +0.1, distractor: -0.05).
                          Large log files cannot be INSPECT_FILE'd — use SEARCH_LOGS instead.
    3. SEARCH_LOGS      — agent queries a large log file by field/value filter.
                          Productive search of relevant file: +0.1. Empty or distractor: -0.05.
    4. SUBMIT_DECISION  — agent renders verdict; episode ends (done=True).

IMPORTANT — step() returns ONLY the delta reward for that action.
inference.py sums all step rewards. Do NOT return cumulative totals.

Reward structure:
    INSPECT relevant file (first time):         +0.1  per file (capped at 0.3 total)
    INSPECT distractor file:                    -0.05 per file
    INSPECT large log file (use SEARCH_LOGS):   -0.05
    INSPECT invalid file:                       -0.1
    INSPECT re-inspection:                       0.0
    SEARCH_LOGS relevant + results found:       +0.1  (shares cap with INSPECT rewards)
    SEARCH_LOGS relevant + no results:          -0.05
    SEARCH_LOGS distractor/non-relevant file:   -0.05
    SEARCH_LOGS non-large file:                 -0.05
    SEARCH_LOGS re-query (same params):          0.0
    SUBMIT correct decision + reason:           +(1.0 - accumulated_inspect_reward)
        → guarantees total episode reward caps at exactly 1.0
    SUBMIT correct decision + wrong reason:     +0.2 flat (partial credit, done=True)
    SUBMIT wrong decision (APPROVE bad):        -0.5 (done=True)
    SUBMIT wrong decision (REJECT good):        -0.3 (done=True)

accumulated_inspect_reward tracks ONLY positive inspect/search rewards (+0.1 each, cap 0.3).
Distractor penalties reduce the inference.py sum but do NOT reduce the submit reward.

Example — perfect play, Task 3 (3 relevant files inspected before cap):
    step 1: INSPECT hr_terminations.json     → +0.1  (accumulated=0.1)
    step 2: INSPECT aws_users.json           → +0.1  (accumulated=0.2)
    step 3: INSPECT github_users.json        → +0.1  (accumulated=0.3)
    step 4: INSPECT prod_db_users.json       →  0.0  (cap reached)
    step 5: SUBMIT REJECT INCOMPLETE_REVOCATION → +(1.0 - 0.3) = +0.7
    inference sum: 0.1+0.1+0.1+0.0+0.7 = 1.0  ✓

Example — SEARCH_LOGS task (Task 11):
    step 1: INSPECT hr_terminations.json     → +0.1  (accumulated=0.1)
    step 2: SEARCH_LOGS aws_cloudtrail_full_log.json username=alice_dev → +0.1  (accumulated=0.2)
    step 3: SUBMIT REJECT INCOMPLETE_REVOCATION → +(1.0 - 0.2) = +0.8
    inference sum: 0.1+0.1+0.8 = 1.0  ✓
"""

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SOC2Action, SOC2Observation
    from .tasks import ALL_TASKS, EVIDENCE, LARGE_FILES, AuditTask, grade_decision
except ImportError:
    from models import SOC2Action, SOC2Observation
    from server.tasks import ALL_TASKS, EVIDENCE, LARGE_FILES, AuditTask, grade_decision

MAX_INSPECT_REWARD = 0.3    # cap on positive inspect/search rewards (3 × 0.1)
MAX_STEPS_PER_EPISODE = 15  # server-side hard limit — prevents runaway episodes


class SOC2Environment(Environment):
    """
    SOC 2 Evidence Auditor RL environment.

    The state space includes:
    - Which task is active (control requirement + full file list)
    - Which files have been inspected (with their contents)
    - Which log searches have been performed (with their results)
    - Cumulative reward so far
    - Episode active/done status

    The action space is:
    - INSPECT_FILE:     read one file (relevant: +0.1 capped, distractors: -0.05)
    - SEARCH_LOGS:      query a large log file by field=value (productive: +0.1, empty: -0.05)
    - SUBMIT_DECISION:  render verdict (done=True)

    File lists contain both relevant files and distractors. Large log files (in LARGE_FILES)
    must be queried via SEARCH_LOGS — INSPECT_FILE on them returns an error.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state: State = State(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[AuditTask] = None
        self._inspected_files: dict = {}
        self._accumulated_inspect_reward: float = 0.0  # positive inspect/search rewards only
        self._cumulative_reward: float = 0.0           # running total for agent display
        self._audit_status: str = "IN_PROGRESS"
        self._episode_active: bool = False

    def reset(  # type: ignore[override]
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SOC2Observation:
        """
        Start a new audit episode.

        Args:
            seed:       Random seed for reproducible task selection.
            episode_id: Override the episode UUID.
            task_id:    Pin to a specific task. If None, picks randomly.

        Returns:
            Initial SOC2Observation with full file list and empty inspected_files.
        """
        rng = random.Random(seed)

        if task_id is not None:
            matches = [t for t in ALL_TASKS if t.task_id == task_id]
            if not matches:
                valid = [t.task_id for t in ALL_TASKS]
                raise ValueError(f"Unknown task_id {task_id!r}. Valid: {valid}")
            self._task = matches[0]
        else:
            self._task = rng.choice(ALL_TASKS)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._inspected_files = {}
        self._accumulated_inspect_reward = 0.0
        self._cumulative_reward = 0.0
        self._audit_status = "IN_PROGRESS"
        self._episode_active = True

        num_relevant = len(self._task.relevant_files)
        num_distractors = len(self._task.available_files) - num_relevant
        has_large = any(f in LARGE_FILES for f in self._task.available_files)
        large_note = " Large log files require SEARCH_LOGS (not INSPECT_FILE)." if has_large else ""

        return SOC2Observation(
            task_id=self._task.task_id,
            control_requirement=self._task.control_requirement,
            available_files=list(self._task.available_files),
            inspected_files={},
            audit_status="IN_PROGRESS",
            step_reward=0.0,
            cumulative_reward=0.0,
            done=False,
            reward=0.0,
            message=(
                f"Audit task '{self._task.task_id}' started [{self._task.difficulty}]. "
                f"{len(self._task.available_files)} files available "
                f"({num_relevant} relevant, {num_distractors} distractors — identify them by name).{large_note} "
                f"Max episode reward: 1.0. "
                f"INSPECT relevant files (+0.1 each, max 0.3 total), ignore distractors (-0.05 each). "
                f"Then SUBMIT_DECISION. Control: {self._task.control_requirement}"
            ),
        )

    def step(  # type: ignore[override]
        self,
        action: SOC2Action,
        **kwargs: Any,
    ) -> SOC2Observation:
        if self._task is None or not self._episode_active:
            raise RuntimeError("Call reset() before step().")

        # Server-side hard step limit
        if self._state.step_count >= MAX_STEPS_PER_EPISODE:
            step_reward = -0.1
            self._cumulative_reward += step_reward
            self._episode_active = False
            return self._build_obs(
                step_reward=step_reward,
                done=True,
                message=(
                    f"TIMEOUT: Maximum steps ({MAX_STEPS_PER_EPISODE}) reached without "
                    f"SUBMIT_DECISION. -0.1 penalty. Episode ended."
                ),
            )

        self._state.step_count += 1

        if action.type == "INSPECT_FILE":
            return self._handle_inspect(action)
        elif action.type == "SEARCH_LOGS":
            return self._handle_search_logs(action)
        elif action.type == "SUBMIT_DECISION":
            return self._handle_submit(action)
        else:
            step_reward = -0.1
            self._cumulative_reward += step_reward
            return self._build_obs(
                step_reward=step_reward,
                done=False,
                message=f"Unknown action type '{action.type}'. Use INSPECT_FILE, SEARCH_LOGS, or SUBMIT_DECISION.",
            )

    def _handle_inspect(self, action: SOC2Action) -> SOC2Observation:
        file_name = action.file_name

        if not file_name:
            step_reward = -0.1
            self._cumulative_reward += step_reward
            return self._build_obs(
                step_reward=step_reward,
                done=False,
                message="INSPECT_FILE requires a 'file_name'. Provide one from available_files.",
            )

        if file_name not in self._task.available_files:
            step_reward = -0.1
            self._cumulative_reward += step_reward
            return self._build_obs(
                step_reward=step_reward,
                done=False,
                message=(
                    f"'{file_name}' is not available for this task. "
                    f"Available: {self._task.available_files}"
                ),
            )

        # Block direct inspection of large log files
        if file_name in LARGE_FILES:
            step_reward = -0.05
            self._cumulative_reward += step_reward
            return self._build_obs(
                step_reward=step_reward,
                done=False,
                message=(
                    f"'{file_name}' is too large to inspect directly (-0.05). "
                    f"Use SEARCH_LOGS with query_field and query_value to filter relevant events. "
                    f"Example: {{\"type\": \"SEARCH_LOGS\", \"file_name\": \"{file_name}\", "
                    f"\"query_field\": \"username\", \"query_value\": \"<value>\"}}"
                ),
            )

        if file_name in self._inspected_files:
            return self._build_obs(
                step_reward=0.0,
                done=False,
                message=f"'{file_name}' already inspected. No additional reward.",
            )

        # First-time inspection — relevant or distractor?
        self._inspected_files[file_name] = EVIDENCE[file_name]
        is_relevant = file_name in self._task.relevant_files

        if is_relevant:
            if self._accumulated_inspect_reward >= MAX_INSPECT_REWARD:
                step_reward = 0.0
                msg = (
                    f"Inspected relevant file '{file_name}'. Content added to inspected_files. "
                    f"Inspect reward cap ({MAX_INSPECT_REWARD}) already reached — no additional reward."
                )
            else:
                step_reward = 0.1
                self._accumulated_inspect_reward += step_reward
                remaining = MAX_INSPECT_REWARD - self._accumulated_inspect_reward
                msg = (
                    f"Inspected relevant file '{file_name}'. +0.1 reward. "
                    f"Inspect budget remaining: {remaining:.1f}. "
                    f"Examine the content against the control requirement."
                )
        else:
            step_reward = -0.05
            msg = (
                f"'{file_name}' is a distractor — not relevant to this control. "
                f"-0.05 penalty. Focus on files related to the control requirement."
            )

        self._cumulative_reward += step_reward
        return self._build_obs(step_reward=step_reward, done=False, message=msg)

    def _handle_search_logs(self, action: SOC2Action) -> SOC2Observation:
        file_name = action.file_name
        query_field = action.query_field
        query_value = action.query_value

        if not file_name or not query_field or not query_value:
            step_reward = -0.1
            self._cumulative_reward += step_reward
            return self._build_obs(
                step_reward=step_reward,
                done=False,
                message=(
                    "SEARCH_LOGS requires 'file_name', 'query_field', and 'query_value'. "
                    "Example: {\"type\": \"SEARCH_LOGS\", \"file_name\": \"aws_cloudtrail_full_log.json\", "
                    "\"query_field\": \"username\", \"query_value\": \"alice_dev\"}"
                ),
            )

        if file_name not in self._task.available_files:
            step_reward = -0.1
            self._cumulative_reward += step_reward
            return self._build_obs(
                step_reward=step_reward,
                done=False,
                message=f"'{file_name}' is not available for this task. Available: {self._task.available_files}",
            )

        if file_name not in LARGE_FILES:
            step_reward = -0.05
            self._cumulative_reward += step_reward
            return self._build_obs(
                step_reward=step_reward,
                done=False,
                message=(
                    f"'{file_name}' is not a large log file — use INSPECT_FILE instead. "
                    f"-0.05 penalty. Large files requiring SEARCH_LOGS: {sorted(LARGE_FILES & set(self._task.available_files))}"
                ),
            )

        # Check for repeated identical search
        search_key = f"{file_name}?{query_field}={query_value}"
        if search_key in self._inspected_files:
            return self._build_obs(
                step_reward=0.0,
                done=False,
                message=f"Search '{search_key}' already performed. No additional reward.",
            )

        # Execute the search — filter events by query_field == query_value
        file_data = EVIDENCE[file_name]
        events = file_data.get("events", [])
        matches = [e for e in events if str(e.get(query_field, "")) == str(query_value)]

        self._inspected_files[search_key] = {
            "query": {"field": query_field, "value": query_value},
            "matched_events": len(matches),
            "results": matches,
        }

        is_relevant = file_name in self._task.relevant_files

        if is_relevant and matches:
            if self._accumulated_inspect_reward >= MAX_INSPECT_REWARD:
                step_reward = 0.0
                msg = (
                    f"SEARCH_LOGS '{file_name}' found {len(matches)} event(s) where "
                    f"{query_field}={query_value!r}. Results in inspected_files. "
                    f"Inspect reward cap reached — no additional reward."
                )
            else:
                step_reward = 0.1
                self._accumulated_inspect_reward += step_reward
                remaining = MAX_INSPECT_REWARD - self._accumulated_inspect_reward
                msg = (
                    f"SEARCH_LOGS '{file_name}' found {len(matches)} event(s) where "
                    f"{query_field}={query_value!r}. +0.1 reward. "
                    f"Inspect budget remaining: {remaining:.1f}. "
                    f"Review the matched events against the control requirement."
                )
        elif is_relevant and not matches:
            step_reward = -0.05
            msg = (
                f"SEARCH_LOGS '{file_name}': no events found where {query_field}={query_value!r}. "
                f"-0.05 penalty. Try a different query_field or query_value."
            )
        else:
            # Distractor file or non-relevant
            step_reward = -0.05
            msg = (
                f"'{file_name}' is not relevant to this control. "
                f"-0.05 penalty. Focus your search on files related to the control requirement."
            )

        self._cumulative_reward += step_reward
        return self._build_obs(step_reward=step_reward, done=False, message=msg)

    def _handle_submit(self, action: SOC2Action) -> SOC2Observation:
        if action.decision is None or action.reason is None:
            step_reward = -0.1
            self._cumulative_reward += step_reward
            return self._build_obs(
                step_reward=step_reward,
                done=False,
                message=(
                    "SUBMIT_DECISION requires both 'decision' (APPROVE|REJECT) "
                    "and 'reason' (MISSING_TIMESTAMP|SLA_VIOLATION|MISSING_APPROVAL|"
                    "INCOMPLETE_REVOCATION|POLICY_VIOLATION|NONE)."
                ),
            )

        correct_decision = action.decision == self._task.correct_decision
        correct_reason = action.reason == self._task.correct_reason
        num_inspected = len(self._inspected_files)

        if correct_decision and correct_reason:
            step_reward = round(1.0 - self._accumulated_inspect_reward, 10)
            msg = (
                f"CORRECT: {action.decision} / {action.reason}. "
                f"Inspected/searched {num_inspected} file(s). "
                f"Submit reward = 1.0 - {self._accumulated_inspect_reward:.1f} = {step_reward:.2f}. "
                f"Episode total will be exactly 1.0 minus any distractor penalties."
            )
        elif correct_decision and not correct_reason:
            step_reward = 0.2
            msg = (
                f"PARTIAL: Decision '{action.decision}' is correct but reason "
                f"'{action.reason}' is wrong. Expected '{self._task.correct_reason}'. "
                f"+0.2 partial credit only."
            )
        elif not correct_decision and self._task.correct_decision == "REJECT":
            step_reward = -0.5
            msg = (
                f"CRITICAL ERROR: You APPROVED evidence that should be REJECTED "
                f"(reason: {self._task.correct_reason}). "
                f"This would constitute an audit failure in a real SOC 2 review."
            )
        else:
            step_reward = -0.3
            msg = "INCORRECT: You REJECTED evidence that should be APPROVED."

        self._cumulative_reward += step_reward
        self._audit_status = action.decision
        self._episode_active = False

        return self._build_obs(step_reward=step_reward, done=True, message=msg)

    def _build_obs(self, step_reward: float, done: bool, message: str) -> SOC2Observation:
        return SOC2Observation(
            task_id=self._task.task_id,
            control_requirement=self._task.control_requirement,
            available_files=list(self._task.available_files),
            inspected_files=dict(self._inspected_files),
            audit_status=self._audit_status,
            step_reward=step_reward,
            cumulative_reward=self._cumulative_reward,
            done=done,
            reward=float(step_reward),   # DELTA only — inference.py sums these up
            message=message,
        )

    @property
    def state(self) -> State:
        return self._state
