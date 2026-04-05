"""
SOC 2 Evidence Auditor — Task definitions, evidence data, and deterministic grader.

Reward structure (max 1.0 per episode):
    INSPECT relevant file (first time):         +0.1  (capped at 0.3 total)
    INSPECT distractor file:                    -0.05
    INSPECT invalid/unknown file:               -0.1
    INSPECT large log file (use SEARCH_LOGS):   -0.05
    SEARCH_LOGS relevant file + results found:  +0.1  (capped at 0.3 with INSPECT rewards)
    SEARCH_LOGS relevant file + no results:     -0.05
    SEARCH_LOGS non-relevant/distractor file:   -0.05
    SEARCH_LOGS non-large file:                 -0.05
    SUBMIT correct decision + reason:     +(1.0 - accumulated_inspect_reward) (done=True)
    SUBMIT correct decision + wrong reason: +0.2 (done=True)
    SUBMIT wrong decision (APPROVE bad):  -0.5  (done=True)
    SUBMIT wrong decision (REJECT good):  -0.3  (done=True)

Maximum per episode = 0.3 (inspect cap) + 0.7 (correct submit) = 1.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple



LARGE_FILES: Set[str] = {
    "aws_cloudtrail_full_log.json",
}



_DISTRACTORS: Dict[str, Any] = {
    "slack_engineering_channel.json": {
        "channel": "#engineering",
        "export_date": "2024-10-15",
        "messages": [
            {"user": "alice", "text": "Deploying hotfix now", "ts": "2024-10-15T14:00:00Z"},
            {"user": "bob", "text": "LGTM, go ahead", "ts": "2024-10-15T14:01:00Z"},
            {"user": "charlie", "text": "Tests passing", "ts": "2024-10-15T14:05:00Z"},
        ],
        "_note": "DISTRACTOR: Slack logs irrelevant to this control.",
    },
    "q3_sprint_planning.json": {
        "sprint": "Q3-Sprint-7",
        "team": "Platform Engineering",
        "start_date": "2024-10-01",
        "end_date": "2024-10-14",
        "tickets": ["PLAT-441", "PLAT-442", "PLAT-443"],
        "velocity": 42,
        "_note": "DISTRACTOR: Sprint planning irrelevant to security controls.",
    },
    "ci_cd_pipeline_config.json": {
        "pipeline": "main-deploy",
        "trigger": "push to main",
        "stages": ["lint", "test", "build", "deploy"],
        "notifications": "slack:#deployments",
        "last_run": "2024-10-15T14:32:00Z",
        "status": "SUCCESS",
        "_note": "DISTRACTOR: CI/CD config is not the evidence needed here.",
    },
    "code_style_guidelines.json": {
        "document": "Engineering Code Style Guidelines v2.3",
        "last_updated": "2024-09-01",
        "rules": ["PEP8 for Python", "ESLint for JS", "Black formatter"],
        "enforced_by": "pre-commit hooks",
        "_note": "DISTRACTOR: Style guidelines are irrelevant to this audit.",
    },
    "lunch_menu_october.json": {
        "document": "Office Lunch Menu — October 2024",
        "monday": "Grilled chicken + salad",
        "tuesday": "Pasta primavera",
        "wednesday": "Tacos",
        "thursday": "Sushi",
        "friday": "Pizza",
        "_note": "DISTRACTOR: Obviously irrelevant.",
    },
    "it_helpdesk_tickets.json": {
        "export_date": "2024-10-15",
        "open_tickets": [
            {"id": "IT-9901", "subject": "VPN slow connection", "status": "open"},
            {"id": "IT-9902", "subject": "New laptop setup", "status": "closed"},
            {"id": "IT-9903", "subject": "Printer not working", "status": "open"},
        ],
        "_note": "DISTRACTOR: Helpdesk tickets are not evidence for this control.",
    },
    "employee_handbook.json": {
        "document": "Employee Handbook v4.1",
        "sections": ["Code of Conduct", "PTO Policy", "Benefits", "Remote Work"],
        "last_updated": "2024-01-15",
        "hr_contact": "hr@company.com",
        "_note": "DISTRACTOR: The handbook defines policies but is not audit evidence.",
    },
    "slack_hr_channel.json": {
        "channel": "#hr-announcements",
        "export_date": "2024-10-05",
        "messages": [
            {"user": "hr-admin", "text": "Welcome new hires!", "ts": "2024-10-01T09:30:00Z"},
            {"user": "hr-admin", "text": "Reminder: benefits enrollment deadline Oct 31", "ts": "2024-10-03T10:00:00Z"},
        ],
        "_note": "DISTRACTOR: HR Slack messages are not structured audit evidence.",
    },
    "office_supply_orders.json": {
        "order_id": "PO-2024-0231",
        "items": ["Printer paper x10", "Pens x50", "Sticky notes x20"],
        "total_usd": 87.40,
        "approved_by": "office-manager",
        "_note": "DISTRACTOR: Office supply orders irrelevant to security audit.",
    },
    "q4_headcount_plan.json": {
        "document": "Q4 2024 Headcount Plan",
        "engineering_headcount": 42,
        "planned_hires": 8,
        "planned_exits": 3,
        "budget_approved": True,
        "_note": "DISTRACTOR: Headcount planning is not SOC 2 evidence.",
    },
    "zoom_license_report.json": {
        "report_date": "2024-10-01",
        "total_licenses": 150,
        "active_licenses": 138,
        "unused_licenses": 12,
        "cost_per_license_usd": 14.99,
        "_note": "DISTRACTOR: Zoom license report irrelevant to access revocation audit.",
    },
    "jira_project_members.json": {
        "project": "PLATFORM",
        "members": [
            {"user": "alice_dev", "role": "Developer", "last_active": "2024-09-30"},
            {"user": "bob_eng", "role": "Tech Lead", "last_active": "2024-10-14"},
        ],
        "_note": "DISTRACTOR: Jira membership does not constitute production system access evidence.",
    },
    "slack_workspace_users.json": {
        "workspace": "acme-corp",
        "export_date": "2024-10-06",
        "total_users": 203,
        "active_users": 198,
        "deactivated_users": ["alice_dev", "old_contractor_1"],
        "_note": "DISTRACTOR: Slack deactivation is not production system access. Control requires AWS+GitHub+DB.",
    },
    "employee_benefits_portal.json": {
        "portal": "BambooHR",
        "active_employees": 201,
        "on_leave": 3,
        "terminated_this_quarter": 5,
        "_note": "DISTRACTOR: Benefits portal is HR data, not a production access system.",
    },
    "q3_marketing_goals.json": {
        "document": "Q3 Marketing OKRs",
        "objective": "Grow enterprise pipeline by 40%",
        "key_results": ["10 enterprise demos", "2 case studies", "1 conference"],
        "_note": "DISTRACTOR: Marketing goals are irrelevant to a security audit.",
    },
}

_RELEVANT: Dict[str, Any] = {
    # ── Task 1: PR Approval Check ──────────────────────────────────────────
    "pull_request_log.json": {
        "pr_id": "PR-2024-0891",
        "title": "Fix authentication bypass vulnerability",
        "author": "dev-charlie",
        "merged_by": "alice",
        "merged_at": "2024-10-15T14:30:00Z",
        "approved_by": None,
        "approvals_count": 0,
        "branch": "hotfix/auth-fix",
        "target_branch": "main",
        "review_comments": 0,
        "ci_passed": True,
        "_note": "FLAW: approved_by is null — merged with zero approvals.",
    },
    # Compliant variant — used by pr_approval_compliant task
    "pull_request_log_compliant.json": {
        "pr_id": "PR-2024-0892",
        "title": "Add rate limiting to public API endpoints",
        "author": "dev-diana",
        "merged_by": "bob-lead",
        "merged_at": "2024-10-16T11:00:00Z",
        "approved_by": "bob-lead",
        "approvals_count": 2,
        "branch": "feature/rate-limiting",
        "target_branch": "main",
        "review_comments": 4,
        "ci_passed": True,
        "_note": "COMPLIANT: approved_by is set, approvals_count=2.",
    },

    # ── Task 2: Access Revocation SLA ──────────────────────────────────────
    "hr_termination_ticket.json": {
        "ticket_id": "HR-2024-4421",
        "employee_id": "EMP-0042",
        "employee_name": "Bob Martinez",
        "department": "Engineering",
        "termination_type": "Voluntary",
        "termination_date": "2024-10-01T09:00:00Z",
        "ticket_created_by": "hr-admin",
        "status": "CLOSED",
        "last_working_day": "2024-10-01",
    },
    "aws_iam_audit_log.json": {
        "employee_id": "EMP-0042",
        "employee_name": "Bob Martinez",
        "access_removed_date": "2024-10-05T11:30:00Z",
        "hours_since_termination": 98.5,
        "removed_by": "it-admin",
        "systems_revoked": ["AWS-Console", "GitHub", "Slack", "Jira", "VPN"],
        "confirmation_ticket": "IT-2024-8823",
        "_note": "FLAW: Access removed 98.5 hours after termination — exceeds 24-hour SLA.",
    },
    # Borderline variant — 26h, still a violation
    "aws_iam_audit_log_26h.json": {
        "employee_id": "EMP-0042",
        "employee_name": "Bob Martinez",
        "access_removed_date": "2024-10-02T11:00:00Z",
        "hours_since_termination": 26.0,
        "removed_by": "it-admin",
        "systems_revoked": ["AWS-Console", "GitHub", "Slack", "Jira", "VPN"],
        "confirmation_ticket": "IT-2024-8824",
        "_note": "FLAW: Access removed 26 hours after termination — exceeds 24-hour SLA by 2 hours.",
    },
    # Compliant variant — 18h, within SLA
    "aws_iam_audit_log_compliant.json": {
        "employee_id": "EMP-0043",
        "employee_name": "Carol Singh",
        "access_removed_date": "2024-10-02T03:00:00Z",
        "hours_since_termination": 18.0,
        "removed_by": "it-admin",
        "systems_revoked": ["AWS-Console", "GitHub", "Slack", "Jira", "VPN"],
        "confirmation_ticket": "IT-2024-8825",
        "_note": "COMPLIANT: Access removed 18 hours after termination — within 24-hour SLA.",
    },

    # ── Task 3: Multi-System Access Revocation (Hard) ─────────────────────
    "hr_terminations.json": {
        "report_date": "2024-10-01",
        "terminations": [
            {
                "employee_id": "EMP-0099",
                "username": "alice_dev",
                "full_name": "Alice Chen",
                "department": "Engineering",
                "termination_date": "2024-10-01T17:00:00Z",
                "termination_type": "Resignation",
            }
        ],
    },
    "aws_users.json": {
        "report_date": "2024-10-02",
        "active_users": ["bob_eng", "carol_ops", "david_sre", "eve_dev"],
        "recently_removed": ["alice_dev"],
        "note": "alice_dev access removed 2024-10-02T09:00:00Z",
    },
    "github_users.json": {
        "report_date": "2024-10-02",
        "org": "acme-corp",
        "active_members": ["bob_eng", "carol_ops", "david_sre", "eve_dev"],
        "recently_removed": ["alice_dev"],
        "note": "alice_dev removed from org 2024-10-02T09:15:00Z",
    },
    "prod_db_users.json": {
        "report_date": "2024-10-06",
        "database": "prod-postgres-primary",
        "active_users": ["bob_eng", "carol_ops", "david_sre", "alice_dev", "eve_dev"],
        "last_audited": "2024-10-06T08:00:00Z",
        "_note": "FLAW: alice_dev is still ACTIVE in production DB — access was never revoked.",
    },
    # Compliant variant — alice_dev removed from all systems including DB
    "prod_db_users_compliant.json": {
        "report_date": "2024-10-02",
        "database": "prod-postgres-primary",
        "active_users": ["bob_eng", "carol_ops", "david_sre", "eve_dev"],
        "recently_removed": ["alice_dev"],
        "last_audited": "2024-10-02T10:00:00Z",
        "_note": "COMPLIANT: alice_dev removed from prod DB 2024-10-02T09:45:00Z — all systems revoked.",
    },
    # HR termination for compliant SLA task (Carol Singh)
    "hr_termination_ticket_carol.json": {
        "ticket_id": "HR-2024-4422",
        "employee_id": "EMP-0043",
        "employee_name": "Carol Singh",
        "department": "Marketing",
        "termination_type": "Voluntary",
        "termination_date": "2024-10-01T09:00:00Z",
        "ticket_created_by": "hr-admin",
        "status": "CLOSED",
        "last_working_day": "2024-10-01",
    },

    # ── Task 4: MFA Enforcement ────────────────────────────────────────────
    "cloudtrail_login_events.json": {
        "event_id": "CT-20241012-8823",
        "account_id": "123456789012",
        "user_identity": "root",
        "event_type": "ConsoleLogin",
        "timestamp": "2024-10-12T08:15:00Z",
        "mfa_used": False,
        "source_ip": "203.0.113.42",
        "_note": "FLAW: Root account login without MFA.",
    },

    # ── Task 5: Encryption at Rest ─────────────────────────────────────────
    "s3_bucket_policy.json": {
        "bucket_name": "acme-prod-customer-data",
        "region": "us-east-1",
        "versioning_enabled": True,
        "public_access_blocked": True,
        "server_side_encryption": None,
        "kms_key_id": None,
        "data_classification": "CONFIDENTIAL",
        "_note": "FLAW: server_side_encryption is null.",
    },

    # ── Task 6: Incident Response SLA ─────────────────────────────────────
    "incident_response_log.json": {
        "incident_id": "INC-2024-0088",
        "severity": "HIGH",
        "detected_at": "2024-09-15T02:00:00Z",
        "reported_to_security_team_at": "2024-09-15T07:30:00Z",
        "hours_to_escalate": 5.5,
        "escalation_sla_hours": 4,
        "status": "RESOLVED",
        "_note": "FLAW: 5.5 hours to escalate, exceeds 4-hour SLA.",
    },

    # ── Task 7: Vendor Security Review ────────────────────────────────────
    "vendor_review_register.json": {
        "vendor_id": "VND-0034",
        "vendor_name": "CloudUtils Inc.",
        "contract_start": "2024-01-15",
        "last_security_review": None,
        "questionnaire_on_file": False,
        "data_access_level": "PII",
        "_note": "FLAW: PII vendor has no questionnaire on file.",
    },

    # ── Task 8: Access Review Timestamp (hallucination trap) ──────────────
    "access_review_report.json": {
        "review_id": "AR-Q3-2024",
        "review_period": "Q3 2024 (July-September)",
        "reviewed_by": "security-team",
        "review_status": "COMPLETE",
        "users_reviewed": 47,
        "users_deprovisioned": 3,
        "manually_signed": "2024-10-01",
        "sign_off_name": "Jane Smith, CISO",
        "_note": "FLAW: No system_timestamp field — manually_signed is insufficient.",
    },

    # ── Task 8b: Access Review Compliant (APPROVE trap) ───────────────────
    "access_review_report_compliant.json": {
        "review_id": "AR-Q2-2024",
        "review_period": "Q2 2024 (April-June)",
        "reviewed_by": "security-team",
        "review_status": "COMPLETE",
        "users_reviewed": 43,
        "users_deprovisioned": 2,
        "system_timestamp": "2024-07-02T10:32:44Z",
        "sign_off_name": "Jane Smith, CISO",
        "_note": "COMPLIANT: system_timestamp present — access review properly documented.",
    },

    # ── Task 9: Change Management ─────────────────────────────────────────
    "change_management_ticket.json": {
        "ticket_id": "CHG-2024-1122",
        "title": "Database schema migration — add PII columns",
        "risk_level": "HIGH",
        "cab_approval_required": True,
        "cab_approved": False,
        "cab_approval_date": None,
        "implemented_at": "2024-11-02T14:00:00Z",
        "_note": "FLAW: HIGH-risk change implemented without CAB approval.",
    },

    # ── Task 10: Password Policy ───────────────────────────────────────────
    "password_policy_audit.json": {
        "audit_date": "2024-10-15",
        "total_accounts_audited": 213,
        "accounts_compliant": 146,
        "compliance_percentage": 68.5,
        "policy_requirement": "100% compliance required",
        "_note": "FLAW: Only 68.5% compliant, policy requires 100%.",
    },

    # ── Task 11: CloudTrail Privileged Access Audit (SEARCH_LOGS required) ─
    "aws_cloudtrail_full_log.json": {
        "_large": True,
        "log_type": "aws_cloudtrail",
        "account_id": "123456789012",
        "region": "us-east-1",
        "export_period": "2024-10-01 to 2024-10-05",
        "total_events": 28,
        "events": [
            # Oct 1 before termination (alice_dev terminated 2024-10-01T17:00:00Z)
            {"event_id": "CT-001", "username": "bob_eng",   "event_name": "DescribeInstances",     "timestamp": "2024-10-01T08:00:00Z", "source_ip": "10.0.1.5"},
            {"event_id": "CT-002", "username": "carol_ops", "event_name": "PutObject",              "timestamp": "2024-10-01T09:00:00Z", "source_ip": "10.0.1.6", "resource": "s3://backups"},
            {"event_id": "CT-003", "username": "alice_dev", "event_name": "DescribeInstances",     "timestamp": "2024-10-01T10:00:00Z", "source_ip": "192.168.1.22"},
            {"event_id": "CT-004", "username": "david_sre", "event_name": "CreateSnapshot",        "timestamp": "2024-10-01T11:00:00Z", "source_ip": "10.0.1.8", "resource": "vol-0abc123"},
            {"event_id": "CT-005", "username": "eve_dev",   "event_name": "ListBuckets",           "timestamp": "2024-10-01T12:00:00Z", "source_ip": "10.0.1.9"},
            {"event_id": "CT-006", "username": "alice_dev", "event_name": "GetSecretValue",        "timestamp": "2024-10-01T14:00:00Z", "source_ip": "192.168.1.22", "resource": "arn:aws:secretsmanager:::secret:prod-db-creds"},
            {"event_id": "CT-007", "username": "bob_eng",   "event_name": "DescribeSecurityGroups","timestamp": "2024-10-01T15:00:00Z", "source_ip": "10.0.1.5"},
            {"event_id": "CT-008", "username": "carol_ops", "event_name": "DescribeInstances",     "timestamp": "2024-10-01T16:00:00Z", "source_ip": "10.0.1.6"},
            # alice_dev terminated 2024-10-01T17:00:00Z — events below are VIOLATIONS
            {"event_id": "CT-009", "username": "david_sre", "event_name": "PutObject",             "timestamp": "2024-10-01T18:00:00Z", "source_ip": "10.0.1.8"},
            {"event_id": "CT-010", "username": "eve_dev",   "event_name": "DescribeInstances",     "timestamp": "2024-10-01T20:00:00Z", "source_ip": "10.0.1.9"},
            # Oct 2 — alice_dev API key still active (console access was revoked but API key was not)
            {"event_id": "CT-011", "username": "alice_dev", "event_name": "GetSecretValue",        "timestamp": "2024-10-02T07:30:00Z", "source_ip": "203.0.113.77", "resource": "arn:aws:secretsmanager:::secret:prod-db-creds", "_note": "VIOLATION: API call 14.5h after termination — API key not revoked"},
            {"event_id": "CT-012", "username": "bob_eng",   "event_name": "DescribeInstances",     "timestamp": "2024-10-02T09:00:00Z", "source_ip": "10.0.1.5"},
            {"event_id": "CT-013", "username": "carol_ops", "event_name": "CreateSnapshot",        "timestamp": "2024-10-02T10:00:00Z", "source_ip": "10.0.1.6"},
            {"event_id": "CT-014", "username": "alice_dev", "event_name": "ListBuckets",           "timestamp": "2024-10-02T14:15:00Z", "source_ip": "203.0.113.77", "_note": "VIOLATION: API call 21.25h after termination"},
            {"event_id": "CT-015", "username": "david_sre", "event_name": "DescribeInstances",     "timestamp": "2024-10-02T15:00:00Z", "source_ip": "10.0.1.8"},
            {"event_id": "CT-016", "username": "eve_dev",   "event_name": "PutObject",             "timestamp": "2024-10-02T16:00:00Z", "source_ip": "10.0.1.9"},
            # Oct 3 — alice_dev still accessing AWS programmatically
            {"event_id": "CT-017", "username": "alice_dev", "event_name": "AssumeRole",            "timestamp": "2024-10-03T09:45:00Z", "source_ip": "203.0.113.77", "resource": "arn:aws:iam::123456789012:role/AdminRole", "_note": "VIOLATION: Role assumption 40.75h after termination"},
            {"event_id": "CT-018", "username": "bob_eng",   "event_name": "ListBuckets",           "timestamp": "2024-10-03T10:00:00Z", "source_ip": "10.0.1.5"},
            {"event_id": "CT-019", "username": "carol_ops", "event_name": "DescribeInstances",     "timestamp": "2024-10-03T11:00:00Z", "source_ip": "10.0.1.6"},
            {"event_id": "CT-020", "username": "david_sre", "event_name": "GetSecretValue",        "timestamp": "2024-10-03T12:00:00Z", "source_ip": "10.0.1.8"},
            {"event_id": "CT-021", "username": "eve_dev",   "event_name": "DescribeInstances",     "timestamp": "2024-10-04T08:00:00Z", "source_ip": "10.0.1.9"},
            {"event_id": "CT-022", "username": "bob_eng",   "event_name": "CreateSnapshot",        "timestamp": "2024-10-04T09:00:00Z", "source_ip": "10.0.1.5"},
            {"event_id": "CT-023", "username": "carol_ops", "event_name": "PutObject",             "timestamp": "2024-10-04T10:00:00Z", "source_ip": "10.0.1.6"},
            {"event_id": "CT-024", "username": "david_sre", "event_name": "DescribeInstances",     "timestamp": "2024-10-04T11:00:00Z", "source_ip": "10.0.1.8"},
            {"event_id": "CT-025", "username": "eve_dev",   "event_name": "GetSecretValue",        "timestamp": "2024-10-05T08:00:00Z", "source_ip": "10.0.1.9"},
            {"event_id": "CT-026", "username": "bob_eng",   "event_name": "DescribeInstances",     "timestamp": "2024-10-05T09:00:00Z", "source_ip": "10.0.1.5"},
            {"event_id": "CT-027", "username": "carol_ops", "event_name": "ListBuckets",           "timestamp": "2024-10-05T10:00:00Z", "source_ip": "10.0.1.6"},
            {"event_id": "CT-028", "username": "david_sre", "event_name": "PutObject",             "timestamp": "2024-10-05T11:00:00Z", "source_ip": "10.0.1.8"},
        ],
    },
}

# Combined evidence store
EVIDENCE: Dict[str, Any] = {**_RELEVANT, **_DISTRACTORS}



@dataclass
class AuditTask:
    task_id: str
    control_requirement: str
    available_files: List[str]       # all files the agent can see (relevant + distractors)
    relevant_files: List[str]        # subset that actually contain evidence for this control
    correct_decision: str
    correct_reason: str
    difficulty: str
    description: str


# ---------------------------------------------------------------------------
# GRADED_TASKS — 3 official hackathon tasks with distractors
# ---------------------------------------------------------------------------

GRADED_TASKS: List[AuditTask] = [

    # ── Task 1: Easy — 1 relevant file, 5 distractors ─────────────────────
    AuditTask(
        task_id="pr_approval_check",
        control_requirement=(
            "All code changes to production branches require documented peer review "
            "approval from at least one team member before merging. "
            "Merges without recorded approvals are non-compliant."
        ),
        available_files=[
            "pull_request_log.json",          # RELEVANT
            "slack_engineering_channel.json",  # distractor
            "q3_sprint_planning.json",         # distractor
            "ci_cd_pipeline_config.json",      # distractor
            "code_style_guidelines.json",      # distractor
            "lunch_menu_october.json",         # distractor
        ],
        relevant_files=["pull_request_log.json"],
        correct_decision="REJECT",
        correct_reason="MISSING_APPROVAL",
        difficulty="easy",
        description=(
            "A PR was merged to main with zero approvals. Agent must identify "
            "pull_request_log.json as the relevant file (ignoring 5 distractors), "
            "find approved_by=null, and REJECT for MISSING_APPROVAL."
        ),
    ),

    # ── Task 2: Medium — 2 relevant files, 5 distractors ──────────────────
    AuditTask(
        task_id="access_revocation_sla",
        control_requirement=(
            "Upon employee termination, all system access must be revoked within 24 hours "
            "of the recorded termination datetime, as mandated by HR policy ITC-07. "
            "Evidence must include both the HR termination record and the IAM audit log."
        ),
        available_files=[
            "hr_termination_ticket.json",  # RELEVANT
            "aws_iam_audit_log.json",      # RELEVANT
            "it_helpdesk_tickets.json",    # distractor
            "employee_handbook.json",      # distractor
            "slack_hr_channel.json",       # distractor
            "office_supply_orders.json",   # distractor
            "q4_headcount_plan.json",      # distractor
        ],
        relevant_files=["hr_termination_ticket.json", "aws_iam_audit_log.json"],
        correct_decision="REJECT",
        correct_reason="SLA_VIOLATION",
        difficulty="medium",
        description=(
            "HR records termination at 2024-10-01T09:00Z. IAM log shows access removed "
            "98.5 hours later. Agent must find both relevant files among 7 total, "
            "cross-reference timestamps, compute 98.5h > 24h, and REJECT for SLA_VIOLATION."
        ),
    ),

    # ── Task 3: Hard — 4 relevant files, 4 distractors ────────────────────
    AuditTask(
        task_id="multi_system_access_revocation",
        control_requirement=(
            "Upon employee termination, access to ALL production systems — including "
            "AWS, GitHub, and the Production Database — must be revoked within 24 hours. "
            "Evidence must confirm revocation across every system. "
            "Partial revocation (access remaining in any one system) is non-compliant."
        ),
        available_files=[
            "hr_terminations.json",         # RELEVANT — shows alice_dev terminated
            "aws_users.json",               # RELEVANT — alice_dev removed (good)
            "github_users.json",            # RELEVANT — alice_dev removed (good)
            "prod_db_users.json",           # RELEVANT — alice_dev STILL ACTIVE (the flaw!)
            "slack_workspace_users.json",   # distractor
            "zoom_license_report.json",     # distractor
            "jira_project_members.json",    # distractor
            "employee_benefits_portal.json", # distractor
        ],
        relevant_files=[
            "hr_terminations.json",
            "aws_users.json",
            "github_users.json",
            "prod_db_users.json",
        ],
        correct_decision="REJECT",
        correct_reason="INCOMPLETE_REVOCATION",
        difficulty="hard",
        description=(
            "alice_dev was terminated on 2024-10-01. AWS and GitHub access were revoked. "
            "BUT prod_db_users.json still lists alice_dev as active — revocation was incomplete. "
            "Agent must inspect all 4 system files (not just 1-2) to find the one failing system. "
            "Distractors include Slack/Zoom/Jira which are not production systems under this control."
        ),
    ),
]


# ---------------------------------------------------------------------------
# EXTRA_TASKS — extended pool for random episodes
# ---------------------------------------------------------------------------

EXTRA_TASKS: List[AuditTask] = [
    AuditTask(
        task_id="mfa_enforcement_check",
        control_requirement=(
            "All AWS console logins must use multi-factor authentication (MFA). "
            "Any console login without MFA is non-compliant."
        ),
        available_files=[
            "cloudtrail_login_events.json",
            "slack_engineering_channel.json",
            "q3_marketing_goals.json",
            "lunch_menu_october.json",
        ],
        relevant_files=["cloudtrail_login_events.json"],
        correct_decision="REJECT",
        correct_reason="POLICY_VIOLATION",
        difficulty="easy",
        description="Root login detected without MFA in CloudTrail. MFA is a mandatory policy control, not an approval gate.",
    ),
    AuditTask(
        task_id="encryption_at_rest_check",
        control_requirement=(
            "All S3 buckets containing confidential data must have server-side encryption enabled."
        ),
        available_files=[
            "s3_bucket_policy.json",
            "ci_cd_pipeline_config.json",
            "office_supply_orders.json",
            "lunch_menu_october.json",
        ],
        relevant_files=["s3_bucket_policy.json"],
        correct_decision="REJECT",
        correct_reason="POLICY_VIOLATION",
        difficulty="easy",
        description="S3 bucket with CONFIDENTIAL data has server_side_encryption=null — mandatory encryption policy not enforced.",
    ),
    AuditTask(
        task_id="incident_response_sla",
        control_requirement=(
            "HIGH severity security incidents must be escalated to the security team within 4 hours."
        ),
        available_files=[
            "incident_response_log.json",
            "slack_hr_channel.json",
            "q3_sprint_planning.json",
            "lunch_menu_october.json",
        ],
        relevant_files=["incident_response_log.json"],
        correct_decision="REJECT",
        correct_reason="SLA_VIOLATION",
        difficulty="medium",
        description="HIGH incident took 5.5 hours to escalate, exceeding 4-hour SLA.",
    ),
    AuditTask(
        task_id="vendor_security_review",
        control_requirement=(
            "All vendors with PII access must have a completed security questionnaire on file."
        ),
        available_files=[
            "vendor_review_register.json",
            "employee_handbook.json",
            "q4_headcount_plan.json",
            "lunch_menu_october.json",
        ],
        relevant_files=["vendor_review_register.json"],
        correct_decision="REJECT",
        correct_reason="MISSING_APPROVAL",
        difficulty="medium",
        description="PII vendor CloudUtils Inc. has questionnaire_on_file=False.",
    ),
    AuditTask(
        task_id="access_review_timestamp",
        control_requirement=(
            "Quarterly access reviews must be documented with a system-generated timestamp. "
            "Manual sign-off dates are not sufficient."
        ),
        available_files=[
            "access_review_report.json",
            "slack_engineering_channel.json",
            "q3_marketing_goals.json",
            "employee_handbook.json",
            "lunch_menu_october.json",
        ],
        relevant_files=["access_review_report.json"],
        correct_decision="REJECT",
        correct_reason="MISSING_TIMESTAMP",
        difficulty="hard",
        description="Review looks complete but has no system_timestamp — hallucination trap.",
    ),
    AuditTask(
        task_id="access_review_compliant",
        control_requirement=(
            "Quarterly access reviews must be documented with a system-generated timestamp. "
            "Manual sign-off dates are not sufficient."
        ),
        available_files=[
            "access_review_report_compliant.json",
            "slack_engineering_channel.json",
            "q3_marketing_goals.json",
            "employee_handbook.json",
            "lunch_menu_october.json",
        ],
        relevant_files=["access_review_report_compliant.json"],
        correct_decision="APPROVE",
        correct_reason="NONE",
        difficulty="medium",
        description=(
            "Q2 access review has system_timestamp='2024-07-02T10:32:44Z' — fully compliant. "
            "Tests the agent's ability to APPROVE when evidence is complete, not just REJECT."
        ),
    ),
    AuditTask(
        task_id="change_management_approval",
        control_requirement=(
            "All HIGH-risk production changes must receive Change Advisory Board (CAB) approval."
        ),
        available_files=[
            "change_management_ticket.json",
            "q3_sprint_planning.json",
            "office_supply_orders.json",
            "lunch_menu_october.json",
        ],
        relevant_files=["change_management_ticket.json"],
        correct_decision="REJECT",
        correct_reason="MISSING_APPROVAL",
        difficulty="medium",
        description="HIGH-risk DB change implemented without CAB approval.",
    ),
    AuditTask(
        task_id="password_policy_compliance",
        control_requirement=(
            "Password policy audit must show 100% compliance across all accounts."
        ),
        available_files=[
            "password_policy_audit.json",
            "it_helpdesk_tickets.json",
            "q4_headcount_plan.json",
            "lunch_menu_october.json",
        ],
        relevant_files=["password_policy_audit.json"],
        correct_decision="REJECT",
        correct_reason="POLICY_VIOLATION",
        difficulty="medium",
        description="Only 68.5% compliance, policy requires 100% — mandatory policy threshold not met.",
    ),
    # ── Compliant variants — APPROVE tasks for training balance ───────────
    AuditTask(
        task_id="pr_approval_compliant",
        control_requirement=(
            "All code changes to production branches require documented peer review "
            "approval from at least one team member before merging. "
            "Merges without recorded approvals are non-compliant."
        ),
        available_files=[
            "pull_request_log_compliant.json",   # RELEVANT — 2 approvals, compliant
            "slack_engineering_channel.json",     # distractor
            "q3_sprint_planning.json",            # distractor
            "ci_cd_pipeline_config.json",         # distractor
            "lunch_menu_october.json",            # distractor
        ],
        relevant_files=["pull_request_log_compliant.json"],
        correct_decision="APPROVE",
        correct_reason="NONE",
        difficulty="easy",
        description=(
            "PR-2024-0892 has approved_by='bob-lead' and approvals_count=2. "
            "Fully compliant. Tests whether the agent can APPROVE when evidence is clean "
            "rather than reflexively REJECT."
        ),
    ),
    AuditTask(
        task_id="access_revocation_sla_borderline",
        control_requirement=(
            "Upon employee termination, all system access must be revoked within 24 hours "
            "of the recorded termination datetime, as mandated by HR policy ITC-07. "
            "Evidence must include both the HR termination record and the IAM audit log."
        ),
        available_files=[
            "hr_termination_ticket.json",     # RELEVANT
            "aws_iam_audit_log_26h.json",     # RELEVANT — 26h, just over SLA
            "it_helpdesk_tickets.json",       # distractor
            "employee_handbook.json",         # distractor
            "slack_hr_channel.json",          # distractor
            "office_supply_orders.json",      # distractor
        ],
        relevant_files=["hr_termination_ticket.json", "aws_iam_audit_log_26h.json"],
        correct_decision="REJECT",
        correct_reason="SLA_VIOLATION",
        difficulty="medium",
        description=(
            "Access removed 26 hours after termination — only 2 hours over the 24h SLA. "
            "Borderline case that tests whether the agent applies the rule literally (26 > 24)."
        ),
    ),
    AuditTask(
        task_id="access_revocation_sla_compliant",
        control_requirement=(
            "Upon employee termination, all system access must be revoked within 24 hours "
            "of the recorded termination datetime, as mandated by HR policy ITC-07. "
            "Evidence must include both the HR termination record and the IAM audit log."
        ),
        available_files=[
            "hr_termination_ticket_carol.json",   # RELEVANT
            "aws_iam_audit_log_compliant.json",   # RELEVANT — 18h, within SLA
            "it_helpdesk_tickets.json",           # distractor
            "employee_handbook.json",             # distractor
            "office_supply_orders.json",          # distractor
            "lunch_menu_october.json",            # distractor
        ],
        relevant_files=["hr_termination_ticket_carol.json", "aws_iam_audit_log_compliant.json"],
        correct_decision="APPROVE",
        correct_reason="NONE",
        difficulty="medium",
        description=(
            "Carol Singh terminated 2024-10-01T09:00Z. Access removed 18h later — within SLA. "
            "Fully compliant. Tests that agent doesn't REJECT compliant SLA evidence."
        ),
    ),
    AuditTask(
        task_id="multi_system_access_revocation_compliant",
        control_requirement=(
            "Upon employee termination, access to ALL production systems — including "
            "AWS, GitHub, and the Production Database — must be revoked within 24 hours. "
            "Evidence must confirm revocation across every system. "
            "Partial revocation (access remaining in any one system) is non-compliant."
        ),
        available_files=[
            "hr_terminations.json",            # RELEVANT
            "aws_users.json",                  # RELEVANT — alice_dev removed ✓
            "github_users.json",               # RELEVANT — alice_dev removed ✓
            "prod_db_users_compliant.json",    # RELEVANT — alice_dev removed ✓
            "slack_workspace_users.json",      # distractor
            "zoom_license_report.json",        # distractor
            "jira_project_members.json",       # distractor
            "employee_benefits_portal.json",   # distractor
        ],
        relevant_files=[
            "hr_terminations.json",
            "aws_users.json",
            "github_users.json",
            "prod_db_users_compliant.json",
        ],
        correct_decision="APPROVE",
        correct_reason="NONE",
        difficulty="hard",
        description=(
            "alice_dev terminated. AWS, GitHub, AND prod DB access all revoked. "
            "All 4 system files show proper revocation. Agent must inspect all systems "
            "and APPROVE — not REJECT — when evidence is complete across all systems."
        ),
    ),
    # ── Task 11: CloudTrail Privileged Access Audit (requires SEARCH_LOGS) ──
    AuditTask(
        task_id="cloudtrail_privileged_access_audit",
        control_requirement=(
            "All AWS credentials — including programmatic API keys and access tokens — "
            "must be fully revoked within 24 hours of employee termination. "
            "Any API call authenticated under employee credentials after their recorded "
            "termination datetime constitutes an incomplete revocation violation."
        ),
        available_files=[
            "hr_terminations.json",             # RELEVANT — alice_dev terminated 2024-10-01T17:00:00Z
            "aws_cloudtrail_full_log.json",     # RELEVANT (LARGE) — 28 events, query with SEARCH_LOGS
            "slack_workspace_users.json",       # distractor
            "zoom_license_report.json",         # distractor
            "jira_project_members.json",        # distractor
            "employee_benefits_portal.json",    # distractor
        ],
        relevant_files=["hr_terminations.json", "aws_cloudtrail_full_log.json"],
        correct_decision="REJECT",
        correct_reason="INCOMPLETE_REVOCATION",
        difficulty="hard",
        description=(
            "alice_dev terminated 2024-10-01T17:00:00Z. Console access was revoked but API key "
            "was not. CloudTrail shows alice_dev API calls on 2024-10-02 (CT-011, CT-014) and "
            "2024-10-03 (CT-017), proving incomplete credential revocation. "
            "Agent must INSPECT hr_terminations.json to find username, then "
            "SEARCH_LOGS aws_cloudtrail_full_log.json with username=alice_dev to find violations."
        ),
    ),
]

ALL_TASKS: List[AuditTask] = GRADED_TASKS + EXTRA_TASKS


# ---------------------------------------------------------------------------
# Deterministic grader
# ---------------------------------------------------------------------------

def grade_decision(
    task: AuditTask,
    decision: str,
    reason: str,
    inspected_any_file: bool,
    num_inspected: int,
) -> Tuple[float, str]:
    """
    Compute the SUBMIT_DECISION step reward. 100% deterministic.

    The environment uses this logic inline; this function exists for
    external evaluation and testing.

    Reward table:
        Correct decision + correct reason          +(1.0 - accumulated_inspect_reward)
        Correct decision + wrong reason            +0.2
        Wrong decision: APPROVE non-compliant      -0.5
        Wrong decision: REJECT compliant           -0.3

    The accumulated inspect reward (max 0.3) is tracked in the environment.
    Total max = 0.3 (inspect cap) + 0.7 (correct submit) = 1.0
    """
    if decision == task.correct_decision:
        if reason == task.correct_reason:
            base = 0.7  # placeholder — real value = 1.0 - accumulated_inspect_reward
            msg = (
                f"CORRECT AUDIT DECISION: {decision} with reason '{reason}'. "
                f"Inspected {num_inspected} file(s). Full submission credit awarded."
            )
        else:
            base = 0.2
            msg = (
                f"PARTIALLY CORRECT: '{decision}' is right but reason '{reason}' is wrong. "
                f"Expected '{task.correct_reason}'. Partial credit only."
            )
    else:
        if task.correct_decision == "REJECT" and decision == "APPROVE":
            base = -0.5
            msg = (
                f"CRITICAL ERROR: You APPROVED non-compliant evidence. "
                f"Correct answer: REJECT with reason '{task.correct_reason}'. "
                f"This would constitute an audit failure in a real SOC 2 review."
            )
        else:
            base = -0.3
            msg = "INCORRECT: You REJECTED evidence that should be APPROVED."

    return (base, msg)
