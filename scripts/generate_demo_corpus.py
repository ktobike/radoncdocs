from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demo"
POLICY_DIR = DATA / "Policies and Procedures"
JOURNAL_DIR = DATA / "RedJournal"
LOGO = ROOT / "ui" / "assets" / "radoncdocs-mark.png"
MANIFEST = DATA / "manifest.json"

NAVY = colors.HexColor("#12213A")
BURGUNDY = colors.HexColor("#8B3D48")
BURGUNDY_DARK = colors.HexColor("#6F2F3A")
LAVENDER = colors.HexColor("#E9E7F0")
INK = colors.HexColor("#202635")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#D9DEE8")
PALE = colors.HexColor("#F6F7FA")
WHITE = colors.white

DISCLAIMER = "Synthetic portfolio demo document. Not clinical, legal, security, or operational guidance."


def slug_filename(title: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9 _-]+", "", title).strip()
    return f"{safe}.pdf"


def _styles():
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle", parent=s["Title"], fontName="Helvetica-Bold", fontSize=20,
            leading=24, textColor=NAVY, spaceAfter=10, alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=s["Normal"], fontName="Helvetica", fontSize=9.5,
            leading=13, textColor=MUTED, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1", parent=s["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
            leading=14, textColor=BURGUNDY_DARK, spaceBefore=10, spaceAfter=5,
            uppercase=True,
        ),
        "body": ParagraphStyle(
            "Body", parent=s["BodyText"], fontName="Helvetica", fontSize=9.6,
            leading=14.2, textColor=INK, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=s["BodyText"], fontName="Helvetica", fontSize=9.5,
            leading=14, leftIndent=13, firstLineIndent=-8, textColor=INK, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "Small", parent=s["BodyText"], fontName="Helvetica", fontSize=8.3,
            leading=11, textColor=MUTED,
        ),
        "abstract": ParagraphStyle(
            "Abstract", parent=s["BodyText"], fontName="Helvetica", fontSize=9.6,
            leading=14.2, textColor=INK, leftIndent=8, rightIndent=8, spaceAfter=8,
        ),
    }


def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = letter
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(0.65 * inch, h - 0.52 * inch, w - 0.65 * inch, h - 0.52 * inch)
    canvas.setFont("Helvetica-Bold", 7.6)
    canvas.setFillColor(NAVY)
    canvas.drawString(0.68 * inch, h - 0.40 * inch, "RadOncDocs - DEMO CORPUS")
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(w - 0.68 * inch, h - 0.40 * inch, "Synthetic documents")

    canvas.setStrokeColor(LINE)
    canvas.line(0.65 * inch, 0.52 * inch, w - 0.65 * inch, 0.52 * inch)
    canvas.setFont("Helvetica", 6.9)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.68 * inch, 0.34 * inch, DISCLAIMER)
    canvas.drawRightString(w - 0.68 * inch, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _doc(path: Path, title: str, author: str, subject: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = BaseDocTemplate(
        str(path), pagesize=letter,
        leftMargin=0.72 * inch, rightMargin=0.72 * inch,
        topMargin=0.72 * inch, bottomMargin=0.72 * inch,
        title=title, author=author, subject=subject,
    )
    frame = Frame(d.leftMargin, d.bottomMargin, d.width, d.height, id="normal")
    d.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=_header_footer))
    return d


def meta_table(st, rows):
    data = []
    for label, value in rows:
        data.append([
            Paragraph(f"<b>{label}</b>", st["small"]),
            Paragraph(value, st["small"]),
        ])
    t = Table(data, colWidths=[1.2 * inch, 5.55 * inch], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def add_section(story, st, heading: str, content):
    story.append(Paragraph(heading.upper(), st["h1"]))
    if isinstance(content, str):
        story.append(Paragraph(content, st["body"]))
    else:
        for item in content:
            value = str(item)
            if re.match(r"^\d+[.)]\s+", value):
                story.append(Paragraph(value, st["bullet"]))
            else:
                story.append(Paragraph(f"• {value}", st["bullet"]))


POLICIES = [
    {
        "title": "Remote Access and MFA Policy",
        "category": "Access & Security", "owner": "Research IT", "updated": "2026-08-18",
        "description": "Requirements for remote access to research systems, including approval, MFA, and user responsibilities.",
        "purpose": "Define the minimum controls required when staff access Demo Health research systems from outside an approved campus network.",
        "scope": "Applies to staff, trainees, collaborators, and service accounts that connect to research applications, file shares, compute resources, or review tools from a remote network.",
        "responsibilities": [
            "Users must protect credentials, use approved devices, complete multi-factor authentication, and report suspected access issues to Research IT.",
            "Managers approve business need and verify that access remains appropriate during quarterly reviews.",
            "Research IT provisions least-privilege access, maintains authentication controls, and records access changes in the audit log.",
        ],
        "procedure": [
            "Submit an access request through the Demo Service Portal and identify the systems needed.",
            "Obtain manager or project-owner approval before provisioning begins.",
            "Enroll the approved device and account in multi-factor authentication.",
            "Verify access with the project owner before handling restricted or study-specific data.",
            "Use the institutional VPN when a system is not exposed through an approved secure gateway.",
            "Report lost devices, suspected credential compromise, or unusual sign-in prompts immediately.",
        ],
        "requirements": [
            "Shared accounts are not permitted for routine interactive access.",
            "Remote sessions must lock automatically after 15 minutes of inactivity.",
            "Downloading restricted data to unmanaged personal storage is prohibited.",
        ],
        "exceptions": "Emergency access may be granted for time-sensitive research operations, but it must be documented and reviewed by Research IT within one business day.",
    },
    {
        "title": "Approved Device Security Standard",
        "category": "Access & Security", "owner": "Research IT", "updated": "2026-07-29",
        "description": "Baseline controls for laptops and workstations used with research data and internal applications.",
        "purpose": "Establish a consistent security baseline for endpoints that access synthetic-demo equivalents of research systems.",
        "scope": "Covers institution-managed laptops, approved workstations, and project-owned systems used for development, review, or analysis.",
        "responsibilities": [
            "Device owners keep operating systems and supported applications patched.",
            "Users must not disable endpoint protection, disk encryption, or screen-lock controls.",
            "Research IT maintains the approved-device inventory and investigates compliance failures.",
        ],
        "procedure": [
            "Register the device in the project inventory before first use.",
            "Confirm full-disk encryption and endpoint protection are enabled.",
            "Apply critical security updates within seven days of release unless a documented compatibility hold exists.",
            "Remove local copies of restricted data when the project no longer requires them.",
            "Revalidate device compliance after major operating-system upgrades.",
        ],
        "requirements": [
            "Automatic screen lock must be enabled.",
            "Local administrator use should be limited to approved maintenance tasks.",
            "Unsupported operating systems may not connect to protected research resources.",
        ],
        "exceptions": "Specialized imaging workstations may receive a temporary patch deferral when validated software requires a fixed operating-system version.",
    },
    {
        "title": "Imaging Data De-identification SOP",
        "category": "Data Governance", "owner": "Imaging Informatics", "updated": "2026-08-30",
        "description": "Procedure for creating de-identified imaging datasets before research development or external sharing.",
        "purpose": "Reduce disclosure risk by removing direct identifiers and replacing study identifiers before imaging data enter the research workspace.",
        "scope": "Applies to CT, MR, dose, structure, and derived image objects used in model development, quality assurance, or demonstration workflows.",
        "responsibilities": [
            "The data preparer runs the approved de-identification profile and preserves only fields required for the research task.",
            "A second reviewer verifies a sample of output files before release to the project workspace.",
            "The project owner documents any retained date or device fields that are scientifically necessary.",
        ],
        "procedure": [
            "Copy source data into the restricted staging area and record the intake batch identifier.",
            "Run the approved DICOM de-identification profile and replace patient identifiers with project-specific study IDs.",
            "Remove burned-in annotations or quarantine images that cannot be safely cleaned.",
            "Validate that expected study, series, geometry, and modality fields remain intact.",
            "Review a random sample of at least ten studies or ten percent of the batch, whichever is larger.",
            "Move the verified batch into the research workspace and retain the re-identification key only in the restricted linkage location.",
        ],
        "requirements": [
            "Free-text comments and secondary-capture images require additional review.",
            "Re-identification keys must never be stored beside the de-identified dataset.",
            "Failures are logged and the affected files remain in staging until resolved.",
        ],
        "exceptions": "Projects with an approved protocol requiring limited identifiers must document the retained fields and access controls before processing.",
    },
    {
        "title": "Imaging Data Intake Procedure",
        "category": "Data Operations", "owner": "Data Engineering", "updated": "2026-09-02",
        "description": "Standard intake checks for new imaging batches before model development or review.",
        "purpose": "Ensure new imaging batches are complete, traceable, and technically usable before they are accepted into a research dataset.",
        "scope": "Applies to imported imaging studies, derived segmentations, dose objects, and associated manifests.",
        "responsibilities": [
            "Data Engineering verifies file integrity, required modalities, and manifest consistency.",
            "Project owners resolve ambiguous cohort membership or missing expected series.",
            "Model developers consume only batches that have passed intake status checks.",
        ],
        "procedure": [
            "Create a batch folder using the approved naming convention and record source, date, and expected study count.",
            "Run checksum and file-readability validation across all received objects.",
            "Compare the observed study count and modality inventory with the intake manifest.",
            "Flag duplicate study identifiers, missing series, corrupt objects, and unsupported transfer syntaxes.",
            "Generate an intake report and resolve blocking issues before marking the batch Ready.",
        ],
        "requirements": [
            "Every accepted batch must have a machine-readable manifest.",
            "Blocking QC failures must be resolved or explicitly excluded.",
            "Intake reports are retained with the dataset version.",
        ],
        "exceptions": "Exploratory one-off samples may be marked Experimental, but they cannot be used for formal evaluation until full intake checks pass.",
    },
    {
        "title": "Contour Annotation QA Procedure",
        "category": "Annotation & Review", "owner": "Clinical AI Operations", "updated": "2026-08-12",
        "description": "Quality checks for segmentation annotations before training or evaluation.",
        "purpose": "Define a repeatable process for detecting missing, mislabeled, geometrically invalid, or inconsistent segmentation annotations.",
        "scope": "Applies to manual and model-assisted annotations prepared for segmentation training, validation, or blinded review.",
        "responsibilities": [
            "Annotators follow the project label dictionary and document intentionally absent structures.",
            "QA reviewers investigate automated validation flags and record final disposition.",
            "Dataset maintainers publish only annotation versions that have completed QA.",
        ],
        "procedure": [
            "Verify every case maps to a valid image study and expected label set.",
            "Check for empty masks, duplicate labels, disconnected islands, and coordinates outside the image volume.",
            "Compare mask geometry with the corresponding image orientation and spacing.",
            "Review flagged cases visually and correct confirmed annotation errors.",
            "Record exclusions, corrections, reviewer initials, and the resulting dataset version.",
        ],
        "requirements": [
            "Label identifiers must be stable across training folds.",
            "Automated checks do not replace visual review for high-impact abnormalities.",
            "Corrections must create a new version rather than silently overwrite a released dataset.",
        ],
        "exceptions": "Pilot labels used only for pipeline development may bypass full review if they are clearly marked non-evaluation data.",
    },
    {
        "title": "Experimental Model Review Procedure",
        "category": "Model Governance", "owner": "AI Evaluation", "updated": "2026-09-04",
        "description": "Blinded review workflow for experimental model outputs before any downstream study decision.",
        "purpose": "Standardize how reviewers inspect experimental model outputs while preserving blinding, traceability, and comparable scoring.",
        "scope": "Applies to internal feasibility studies comparing model-generated outputs with reference or clinician-generated results.",
        "responsibilities": [
            "The study coordinator prepares blinded cases and confirms randomization before review begins.",
            "Reviewers score only the displayed output and must not attempt to infer model identity.",
            "The analysis lead locks the review dataset before unblinding or aggregate analysis.",
        ],
        "procedure": [
            "Confirm the study manifest, case count, structures, and review rubric are finalized.",
            "Generate blinded labels for each output and randomize presentation order.",
            "Load the assigned case in the review portal and verify the image and overlay render correctly.",
            "Score acceptability using the study rubric and record optional comments for visible errors.",
            "Submit each case before moving to the next item; submitted scores remain read-only to the reviewer.",
            "Lock the dataset and export the blinded results for analysis after all assigned reviews are complete.",
        ],
        "requirements": [
            "Reviewers must not see source/model identity during blinded scoring.",
            "The portal records reviewer, case, timestamp, score, and study version.",
            "Rendering or alignment problems are reported as technical issues rather than scored as model failures.",
        ],
        "exceptions": "A case may be removed from analysis when a documented technical display failure prevents valid review.",
    },
    {
        "title": "Model Validation and Release Checklist",
        "category": "Model Governance", "owner": "AI Evaluation", "updated": "2026-08-25",
        "description": "Required validation evidence before an experimental model can be promoted to a shared research workflow.",
        "purpose": "Provide a release gate so model artifacts, preprocessing, evaluation, and known failure modes are documented before shared use.",
        "scope": "Applies to research models moving from individual development into a team-accessible inference or review environment.",
        "responsibilities": [
            "Model owners prepare the validation package and reproduce the final evaluation run.",
            "A reviewer not responsible for the final training run checks metrics, data split integrity, and artifact versioning.",
            "The platform owner confirms deployment configuration and rollback instructions.",
        ],
        "procedure": [
            "Freeze the model checkpoint, configuration, preprocessing version, and dataset split identifiers.",
            "Re-run held-out evaluation using the release candidate artifact.",
            "Compare primary metrics with the acceptance criteria defined before evaluation.",
            "Review representative successes and failures, including edge cases and known weak cohorts.",
            "Verify inference can be reproduced from a clean environment using documented commands.",
            "Approve, reject, or return the candidate with required remediation items.",
        ],
        "requirements": [
            "Release notes must include intended use, excluded use, known limitations, and artifact checksum.",
            "Training data and held-out evaluation data must remain separated.",
            "A model cannot be promoted solely because aggregate metrics improved.",
        ],
        "exceptions": "Exploratory checkpoints may be shared within a development sandbox if clearly labeled experimental and excluded from formal study use.",
    },
    {
        "title": "Incident Reporting and Escalation Procedure",
        "category": "Access & Security", "owner": "Research IT", "updated": "2026-06-17",
        "description": "How to report suspected security, privacy, data-integrity, or system-availability incidents.",
        "purpose": "Ensure suspected incidents are reported quickly, preserved for investigation, and escalated according to impact.",
        "scope": "Covers account compromise, accidental data exposure, lost devices, unexpected data changes, and service outages affecting research workflows.",
        "responsibilities": [
            "Any user who notices a suspected incident reports it immediately rather than attempting an undocumented fix.",
            "Research IT triages technical impact and preserves relevant logs.",
            "Project owners assess study impact and coordinate any required data or workflow hold.",
        ],
        "procedure": [
            "Stop the affected activity if continuing could increase exposure or corrupt data.",
            "Record what was observed, when it occurred, the system involved, and any actions already taken.",
            "Contact Research IT through the incident channel and mark urgent events appropriately.",
            "Do not delete logs, wipe devices, or modify affected files unless directed by the incident lead.",
            "Follow remediation instructions and document when normal activity resumes.",
        ],
        "requirements": [
            "Potential credential theft and unintended restricted-data disclosure are treated as urgent.",
            "Incident records must distinguish confirmed impact from suspected impact.",
            "Post-incident actions are tracked to closure.",
        ],
        "exceptions": "Routine software bugs without security, data-integrity, or availability impact follow the standard issue-tracking workflow instead.",
    },
    {
        "title": "Data Sharing Request Policy",
        "category": "Data Governance", "owner": "Research Operations", "updated": "2026-07-08",
        "description": "Approval and packaging requirements before research data or model outputs are shared outside a project workspace.",
        "purpose": "Create a documented decision path for internal cross-team and external data-sharing requests.",
        "scope": "Applies to raw data, de-identified data, derived labels, model outputs, evaluation exports, and associated metadata.",
        "responsibilities": [
            "Requestors identify the recipient, purpose, requested fields, and required duration.",
            "Project owners confirm the request is consistent with study and data-use constraints.",
            "Data Engineering prepares the approved package and records the transfer checksum.",
        ],
        "procedure": [
            "Submit a sharing request describing recipient, purpose, files, and expected retention period.",
            "Confirm the minimum necessary data elements and remove unrelated fields or files.",
            "Verify de-identification status and any additional project restrictions.",
            "Obtain documented project-owner approval before packaging.",
            "Transfer through an approved mechanism and record package checksum and completion date.",
        ],
        "requirements": [
            "Personal email and consumer file-sharing accounts are not approved transfer methods.",
            "Recipients should receive only the approved package, not broad workspace access.",
            "Sharing records are retained with the project documentation.",
        ],
        "exceptions": "Public synthetic demonstration data may be shared without project-specific approval when it contains no restricted source material.",
    },
    {
        "title": "Research Workspace Access Matrix",
        "category": "Access & Security", "owner": "Research Operations", "updated": "2026-05-21",
        "description": "Role-based access expectations for development, staging, review, and archive workspaces.",
        "purpose": "Define least-privilege defaults across common research workspace roles.",
        "scope": "Applies to project developers, data engineers, reviewers, coordinators, and read-only collaborators.",
        "responsibilities": [
            "Project owners assign roles based on current responsibilities rather than anticipated future need.",
            "Workspace administrators review memberships quarterly.",
            "Users request temporary elevation when a task requires capabilities outside their standard role.",
        ],
        "procedure": [
            "Select the role that matches the user's primary project function.",
            "Grant development write access only to users maintaining code or model artifacts.",
            "Grant review access to users scoring cases without exposing development-only metadata.",
            "Time-limit temporary administrative access and record the reason for elevation.",
            "Remove access when a user leaves the project or no longer needs the role.",
        ],
        "requirements": [
            "Reviewer accounts must not inherit model-training privileges by default.",
            "Service accounts require a named owner.",
            "Dormant accounts are disabled after the configured inactivity threshold.",
        ],
        "exceptions": "Small pilot teams may combine development and operations roles, but privileged actions should remain attributable to an individual account.",
    },
    {
        "title": "File Naming and Versioning Standard",
        "category": "Data Operations", "owner": "Data Engineering", "updated": "2026-08-05",
        "description": "Naming and versioning conventions for datasets, model outputs, reports, and review exports.",
        "purpose": "Make project artifacts easy to trace, compare, and reproduce without relying on informal folder knowledge.",
        "scope": "Applies to shared datasets, model checkpoints, inference outputs, review exports, manifests, and reports.",
        "responsibilities": [
            "Artifact creators use stable identifiers and avoid ambiguous names such as final, latest, or new.",
            "Dataset maintainers increment versions when content changes.",
            "Analysis owners record the exact input versions used for reported results.",
        ],
        "procedure": [
            "Use the pattern project_artifact_YYYYMMDD_vNN for release-like artifacts.",
            "Keep immutable released versions separate from temporary working files.",
            "Record model checkpoint, preprocessing version, and dataset version in inference manifests.",
            "Include reviewer-study version and export timestamp in blinded review outputs.",
            "Archive superseded release artifacts rather than silently replacing them.",
        ],
        "requirements": [
            "Identifiers must be machine-safe and should avoid spaces when consumed by automated pipelines.",
            "The same version label must never refer to two different contents.",
            "Checksums are recommended for model and dataset releases.",
        ],
        "exceptions": "Local scratch files may use shorter names, but they must not be mistaken for released artifacts.",
    },
    {
        "title": "Audit Logging and Review Standard",
        "category": "Access & Security", "owner": "Platform Engineering", "updated": "2026-07-14",
        "description": "Minimum events and review practices for internal research applications that modify study state.",
        "purpose": "Preserve enough application history to reconstruct important changes and review actions.",
        "scope": "Applies to internal applications that manage uploads, model runs, study assignments, scores, or administrative changes.",
        "responsibilities": [
            "Application owners identify high-value events and store timestamps, actor identifiers, object identifiers, and outcomes.",
            "Administrators investigate failed privileged actions and unusual bursts of changes.",
            "Project teams retain logs long enough to support study reconstruction and debugging.",
        ],
        "procedure": [
            "Log successful and failed sign-in events when authentication is handled by the application.",
            "Log create, update, delete, upload, export, model-run, and review-submission events.",
            "Avoid writing secrets, passwords, tokens, or unnecessary sensitive payloads to logs.",
            "Review privileged-action logs monthly for active study systems.",
            "Export or archive logs before decommissioning a study environment.",
        ],
        "requirements": [
            "System clocks should be synchronized so events can be correlated.",
            "Logs should be append-oriented and protected from routine user modification.",
            "Application errors should include enough context for debugging without leaking confidential content.",
        ],
        "exceptions": "Short-lived local prototypes may use simplified logging until they accept shared data or multi-user activity.",
    },
    {
        "title": "HPC Job Submission Guidelines",
        "category": "Compute & Infrastructure", "owner": "Research Computing", "updated": "2026-06-28",
        "description": "Resource, logging, and reproducibility expectations for model training and large inference jobs.",
        "purpose": "Help research jobs use shared compute predictably while preserving enough information to reproduce completed runs.",
        "scope": "Applies to scheduled GPU and CPU jobs used for preprocessing, training, evaluation, and inference.",
        "responsibilities": [
            "Job owners request realistic resources, monitor failed jobs, and clean unnecessary temporary artifacts.",
            "Research Computing maintains queues, node health, and published limits.",
            "Project leads define priority when multiple project jobs compete for limited accelerators.",
        ],
        "procedure": [
            "Estimate memory, CPU, GPU, and wall-time needs from a small representative run.",
            "Submit jobs with explicit environment, code revision, configuration, and output location.",
            "Write stdout and stderr logs to a persistent run directory.",
            "Use checkpointing for training jobs that exceed routine queue windows.",
            "Record exit status and primary metrics in the experiment manifest.",
        ],
        "requirements": [
            "Do not request GPUs for CPU-only preprocessing tasks.",
            "Interactive sessions are for debugging and short experiments, not unattended long-running training.",
            "Large temporary files should use designated scratch storage.",
        ],
        "exceptions": "Urgent debugging may temporarily exceed normal interactive limits when coordinated with Research Computing.",
    },
    {
        "title": "Model Inference Runbook",
        "category": "Model Operations", "owner": "AI Platform", "updated": "2026-09-01",
        "description": "Repeatable steps for running a released research model across a new dataset batch.",
        "purpose": "Reduce inference errors by standardizing environment checks, input validation, output manifests, and post-run QC.",
        "scope": "Applies to batch inference performed with a frozen research-model artifact.",
        "responsibilities": [
            "The operator confirms the model version and dataset batch before starting.",
            "The pipeline validates inputs and records failures without silently skipping them.",
            "The project owner reviews the run report before outputs enter a formal evaluation workflow.",
        ],
        "procedure": [
            "Activate the documented environment and verify the expected model checksum.",
            "Validate the input manifest, required modalities, and preprocessing prerequisites.",
            "Run inference with a unique run identifier and write outputs to a new versioned directory.",
            "Capture per-case status, warnings, runtime, and output location in the inference manifest.",
            "Review failed and outlier cases before marking the run complete.",
            "Publish only the approved output directory to downstream review tools.",
        ],
        "requirements": [
            "Inference runs must be reproducible from the recorded code revision and configuration.",
            "Existing released outputs are never overwritten in place.",
            "Geometry validation is required when outputs are displayed over source imaging.",
        ],
        "exceptions": "Single-case debugging runs may use temporary directories but cannot be mistaken for released batch outputs.",
    },
    {
        "title": "Dataset Quality Control SOP",
        "category": "Data Operations", "owner": "Data Engineering", "updated": "2026-08-22",
        "description": "Dataset-level QC checks before a cohort is frozen for training or evaluation.",
        "purpose": "Detect cohort, file, label, and split problems before they become model-training or evaluation errors.",
        "scope": "Applies to curated imaging datasets, associated labels, metadata, and predefined training or evaluation splits.",
        "responsibilities": [
            "Dataset maintainers run automated QC and review the resulting exceptions.",
            "Model developers report unexpected input patterns rather than patching around undocumented errors.",
            "Project owners approve exclusions that change the analysis cohort.",
        ],
        "procedure": [
            "Verify expected case count and unique study identifiers.",
            "Check for duplicate images, duplicate labels, missing required files, and unreadable objects.",
            "Validate image dimensions, spacing, orientation, and label geometry ranges.",
            "Confirm split membership is mutually exclusive and that held-out cases do not appear in training data.",
            "Review class and cohort distributions for unexpected imbalance after exclusions.",
            "Freeze the manifest and QC report together as the dataset release.",
        ],
        "requirements": [
            "All exclusions require a documented reason.",
            "QC code and thresholds are versioned with the dataset release.",
            "Held-out evaluation membership should remain stable once formal evaluation begins.",
        ],
        "exceptions": "Exploratory datasets may change frequently, but formal metrics should reference a frozen manifest.",
    },
    {
        "title": "Research Data Retention Policy",
        "category": "Data Governance", "owner": "Research Operations", "updated": "2026-04-16",
        "description": "Retention and disposition expectations for active, archived, and temporary research artifacts.",
        "purpose": "Keep required research records while reducing unnecessary accumulation of temporary or duplicated data.",
        "scope": "Applies to project datasets, derived model outputs, logs, review exports, manifests, and temporary processing artifacts.",
        "responsibilities": [
            "Project owners identify records that must remain available for study reconstruction.",
            "Data maintainers separate durable release artifacts from disposable caches and scratch data.",
            "Users do not independently delete shared study records without project-owner approval.",
        ],
        "procedure": [
            "Classify artifacts as active, release record, temporary, cache, or archive.",
            "Retain released datasets, manifests, final model artifacts, and formal review exports according to the project retention plan.",
            "Remove temporary intermediate files after validation when they can be regenerated safely.",
            "Review archived workspaces annually for continued retention need.",
            "Document final disposition when a project closes.",
        ],
        "requirements": [
            "Deletion scripts should support dry-run or inventory review before destructive action.",
            "Re-identification linkage files follow stricter restricted-storage controls.",
            "Backups do not replace a defined retention and disposition plan.",
        ],
        "exceptions": "Legal, contractual, or protocol-specific requirements may extend the normal project retention period.",
    },
    {
        "title": "Software Change Review Procedure",
        "category": "Software Engineering", "owner": "Platform Engineering", "updated": "2026-09-03",
        "description": "Lightweight review and validation process for changes to shared research applications.",
        "purpose": "Reduce regressions in shared tools while keeping research software iteration practical.",
        "scope": "Applies to backend, frontend, pipeline, configuration, and schema changes that affect shared environments.",
        "responsibilities": [
            "Change authors describe intent, risks, test evidence, and any migration or rollback needs.",
            "Reviewers focus on correctness, maintainability, security-sensitive behavior, and user impact.",
            "Application owners decide when a change requires staged validation before release.",
        ],
        "procedure": [
            "Create a focused branch or change set linked to the relevant issue or task.",
            "Run automated tests, formatting or lint checks, and a local functional test.",
            "Request review from at least one teammate for changes affecting shared behavior.",
            "Address review findings and document intentionally deferred issues.",
            "Deploy to a staging or local demo environment when the change affects user workflows.",
            "Confirm the health check and one representative end-to-end workflow after release.",
        ],
        "requirements": [
            "Secrets and restricted datasets must not be committed to version control.",
            "Breaking API or schema changes require a migration plan.",
            "Rollback should be possible for changes with meaningful operational risk.",
        ],
        "exceptions": "Small documentation-only changes may be merged without functional testing when they cannot alter runtime behavior.",
    },
    {
        "title": "Clinician Feedback Capture Procedure",
        "category": "Annotation & Review", "owner": "Clinical AI Operations", "updated": "2026-08-27",
        "description": "Consistent capture of reviewer comments, issue tags, and follow-up for AI evaluation studies.",
        "purpose": "Turn qualitative reviewer observations into traceable study feedback without altering blinded scoring rules.",
        "scope": "Applies to comments recorded during or immediately after structured model-output review.",
        "responsibilities": [
            "Reviewers use predefined issue tags where possible and avoid revealing patient identity in comments.",
            "Study coordinators triage technical issues separately from model-quality observations.",
            "Analysis leads summarize recurring themes only after the blinded review dataset is locked.",
        ],
        "procedure": [
            "Select the closest predefined issue tag before entering free-text detail.",
            "Describe the visible problem in terms of location, extent, or review impact rather than presumed model cause.",
            "Flag rendering, alignment, or missing-data issues as technical when they prevent valid scoring.",
            "Submit feedback with the associated blinded case and structure identifier.",
            "Group recurring feedback themes after study lock and link them to quantitative results where appropriate.",
        ],
        "requirements": [
            "Free-text fields should not contain direct identifiers.",
            "Technical display failures remain distinguishable from model-quality failures.",
            "Reviewer comments cannot be edited after study lock without an auditable correction record.",
        ],
        "exceptions": "General usability feedback not tied to a case may be recorded outside the blinded scoring workflow.",
    },
    {
        "title": "Review Portal Study Setup Guide",
        "category": "Study Operations", "owner": "AI Evaluation", "updated": "2026-09-05",
        "description": "Setup sequence for creating a new blinded evaluation study in the review portal.",
        "purpose": "Ensure study definitions, assignments, overlays, and scoring rules are complete before reviewers receive access.",
        "scope": "Applies to new feasibility and comparison studies configured in the internal review application.",
        "responsibilities": [
            "Study coordinators define the case set, review questions, and assignment plan.",
            "Technical owners validate image and overlay rendering using known reference cases.",
            "Reviewers receive access only after the study is marked Ready.",
        ],
        "procedure": [
            "Create the study record and assign a descriptive versioned name.",
            "Import the case manifest and validate that every case resolves to the expected imaging study.",
            "Attach reference and experimental outputs using blinded labels.",
            "Configure scoring fields, required comments, and reviewer assignments.",
            "Run a technical validation pass across representative cases and all supported view modes.",
            "Freeze the configuration and mark the study Ready before reviewer invitations are sent.",
        ],
        "requirements": [
            "Study configuration changes after review begins require a new version or documented amendment.",
            "Blinded labels must not encode model identity.",
            "At least one known-good case should be checked end to end before launch.",
        ],
        "exceptions": "Training-only mock studies may remain editable while reviewers learn the interface.",
    },
    {
        "title": "Imaging Overlay Validation SOP",
        "category": "Imaging Systems", "owner": "Demo Imaging Platform", "updated": "2026-09-06",
        "description": "Troubleshooting and validation steps for image overlays that appear shifted, flipped, rotated, or otherwise misaligned.",
        "purpose": "Provide a repeatable way to separate display, geometry, and coordinate-system problems before an overlay is used for model review or scoring.",
        "scope": "Applies to segmentation, dose, heatmap, and other derived overlays displayed on volumetric source imaging in research review tools.",
        "responsibilities": [
            "Reviewers flag visible alignment problems as technical issues and do not score the affected output until the display is validated.",
            "Technical owners compare source-image and overlay geometry, reproduce the issue on a known case, and document the corrective change.",
            "Study coordinators replace or revalidate affected review items before they return to an active blinded study.",
        ],
        "procedure": [
            "Confirm the source image and overlay belong to the same study, series, and released output version.",
            "Compare array dimensions, slice count, voxel spacing, and physical image bounds for the source image and overlay.",
            "Verify orientation vectors, image origin, slice order, and row-column or axis conventions before applying any coordinate transform.",
            "Run a known-point pixel-to-patient round-trip check and compare at least one landmark against a known-good reference case.",
            "If the mismatch remains, disable the overlay for review, capture a technical-issue record, and trace the transform from source coordinates through the displayed viewport.",
            "After a correction, rerun the geometry checks and regression case before the overlay is returned to reviewers.",
        ],
        "requirements": [
            "A visibly misaligned overlay must not be treated as a model-quality failure until the display pipeline is validated.",
            "Geometry checks should cover orientation, origin, spacing, dimensions, and coordinate transforms rather than array shape alone.",
            "Corrections to shared transform logic require a regression test using a known reference point or contour.",
        ],
        "exceptions": "Exploratory debugging views may display unvalidated overlays when they are clearly marked as technical diagnostics and are not used for formal scoring.",
    },
]

JOURNALS = [
    {
        "title": "Automated Contour Quality Assessment in a Multi-Reader Imaging Review Workflow",
        "category": "Segmentation", "owner": "Demo Imaging AI Group", "updated": "2026-07-18", "year": 2026,
        "description": "Synthetic retrospective study of automated quality signals for contour review.",
        "abstract": "Purpose: To evaluate whether lightweight geometric quality signals can prioritize segmentation cases for reviewer attention. In a synthetic retrospective cohort of 240 imaging studies, an automated quality score was compared with blinded reviewer assessments. The score identified 82% of cases rated as requiring major correction while preserving a 41% reduction in the number of cases needing immediate manual inspection.",
        "methods": "We assembled 240 de-identified synthetic imaging studies with one reference contour set and one perturbed contour set per case. Three reviewers assigned acceptability labels using a four-point rubric. Features included overlap, surface distance, connected-component count, volume ratio, and boundary irregularity. A logistic model was trained on 160 studies, tuned on 40, and evaluated on a held-out set of 40 studies. Thresholds were selected before held-out evaluation.",
        "results": "On the held-out set, the quality score achieved an area under the receiver operating characteristic curve of 0.89. At the selected operating point, sensitivity for major-correction cases was 0.82 and specificity was 0.76. Using the score as a triage tool reduced immediate manual review volume by 41% while missing 3 of 17 major-correction cases. Most false negatives involved small but clinically important boundary deviations that produced modest global geometric changes.",
        "discussion": "Automated quality features were useful for prioritization but were not sufficient to replace human review. The strongest value came from routing obvious low-risk cases away from the front of the queue while preserving reviewer attention for uncertain or unusual outputs. Errors clustered in cases where local shape differences mattered more than global overlap.",
        "conclusion": "A compact geometric quality model can support review triage in a controlled research workflow, but high-impact cases still require direct visual assessment.",
        "limitations": "The study used synthetic perturbations and a modest held-out cohort. The operating threshold should not be interpreted as clinically validated.",
    },
    {
        "title": "Deep Learning Dose Prediction for Partial Breast Irradiation: A Synthetic Feasibility Study",
        "category": "Dose Prediction", "owner": "Demo Radiation AI Group", "updated": "2026-08-06", "year": 2026,
        "description": "Synthetic feasibility study of a 3D neural network for voxel-level dose prediction.",
        "abstract": "Purpose: To assess the feasibility of three-dimensional dose prediction for partial breast irradiation using a compact convolutional neural network. A synthetic cohort of 96 treatment-planning studies was divided into training, validation, and held-out evaluation sets. The model achieved a held-out mean absolute error of 1.9 Gy and met predefined dose-constraint agreement in 90.8% of evaluated structures.",
        "methods": "The model consumed planning CT, target masks, and selected organ masks on a common voxel grid. Sixty-eight studies were used for training, 14 for validation, and 14 for held-out evaluation. Training used a combined voxel-wise absolute-error and structure-aware penalty. Experiments tracked preprocessing versions, random seeds, checkpoints, validation metrics, and constraint-level agreement. No held-out case was used for threshold tuning.",
        "results": "Held-out voxel-wise mean absolute error was 1.9 Gy. Constraint agreement across predefined target and organ metrics was 90.8%. Prediction error was lowest in high-dose target regions and larger near steep dose gradients and image-volume edges. Two held-out studies showed visually plausible dose distributions despite failing multiple structure-level criteria, illustrating that aggregate appearance alone was insufficient for validation.",
        "discussion": "The results support continued investigation of model-assisted dose estimation as a research tool. The evaluation also showed the importance of combining voxel-level error, structure-level constraints, and case-level visual review rather than relying on a single score.",
        "conclusion": "Three-dimensional dose prediction was technically feasible in this synthetic cohort, but release decisions should consider both numerical agreement and failure-mode review.",
        "limitations": "The cohort was synthetic and intentionally small. Results are not evidence of clinical performance or generalization to other treatment settings.",
    },
    {
        "title": "Clinician-AI Agreement in Blinded Segmentation Review",
        "category": "Human Evaluation", "owner": "Demo Clinical AI Group", "updated": "2026-08-21", "year": 2026,
        "description": "Synthetic blinded reader study comparing AI-generated and manually prepared segmentation outputs.",
        "abstract": "Purpose: To characterize reviewer agreement when evaluating segmentation outputs under blinded presentation. Twenty-five synthetic cases, three structures per case, and three specialist reviewers produced 225 assessments. Reviewers rated acceptability without knowing whether an output was generated by a model or prepared manually. Overall inter-reader agreement was moderate, and disagreements were concentrated in borderline boundary-placement cases.",
        "methods": "A browser-based review tool randomized output labels and presentation order. Each reviewer evaluated 75 case-structure combinations using the same ordinal acceptability rubric. The application recorded blinded item identifier, score, optional comment, and timestamp. Technical rendering failures were tracked separately and excluded before analysis. Agreement was summarized with percent exact agreement and a weighted multi-reader statistic.",
        "results": "Across 225 valid assessments, exact reader agreement on paired items was 78%. Reviewers were most consistent for clearly acceptable and clearly unacceptable examples. Borderline cases generated the largest spread of ratings and the longest comments. Removing technical display issues before scoring prevented several alignment problems from being misclassified as segmentation failures.",
        "discussion": "Blinding and structured data capture made model-versus-reference comparisons easier to interpret, but reader variability remained an important part of the result. The findings emphasize that reviewer disagreement can reflect ambiguous cases rather than simple model error.",
        "conclusion": "A structured blinded review workflow can support reproducible human evaluation of segmentation outputs while keeping technical failures distinct from model-quality judgments.",
        "limitations": "This synthetic study used a small number of reviewers and cases and was designed to demonstrate workflow behavior rather than clinical efficacy.",
    },
    {
        "title": "Uncertainty Estimation for Three-Dimensional Medical Image Segmentation",
        "category": "Segmentation", "owner": "Demo Imaging AI Group", "updated": "2026-06-12", "year": 2026,
        "description": "Synthetic study comparing uncertainty signals with segmentation-error patterns.",
        "abstract": "Purpose: To test whether model uncertainty maps identify regions likely to contain segmentation error. A synthetic dataset of 180 three-dimensional studies was evaluated using ensemble variance and test-time augmentation. High-uncertainty regions overlapped 71% of large boundary errors but were less informative for systematic small-volume bias.",
        "methods": "Five segmentation models were trained with identical architecture and different random seeds. For each held-out study, voxel-wise predictive variance was computed across ensemble outputs and augmented inference passes. Error regions were defined from reference masks using surface distance and connected-component rules. Uncertainty thresholds were selected on validation studies and fixed before evaluation.",
        "results": "High-uncertainty regions captured 71% of boundary errors larger than the predefined tolerance while covering 23% of segmented voxels. Performance was weaker for consistent under-segmentation near low-contrast boundaries, where all models made similar errors and therefore produced low disagreement. Case-level uncertainty ranking improved reviewer yield compared with random ordering.",
        "discussion": "Uncertainty is useful when models disagree, but model consensus does not guarantee correctness. Review systems should avoid presenting low uncertainty as a guarantee of quality and should combine uncertainty with geometric and data-quality checks.",
        "conclusion": "Uncertainty maps can help prioritize review, particularly for localized boundary failures, but should be treated as one signal among several.",
        "limitations": "The study used simulated reference perturbations and did not test prospective reviewer behavior.",
    },
    {
        "title": "Workflow Efficiency of Source-Cited Policy Search for Research Operations",
        "category": "Information Retrieval", "owner": "Demo Research Informatics", "updated": "2026-09-01", "year": 2026,
        "description": "Synthetic usability evaluation of hybrid lexical-semantic document retrieval with page citations.",
        "abstract": "Purpose: To compare manual document lookup with a source-cited hybrid search workflow for routine research-operations questions. Twelve synthetic policies and procedures were queried with 48 task questions. Hybrid BM25 and embedding retrieval reduced median time to locate the relevant passage from 96 seconds with manual browsing to 18 seconds with cited search while preserving direct access to the source page.",
        "methods": "The document set contained security, data-governance, model-review, and operations procedures. Six participants completed matched lookup tasks using manual PDF browsing and the search prototype in counterbalanced order. The system combined lexical and embedding retrieval, returned a short extractive answer, and displayed document title, section, page, and source snippet. Accuracy was scored against a pre-authored answer key.",
        "results": "Median lookup time was 18 seconds with the prototype and 96 seconds with manual browsing. Answer-key accuracy was 92% for cited search and 88% for manual browsing. Participants used the source panel most often for multi-step procedures and questions involving exceptions. Low-confidence retrieval failures occurred primarily when two procedures used similar language for different workflows.",
        "discussion": "The largest usability benefit came from narrowing the search space while keeping the supporting document visible. Citation presentation reduced the need to trust generated prose alone and made it easier to verify ambiguous answers.",
        "conclusion": "Hybrid retrieval with page-level citations can materially shorten routine policy lookup in a controlled synthetic environment.",
        "limitations": "The document set and participants were synthetic or simulated for portfolio demonstration. Timing values are illustrative and not institutional performance claims.",
    },
    {
        "title": "Multi-Label Segmentation of Overlapping Anatomy in Volumetric Imaging",
        "category": "Segmentation", "owner": "Demo Imaging AI Group", "updated": "2026-07-30", "year": 2026,
        "description": "Synthetic evaluation of multi-channel label handling for overlapping segmentation targets.",
        "abstract": "Purpose: To evaluate a multi-label segmentation pipeline for anatomy that cannot be represented reliably as mutually exclusive classes. A synthetic cohort of 482 volumetric imaging studies was evaluated with five-fold cross-validation across 15 overlapping targets. Multi-channel label handling improved representation of nested and intersecting structures and reduced preprocessing conflicts.",
        "methods": "Each target was encoded as an independent binary channel rather than a single exclusive class map. Preprocessing validated image geometry, channel count, label presence, and fold assignment. Five-fold cross-validation used the same training schedule and augmentation policy across folds. Primary evaluation summarized per-target Dice similarity and failure counts for missing or invalid outputs.",
        "results": "The multi-label representation eliminated class-overwrite errors observed in the exclusive-label baseline. Mean performance was stable across folds, while the largest remaining errors occurred in small structures with low contrast and inconsistent annotation boundaries. Automated validation caught 23 malformed label sets before training.",
        "discussion": "Representing overlapping targets explicitly simplified data preparation and reduced a class of silent label conflicts. The remaining performance variation was driven more by anatomy and annotation quality than by channel encoding.",
        "conclusion": "Multi-channel label handling is a practical approach for overlapping segmentation targets in volumetric research pipelines.",
        "limitations": "This synthetic study does not establish performance for any specific clinical anatomy or model architecture.",
    },
    {
        "title": "Human Factors in AI-Assisted Imaging Review: Interface Cues, Blinding, and Reviewer Confidence",
        "category": "Human Factors", "owner": "Demo Clinical AI Group", "updated": "2026-08-14", "year": 2026,
        "description": "Synthetic human-factors study of interface design choices in blinded AI review.",
        "abstract": "Purpose: To examine how interface cues influence reviewer confidence and scoring behavior during AI-assisted imaging evaluation. In a synthetic crossover study, 18 participants reviewed 72 image-output pairs using two interface variants. The variant with explicit source separation, persistent orientation labels, and a technical-issue pathway reduced accidental unblinding and improved confidence calibration.",
        "methods": "Participants completed matched review tasks in a basic viewer and a structured review interface. Both interfaces displayed identical image content. The structured interface separated score controls from technical issue reporting, preserved orientation context during navigation, and hid model-identifying metadata. Outcomes included task time, confidence, technical-error reporting, and accidental unblinding events.",
        "results": "The structured interface reduced accidental unblinding events from 11 to 2 and increased appropriate technical-issue reporting. Median task time changed minimally, but confidence was better aligned with reviewer agreement. Participants cited persistent orientation and clearer distinction between technical failures and output-quality scoring as the most useful changes.",
        "discussion": "Small interface decisions can affect the validity of human evaluation even when the underlying model output is unchanged. Review tools should treat blinding and technical-state communication as study-design features rather than cosmetic details.",
        "conclusion": "Purpose-built review interfaces can improve the reliability of blinded AI evaluation without materially increasing task time.",
        "limitations": "The study used synthetic cases and simulated workflow conditions rather than a production clinical environment.",
    },
    {
        "title": "Robust Coordinate Validation for Medical Imaging Overlays",
        "category": "Imaging Systems", "owner": "Demo Imaging Platform", "updated": "2026-09-03", "year": 2026,
        "description": "Synthetic engineering evaluation of coordinate-system checks for image-overlay alignment.",
        "abstract": "Purpose: To evaluate a coordinate-validation suite for detecting spatial misalignment between source images and derived overlays. Twenty synthetic failure cases were created by altering row-column direction handling, origin, spacing, or axis order. Boundary landmark tests detected all severe transformations and reduced the median time to isolate a coordinate bug from 46 minutes to 9 minutes in a controlled debugging exercise.",
        "methods": "The validation suite compared image dimensions, voxel spacing, orientation vectors, origin, expected physical bounds, and round-trip pixel-to-patient transformations. Synthetic faults included swapped row and column direction vectors, reversed slice order, incorrect origin offsets, and axis transposition. Engineers debugged matched cases with and without the validation report.",
        "results": "All 12 severe misalignment faults and 6 of 8 subtle faults were detected automatically. The largest geometric error came from a row-column direction swap that preserved file readability but displaced overlay coordinates. Adding known-point round-trip tests made the failure immediately reproducible and prevented recurrence in regression testing.",
        "discussion": "Spatial bugs often occur at boundaries between otherwise valid components. Validating coordinate assumptions explicitly is more reliable than checking only file loading or array shape.",
        "conclusion": "A small set of geometry invariants and known-point tests can materially improve reliability of imaging-overlay pipelines.",
        "limitations": "The experiment used synthetic transforms and did not cover every vendor-specific imaging edge case.",
    },
]


def write_policy(item: dict, path: Path):
    st = _styles()
    doc = _doc(path, item["title"], item["owner"], f"Synthetic policy - {item['category']}")
    story = []
    if LOGO.exists():
        head = Table([[Image(str(LOGO), width=0.52 * inch, height=0.53 * inch), Paragraph(item["title"], st["title"])]], colWidths=[0.68 * inch, 6.08 * inch], hAlign="LEFT")
        head.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 0), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
        story.append(head)
    else:
        story.append(Paragraph(item["title"], st["title"]))
    story.append(Paragraph(item["description"], st["subtitle"]))
    story.append(meta_table(st, [
        ("Document type", "Policy / Procedure"),
        ("Category", item["category"]),
        ("Owner", item["owner"]),
        ("Last updated", item["updated"]),
        ("Status", "Current - synthetic demo"),
    ]))
    story.append(Spacer(1, 8))
    add_section(story, st, "Purpose", item["purpose"])
    add_section(story, st, "Scope", item["scope"])
    add_section(story, st, "Responsibilities", item["responsibilities"])
    add_section(story, st, "Procedure", [f"{i+1}. {step}" for i, step in enumerate(item["procedure"])])
    add_section(story, st, "Requirements", item["requirements"])
    add_section(story, st, "Exceptions", item["exceptions"])
    add_section(story, st, "Related Documents", [
        "Use the current version of related security, data-quality, and software-change procedures when the workflow touches those areas.",
        "Where requirements conflict, the more restrictive documented control applies until the project owner resolves the discrepancy.",
    ])
    doc.build(story)


def write_journal(item: dict, path: Path):
    st = _styles()
    doc = _doc(path, item["title"], item["owner"], f"Synthetic research article - {item['category']}")
    story = []
    if LOGO.exists():
        head = Table([[Image(str(LOGO), width=0.52 * inch, height=0.53 * inch), Paragraph(item["title"], st["title"])]], colWidths=[0.68 * inch, 6.08 * inch], hAlign="LEFT")
        head.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 0), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
        story.append(head)
    else:
        story.append(Paragraph(item["title"], st["title"]))
    story.append(Paragraph(f"{item['owner']} · {item['year']} · Research article", st["subtitle"]))
    story.append(meta_table(st, [
        ("Article type", "Synthetic research article"),
        ("Topic", item["category"]),
        ("Year", str(item["year"])),
        ("Collection", "Demo Research Library"),
        ("Status", "Public demo"),
    ]))
    story.append(Spacer(1, 10))
    add_section(story, st, "ABSTRACT", item["abstract"])
    add_section(story, st, "INTRODUCTION", (
        "Research teams increasingly rely on software systems that combine imaging data, model outputs, and human review. "
        "This article belongs to the synthetic RadOncDocs demo corpus, created so search, citation, and summarization workflows can be exercised without exposing real institutional documents or patient information. "
        "The study design and numerical results are illustrative rather than clinical claims."
    ))
    add_section(story, st, "METHODS", item["methods"])
    add_section(story, st, "RESULTS", item["results"])
    add_section(story, st, "DISCUSSION", item["discussion"])
    add_section(story, st, "CONCLUSIONS", item["conclusion"])
    add_section(story, st, "LIMITATIONS", item["limitations"])
    add_section(story, st, "DATA AVAILABILITY", "All examples described in this article are synthetic and included only to exercise the public RadOncDocs demonstration workflow.")
    doc.build(story)


def main():
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    # Replace the tiny starter corpus with this generated set.
    for p in DATA.rglob("*.pdf"):
        p.unlink()

    manifest = []
    for i, item in enumerate(POLICIES, start=1):
        fname = slug_filename(item["title"])
        path = POLICY_DIR / fname
        write_policy(item, path)
        manifest.append({
            "id": f"POL-{i:03d}", "file": fname,
            "rel": str(path.relative_to(DATA)).replace("\\", "/"),
            "title": item["title"], "doc_type": "policy", "category": item["category"],
            "owner": item["owner"], "updated": item["updated"], "status": "Current",
            "description": item["description"],
            "tags": sorted({item["category"].lower(), "policy", "synthetic"}),
        })

    for i, item in enumerate(JOURNALS, start=1):
        fname = slug_filename(item["title"])
        path = JOURNAL_DIR / fname
        write_journal(item, path)
        manifest.append({
            "id": f"JRN-{i:03d}", "file": fname,
            "rel": str(path.relative_to(DATA)).replace("\\", "/"),
            "title": item["title"], "doc_type": "journal", "category": item["category"],
            "owner": item["owner"], "updated": item["updated"], "status": "Published",
            "year": item["year"], "description": item["description"],
            "tags": sorted({item["category"].lower(), "journal", "synthetic"}),
        })

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(POLICIES)} policies and {len(JOURNALS)} research articles ({len(manifest)} PDFs total).")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
