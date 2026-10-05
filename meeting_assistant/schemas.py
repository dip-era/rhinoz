"""Pydantic schemas for everything passed between stages.

Three groups:
  1. Pipeline data (Word, Segment, Transcript, CandidateSpan, EditVerdict, ...)
  2. LLM output contracts (*Out models) - what the LLMs are allowed to return
  3. The canonical MeetingRecord (JSON) from which the Markdown is rendered
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


def _clamp01(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.5
    if f > 1.0 and f <= 100.0:  # model answered in percent
        f = f / 100.0
    return max(0.0, min(1.0, f))


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return list(v)


def _none_if_blank(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "unspecified", "n/a"}:
        return None
    return v


# ============================================================================
# 1. Pipeline data
# ============================================================================


class Word(BaseModel):
    id: int  # global index == position in Transcript.words
    text: str
    start: float
    end: float
    prob: float
    speaker: Optional[str] = None


class Segment(BaseModel):
    id: str  # "S0001"
    start: float
    end: float
    speaker: Optional[str] = None
    word_ids: list[int]  # contiguous global word ids
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


class Transcript(BaseModel):
    words: list[Word]
    segments: list[Segment]
    duration: float
    language: str = "en"
    asr_model: str
    diarized: bool = False

    def segment_words(self, seg: Segment) -> list[Word]:
        return [self.words[i] for i in seg.word_ids]

    def segment_map(self) -> dict[str, Segment]:
        return {s.id: s for s in self.segments}


class GlossaryTerm(BaseModel):
    term: str
    source: Literal["user", "attendee", "llm"]
    category: Literal["term", "name"] = "term"
    heard_as: list[str] = []
    segment_ids: list[str] = []
    rationale: str = ""


EvidenceSource = Literal["low_confidence", "phonetic", "llm_flag"]


class CandidateSpan(BaseModel):
    span_id: str  # "C0001"
    segment_id: str
    word_start: int  # global word id, inclusive
    word_end: int  # global word id, exclusive
    text: str
    sources: list[EvidenceSource]
    mean_prob: float
    min_prob: float
    phonetic_term: Optional[str] = None
    phonetic_score: Optional[float] = None


class EditVerdict(BaseModel):
    """One proposed edit + all evidence + the deterministic verdict. Logged for the UI diff."""

    edit_id: str
    span_id: Optional[str] = None
    segment_id: str
    original: str  # as proposed by the LLM
    located_text: Optional[str] = None  # the actual transcript words that will be replaced
    replacement: str
    edit_type: Literal["acoustic", "formatting"]
    claimed_edit_type: str
    reason: str = ""
    confidence: float = 0.5
    word_start: Optional[int] = None
    word_end: Optional[int] = None
    sources: list[str] = []
    mean_prob: Optional[float] = None
    glossary_backed: bool = False
    glossary_term: Optional[str] = None
    phonetic_score: Optional[float] = None
    protected_hits: list[str] = []
    strict: bool = False
    precheck_failed: Optional[str] = None
    acoustic_logp_original: Optional[float] = None
    acoustic_logp_edit: Optional[float] = None
    acoustic_delta: Optional[float] = None
    acoustic_allowed_drop: Optional[float] = None
    accepted: bool = False
    verdict_reason: str = ""


class RefinedSegment(BaseModel):
    id: str
    start: float
    end: float
    speaker: Optional[str] = None
    text: str
    raw_text: str
    edit_ids: list[str] = []


# ---- lifecycle state (Stage 3) ----------------------------------------------


class LifecycleEvent(BaseModel):
    status: str
    segment_id: str
    quote: str
    start: float
    speaker: Optional[str] = None


class Proposal(BaseModel):
    proposal_id: str  # "P1"
    description: str
    segment_id: str
    quote: str
    proposed_by: Optional[str] = None  # speaker label, only if diarized
    status: Literal["proposed", "accepted", "rejected", "deferred", "unresolved"] = "proposed"
    history: list[LifecycleEvent] = []
    notes: list[str] = []


class OwnerEvidence(BaseModel):
    owner_text: Optional[str] = None
    owner_is_speaker: bool = False
    segment_id: str
    quote: str


class DeadlineEvidence(BaseModel):
    deadline_text: str
    segment_id: str
    quote: str


class NameHint(BaseModel):
    name: str
    segment_id: str
    quote: str


class Task(BaseModel):
    task_id: str  # "T1"
    description: str
    kind: Literal["self_commitment", "assignment", "open_task", "request"]
    segment_id: str
    quote: str
    status: Literal["confirmed", "open", "declined"]
    explicit_confirmation: bool = False
    linked_proposal: Optional[str] = None
    owner_evidence: list[OwnerEvidence] = []
    deadline_evidence: list[DeadlineEvidence] = []
    name_hints: list[NameHint] = []
    history: list[LifecycleEvent] = []
    notes: list[str] = []


# ============================================================================
# 2. LLM output contracts
# ============================================================================


class GlossaryTermOut(BaseModel):
    term: str
    heard_as: list[str] = []
    segment_ids: list[str] = []
    rationale: str = ""

    _l1 = field_validator("heard_as", "segment_ids", mode="before")(_as_list)


class GlossaryOut(BaseModel):
    terms: list[GlossaryTermOut] = []


class ProposedEdit(BaseModel):
    span_id: Optional[str] = None
    segment_id: str
    original: str
    replacement: str
    edit_type: str = "acoustic"
    reason: str = ""
    confidence: float = 0.5

    _c = field_validator("confidence", mode="before")(_clamp01)
    _s = field_validator("span_id", mode="before")(_none_if_blank)

    @field_validator("edit_type", mode="before")
    @classmethod
    def _et(cls, v):
        v = str(v or "acoustic").lower().strip()
        return "formatting" if v.startswith("format") else "acoustic"


class EditProposalBatch(BaseModel):
    edits: list[ProposedEdit] = []


class SpeechActOut(BaseModel):
    segment_id: str
    acts: list[str] = []

    _l = field_validator("acts", mode="before")(_as_list)


class NewProposalOut(BaseModel):
    ref: str
    description: str
    segment_id: str
    quote: str = ""


class ProposalUpdateOut(BaseModel):
    id: str
    status: str
    segment_id: str
    quote: str = ""


class _OwnerFields(BaseModel):
    owner_text: Optional[str] = None
    owner_is_speaker: bool = False
    deadline_text: Optional[str] = None
    context_name: Optional[str] = None
    context_quote: Optional[str] = None

    _n = field_validator("owner_text", "deadline_text", "context_name", "context_quote", mode="before")(_none_if_blank)

    @field_validator("owner_is_speaker", mode="before")
    @classmethod
    def _b(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in {"true", "yes", "1"}
        return bool(v)


class NewTaskOut(_OwnerFields):
    ref: str
    description: str
    kind: str = "request"
    segment_id: str
    quote: str = ""
    linked_proposal: Optional[str] = None

    _lp = field_validator("linked_proposal", mode="before")(_none_if_blank)


class TaskUpdateOut(_OwnerFields):
    id: str
    event: str
    segment_id: str
    quote: str = ""


class DocChunkOut(BaseModel):
    speech_acts: list[SpeechActOut] = []
    new_proposals: list[NewProposalOut] = []
    proposal_updates: list[ProposalUpdateOut] = []
    new_tasks: list[NewTaskOut] = []
    task_updates: list[TaskUpdateOut] = []


class NoteOut(BaseModel):
    text: str
    segment_ids: list[str] = []

    _l = field_validator("segment_ids", mode="before")(_as_list)


class NotesOut(BaseModel):
    notes: list[NoteOut] = []


class MinutePointOut(BaseModel):
    text: str
    segment_ids: list[str] = []

    _l = field_validator("segment_ids", mode="before")(_as_list)


class MinutesSectionOut(BaseModel):
    topic: str
    points: list[MinutePointOut] = []


class SummaryOut(BaseModel):
    summary: str
    minutes: list[MinutesSectionOut] = []


# ============================================================================
# 3. Canonical meeting record
# ============================================================================


class Provenance(BaseModel):
    segment_ids: list[str]
    start: float
    end: float
    quotes: list[str] = []


class Decision(BaseModel):
    id: str
    decision: str
    proposed_by: Optional[str] = None
    accepted_by: Optional[str] = None
    source_proposal: str
    provenance: Provenance
    history: list[LifecycleEvent] = []
    flags: list[str] = []


class ActionItem(BaseModel):
    id: str
    task: str
    owner: str = "unspecified"
    owner_source: Literal["stated", "speaker_label", "unspecified"] = "unspecified"
    owner_annotation: Optional[str] = None  # context-inferred hint; NEVER the owner
    deadline: str = "unspecified"  # verbatim as spoken
    kind: str
    status: str
    provenance: Provenance
    flags: list[str] = []
    verification_notes: list[str] = []


class ProposalSummary(BaseModel):
    id: str
    proposal: str
    status: str
    proposed_by: Optional[str] = None
    provenance: Provenance


class MinutePoint(BaseModel):
    text: str
    segment_ids: list[str] = []
    cited: bool = True


class MinutesSection(BaseModel):
    topic: str
    points: list[MinutePoint] = []


class ModelInfo(BaseModel):
    asr: str
    diarization: Optional[str] = None
    acoustic_verifier: Optional[str] = None
    refiner_llm: str
    documenter_llm: str
    nli: Optional[str] = None


class MeetingRecord(BaseModel):
    schema_version: str = "1.0"
    source_file: str
    created_at: str
    duration_sec: float
    diarized: bool
    models: ModelInfo
    summary: str = ""
    minutes: list[MinutesSection] = []
    decisions: list[Decision] = []
    action_items: list[ActionItem] = []
    unconfirmed_requests: list[ActionItem] = []
    rejected_proposals: list[ProposalSummary] = []
    deferred_proposals: list[ProposalSummary] = []
    unresolved_proposals: list[ProposalSummary] = []
    raw_transcript: Transcript
    refined_transcript: list[RefinedSegment]
    glossary: list[GlossaryTerm] = []
    candidates: list[CandidateSpan] = []
    edits: list[EditVerdict] = []
    speech_acts: dict[str, list[str]] = {}
    settings_snapshot: dict = Field(default_factory=dict)
    warnings: list[str] = []
