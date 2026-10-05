"""Offline tests: no GPU, no audio models, no API key. A fake LLM returns canned JSON.

    python -m pytest tests -q        (or)        python tests/test_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_assistant.candidates import find_candidates
from meeting_assistant.config import Settings
from meeting_assistant.documentation import extract_lifecycle
from meeting_assistant.guards import extract_numbers, protected_changes
from meeting_assistant.phonetics import term_match_score
from meeting_assistant.record import to_markdown
from meeting_assistant.refine import Thresholds, apply_edits, finalize, prepare_verdicts
from meeting_assistant.schemas import (
    DocChunkOut,
    EditProposalBatch,
    GlossaryTerm,
    RefinedSegment,
    Segment,
    Transcript,
    Word,
)
from meeting_assistant.utils import find_verbatim
from meeting_assistant.verification import verify


class FakeLLM:
    def __init__(self, replies: dict):
        self.replies = replies

    def call_json(self, system, user, schema, **kw):
        return schema.model_validate(self.replies[schema.__name__])


def make_transcript(lines: list[tuple[str, list[float] | None, str | None]], diarized=False) -> Transcript:
    words, segs, t = [], [], 0.0
    for k, (text, probs, spk) in enumerate(lines, 1):
        ids = []
        for i, w in enumerate(text.split()):
            p = probs[i] if probs else 0.95
            ids.append(len(words))
            words.append(Word(id=len(words), text=w, start=t, end=t + 0.3, prob=p, speaker=spk))
            t += 0.35
        segs.append(Segment(id=f"S{k:04d}", start=words[ids[0]].start, end=words[ids[-1]].end, speaker=spk,
                            word_ids=ids, text=text))
    return Transcript(words=words, segments=segs, duration=t, asr_model="test", diarized=diarized)


# ---------------------------------------------------------------------------
def test_phonetic_ngram_match():
    assert term_match_score("cube earnest", "Kubernetes") >= 0.8
    assert term_match_score("graph ana", "Grafana") >= 0.8
    assert term_match_score("banana", "Kubernetes") < 0.5


def test_numbers_and_protected():
    assert extract_numbers("we moved forty two services".split()) == ["42"]
    assert extract_numbers("zero point eight one".split()) == ["0.81"]
    assert extract_numbers("two hundred and five".split()) == ["205"]
    assert extract_numbers("one and the".split()) == ["1"]
    toks = "we will not deploy 15 pods".split()
    assert "negation" in protected_changes(toks, 1, 3, "will", set(), set())
    assert "modal" in protected_changes(toks, 1, 3, "might not", set(), set())
    assert "number" in protected_changes(toks, 4, 5, "50", set(), set())
    assert protected_changes(toks, 4, 5, "fifteen", set(), set()) == []


def test_find_verbatim():
    assert find_verbatim("by friday", "I'll rewrite them by Friday.") == "by Friday"
    assert find_verbatim("priya", "Thanks, Priya's team will do it") == "Priya"  # transcript casing, possessive dropped
    assert find_verbatim("Priya", "Thanks, Pria, good work") is None  # too far: owner would be dropped
    assert find_verbatim("next quarter", "I'll do it tomorrow") is None


def test_refinement_evidence_gating():
    s = Settings()
    tr = make_transcript([
        ("we need to deploy the cube earnest pods by friday", [0.9, .9, .9, .9, .9, .3, .25, .9, .9, .9], None),
        ("we will not migrate the database", None, None),
    ])
    glossary = [GlossaryTerm(term="Kubernetes", source="user")]
    cands = find_candidates(tr, glossary, s)
    assert any(c.phonetic_term == "Kubernetes" and c.text == "cube earnest" for c in cands)

    batch = EditProposalBatch.model_validate({"edits": [
        {"span_id": cands[0].span_id, "segment_id": "S0001", "original": "cube earnest", "replacement": "Kubernetes",
         "edit_type": "acoustic", "reason": "pods context", "confidence": 0.9},
        {"span_id": None, "segment_id": "S0002", "original": "will not", "replacement": "will",
         "edit_type": "acoustic", "reason": "bad", "confidence": 0.9},
        {"span_id": None, "segment_id": "S0002", "original": "the data warehouse", "replacement": "the DB",
         "edit_type": "formatting", "reason": "hallucinated original", "confidence": 0.9},
    ]})
    thr = Thresholds.from_settings(s)
    verdicts = prepare_verdicts(batch.edits, tr, cands, glossary, thr)
    finalize(verdicts, thr, acoustic_available=False)
    v1, v2, v3 = verdicts
    assert v1.accepted and v1.glossary_backed, v1.verdict_reason
    assert not v2.accepted and "negation" in v2.protected_hits
    assert not v3.accepted and "not found" in v3.verdict_reason
    refined = apply_edits(tr, verdicts)
    assert refined[0].text == "we need to deploy the Kubernetes pods by friday"
    assert refined[1].text == tr.segments[1].text

    # with acoustic evidence: protected edits need a strong positive margin
    v2.acoustic_delta = 0.2
    v1.acoustic_delta = -2.0  # rare term penalised by Whisper's LM prior, but low conf + glossary allow it
    finalize(verdicts, thr, acoustic_available=True)
    assert v1.accepted and not v2.accepted


def test_lifecycle_and_owner_rules():
    s = Settings()
    segs = [
        RefinedSegment(id="S0001", start=0, end=2, speaker="SPEAKER_00", text="I propose we raise the threshold to 500 milliseconds.", raw_text=""),
        RefinedSegment(id="S0002", start=2, end=4, speaker="SPEAKER_01", text="No objection. Let's do it.", raw_text=""),
        RefinedSegment(id="S0003", start=4, end=6, speaker="SPEAKER_00", text="What if we switch to Traefik?", raw_text=""),
        RefinedSegment(id="S0004", start=6, end=8, speaker="SPEAKER_01", text="No, let's not switch this quarter.", raw_text=""),
        RefinedSegment(id="S0005", start=8, end=10, speaker="SPEAKER_01", text="I'll rewrite the annotations by Thursday.", raw_text=""),
        RefinedSegment(id="S0006", start=10, end=12, speaker="SPEAKER_00", text="Someone needs to update Terraform. Thanks Rahul.", raw_text=""),
    ]
    reply = DocChunkOut.model_validate({
        "speech_acts": [{"segment_id": "S0001", "acts": ["proposal"]}, {"segment_id": "S0002", "acts": ["agreement"]},
                        {"segment_id": "S0003", "acts": ["proposal"]}, {"segment_id": "S0004", "acts": ["objection"]},
                        {"segment_id": "S0005", "acts": ["commitment"]}, {"segment_id": "S0006", "acts": ["assignment"]}],
        "new_proposals": [
            {"ref": "N1", "description": "Raise the threshold to 500 ms", "segment_id": "S0001", "quote": "raise the threshold to 500 milliseconds"},
            {"ref": "N2", "description": "Switch to Traefik", "segment_id": "S0003", "quote": "switch to Traefik"},
        ],
        "proposal_updates": [
            {"id": "N1", "status": "accepted", "segment_id": "S0002", "quote": "No objection. Let's do it."},
            {"id": "N2", "status": "rejected", "segment_id": "S0004", "quote": "let's not switch this quarter"},
        ],
        "new_tasks": [
            {"ref": "M1", "description": "Rewrite the annotations", "kind": "self_commitment", "segment_id": "S0005",
             "quote": "I'll rewrite the annotations by Thursday", "owner_is_speaker": True, "deadline_text": "by Thursday",
             "context_name": "Rahul", "context_quote": "Thanks Rahul"},
            {"ref": "M2", "description": "Update Terraform", "kind": "open_task", "segment_id": "S0006",
             "quote": "Someone needs to update Terraform", "owner_text": "Priya", "deadline_text": "by Monday"},
        ],
        "task_updates": [],
    })
    life = extract_lifecycle(segs, True, FakeLLM({"DocChunkOut": reply.model_dump()}), s)
    out = verify(life, segs, True, s)
    assert [d.decision for d in out.decisions] == ["Raise the threshold to 500 ms"]
    assert [p.proposal for p in out.rejected] == ["Switch to Traefik"]
    a1, a2 = out.action_items
    assert (a1.owner, a1.owner_source, a1.deadline) == ("SPEAKER_01", "speaker_label", "by Thursday")
    assert a1.owner_annotation and "Rahul" in a1.owner_annotation  # hint, not owner
    # invented owner/deadline (not in the cited segment) are removed by the deterministic check
    assert (a2.owner, a2.deadline) == ("unspecified", "unspecified")

    # without diarization the self-commitment owner must be unspecified
    life2 = extract_lifecycle(segs, False, FakeLLM({"DocChunkOut": reply.model_dump()}), s)
    out2 = verify(life2, segs, False, s)
    assert out2.action_items[0].owner == "unspecified"


def test_markdown_from_json_renders_empty_lists():
    from meeting_assistant.schemas import MeetingRecord, ModelInfo

    tr = make_transcript([("hello there", None, None)])
    rec = MeetingRecord(source_file="x.wav", created_at="now", duration_sec=1.0, diarized=False,
                        models=ModelInfo(asr="a", refiner_llm="b", documenter_llm="c"), raw_transcript=tr,
                        refined_transcript=apply_edits(tr, []))
    md = to_markdown(rec)
    assert "No decisions were reached" in md and "No action items were assigned" in md



def test_speaker_naming_rules():
    from meeting_assistant.record import transcript_text
    from meeting_assistant.speakers import name_map, resolve_speakers
    from meeting_assistant.verification import verify as _verify

    s = Settings()
    tr = make_transcript([
        ("Hi everyone, I'm Rose, the project manager.", None, "SPEAKER_00"),
        ("Bob, can you check the logs?", None, "SPEAKER_00"),
        ("Sure, I'll check them by Friday.", None, "SPEAKER_01"),
        ("Thanks, Carol.", None, "SPEAKER_01"),
        ("Let's also ask Dave about it.", None, "SPEAKER_02"),
    ], diarized=True)
    claims = {"claims": [
        {"speaker": "SPEAKER_00", "name": "Rose", "role": "project manager", "kind": "self_intro",
         "segment_id": "S0001", "quote": "I'm Rose, the project manager"},
        {"speaker": "SPEAKER_01", "name": "Bob", "kind": "addressed", "segment_id": "S0002",
         "quote": "Bob, can you check the logs?"},
        # invalid: SPEAKER_02 does not answer right after "Thanks, Carol" (S0004 is SPEAKER_01 talking)
        {"speaker": "SPEAKER_00", "name": "Carol", "kind": "addressed", "segment_id": "S0004", "quote": "Thanks, Carol."},
        # invalid: Dave is only mentioned, and the name is not a self-introduction by SPEAKER_02
        {"speaker": "SPEAKER_02", "name": "Dave", "kind": "self_intro", "segment_id": "S0002", "quote": "ask Dave"},
    ]}
    ids, warns = resolve_speakers(tr, FakeLLM({"SpeakerIdOut": claims}), s)
    by = {i.label: i for i in ids}
    assert (by["SPEAKER_00"].display_name, by["SPEAKER_00"].confidence) == ("Rose (Project Manager)", "high")
    assert (by["SPEAKER_01"].display_name, by["SPEAKER_01"].confidence) == ("Bob", "low")
    assert by["SPEAKER_02"].display_name == "SPEAKER_02" and len(warns) == 2
    txt = transcript_text(tr.segments, True, name_map(ids))
    assert "Rose (Project Manager): Hi everyone" in txt and "SPEAKER_02: Let's" in txt

    # owners: an "addressed"-only name never becomes the owner, a self-introduced one does
    segs = [RefinedSegment(id=x.id, start=x.start, end=x.end, speaker=x.speaker, text=x.text, raw_text=x.text)
            for x in tr.segments]
    reply = DocChunkOut.model_validate({
        "speech_acts": [{"segment_id": "S0003", "acts": ["commitment"]}],
        "new_tasks": [{"ref": "M1", "description": "Check the logs", "kind": "self_commitment", "segment_id": "S0003",
                       "quote": "I'll check them by Friday", "owner_is_speaker": True, "deadline_text": "by Friday"}],
    })
    life = extract_lifecycle(segs, True, FakeLLM({"DocChunkOut": reply.model_dump()}), s)
    a = _verify(life, segs, True, s, speakers=by).action_items[0]
    assert (a.owner, a.owner_source) == ("SPEAKER_01", "speaker_label") and "probably Bob" in a.owner_annotation
    by["SPEAKER_01"].confidence = "high"
    by["SPEAKER_01"].evidence[0].kind = "self_intro"
    a = _verify(life, segs, True, s, speakers=by).action_items[0]
    assert (a.owner, a.owner_source) == ("Bob", "speaker_name")

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
