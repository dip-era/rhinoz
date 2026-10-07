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
        {"span_id": None, "segment_id": "S0002", "original": "will not", "replacement": "will now",
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


def test_capitalisation_guard():
    from meeting_assistant.refine import capitalisation_problem

    assert capitalisation_problem("battery", "Battery", set())  # ordinary word: rejected
    assert capitalisation_problem("remote control.", "Remote control.", set())
    assert capitalisation_problem("tv", "TV", set()) is None  # acronym: allowed
    assert capitalisation_problem("postgresql", "PostgreSQL", set()) is None  # internal capitals: allowed
    assert capitalisation_problem("grafana", "Grafana", {"grafana"}) is None  # user-supplied term: allowed


def test_quote_matching_ellipsis_and_stutter():
    from meeting_assistant.utils import fuzzy_contains

    hay = "okay so maybe have like one remote that has the main functions and another remote remote with all the special things"
    assert fuzzy_contains("one remote that has the main functions ... and another remote with all the special things", hay)
    assert not fuzzy_contains("one remote ... a banana phone with lasers", hay)


def test_announced_decisions_followups_and_split_turns():
    s = Settings()
    segs = [
        RefinedSegment(id="S0001", start=0, end=2, speaker="SPEAKER_00", text="We'd like to sell it for about 25 euro.", raw_text=""),
        RefinedSegment(id="S0002", start=2, end=4, speaker="SPEAKER_00", text="Introduce yourself and draw your favourite animal.", raw_text=""),
        RefinedSegment(id="S0003", start=4, end=5, speaker="SPEAKER_01", text="I'll look into", raw_text=""),
        RefinedSegment(id="S0004", start=5, end=6, speaker="SPEAKER_00", text="it if I can.", raw_text=""),
        RefinedSegment(id="S0005", start=6, end=8, speaker="SPEAKER_00", text="Should the price be 30 euro?", raw_text=""),
    ]
    reply = DocChunkOut.model_validate({
        "speech_acts": [{"segment_id": "S0005", "acts": ["question"]}],
        "new_decisions": [
            {"ref": "D1", "description": "Sell the remote for 25 euro", "segment_id": "S0001", "quote": "sell it for about 25 euro"},
            {"ref": "D2", "description": "Price is 30 euro", "segment_id": "S0005", "quote": "Should the price be 30 euro?"},
        ],
        "new_tasks": [
            {"ref": "M1", "description": "Introduce yourself", "kind": "open_task", "segment_id": "S0002",
             "quote": "Introduce yourself", "is_followup": False},
            {"ref": "M2", "description": "Look into the remote scope", "kind": "self_commitment", "segment_id": "S0003",
             "quote": "I'll look into", "owner_is_speaker": True},
        ],
    })
    life = extract_lifecycle(segs, True, FakeLLM({"DocChunkOut": reply.model_dump()}), s)
    out = verify(life, segs, True, s)
    assert [(d.decision, d.basis) for d in out.decisions] == [("Sell the remote for 25 euro", "announced")]
    assert [a.task for a in out.action_items] == ["Look into the remote scope"]  # the ice-breaker is not a task
    a = out.action_items[0]
    assert a.owner == "unspecified" and "split" in " ".join(a.verification_notes)  # diarization cut the sentence


def test_review_pass_recovers_skipped_task():
    s = Settings()
    segs = [RefinedSegment(id="S0001", start=0, end=2, speaker="SPEAKER_00",
                           text="The industrial designer will do the working design.", raw_text="")]
    first = {"speech_acts": [{"segment_id": "S0001", "acts": ["assignment"]}]}  # tagged, but no task emitted
    second = {"new_tasks": [{"ref": "M1", "description": "Do the working design", "kind": "assignment",
                             "segment_id": "S0001", "quote": "The industrial designer will do the working design",
                             "owner_text": "The industrial designer"}]}

    class TwoReplies:
        def __init__(self):
            self.replies = [first, second]

        def call_json(self, system, user, schema, **kw):
            return schema.model_validate(self.replies.pop(0))

    life = extract_lifecycle(segs, True, TwoReplies(), s)
    out = verify(life, segs, True, s)
    assert [(a.task, a.owner) for a in out.action_items] == [("Do the working design", "The industrial designer")]


def test_supervisor_vetoes_edits():
    from meeting_assistant.refine import supervise

    s = Settings()
    tr = make_transcript([("we have a profit aim of 50 million", None, None), ("one for the vcr and the tv", None, None)])
    batch = EditProposalBatch.model_validate({"edits": [
        {"segment_id": "S0001", "original": "aim", "replacement": "AIM", "edit_type": "formatting", "confidence": 0.9},
        {"segment_id": "S0002", "original": "vcr", "replacement": "VCR", "edit_type": "formatting", "confidence": 0.9},
        {"segment_id": "S0002", "original": "tv", "replacement": "TV", "edit_type": "formatting", "confidence": 0.9},
    ]})
    thr = Thresholds.from_settings(s)
    verdicts = prepare_verdicts(batch.edits, tr, [], [], thr)
    reviews = {"reviews": [
        {"edit_id": "E0001", "verdict": "reject", "casing_ok": False, "grammar_ok": True, "meaning_ok": True,
         "reason": "'aim' is an ordinary word"},
        {"edit_id": "E0002", "verdict": "approve", "casing_ok": True, "grammar_ok": True, "meaning_ok": True,
         "reason": "VCR is an acronym"},
        # E0003 not reviewed -> must be rejected (fail closed)
    ]}
    supervise(verdicts, tr, FakeLLM({"SupervisorOut": reviews}), s, lambda *a: None)
    finalize(verdicts, thr, acoustic_available=False)
    assert [v.accepted for v in verdicts] == [False, True, False]
    assert "supervisor rejected (casing)" in verdicts[0].verdict_reason
    assert verdicts[2].supervisor_verdict == "not reviewed"
    assert apply_edits(tr, verdicts)[0].text == "we have a profit aim of 50 million"


def test_llm_client_recovers_when_reasoning_eats_the_budget(tmp_path):
    from types import SimpleNamespace

    from meeting_assistant.llm_client import LLMClient
    from meeting_assistant.schemas import GlossaryOut

    calls = []

    def create(**params):
        calls.append(dict(params))
        if params.get("reasoning_effort") != "low":  # what gpt-oss-20b did: all tokens spent reasoning
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=""))])
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                                                        message=SimpleNamespace(content='{"terms": []}'))])

    llm = LLMClient("openai/gpt-oss-20b", "dummy-key", "LLM #1", cache_dir=tmp_path, reasoning_effort="medium")
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert llm.call_json("sys", "user", GlossaryOut).terms == []
    assert [c.get("reasoning_effort") for c in calls] == ["medium", "low"]
    cached = list((tmp_path / "llm").glob("*.json"))
    assert len(cached) == 1 and "terms" in cached[0].read_text()  # only the good reply is cached


def test_role_stated_separately_from_name():
    from meeting_assistant.speakers import resolve_speakers

    s = Settings()
    tr = make_transcript([
        ("My name is Alima Bucintini.", None, "SPEAKER_02"),
        ("How do you spell your name?", None, "SPEAKER_03"),
        ("Oh, and I guess I'm the industrial designer on this project.", None, "SPEAKER_02"),
    ], diarized=True)
    claims = {"claims": [
        {"speaker": "SPEAKER_02", "name": "Alima Bucintini", "kind": "self_intro", "segment_id": "S0001",
         "quote": "My name is Alima Bucintini"},
        {"speaker": "SPEAKER_02", "name": "", "role": "industrial designer", "kind": "self_role", "segment_id": "S0003",
         "quote": "I'm the industrial designer on this project"},
        # invalid: S0002 is not spoken by SPEAKER_02
        {"speaker": "SPEAKER_02", "name": "", "role": "designer", "kind": "self_role", "segment_id": "S0002", "quote": "spell"},
    ]}
    ids, warns = resolve_speakers(tr, FakeLLM({"SpeakerIdOut": claims}), s)
    by = {i.label: i for i in ids}
    assert by["SPEAKER_02"].display_name == "Alima Bucintini (Industrial Designer)"
    assert by["SPEAKER_03"].display_name == "SPEAKER_03" and len(warns) == 1


def test_transcript_paragraphs_per_speaker():
    from meeting_assistant.record import transcript_text

    tr = make_transcript([
        ("Okay, good morning.", None, "SPEAKER_00"),
        ("My name is Rose.", None, "SPEAKER_00"),
        ("Hi Rose.", None, "SPEAKER_01"),
        ("Let's start.", None, "SPEAKER_00"),
    ], diarized=True)
    paras = transcript_text(tr.segments, True, {"SPEAKER_00": "Rose"}).strip().split("\n\n")
    assert len(paras) == 3
    assert "(S0001-S0002) Rose: Okay, good morning. My name is Rose." in paras[0]
    assert "SPEAKER_01: Hi Rose." in paras[1]


def test_sentence_units_for_voice_recheck():
    from meeting_assistant.diarization import _sentence_units

    tr = make_transcript([("I have no artistic talent. How do you spell your name? A-L-I-M-A, thanks.", None, "SPEAKER_02")],
                         diarized=True)
    units = [" ".join(tr.words[i].text for i in u) for u in _sentence_units(tr.words)]
    assert units == ["I have no artistic talent.", "How do you spell your name?", "A-L-I-M-A, thanks."]


def test_name_variants_merge_and_pattern_backstop():
    from meeting_assistant.speakers import resolve_speakers

    s = Settings()
    tr = make_transcript([
        ("My name is Rose Lindgren. I'll be the project manager.", None, "SPEAKER_03"),
        ("introduce myself my name is Alima Bucintini I'm from Maine", None, "SPEAKER_02"),
        ("I'm Rose and I love coyotes.", None, "SPEAKER_03"),
    ], diarized=True)
    claims = {"claims": [  # the LLM misses Alima entirely and names Rose twice, once by first name only
        {"speaker": "SPEAKER_03", "name": "Rose Lindgren", "role": "project manager", "kind": "self_intro",
         "segment_id": "S0001", "quote": "My name is Rose Lindgren. I'll be the project manager."},
        {"speaker": "SPEAKER_03", "name": "Rose", "kind": "self_intro", "segment_id": "S0003", "quote": "I'm Rose"},
    ]}
    ids, _ = resolve_speakers(tr, FakeLLM({"SpeakerIdOut": claims}), s)
    by = {i.label: i for i in ids}
    assert by["SPEAKER_03"].display_name == "Rose Lindgren (Project Manager)"  # no tie between "Rose" variants
    assert by["SPEAKER_02"].display_name == "Alima Bucintini"  # found by the "my name is" pattern


def test_double_encoded_llm_json_is_accepted():
    import json

    from meeting_assistant.schemas import SummaryOut

    sec = {"topic": "Project objectives", "points": [{"text": "Remote must be original", "segment_ids": ["S0061"]}]}
    reply = {"summary": "Kickoff.", "minutes": [sec, json.dumps({"topic": "Design workflow", "points": ["Three phases"]}), ""]}
    out = SummaryOut.model_validate(reply)  # gpt-oss-120b returned sections as JSON strings plus an empty string
    assert [m.topic for m in out.minutes] == ["Project objectives", "Design workflow"]
    assert out.minutes[1].points[0].text == "Three phases"


def test_spelled_out_name_wins_everywhere():
    from meeting_assistant.glossary import build_glossary
    from meeting_assistant.speakers import resolve_speakers
    from meeting_assistant.utils import spelled_words

    assert spelled_words("A -L -I -M -A.") == spelled_words("A. L. I. M. A.") == spelled_words("a-l-i-m-a") == ["Alima"]
    assert spelled_words("I have a TV in the U .S.") == []

    s = Settings()
    tr = make_transcript([
        ("My name is Eliema, which is me.", None, "SPEAKER_04"),
        ("How do you spell your name?", None, "SPEAKER_02"),
        ("A -L -I -M -A.", None, "SPEAKER_04"),
    ], diarized=True)
    # the LLM copies the misheard name from the transcript, as it should
    claims = {"claims": [{"speaker": "SPEAKER_04", "name": "Eliema", "kind": "self_intro", "segment_id": "S0001",
                          "quote": "My name is Eliema"}]}
    ids, _ = resolve_speakers(tr, FakeLLM({"SpeakerIdOut": claims}), s, attendees=["Elena"])
    sp = next(i for i in ids if i.label == "SPEAKER_04")
    assert sp.name == "Alima" and any(e.kind == "spelled" for e in sp.evidence)  # spelling beats transcript + attendee list

    gl, _ = build_glossary(tr, [], [], None, s)
    assert [(g.term, g.source, g.category) for g in gl] == [("Alima", "spelled", "name")]

    # refinement: the misheard name is corrected toward the verified spelling and the supervisor is told about it
    seen = {}

    class Sup:
        def call_json(self, system, user, schema, **kw):
            seen["user"] = user
            return schema.model_validate({"reviews": [{"edit_id": "E0001", "verdict": "approve", "casing_ok": True,
                                                       "grammar_ok": True, "meaning_ok": True, "reason": "verified spelling"}]})

    batch = EditProposalBatch.model_validate({"edits": [{"segment_id": "S0001", "original": "Eliema,", "replacement": "Alima",
                                                         "edit_type": "acoustic", "confidence": 0.9}]})
    thr = Thresholds.from_settings(s)
    verdicts = prepare_verdicts(batch.edits, tr, [], gl, thr)
    from meeting_assistant.refine import supervise
    supervise(verdicts, tr, Sup(), s, lambda *a: None, ["Alima"])
    finalize(verdicts, thr, acoustic_available=False)
    assert "VERIFIED SPELLINGS: Alima" in seen["user"]
    assert apply_edits(tr, verdicts)[0].text == "My name is Alima, which is me."


def test_numbers_in_llm_wording_must_be_spoken():
    from meeting_assistant.verification import _unsupported_numbers

    seq = [RefinedSegment(id="S0001", start=0, end=1, text="we need a maximum of 12 .50 per unit", raw_text=""),
           RefinedSegment(id="S0002", start=1, end=2, text="and a target of fifty thousand sales", raw_text="")]
    order = {"S0001": 0, "S0002": 1}
    assert _unsupported_numbers("Maximum cost is 1250 per unit", ["S0001"], seq, order) == ["1250"]  # misread
    assert _unsupported_numbers("Maximum cost is 12.50 per unit", ["S0001"], seq, order) == []
    assert _unsupported_numbers("Target of 50000 sales", ["S0002"], seq, order) == []  # spelled-out number matches
    assert _unsupported_numbers("Target of 60000 sales", ["S0002"], seq, order) == ["60000"]  # invented


def test_wrong_segment_id_is_re_anchored():
    from meeting_assistant.speakers import resolve_speakers, usable_role

    s = Settings()
    lines = [("Okay, let's begin.", None, "SPEAKER_00")] * 3 + [
        ("Very good. And as you already know, I'm Betsy, I'm the project manager for today.", None, "SPEAKER_01"),
        ("My role is the main responsibility is user interface.", None, "SPEAKER_02"),
        ("And my role is to design a television remote control.", None, "SPEAKER_02"),
    ] + [("The functional design is individual work.", None, "SPEAKER_01")] * 3
    tr = make_transcript(lines, diarized=True)
    claims = {"claims": [  # the LLM mis-copies the segment id (S0004 -> S0009), as gpt-oss did on a real meeting
        {"speaker": "SPEAKER_01", "name": "Betsy", "role": "project manager", "kind": "self_intro",
         "segment_id": "S0009", "quote": "I'm Betsy, I'm the project manager for today"},
        {"speaker": "SPEAKER_02", "name": "", "role": "user interface", "kind": "self_role", "segment_id": "S0005",
         "quote": "My role is the main responsibility is user interface"},
        {"speaker": "SPEAKER_02", "name": "", "role": "design a television remote control", "kind": "self_role",
         "segment_id": "S0006", "quote": "my role is to design a television remote control"},
    ]}
    ids, warns = resolve_speakers(tr, FakeLLM({"SpeakerIdOut": claims}), s)
    by = {i.label: i for i in ids}
    assert by["SPEAKER_01"].display_name == "Betsy (Project Manager)"
    assert by["SPEAKER_02"].display_name == "SPEAKER_02 (User Interface)"  # an activity is not a role
    assert any("re-anchored" in w for w in warns)
    assert usable_role("user interface designer") and not usable_role("to design a remote")

    # documentation stage: a decision citing the wrong segment is moved to the segment holding its quote
    segs = [RefinedSegment(id=x.id, start=x.start, end=x.end, speaker=x.speaker, text=x.text, raw_text=x.text)
            for x in tr.segments]
    reply = DocChunkOut.model_validate({"new_decisions": [{"ref": "D1", "description": "Functional design is individual work",
                                                            "segment_id": "S0002", "quote": "functional design is individual work"}]})
    life = extract_lifecycle(segs, True, FakeLLM({"DocChunkOut": reply.model_dump()}), s)
    assert [p.segment_id for p in life.proposals] == ["S0007"] and any("re-anchored" in w for w in life.warnings)

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
