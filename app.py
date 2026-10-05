"""Streamlit UI - step-by-step:

    upload  -> Stage 0 validation runs automatically (OK message or clear error)
    button  -> Stage 1: raw transcript            (shown + downloadable)
    button  -> Stage 2: refined transcript        (diff, edit log, downloadable)
    button  -> Stage 3: summary, minutes, decisions, action items (+ all downloads)

Run:  streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import difflib
import html
import traceback
from pathlib import Path
from typing import Callable

import streamlit as st

from meeting_assistant.audio_io import clip_wav_bytes
from meeting_assistant.config import ASR_MODEL, Settings
from meeting_assistant.errors import PipelineError
from meeting_assistant.pipeline import stage0_validate, stage1_transcribe, stage2_refine, stage3_document
from meeting_assistant.record import to_markdown, transcript_text
from meeting_assistant.utils import fmt_ts

st.set_page_config(page_title="Evidence-Traced Meeting Assistant", layout="wide")


def _diff_html(raw: str, refined: str) -> str:
    a, b = raw.split(), refined.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape(" ".join(a[i1:i2])))
        else:
            if i2 > i1:
                out.append(f"<del style='background:#fdd;color:#900'>{html.escape(' '.join(a[i1:i2]))}</del>")
            if j2 > j1:
                out.append(f"<ins style='background:#dfd;color:#060;text-decoration:none'>{html.escape(' '.join(b[j1:j2]))}</ins>")
    return " ".join(out)


def _run_stage(label: str, fn: Callable[[Callable], object]) -> bool:
    """Run one stage inside a live status box; show a clear error on failure."""
    with st.status(f"{label}...", expanded=True) as status:
        bar = st.progress(0.0)

        def cb(stage: str, msg: str, frac: float | None = None):
            status.update(label=f"{stage}: {msg}")
            st.write(f"**{stage}** - {msg}")
            if frac is not None:
                bar.progress(min(max(frac, 0.0), 1.0))

        try:
            fn(cb)
            status.update(label=f"{label} - done", state="complete", expanded=False)
            return True
        except PipelineError as e:
            status.update(label=f"{label} - failed", state="error")
            st.error(f"**{e.stage}:** {e}")
        except Exception as e:
            status.update(label=f"{label} - failed (unexpected error)", state="error")
            st.error(f"Unexpected error: {type(e).__name__}: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
    return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
base = Settings.from_env()
with st.sidebar:
    st.header("Settings")
    st.caption(f"Whisper model: `{ASR_MODEL}`")
    diarize = st.checkbox("Speaker diarization (pyannote, needs HF_TOKEN)", value=base.diarize)
    acoustic = st.checkbox("Acoustic verification of edits", value=base.acoustic_check)
    nli = st.checkbox("NLI support flags (DeBERTa)", value=base.nli_check)
    st.caption(f"LLM #1 (refinement): `{base.llm1_model}`")
    st.caption(f"LLM #2 (documentation): `{base.llm2_model}`")
    if not base.groq_api_key:
        st.error("GROQ_API_KEY is not set - Stages 2 and 3 need it (add it to .env)")
    st.divider()
    glossary_text = st.text_area("Agenda / domain terms (optional, used from Stage 1)",
                                 placeholder="Kubernetes, Grafana, PostgreSQL ...", height=110)
    attendees_text = st.text_area("Attendee names (optional, used from Stage 1)",
                                  placeholder="Priya, Rahul, Ankit", height=70)

settings = dataclasses.replace(base, diarize=diarize, acoustic_check=acoustic, nli_check=nli)

st.title("Evidence-Traced Meeting Assistant")
st.caption("Every word in the output is traceable back to the audio.")

# ---------------------------------------------------------------------------
# Stage 0 - upload + automatic validation
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload a meeting recording (English)", type=None)
if uploaded is None:
    for k in ("sess", "upload_id", "upload_error"):
        st.session_state.pop(k, None)
    st.stop()

upload_id = f"{uploaded.name}:{uploaded.size}"
if st.session_state.get("upload_id") != upload_id:  # new file -> validate it once
    st.session_state["upload_id"] = upload_id
    st.session_state.pop("sess", None)
    st.session_state.pop("upload_error", None)
    upload_dir = settings.output_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / Path(uploaded.name).name
    path.write_bytes(uploaded.getvalue())
    try:
        with st.spinner("Stage 0: checking the file..."):
            st.session_state["sess"] = stage0_validate(path, settings, lambda *a: None)
    except PipelineError as e:
        st.session_state["upload_error"] = str(e)
    except Exception as e:
        st.session_state["upload_error"] = f"The file could not be processed ({type(e).__name__}: {e})."

if "upload_error" in st.session_state:
    st.error(f"**Stage 0 - file rejected:** {st.session_state['upload_error']}")
    st.stop()

sess = st.session_state["sess"]
sess.settings = settings  # apply the current sidebar options to the next stage that runs
m = sess.meta
st.success(
    f"**Stage 0 - file OK:** `{m['file']}` · {m['format'].upper()} · {fmt_ts(m['duration_sec'])} of audio "
    f"({m.get('codec')}, {m.get('source_sample_rate')} Hz). Ready for transcription."
)
st.audio(str(sess.audio_wav))

# ---------------------------------------------------------------------------
# Stage 1 - raw transcript
# ---------------------------------------------------------------------------
st.header("Stage 1 · Raw transcript")
if st.button("Generate raw transcript", type="primary" if sess.transcript is None else "secondary"):
    _run_stage("Stage 1: transcribing", lambda cb: stage1_transcribe(sess, glossary_text, attendees_text, cb))

if sess.transcript is None:
    st.stop()
tr = sess.transcript
raw_txt = transcript_text(tr.segments, tr.diarized)
st.caption(f"{len(tr.segments)} segments · {len(tr.words)} words · speakers "
           f"{'diarized' if tr.diarized else 'not identified'} · model `{tr.asr_model}`")
st.text_area("Raw transcript (speech-to-text output, before any LLM)", raw_txt, height=300)
st.download_button("Download raw transcript (.txt)", raw_txt, "raw_transcript.txt")

# ---------------------------------------------------------------------------
# Stage 2 - refined transcript
# ---------------------------------------------------------------------------
st.header("Stage 2 · Refined transcript")
if st.button("Generate refined transcript", type="primary" if sess.refinement is None else "secondary"):
    _run_stage("Stage 2: refining terminology", lambda cb: stage2_refine(sess, cb))

if sess.refinement is None:
    st.stop()
ref = sess.refinement
refined_txt = transcript_text(ref.refined_segments, tr.diarized)
n_acc = sum(v.accepted for v in ref.verdicts)
st.caption(f"{n_acc} of {len(ref.verdicts)} proposed edits accepted · {len(ref.candidates)} candidate spans · "
           f"acoustic verifier: {ref.acoustic_model if ref.acoustic_used else 'not used'}")
for w in ref.warnings:
    st.warning(w)

t_diff, t_side, t_ref, t_log = st.tabs(["Changes", "Side by side", "Refined transcript", "Refinement log"])
with t_diff:
    changed = [s for s in ref.refined_segments if s.edit_ids]
    if not changed:
        st.info("No edits were accepted - the refined transcript equals the raw transcript.")
    for s in changed:
        st.markdown(f"`{s.id}` [{fmt_ts(s.start)}] " + _diff_html(s.raw_text, s.text), unsafe_allow_html=True)
with t_side:
    c1, c2 = st.columns(2)
    c1.markdown("**Raw (ASR)**")
    c2.markdown("**Refined**")
    for s in ref.refined_segments:
        spk = f"{s.speaker}: " if s.speaker else ""
        c1.markdown(f"`{s.id}` {spk}{html.escape(s.raw_text)}", unsafe_allow_html=True)
        c2.markdown(f"`{s.id}` {spk}" + _diff_html(s.raw_text, s.text), unsafe_allow_html=True)
with t_ref:
    st.text_area("Refined transcript", refined_txt, height=300)
with t_log:
    st.caption(f"Glossary: {', '.join(g.term for g in ref.glossary) or '(empty)'}")
    rows = [
        {
            "id": e.edit_id, "seg": e.segment_id, "accepted": e.accepted, "original": e.located_text or e.original,
            "replacement": e.replacement, "type": e.edit_type, "evidence": ", ".join(e.sources),
            "ASR p": None if e.mean_prob is None else round(e.mean_prob, 2),
            "Δlogp": None if e.acoustic_delta is None else round(e.acoustic_delta, 2),
            "protected": ", ".join(e.protected_hits), "LLM conf": e.confidence,
            "verdict": e.verdict_reason, "LLM reason": e.reason,
        }
        for e in ref.verdicts
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("LLM #1 proposed no edits.")
st.download_button("Download refined transcript (.txt)", refined_txt, "refined_transcript.txt")

# ---------------------------------------------------------------------------
# Stage 3 - minutes, decisions, action items
# ---------------------------------------------------------------------------
st.header("Stage 3 · Summary, minutes, decisions and action items")
if st.button("Generate meeting record", type="primary" if sess.record is None else "secondary"):
    _run_stage("Stage 3: documenting the meeting", lambda cb: stage3_document(sess, cb))

rec = sess.record
if rec is None:
    st.stop()
audio = sess.audio

if rec.warnings:
    with st.expander(f"⚠ {len(rec.warnings)} pipeline warnings"):
        for w in rec.warnings:
            st.write("- " + w)

t_sum, t_dec, t_act, t_dl = st.tabs(["Summary & minutes", "Decisions", "Action items", "Downloads"])
with t_sum:
    st.subheader("Summary")
    st.write(rec.summary or "_No summary._")
    st.subheader("Minutes")
    for sec in rec.minutes:
        st.markdown(f"**{sec.topic}**")
        for p in sec.points:
            cite = ", ".join(p.segment_ids) if p.segment_ids else "uncited"
            st.markdown(f"- {p.text} <span style='color:gray'>({cite})</span>", unsafe_allow_html=True)
    not_adopted = rec.rejected_proposals + rec.deferred_proposals + rec.unresolved_proposals
    if not_adopted:
        st.subheader("Proposals not adopted")
        for p in not_adopted:
            st.markdown(f"- **{p.status}** - {p.proposal} ({', '.join(p.provenance.segment_ids)})")

with t_dec:
    if not rec.decisions:
        st.info("No decisions were reached (an empty list is a valid result).")
    for d in rec.decisions:
        with st.expander(f"{d.id}: {d.decision}  ·  {fmt_ts(d.provenance.start)}"):
            for f in d.flags:
                st.warning(f)
            st.audio(clip_wav_bytes(audio, d.provenance.start, d.provenance.end), format="audio/wav")
            st.markdown("**Lifecycle**")
            for e in d.history:
                spk = f" ({e.speaker})" if e.speaker else ""
                st.markdown(f"- `{e.status}` at {e.segment_id} [{fmt_ts(e.start)}]{spk}: “{e.quote}”")

with t_act:
    if not rec.action_items:
        st.info("No action items were assigned.")
    for a in rec.action_items:
        owner = a.owner + (" (speaker label)" if a.owner_source == "speaker_label" else "")
        with st.expander(f"{a.id}: {a.task}  ·  owner: {owner}  ·  deadline: {a.deadline}"):
            if a.owner_annotation:
                st.caption(f"Owner annotation (not a stated owner): {a.owner_annotation}")
            for f in a.flags:
                st.warning(f)
            for n in a.verification_notes:
                st.caption("Verification: " + n)
            st.audio(clip_wav_bytes(audio, a.provenance.start, a.provenance.end), format="audio/wav")
            for q in a.provenance.quotes:
                st.markdown(f"> {q}")
    if rec.unconfirmed_requests:
        st.subheader("Requests not confirmed in the meeting")
        for r in rec.unconfirmed_requests:
            st.markdown(f"- {r.id}: {r.task} ({', '.join(r.provenance.segment_ids)})")

with t_dl:
    st.download_button("Meeting record (JSON)", rec.model_dump_json(indent=2), "record.json", "application/json")
    st.download_button("Meeting record (Markdown)", to_markdown(rec), "record.md", "text/markdown")
    st.download_button("Raw transcript (.txt)", raw_txt, "raw_transcript.txt", key="dl_raw_final")
    st.download_button("Refined transcript (.txt)", refined_txt, "refined_transcript.txt", key="dl_ref_final")
    st.caption(f"Also saved to `{sess.output_dir}`")
