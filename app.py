"""Streamlit UI: upload -> process (live status) -> inspect transcripts, diff, decisions
(with playable evidence clips), action items -> download.

Run:  streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import difflib
import html
import traceback
from pathlib import Path

import streamlit as st

from meeting_assistant.audio_io import clip_wav_bytes, read_wav
from meeting_assistant.config import ASR_TO_HF, Settings
from meeting_assistant.errors import PipelineError
from meeting_assistant.pipeline import run_pipeline
from meeting_assistant.record import to_markdown, transcript_text
from meeting_assistant.utils import fmt_ts

st.set_page_config(page_title="Evidence-Traced Meeting Assistant", layout="wide")


@st.cache_data(show_spinner=False)
def _load_audio(path: str):
    return read_wav(path)


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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
base = Settings.from_env()
with st.sidebar:
    st.header("Settings")
    asr_models = list(ASR_TO_HF)
    asr_model = st.selectbox("Whisper model", asr_models, index=asr_models.index(base.asr_model) if base.asr_model in asr_models else 0)
    diarize = st.checkbox("Speaker diarization (pyannote, needs HF_TOKEN)", value=base.diarize)
    acoustic = st.checkbox("Acoustic verification of edits", value=base.acoustic_check)
    nli = st.checkbox("NLI support flags (DeBERTa)", value=base.nli_check)
    st.caption(f"LLM #1 (refinement): `{base.llm1_model}`")
    st.caption(f"LLM #2 (documentation): `{base.llm2_model}`")
    if not base.groq_api_key:
        st.error("GROQ_API_KEY is not set - add it to .env")
    st.divider()
    glossary_text = st.text_area("Agenda / domain terms (optional)", placeholder="Kubernetes, Grafana, PostgreSQL ...", height=110)
    attendees_text = st.text_area("Attendee names (optional)", placeholder="Priya, Rahul, Ankit", height=70)

settings = dataclasses.replace(base, asr_model=asr_model, diarize=diarize, acoustic_check=acoustic, nli_check=nli)

# ---------------------------------------------------------------------------
# Upload + run
# ---------------------------------------------------------------------------
st.title("Evidence-Traced Meeting Assistant")
st.caption("Every word in the output is traceable back to the audio.")

uploaded = st.file_uploader("Upload a meeting recording (English)", type=None)
if st.button("Process recording", type="primary", disabled=uploaded is None):
    upload_dir = settings.output_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / Path(uploaded.name).name
    path.write_bytes(uploaded.getvalue())
    st.session_state.pop("result", None)
    with st.status("Processing...", expanded=True) as status:
        bar = st.progress(0.0)

        def cb(stage: str, msg: str, frac: float | None = None):
            status.update(label=f"{stage}: {msg}")
            st.write(f"**{stage}** - {msg}")
            if frac is not None:
                bar.progress(min(max(frac, 0.0), 1.0))

        try:
            res = run_pipeline(path, glossary_text, attendees_text, settings, cb)
            st.session_state["result"] = res
            status.update(label="Done", state="complete", expanded=False)
        except PipelineError as e:
            status.update(label=f"Failed at {e.stage}", state="error")
            st.error(f"**{e.stage}:** {e}")
        except Exception as e:
            status.update(label="Failed (unexpected error)", state="error")
            st.error(f"Unexpected error: {type(e).__name__}: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

res = st.session_state.get("result")
if not res:
    st.stop()

rec = res.record
audio = _load_audio(str(res.audio_wav))
segmap = {s.id: s for s in rec.refined_transcript}

if rec.warnings:
    with st.expander(f"⚠ {len(rec.warnings)} pipeline warnings"):
        for w in rec.warnings:
            st.write("- " + w)

tabs = st.tabs(["Summary & minutes", "Decisions", "Action items", "Transcripts", "Refinement log", "Downloads"])

with tabs[0]:
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

with tabs[1]:
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

with tabs[2]:
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

with tabs[3]:
    view = st.radio("View", ["Diff (changed segments)", "Side by side", "Raw", "Refined"], horizontal=True)
    if view.startswith("Diff"):
        changed = [s for s in rec.refined_transcript if s.edit_ids]
        if not changed:
            st.info("No edits were accepted - refined transcript equals the raw transcript.")
        for s in changed:
            st.markdown(f"`{s.id}` [{fmt_ts(s.start)}] " + _diff_html(s.raw_text, s.text), unsafe_allow_html=True)
    elif view == "Side by side":
        c1, c2 = st.columns(2)
        c1.markdown("**Raw (ASR)**")
        c2.markdown("**Refined**")
        for s in rec.refined_transcript:
            spk = f"{s.speaker}: " if s.speaker else ""
            c1.markdown(f"`{s.id}` {spk}{s.raw_text}")
            c2.markdown(f"`{s.id}` {spk}" + _diff_html(s.raw_text, s.text), unsafe_allow_html=True)
    elif view == "Raw":
        st.text(transcript_text(rec.raw_transcript.segments, rec.diarized))
    else:
        st.text(transcript_text(rec.refined_transcript, rec.diarized))

with tabs[4]:
    st.caption(
        f"Glossary: {', '.join(g.term for g in rec.glossary) or '(empty)'}  ·  "
        f"{len(rec.candidates)} candidate spans  ·  acoustic verifier: {rec.models.acoustic_verifier or 'not used'}"
    )
    rows = [
        {
            "id": e.edit_id, "seg": e.segment_id, "accepted": e.accepted, "original": e.located_text or e.original,
            "replacement": e.replacement, "type": e.edit_type, "evidence": ", ".join(e.sources),
            "ASR p": None if e.mean_prob is None else round(e.mean_prob, 2),
            "Δlogp": None if e.acoustic_delta is None else round(e.acoustic_delta, 2),
            "protected": ", ".join(e.protected_hits), "LLM conf": e.confidence,
            "verdict": e.verdict_reason, "LLM reason": e.reason,
        }
        for e in rec.edits
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("LLM #1 proposed no edits.")

with tabs[5]:
    st.download_button("Meeting record (JSON)", rec.model_dump_json(indent=2), "record.json", "application/json")
    st.download_button("Meeting record (Markdown)", to_markdown(rec), "record.md", "text/markdown")
    st.download_button("Raw transcript (.txt)", transcript_text(rec.raw_transcript.segments, rec.diarized), "raw_transcript.txt")
    st.download_button("Refined transcript (.txt)", transcript_text(rec.refined_transcript, rec.diarized), "refined_transcript.txt")
    st.caption(f"Also saved to `{res.output_dir}`")
