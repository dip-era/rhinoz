# Evidence-Traced Meeting Assistant

**Every word in the output is traceable back to the audio.**

The usual pipeline is Whisper → prompt → prompt → JSON. Ours doesn't trust either LLM. At each stage the LLM only *proposes* structured claims with citations, and deterministic code decides what survives. It decides using evidence that comes from the audio: word confidences, phonetic matches, Whisper log-likelihoods, and verbatim quotes tied to segment ids.

```
 upload ─► Stage 0  validate + decode (PyAV, no ffmpeg) ─► 16 kHz mono
           Stage 1  faster-whisper (VAD → ASR, word timestamps + probabilities) [+ pyannote diarization]
                         │ raw transcript (segments S0001…, words with p)
           Stage 2  LLM #1 = constrained surgeon
                    glossary (user terms + attendees + LLM-inferred, grounded)
                    candidate spans = low confidence ∪ Double-Metaphone n-gram match ∪ LLM flag
                    LLM #1 returns {span_id, original, replacement, edit_type, reason, confidence}
                    → locate verbatim → protected-token guard → teacher-forced Whisper scoring
                    → deterministic verdict → apply   (every edit + verdict logged)
                         │ refined transcript
           Stage 3a LLM #2 (different model): speech acts + proposal/task lifecycle events per chunk,
                    with a running open-proposals/tasks state carried across chunks
           Stage 4  decisions = proposals that reached "accepted"; owner/deadline must be found verbatim
                    in the cited segment (or come from the speaker label) else "unspecified"; optional NLI flags
           Stage 3b LLM #2: summary + minutes, written to agree with the verified lists
                         │
           record.json (canonical) ─► record.md (rendered FROM the JSON) + raw/refined transcripts
```

## Models and their roles

| Stage | Model | Role |
|---|---|---|
| Speech-to-text | **faster-whisper `large-v3-turbo`** (CTranslate2, int8_float16; the only supported model) | VAD-filtered transcription with word timestamps and probabilities |
| Diarization (optional) | **pyannote/speaker-diarization-3.1** | Speaker labels, used for self-commitment owners ("I'll do it") |
| LLM #1 – refinement | **Groq `llama-3.3-70b-versatile`** | Proposes glossary terms and minimal structured edits. Never rewrites text. |
| Acoustic verifier | **`openai/whisper-large-v3-turbo`** (HF transformers, same checkpoint as the ASR) | Teacher-forced log-likelihood of each segment with vs. without an edit |
| LLM #2 – documentation | **Groq `openai/gpt-oss-120b`** | Speech-act tagging, proposal/task lifecycle events, summary and minutes |
| NLI flags (optional) | **cross-encoder/nli-deberta-v3-small** | Flags weakly supported decisions/tasks in the UI. Never deletes anything. |

You can change all of these in `.env`. Groq's model lineup and free-tier limits change over time, so check console.groq.com.

---

## Setup (Windows, NVIDIA GPU)

> **Use Python 3.11.** This machine currently has only Python 3.14, and torch, ctranslate2 and pyannote may not have working 3.14 wheels.

```powershell
winget install Python.Python.3.11          # or: pip install uv ; uv venv -p 3.11 .venv
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements-optional.txt    # diarization / NLI / TTS / AMI (optional)
copy .env.example .env                      # then add GROQ_API_KEY (and HF_TOKEN for diarization)
python -m pytest tests -q                   # offline tests, no GPU / API key needed
```

* **No system ffmpeg is needed.** Audio is probed and decoded with PyAV, which ships with faster-whisper.
* **cuDNN on Windows.** `asr.py` imports torch before faster-whisper so that torch's bundled cuBLAS/cuDNN DLLs get loaded. If you still see `cudnn_ops64_9.dll not found`, the process crashes hard and can't be caught. Install cuDNN 9 for CUDA 12, or run with `ASR_DEVICE=cpu`.
* **Diarization.** Accept the terms for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on huggingface.co, then set `HF_TOKEN` and `DIARIZE=true`.
* **VRAM (6 GB).** Only one model is in memory at a time: faster-whisper → pyannote → HF Whisper → NLI. Each one is freed before the next loads.

## Run

```powershell
streamlit run app.py                                         # UI
python run_cli.py meeting.mp3 --glossary "Kubernetes, Grafana" --attendees "Priya, Rahul" --diarize
```

Outputs are written to `outputs/<timestamp>_<name>/`: `record.json`, `record.md`, `raw_transcript.txt`, `refined_transcript.txt` and `audio_16k.wav`.

---

## What every Python file does

### `meeting_assistant/` (the pipeline package)

| File | Purpose |
|---|---|
| `__init__.py` | Package marker and the project's core thesis. |
| `config.py` | `Settings` dataclass with every model name, threshold and path, loaded from `.env`. Also `ASR_TO_HF`, which maps each faster-whisper model to the identical HF checkpoint the acoustic verifier uses. |
| `errors.py` | `PipelineError`: an exception whose message is shown verbatim in the UI. Used for empty, unsupported and unreadable files, rate limits, and similar failures. |
| `schemas.py` | Pydantic schemas for everything passed between stages: `Word`/`Segment`/`Transcript`, `CandidateSpan`, `EditVerdict`, lifecycle state (`Proposal`, `Task`), the LLM output contracts (`*Out`), and the canonical `MeetingRecord`. |
| `utils.py` | Shared helpers: text normalization, timestamp formatting, word-budget chunking, `fuzzy_contains` (quote checking), `find_verbatim` (copies owner/deadline text *from the transcript*), and GPU cleanup. |
| `audio_io.py` | **Stage 0.** Checks the extension, empty files, a readable container, an audio track, minimum length and silence. Decodes to 16 kHz mono. Also writes WAVs and cuts the evidence clips for the UI. |
| `asr.py` | **Stage 1.** faster-whisper with `vad_filter=True`, word timestamps and `condition_on_previous_text=False`. User terms go in as `hotwords`. Falls back to CPU if CUDA fails to load. Builds segments (`S0001`…) and words with probabilities. |
| `diarization.py` | Optional pyannote diarization, called directly rather than through WhisperX to avoid its pinned versions. Assigns a speaker to each word by overlap, smooths one-word flips, and splits segments at speaker changes. |
| `speakers.py` | **Speaker naming** (Stage 1, after diarization). The LLM #2 model reads the transcript in chunks of under 2,000 words and claims `self_intro` ("I'm Rose", weight 3) or `addressed` ("Thanks, Bob" and Bob answers next, weight 1) evidence. Each claim is checked against the transcript: the name must be spoken in the cited segment, a self-intro must come from that label, and an addressed person must reply within 2 turns. A weighted vote per label picks the name and role; ties or a name claimed by two labels stay unnamed. Transcripts show `Rose (Project Manager)`, otherwise `SPEAKER_xx`. Only a self-introduced name may become an action-item owner; an addressed-only name is an annotation. |
| `llm_client.py` | Groq wrapper. JSON-only replies, Pydantic validation with up to 2 repair retries, tokens-per-minute pacing, and a **disk cache** to save free-tier quota. Also handles truncation, `json_validate_failed`, and clear errors for 429/401/413. |
| `prompts.py` | All LLM instructions: LLM #1 glossary + constrained edit proposal; LLM #2 speech acts + lifecycle events, notes (map step) and summary/minutes. Each prompt specifies a strict JSON output format. |
| `glossary.py` | Merges user terms, attendee names and LLM-#1-proposed terms. LLM terms are kept only if grounded (the term appears, or a `heard_as` phrase is found verbatim in the cited segment). Builds the Whisper glossary prompt. |
| `phonetics.py` | Double Metaphone on **word n-grams with word boundaries removed** ("cube earnest" ≈ "Kubernetes"), with OSA similarity and guards for short codes. |
| `candidates.py` | Candidate spans = (a) runs of low-confidence words ∪ (b) phonetic n-gram matches against the glossary, with non-max suppression and merging. (c) LLM flags are added in `refine.py`. |
| `guards.py` | Protected-token guard. Detects changes to **numbers** (spelled-out numbers are canonicalized, so "fifteen" == "15"), **negations**, **modal/commitment words** (will/might/should…) and **names**. Also contains the number parser the eval metrics use. |
| `acoustic.py` | `AcousticScorer`: teacher-forced total log-likelihood of a segment's text given its audio, using the same Whisper checkpoint and the same glossary prompt for both hypotheses, with no length normalization. The encoder runs once per segment. |
| `refine.py` | **Stage 2 orchestration:** glossary → candidates → LLM #1 edits → locate verbatim (a hallucinated `original` is rejected) → check formatting claims (a "formatting" edit that changes the spoken words is reclassified as acoustic) → protected-token check → acoustic scoring → `decide()` → overlap resolution → `apply_edits()`. These are pure functions so thresholds can be retuned offline. |
| `documentation.py` | **Stage 3.** `extract_lifecycle`: one LLM-#2 call per chunk that tags speech acts and emits proposal/task events, with open state carried across chunks. Every event needs a verbatim quote from a segment in the current chunk. An acceptance must sit on an agreement/decision/commitment act and can't come from the proposer alone. `summarize`: summary + minutes with segment citations (map-reduce for long meetings). |
| `verification.py` | **Stage 4.** Decisions = accepted proposals only. An owner must be found verbatim (fuzzy) in a cited segment, or be the speaker label of a first-person commitment; otherwise it's `unspecified`. Names inferred from context go only to `owner_annotation`. Deadlines are copied verbatim, never converted to dates. Optional DeBERTa-MNLI adds UI flags. |
| `record.py` | Builds the canonical `MeetingRecord`, renders Markdown **from** it (empty lists render as "No decisions were reached"), writes transcript text files and saves all outputs. |
| `pipeline.py` | Stage functions the UI calls one at a time (`stage0_validate`, `stage1_transcribe`, `stage2_refine`, `stage3_document`, sharing a `PipelineSession`), plus `run_pipeline()` chaining them for the CLI/eval: Stage 0 → 1 → 2 → 3a → 4 → 3b → record, with a progress callback and sequential model loading. LLM clients are created before ASR so a missing key fails fast. |

### Top level

| File | Purpose |
|---|---|
| `app.py` | Step-by-step Streamlit UI: upload → automatic Stage 0 check (OK message or clear error) → button for Stage 1 (raw transcript shown) → button for Stage 2 (refined transcript, diff, edit log) → button for Stage 3 (summary/minutes, decisions, action items, downloads). Tabs for summary/minutes, decisions (**expand to play the audio clip** and see the lifecycle), action items (owner source, annotations, evidence clip), raw/refined/diff transcripts, the refinement log (every edit with ASR p, Δlogp, protected hits and verdict), and downloads. |
| `run_cli.py` | The same pipeline from the command line, for evaluation and debugging. |

### `eval/` (evaluation, our main differentiator)

| File | Purpose |
|---|---|
| `answer_keys/*.json` | Two scripted meetings (speaker turns = reference transcript) with answer keys: glossary terms to score, decisions, *non*-decisions (rejected/deferred/unresolved proposals), and action items with owner/deadline or `null`. They cover jargon, rejected proposals, tasks with no owner, numbers and negations. |
| `make_synthetic_audio.py` | `--print-script` prints a script for your team to read aloud. Without the flag it renders a TTS WAV (pyttsx3) as a **smoke test only**. |
| `metrics.py` | WER (numbers normalized), term error rate on glossary terms, protected-token violations (correct in raw but broken in refined), decision P/R + proposal leaks, action P/R + invented owners/deadlines. |
| `run_eval.py` | Runs the pipeline on each recording that has a key (or `--reuse` existing outputs), then writes `eval/results.md` and per-recording JSON. `--no-acoustic` gives the ablation. |
| `tune_thresholds.py` | Grid-searches `tau_conf` / `tau_glossary` / `strong_margin` **offline** from saved `record.json` files. No models or LLM calls are needed, because every Δlogp is logged. |
| `ami_wer.py` | (NICE) Raw ASR WER on a sample of AMI IHM test utterances. |

### `tests/`

| File | Purpose |
|---|---|
| `test_offline.py` | Runs with no GPU and no key, using a fake LLM. Covers phonetic n-gram matching, number/negation/modal guards, verbatim matching, evidence gating (a glossary edit is accepted; a negation removal and a hallucinated original are rejected), the proposal lifecycle (accepted vs rejected), owner rules (speaker label vs unspecified, context name as annotation only, invented owner/deadline removed), and Markdown for empty lists. |

---

## Key design decisions

1. **Edits, not rewrites.** LLM #1 returns `{span_id, original, replacement, edit_type, reason, confidence}`. An `original` that isn't found verbatim in the segment is rejected as a hallucination. Deletions and rewrite-length replacements are rejected.
2. **Confidence sets the evidence bar, not permission to edit.** Every edit has to pass the acoustic test: `Δlogp = logp(edit) − logp(raw)` must satisfy `Δ ≥ −[tau_conf·(1 − mean word p) + tau_glossary·(phonetic glossary match)]`. Whisper's LM prior penalizes rare terms, which is why phonetic support buys some tolerance.
3. **Protected tokens** (numbers, negations, modals, names) need `Δ ≥ +strong_margin`, meaning the audio must actually *prefer* the edit. Without the acoustic model, these edits are always rejected. Name protection is waived only when the replacement phonetically matches a glossary term or attendee (e.g. a capitalized "Cube Earnest" → "Kubernetes").
4. **Formatting claims are verified.** "formatting" is accepted only if the edit leaves the letters and digits unchanged (`api` → `API`). Anything else is reclassified as acoustic and scored.
5. **Lifecycle events, not prose.** LLM #2 emits events. Code keeps the state and enforces the rules: decisions = accepted proposals only; an acceptance needs an agreement/decision/commitment act from someone other than the proposer; a task linked to a proposal that wasn't accepted is demoted to "unconfirmed".
6. **Owner/deadline text is copied from the transcript** (`find_verbatim`), never from the LLM's wording.
7. **One canonical JSON**, with Markdown rendered from it, so both formats always contain the same decisions and tasks.

## Evaluation results

> **TBD.** Fill this in by running `python -m eval.run_eval` (and `--no-acoustic` for the ablation) on your **human** recordings. The numbers below are placeholders, not results.

| Recording | WER raw | WER refined | Term err raw | Term err refined | Protected violations | Decision P | Decision R | Proposal leaks | Invented owners | Invented deadlines |
|---|---|---|---|---|---|---|---|---|---|---|
| meeting1_platform_sync | – | – | – | – | – | – | – | – | – | – |
| meeting2_ml_sync | – | – | – | – | – | – | – | – | – | – |

To record: `python -m eval.make_synthetic_audio eval/answer_keys/meeting1_platform_sync.json --print-script`, read it with 3 people, save it as the `audio` path in the key, then run the eval. Targets: protected violations = 0 and invented owners/deadlines = 0.

## Known limitations

* **Without diarization**, an owner from "I'll do it" is `unspecified` (with an annotation). Speaker labels are `SPEAKER_00`-style, not names. A name heard in context is shown only as an annotation.
* **Thresholds are tuned on 2–3 recordings.** Keep the grid coarse and treat the values as approximate.
* **Eval item matching uses fuzzy text** (a proxy). `run_eval` prints every match so you can hand-check it.
* **Groq free-tier daily token caps** can stop a run partway. The disk cache means a re-run only pays for the calls that didn't finish.

## Remaining-days plan (with fallbacks)

| Day | Person A (ASR/infra) | Person B (refinement) | Person C (docs/eval) |
|---|---|---|---|
| 2 | Python 3.11 env, CUDA, run end-to-end with `ACOUSTIC_CHECK=false`. **Test pyannote install today.** | Run `tests`, inspect candidates/edits on the first recording | Record the 2 scripted meetings, run `run_eval` |
| 3 | Diarization on; if it still fails by noon → **drop it**, keep `unspecified` + annotations | Acoustic verifier on; tune with `tune_thresholds`. If it's unstable by evening → **fallback rules** (already implemented, automatic) | Fix prompt failures found by eval; NLI flags only if time allows |
| 4 | Freeze code by midday, demo recording, sample outputs | Final ablation table (acoustic on/off) | README results table, demo video |
