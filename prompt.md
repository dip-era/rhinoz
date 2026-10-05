You're a senior ML engineer helping my team build a hackathon submission. Be direct and push back when something won't work. Don't agree just to be agreeable. If something blocking is unclear, ask me before producing output.

## Problem statement
check the file ml_bootcamp_problem_statement.md

## Context
- Time limit: 4 days total. [3] days remaining.
- Compute: [local 24 GB RAM, NVIDIA GeForce RTX GPU with 6 GB dedicated VRAM + 11.9 GB shared GPU memory, Intel UHD integrated graphics, NVMe SSD]. LLM access: [only free api is allowed, preferrably qroq].
- Language: Python.
- The evaluators said they already know the straightforward solution (Whisper → prompt → prompt → JSON) and explicitly want an innovative approach. The rubric still gives 60/100 to faithfulness: transcript accuracy, refinement fidelity, decisions, and action items. So our novelty must live inside the method and improve accuracy, not be bolted-on features. Also we can do innovation in the overall architecture.

## Core thesis
"Every word in the output is traceable back to the audio." Each stage is evidence-driven instead of trusting the LLM.

## Agreed design

Stage 0 – Input handling
- Validate the upload: supported file format (.wav), unsupported format, empty, or unreadable files get a clear UI error (probe with ffmpeg).
- Run VAD before ASR (faster-whisper vad_filter=True, default thresholds) so Whisper doesn't hallucinate text over silence.

Stage 1 – ASR
- faster-whisper with word-level timestamps.
- Diarization via WhisperX (pyannote) is a "should," not a "must." The install must be tested on day 1, because the gated-model token and torch version conflicts are a known time sink.

Stage 2 – Refinement (LLM #1). This is a constrained surgeon, not a rewriter.
- Glossary: built from an optional user-supplied agenda/term list plus terms the LLM proposes from context (e.g. infer "Kubernetes" from "cube earnest" next to "pods" and "deployment").
- Candidate spans are the union of:
  (a) low ASR confidence,
  (b) a phonetic match (Double Metaphone on word n-grams, not single words) against glossary terms,
  (c) an LLM flag.
  Whisper confidence is poorly calibrated, so it sets how much evidence an edit needs, not whether a span can be edited.
- The LLM returns structured edits only, never a rewritten transcript:
  {span_id, original, replacement, edit_type: acoustic|formatting, reason, confidence}
- Protected-token guard: numbers, negations, and names can't change unless the evidence is strong.
- Acoustic verification via teacher-forced scoring. Load the same Whisper checkpoint in HF transformers. Compute the total log-likelihood of the full ~30s segment with vs. without the edit, using the same glossary prompt for both. No length normalization, because both hypotheses explain the same audio. Accept if the candidate is not much worse; tune the threshold on our scripted recordings (Whisper's LM prior penalizes rare terms). Skip this check for formatting-only edits (e.g. "api" → "API").
- Log every edit and its verdict so the UI can show a raw-vs-refined diff.

Stage 3 – Documentation (LLM #2: a DIFFERENT model from LLM #1)
- Tag speech acts in batches of utterances (not one call each): proposal, agreement, objection, commitment, question, info.
- Track each proposal's lifecycle: proposed → accepted / rejected / deferred / unresolved. Carry a running open-proposals state across chunks so a proposal at minute 5 can be resolved at minute 40.
- Decisions = only proposals that reached "accepted." Action items come from commitments and explicit assignments.
- Owner field is strictly what was stated. A self-commitment ("I'll handle it") gets the speaker label. A name inferred from context (e.g. "Thanks, Priya") is shown as a separate, clearly marked annotation, never in the owner field. Otherwise "unspecified."
- Deadlines are kept verbatim as spoken ("by Friday"). Never convert them to dates.
- Every decision and action item carries provenance: segment IDs and timestamps.
- Also produce a concise summary and organized minutes.

Stage 4 – Verification
- Deterministic check: the owner and deadline text must fuzzy-match inside the cited span, or the owner must be speaker-derived. Otherwise set the field to "unspecified."
- Optional DeBERTa-MNLI support check, used as a FLAG in the UI only, never to delete items (it's noisy on conversational speech).

Outputs and UI
- One canonical JSON record. The Markdown version is generated FROM the JSON, so both always agree. Empty decision/action lists are valid.
- Streamlit or Gradio (don't spend time on frontend polish). It needs: upload, processing status, clear errors, raw/refined diff view, click a decision to play its audio clip, and downloads.

Evaluation (our main differentiator)
- 2–3 scripted recordings we make ourselves, with an answer key. Include jargon, rejected proposals, tasks with no owner, numbers, and negations.
- Metrics:
  - term error rate on glossary terms, raw vs. refined
  - protected-token violations (target: 0)
  - decision precision/recall
  - invented owner/deadline count (target: 0)
- Optional: WER on a few AMI meetings via the HF dataset edinburghcstr/ami. Don't attempt AMI decision annotations; parsing and matching them costs too much time.
- A results table goes in the README.

## Priority tiers
- MUST:
  - baseline end-to-end pipeline working by end of day 1 (ugly is fine)
  - constrained edits + protected-token guards
  - lifecycle tracking with cross-chunk state
  - provenance + deterministic check
  - scripted recordings + results table
- SHOULD: teacher-forced acoustic scoring (~1 day including debugging), WhisperX diarization
- NICE: NLI flags, AMI WER
- AVOID: sentiment/emotion analysis, integrations, chatbots over the transcript, fine-tuning, multiple ASR ensembles, fancy frontend.

## Your task
[CHOOSE ONE, e.g.:]
1. Propose a module/file layout and the JSON schemas passed between stages (edits, speech acts, proposal state, final record).
2. Draft the prompts for LLM #1 (edit proposal) and LLM #2 (speech-act tagging + lifecycle), with strict JSON output formats.
3. Create a day-by-day plan split across our team members, with clear fallback points if a SHOULD item fails.

Before starting, flag anything in this design you think is wrong or too risky for our time and resources.