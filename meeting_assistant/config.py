"""All tunable settings in one place. Values come from environment / .env, with defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent

# The only supported ASR model, and the same checkpoint in HF transformers.
# The acoustic verifier must score with the same weights the ASR used.
ASR_MODEL = "large-v3-turbo"
ASR_TO_HF = {ASR_MODEL: "openai/whisper-large-v3-turbo"}


def _env(name: str, default):
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_num(name: str, default, cast=float):
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return cast(v)


@dataclass
class Settings:
    # ---- Stage 1: ASR -------------------------------------------------------
    asr_model: str = ASR_MODEL  # fixed: large-v3-turbo only
    asr_device: str = "auto"  # auto | cuda | cpu
    asr_compute_type: str = "int8_float16"  # used on CUDA; CPU always uses int8
    asr_beam_size: int = 5

    # ---- Diarization (SHOULD) ----------------------------------------------
    diarize: bool = False
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    # sentence-level voice re-check: catches short interjections pyannote folds into someone else's turn
    diarization_recheck: bool = True
    embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    recheck_min_sim: float = 0.45  # the other speaker must match at least this well (cosine)
    recheck_margin: float = 0.20  # ... and this much better than the currently assigned speaker
    recheck_min_dur: float = 0.6  # shorter sentences give unreliable voice embeddings
    hf_token: str | None = None
    num_speakers: int | None = None

    # ---- Speaker naming (needs diarization) -------------------------------
    speaker_naming: bool = True
    speaker_id_model: str | None = None  # default: the LLM #2 model
    speaker_id_chunk_words: int = 2000  # < 2,500 words per call (fits the 8k tokens/minute free tier)

    # ---- LLMs (Groq free tier) ---------------------------------------------
    groq_api_key: str | None = None
    llm1_model: str = "openai/gpt-oss-20b"  # LLM #1: refinement (proposes edits)
    llm2_model: str = "openai/gpt-oss-120b"  # LLM #2: documentation
    supervisor_model: str = "qwen/qwen3.8-27b"  # LLM #3: approves/rejects every refinement edit
    supervise_edits: bool = True
    supervisor_batch: int = 20  # edits reviewed per supervisor call
    llm1_tpm: int = 8000  # tokens/minute pacing (set to your Groq limit)
    llm2_tpm: int = 8000
    supervisor_tpm: int = 6000
    llm1_reasoning_effort: str = "low"  # extraction-style task; higher effort can exhaust the output budget
    llm2_reasoning_effort: str = "medium"  # only sent to reasoning models (gpt-oss)
    use_llm_cache: bool = True

    # ---- Stage 2: refinement -----------------------------------------------
    low_conf: float = 0.5  # word probability below this => candidate span
    phonetic_threshold: float = 0.8  # Double-Metaphone similarity for a glossary match
    max_ngram: int = 3
    glossary_chunk_words: int = 2500
    refine_chunk_words: int = 900
    acoustic_check: bool = True
    acoustic_model: str | None = None  # default: HF twin of asr_model
    # Evidence thresholds (log-likelihood, nats). TUNE these with eval/tune_thresholds.py.
    tau_conf: float = 6.0  # allowed log-lik drop = tau_conf * (1 - mean word prob) ...
    tau_glossary: float = 3.0  # ... + tau_glossary if replacement is a phonetically matching glossary term
    strong_margin: float = 0.5  # protected-token edits must IMPROVE log-lik by this much
    no_acoustic_min_llm_conf: float = 0.75  # fallback rule when acoustic check is unavailable

    # ---- Stage 3: documentation --------------------------------------------
    doc_chunk_words: int = 600  # keeps prompt + output under gpt-oss-120b's 8k tokens/minute free-tier limit
    summary_max_words: int = 1800  # above this, summary uses map-reduce notes

    # ---- Stage 4: verification ---------------------------------------------
    quote_match_threshold: float = 85.0  # rapidfuzz partial_ratio for quotes/owners/deadlines
    nli_check: bool = False
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_threshold: float = 0.5

    # ---- Paths --------------------------------------------------------------
    cache_dir: Path = field(default_factory=lambda: ROOT / ".cache")
    output_dir: Path = field(default_factory=lambda: ROOT / "outputs")

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.asr_device = _env("ASR_DEVICE", s.asr_device)
        s.asr_compute_type = _env("ASR_COMPUTE_TYPE", s.asr_compute_type)
        s.asr_beam_size = _env_num("ASR_BEAM_SIZE", s.asr_beam_size, int)
        s.diarize = _env_bool("DIARIZE", s.diarize)
        s.diarization_model = _env("DIARIZATION_MODEL", s.diarization_model)
        s.diarization_recheck = _env_bool("DIARIZATION_RECHECK", s.diarization_recheck)
        s.hf_token = _env("HF_TOKEN", None)
        ns = _env("NUM_SPEAKERS", None)
        s.num_speakers = int(ns) if ns else None
        s.groq_api_key = _env("GROQ_API_KEY", None)
        s.speaker_naming = _env_bool("SPEAKER_NAMING", s.speaker_naming)
        s.speaker_id_model = _env("SPEAKER_ID_MODEL", None)
        s.llm1_model = _env("LLM1_MODEL", s.llm1_model)
        s.llm2_model = _env("LLM2_MODEL", s.llm2_model)
        s.supervisor_model = _env("SUPERVISOR_MODEL", s.supervisor_model)
        s.supervise_edits = _env_bool("SUPERVISE_EDITS", s.supervise_edits)
        s.supervisor_tpm = _env_num("SUPERVISOR_TPM", s.supervisor_tpm, int)
        s.llm1_tpm = _env_num("LLM1_TPM", s.llm1_tpm, int)
        s.llm2_tpm = _env_num("LLM2_TPM", s.llm2_tpm, int)
        s.llm1_reasoning_effort = _env("LLM1_REASONING_EFFORT", s.llm1_reasoning_effort)
        s.llm2_reasoning_effort = _env("LLM2_REASONING_EFFORT", s.llm2_reasoning_effort)
        s.use_llm_cache = _env_bool("USE_LLM_CACHE", s.use_llm_cache)
        s.low_conf = _env_num("LOW_CONF", s.low_conf)
        s.phonetic_threshold = _env_num("PHONETIC_THRESHOLD", s.phonetic_threshold)
        s.acoustic_check = _env_bool("ACOUSTIC_CHECK", s.acoustic_check)
        s.acoustic_model = _env("ACOUSTIC_MODEL", None)
        s.tau_conf = _env_num("TAU_CONF", s.tau_conf)
        s.tau_glossary = _env_num("TAU_GLOSSARY", s.tau_glossary)
        s.strong_margin = _env_num("STRONG_MARGIN", s.strong_margin)
        s.nli_check = _env_bool("NLI_CHECK", s.nli_check)
        s.nli_model = _env("NLI_MODEL", s.nli_model)
        s.output_dir = Path(_env("OUTPUT_DIR", str(s.output_dir)))
        s.cache_dir = Path(_env("CACHE_DIR", str(s.cache_dir)))
        return s

    @property
    def acoustic_model_id(self) -> str | None:
        return self.acoustic_model or ASR_TO_HF.get(self.asr_model)
