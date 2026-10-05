"""Acoustic verification by teacher-forced scoring.

The same Whisper checkpoint (HF transformers) scores the total log-likelihood of a
whole segment's text given its audio - once as transcribed, once with the edit -
using the same glossary prompt for both. No length normalisation: both hypotheses
explain the same audio. The encoder runs once per segment; only the decoder is re-run.
"""
from __future__ import annotations

import numpy as np

from .utils import free_gpu


class AcousticScorer:
    def __init__(self, model_id: str, device: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_id)
        # torch_dtype (not dtype) works on old and new transformers; loading fp32 would not fit 6 GB.
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        )
        tok = self.processor.tokenizer
        if model_id.endswith(".en"):
            tok.set_prefix_tokens(predict_timestamps=False)
        else:
            tok.set_prefix_tokens(language="english", task="transcribe", predict_timestamps=False)
        self.tok = tok
        self.prefix = list(tok.prefix_tokens)  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        self.sot_prev = tok.convert_tokens_to_ids("<|startofprev|>")
        self.eot = tok.convert_tokens_to_ids("<|endoftext|>")
        self.max_len = int(self.model.config.max_target_positions)

    def encode(self, audio_16k: np.ndarray):
        feats = self.processor.feature_extractor(audio_16k, sampling_rate=16000, return_tensors="pt").input_features
        with self.torch.inference_mode():
            return self.model.get_encoder()(feats.to(self.device, self.dtype)).last_hidden_state

    def logprob(self, enc, text: str, prompt: str | None = None) -> float:
        torch = self.torch
        prompt_ids: list[int] = []
        if prompt:
            p = self.tok.encode(" " + prompt.strip(), add_special_tokens=False)
            prompt_ids = [self.sot_prev] + p[-(self.max_len // 2 - 1) :]
        target = self.tok.encode(" " + text.strip(), add_special_tokens=False) + [self.eot]
        ids = prompt_ids + self.prefix + target
        if len(ids) > self.max_len:
            prompt_ids = []
            ids = self.prefix + target
        if len(ids) > self.max_len:
            raise ValueError("segment text too long for the Whisper decoder")
        n_ctx = len(prompt_ids) + len(self.prefix)
        with torch.inference_mode():
            inp = torch.tensor([ids[:-1]], device=self.device)
            logits = self.model(encoder_outputs=(enc,), decoder_input_ids=inp).logits[0].float()
            logp = torch.log_softmax(logits, dim=-1)
            labels = torch.tensor(ids[1:], device=self.device)
            tok_lp = logp.gather(1, labels[:, None]).squeeze(1)
        # position i predicts ids[i+1]; target tokens start at ids[n_ctx]
        return float(tok_lp[n_ctx - 1 :].sum())

    def close(self) -> None:
        del self.model
        free_gpu()
