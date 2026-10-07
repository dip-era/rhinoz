"""Evidence-traced meeting assistant.

Core thesis: every word in the output is traceable back to the audio. Each stage
produces evidence (word confidences, phonetic matches, acoustic log-likelihoods,
verbatim quotes, segment ids) and deterministic code - not the LLM - decides what
survives into the final record.
"""

__version__ = "0.1.0"
