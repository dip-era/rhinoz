"""Candidate spans = union of (a) low ASR confidence, (b) phonetic glossary match.

(c) LLM flags are added later, when LLM #1 proposes an edit outside every candidate.
Candidates are hints that steer the LLM and set how much evidence an edit needs -
they never decide on their own whether a span may be edited.
"""
from __future__ import annotations

from .config import Settings
from .phonetics import term_match_score
from .schemas import CandidateSpan, GlossaryTerm, Transcript
from .utils import EDGE_PUNCT, norm_word

STOPWORDS = set(
    "a an the and or but so to of in on at for with by from as is are was were be been it its this that these "
    "those i you he she we they me him her us them my your our their um uh yeah ok okay oh like just".split()
)


def find_candidates(transcript: Transcript, glossary: list[GlossaryTerm], settings: Settings) -> list[CandidateSpan]:
    terms = [g.term for g in glossary]
    raw: list[dict] = []  # local spans: seg, ls, le, sources, term, score

    for seg in transcript.segments:
        words = transcript.segment_words(seg)

        # (a) runs of low-confidence words
        i = 0
        while i < len(words):
            if words[i].prob < settings.low_conf:
                j = i
                while j < len(words) and words[j].prob < settings.low_conf:
                    j += 1
                raw.append(dict(seg=seg, ls=i, le=j, sources={"low_confidence"}, term=None, score=None))
                i = j
            else:
                i += 1

        # (b) word n-grams that sound like a glossary term
        if not terms:
            continue
        best: dict[tuple[int, int], tuple[str, float]] = {}
        for n in range(1, settings.max_ngram + 1):
            for i in range(0, len(words) - n + 1):
                toks = [w.text for w in words[i : i + n]]
                normed = [norm_word(t) for t in toks]
                if all((not t) or t in STOPWORDS for t in normed):
                    continue
                heard = " ".join(toks).strip(EDGE_PUNCT)
                for term in terms:
                    if heard == term:
                        continue  # already exactly right
                    s = term_match_score(heard, term)
                    if s >= settings.phonetic_threshold and ((i, i + n) not in best or s > best[(i, i + n)][1]):
                        best[(i, i + n)] = (term, s)
        kept: list[tuple[int, int]] = []
        for (ls, le), (term, s) in sorted(best.items(), key=lambda kv: (-kv[1][1], kv[0][1] - kv[0][0])):
            if any(ls < ke and ks < le for ks, ke in kept):
                continue  # non-max suppression
            kept.append((ls, le))
            raw.append(dict(seg=seg, ls=ls, le=le, sources={"phonetic"}, term=term, score=s))

    # merge overlapping spans within a segment
    merged: list[dict] = []
    for sp in sorted(raw, key=lambda d: (d["seg"].id, d["ls"], d["le"])):
        last = merged[-1] if merged else None
        if last and last["seg"].id == sp["seg"].id and sp["ls"] < last["le"]:
            last["le"] = max(last["le"], sp["le"])
            last["sources"] |= sp["sources"]
            if sp["score"] is not None and (last["score"] is None or sp["score"] > last["score"]):
                last["term"], last["score"] = sp["term"], sp["score"]
        else:
            merged.append(dict(sp, sources=set(sp["sources"])))

    out: list[CandidateSpan] = []
    for k, sp in enumerate(merged, 1):
        seg = sp["seg"]
        ws = transcript.segment_words(seg)[sp["ls"] : sp["le"]]
        probs = [w.prob for w in ws]
        out.append(
            CandidateSpan(
                span_id=f"C{k:04d}",
                segment_id=seg.id,
                word_start=ws[0].id,
                word_end=ws[-1].id + 1,
                text=" ".join(w.text for w in ws),
                sources=sorted(sp["sources"]),
                mean_prob=sum(probs) / len(probs),
                min_prob=min(probs),
                phonetic_term=sp["term"],
                phonetic_score=sp["score"],
            )
        )
    return out
