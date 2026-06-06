"""Five extractive Stage-1 summarizers behind BaseSummarizer.

`sumy` and `nltk` are lazy-imported inside methods (gated by the
`[summarization]` extra), so importing this module does not require them.
NLTK data (`punkt`, `punkt_tab`, `stopwords`) is fetched on first use.
"""

from __future__ import annotations

from collections import defaultdict

from foodscholar.layer_c.base import BaseSummarizer, split_sentences

_NLTK_READY = False


def _ensure_nltk_data() -> None:
    """Download the nltk resources the summarizers need, once per process."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    _NLTK_READY = True


def _concat(chunks: list[str]) -> str:
    return "\n".join(c for c in chunks if c and c.strip())


class NLTKFrequencySummarizer(BaseSummarizer):
    """Word-frequency extractive summarizer (stopword-filtered, normalized)."""

    name = "nltk_freq"

    def __init__(self, n: int = 8) -> None:
        self.n = n

    def summarize(self, chunks: list[str]) -> str:
        text = _concat(chunks)
        sentences = split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) <= self.n:
            return " ".join(sentences)

        _ensure_nltk_data()
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        stop = set(stopwords.words("english"))
        freq: dict[str, float] = defaultdict(float)
        for w in word_tokenize(text.lower()):
            if w.isalpha() and w not in stop:
                freq[w] += 1.0
        if not freq:
            return " ".join(sentences[: self.n])
        peak = max(freq.values())
        for w in freq:
            freq[w] /= peak

        scored: list[tuple[int, float]] = []
        for i, sent in enumerate(sentences):
            words = [w for w in word_tokenize(sent.lower()) if w.isalpha()]
            score = sum(freq.get(w, 0.0) for w in words)
            scored.append((i, score))

        top_idx = sorted(
            (i for i, _ in sorted(scored, key=lambda t: t[1], reverse=True)[: self.n])
        )
        return " ".join(sentences[i] for i in top_idx)


class _SumyBase(BaseSummarizer):
    """Shared parse/tokenize/run/join for the four sumy algorithms.

    Subclasses set `name` and implement `_algo()` returning a sumy summarizer.
    """

    def __init__(self, n: int = 8) -> None:
        self.n = n

    def _algo(self):
        raise NotImplementedError

    def summarize(self, chunks: list[str]) -> str:
        text = _concat(chunks)
        sentences = split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) <= self.n:
            return " ".join(sentences)

        _ensure_nltk_data()
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.parsers.plaintext import PlaintextParser

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        picked = self._algo()(parser.document, self.n)
        out = " ".join(str(s) for s in picked)
        return out or " ".join(sentences[: self.n])


class SumyLexRankSummarizer(_SumyBase):
    name = "lexrank"

    def _algo(self):
        from sumy.summarizers.lex_rank import LexRankSummarizer

        return LexRankSummarizer()


class SumyLsaSummarizer(_SumyBase):
    name = "lsa"

    def _algo(self):
        from sumy.summarizers.lsa import LsaSummarizer

        return LsaSummarizer()


class SumyLuhnSummarizer(_SumyBase):
    name = "luhn"

    def _algo(self):
        from sumy.summarizers.luhn import LuhnSummarizer

        return LuhnSummarizer()


class SumyTextRankSummarizer(_SumyBase):
    name = "textrank"

    def _algo(self):
        from sumy.summarizers.text_rank import TextRankSummarizer

        return TextRankSummarizer()
