from __future__ import annotations
import json

from icu import Locale, BreakIterator
import torch

from daluke import daBERT
from daluke.data import BatchedExamples
# Must be imported for API availability
from .loader import DataLoader
from .masking import MaskedBatchedExamples, mask_ent_batch

class ICUSentenceTokenizer:
    """ Segment text to sentences. """

    def __init__(self, locale: str):

        # ICU includes lists of common abbreviations that can be used to filter, to ignore,
        # these false sentence boundaries for some languages.
        # (http://userguide.icu-project.org/boundaryanalysis)
        if locale in {"en", "de", "es", "it", "pt"}:
            locale += "@ss=standard"
        self.locale = Locale(locale)
        self.breaker = BreakIterator.createSentenceInstance(self.locale)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        """
        ICU's BreakIterator gives boundary indices by counting *codeunits*, not *codepoints*.
        (https://stackoverflow.com/questions/30775689/python-length-of-unicode-string-confusion)
        As a result, something like this can happen.

        ```
        text = "󰡕test."  󰡕 is a non-BMP (Basic Multilingual Plane) character, which consists of two codeunits.
        len(text)
        >>> 6
        icu_tokenizer.span_tokenize(text)
        >>> [(0, 7)]
        ```

        This results in undesirable bugs in following stages.
        So, we have decided to replace non-BMP characters with an arbitrary BMP character, and then run BreakIterator.
        """

        # replace non-BMP characters with a whitespace
        # (https://stackoverflow.com/questions/36283818/remove-characters-outside-of-the-bmp-emojis-in-python-3)
        text = "".join(c if c <= "\uFFFF" else " " for c in text)

        self.breaker.setText(text)
        start_idx = 0
        spans = []
        for end_idx in self.breaker:
            spans.append((start_idx, end_idx))
            start_idx = end_idx
        return spans


def load_entity_vocab(vocab_file: str) -> dict[str, dict[str, int]]:
    """ Loads an entity vocab created by build-entity-vocab
    { "entity": { "id": int, "count": int } } """
    entities = dict()
    with open(vocab_file) as vf:
        for line in vf.readlines():
            if l := line.strip():
                entity = json.loads(l)
                entities[entity["entities"][0][0]] = {
                    "id": entity["id"],
                    "count": entity["count"],
                }
    return entities

def calculate_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """ Calculate word spans from a list of tokens. Excludes punctuation """
    spans = list()
    start, i = -1, 0
    while i < len(tokens):
        if tokens[i].isalnum():
            start = i
        elif tokens[i].startswith("##"):  # '##' marks word continuation token
            while i < len(tokens) and tokens[i].startswith("##") and start != -1:
                i += 1
            i -= 1
            if start != -1:
                spans.append((start, i))
        elif start == i - 1 and start != -1:
            spans.append((start, i))
        i += 1

    return spans

