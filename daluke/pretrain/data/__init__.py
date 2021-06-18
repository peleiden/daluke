from __future__ import annotations

try:
    from icu import Locale, BreakIterator
    icu_available = True
except ImportError:
    icu_available = False

from pelutils.jsonl import load_jsonl


ENTITY_UNK_TOKEN = "[UNK]"
ENTITY_MASK_TOKEN = "[MASK]"

class ICUSentenceTokenizer:
    """ Segment text to sentences. """

    def __init__(self, locale: str):
        if not icu_available:
            raise RuntimeError("Pretrain data generation requires installation of the optional requirement `PyIcU`")

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

def ignore_title(title: str) -> bool:
    return any(title.lower().startswith(word + ":") for word in ("billede", "fil", "kategori"))

def load_entity_vocab(vocab_file: str) -> dict[str, dict[str, int]]:
    """ Loads an entity vocab in .jsonl format created by build-entity-vocab
    { "entity": { "id": int, "count": int } }
    Kategory, images, and file pages are removed """
    entities = dict()
    with open(vocab_file) as vf:
        for entity in load_jsonl(vf):
            if not ignore_title(ent := entity["entities"][0][0]):
                entities[ent] = {
                    "id": entity["id"],
                    "count": entity["count"],
                }
    return entities

def calculate_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """ Calculate word spans from a list of tokens. Excludes punctuation """
    spans = list()
    i = 0
    while i < len(tokens):
        start = i
        if tokens[i].isalnum():
            start = i
            i += 1
            while i < len(tokens) and tokens[i].startswith("##"):
                i += 1
            spans.append((start, i))
        else:
            i += 1

    return spans


# Imported for API availability
from daluke import daBERT
from daluke.data import BatchedExamples
from .loader import DataLoader
from .masking import MaskedBatchedExamples, mask_ent_batch, mask_word_batch
