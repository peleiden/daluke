from __future__ import annotations
import torch

from daluke.data import Words, Entities, Example

# TODO: Use something else than this for tests
def features_from_str(words: list[str], entity_spans: list[tuple[int, int]], entity_vocab: dict[str, int], tokenizer: AutoTokenizer) -> Example:
    word_ids = torch.IntTensor(tokenizer.convert_tokens_to_ids(words))
    ents = (" ".join(words[e[0]:e[1]]) for e in entity_spans)
    ent_ids = torch.IntTensor([entity_vocab.get(ent, entity_vocab["[UNK]"]) for ent in ents])
    return Example(
        words=Words.build(word_ids),
        entities=Entities.build(ent_ids, entity_spans, 128, 30)
    )
