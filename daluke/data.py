from __future__ import annotations
from dataclasses import dataclass

from transformers import AutoTokenizer

import torch

@dataclass
class Words:
    ids: torch.LongTensor
    segments: torch.LongTensor
    attention_mask: torch.LongTensor

    @classmethod
    def build(cls, ids: torch.LongTensor,
            max_len: int=512,
            sep_id: int=3,
            cls_id: int=2,
            pad_id: int=0,
        ):
        """
        For creating a single example
        """
        s = ids.shape[0]
        word_ids = torch.LongTensor(max_len).fill_(pad_id)
        word_ids[:s+2] = torch.cat((torch.LongTensor([cls_id]), ids, torch.LongTensor([sep_id])))
        return cls(
            ids            = word_ids,
            segments       = cls._build_segments(max_len),
            attention_mask = cls._build_att_mask(s+2, max_len),
        )

    @staticmethod
    def _build_att_mask(fill: int, max_size: int):
        att_mask = torch.zeros(max_size, dtype=torch.long)
        att_mask[:fill] = 1
        return att_mask

    @staticmethod
    def _build_segments(max_size: int):
        return torch.zeros(max_size, dtype=torch.long)

@dataclass
class Entities(Words):
    pos: torch.LongTensor

    @classmethod
    def build(cls, ids: torch.LongTensor, spans: list[tuple],
            max_len: int=128,
            max_mention: int=30,
        ):
        """
        For creating a single example

        ids: N ids found from entity vocab used to train the model
        spans: N long list containing start and end of entities
        """
        s = ids.shape[0]
        ent_ids = torch.zeros(max_len, dtype=torch.long)
        ent_ids[:s] = ids

        ent_pos = torch.LongTensor(max_len, max_mention).fill_(-1)
        # TODO: Make faster than for loop
        for i, e in enumerate(spans):
            ent_pos[i, :e[1]-e[0]] = torch.LongTensor(list(range(*e)))
        ent_pos[ent_pos != -1] += 1 #+1 for [cls]

        return cls(
            ids            = ent_ids,
            segments       = cls._build_segments(max_len),
            attention_mask = cls._build_att_mask(s, max_len),
            pos            = ent_pos
        )

@dataclass
class Example:
    """
    A single data example
    """
    words: Words
    entities: Entities

@dataclass
class BatchedExamples(Example):
    """
    Data to be forward passed to daLUKE
    """
    @classmethod
    def build(cls, features: list[Example]):
        return cls(
            words = Words(
                ids             = torch.stack(tuple(f.words.ids for f in features)),
                segments        = torch.stack(tuple(f.words.segments for f in features)),
                attention_mask  = torch.stack(tuple(f.words.attention_mask for f in features)),
            ),
            entities = Entities(
                ids             = torch.stack(tuple(f.entities.ids for f in features)),
                segments        = torch.stack(tuple(f.entities.segments for f in features)),
                attention_mask  = torch.stack(tuple(f.entities.attention_mask for f in features)),
                pos             = torch.stack(tuple(f.entities.pos for f in features)),

            )
        )

def get_special_ids(tokenizer: AutoTokenizer) -> (int, int, int):
    """ Returns seperator id, close id and pad id """
    return tuple(tokenizer.convert_tokens_to_ids(t) for t in (tokenizer.sep_token, tokenizer.cls_token, tokenizer.pad_token))

# FIXME: A lot of the logic in here should call methods implemented in a Dataset class
def features_from_str(words: list[str], entity_spans: list[tuple[int, int]], entity_vocab: dict[str, int], tokenizer: AutoTokenizer) -> Example:
    """
    A one-time feature generator used for inference of a single example - mostly practical as an example.
    words: The sentence as tokenized list of strings e.g. ['Jeg', 'hedder', 'Wolfgang', 'Amadeus', 'Mozart', 'og', 'er', 'fra', 'Salzburg']
    entity_spans: List of start (included) and end (excluded) indices of each entity e.g. [(2, 5), (8, 9)]
    --
    entity_vocab: Maps entity string to entity ids for forward passing
    tokenizer: tokenizer used for word id computation
    """
    sep, cls_, pad = get_special_ids(tokenizer)
    word_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(words))
    ents = (" ".join(words[e[0]:e[1]]) for e in entity_spans)
        # FIXME: Handle tokenization, e.g.: What if the entity is subword?
    ent_ids = torch.LongTensor([entity_vocab.get(ent, entity_vocab["[UNK]"]) for ent in ents])
        # FIXME: Make a class for entity vocab
        # FIXME: Consider entity casing
    return Example(
        words=Words.build(word_ids),
        entities=Entities.build(ent_ids, entity_spans)
    )
