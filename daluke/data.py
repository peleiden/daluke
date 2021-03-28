from __future__ import annotations
from dataclasses import dataclass

from transformers import AutoTokenizer

import torch

@dataclass
class Words:
    ids: torch.Tensor
    segments: torch.Tensor
    attention_mask: torch.Tensor
    N: int

    @classmethod
    def build(cls, ids: torch.Tensor,
            max_len: int=512,
            sep_id: int=3,
            cls_id: int=2,
            pad_id: int=0,
        ):
        """
        For creating a single example
        """
        N = ids.shape[0]
        word_ids = torch.full((max_len,), pad_id, dtype=torch.long)
        word_ids[:N+2] = torch.cat((torch.LongTensor([cls_id]), ids, torch.LongTensor([sep_id])))
        return cls(
            ids            = word_ids,
            segments       = cls._build_segments(max_len),
            attention_mask = cls._build_att_mask(N+2, max_len),
            N              = N
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
    pos: torch.Tensor

    @classmethod
    def build(cls, ids: torch.Tensor, spans: list[tuple],
            max_len: int=128,
            max_mention: int=30,
        ):
        """
        For creating a single example

        ids: N ids found from entity vocab used to train the model
        spans: N long list containing start and end of entities
        """
        N = ids.shape[0]
        ent_ids = torch.zeros(max_len, dtype=torch.long)
        ent_ids[:N] = ids

        ent_pos = torch.full((max_len, max_mention), -1, dtype=torch.long)
        # TODO: Make faster than for loop
        for i, e in enumerate(spans):
            ent_pos[i, :e[1]-e[0]] = torch.LongTensor(list(range(*e)))
        ent_pos[ent_pos != -1] += 1 #+1 for [cls]

        return cls(
            ids            = ent_ids,
            segments       = cls._build_segments(max_len),
            attention_mask = cls._build_att_mask(N, max_len),
            pos            = ent_pos,
            N              = N,
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
    @staticmethod
    def stack(ex: list[Example]) -> (Words, Entities):
        return Words(
            ids             = torch.stack(tuple(e.words.ids for e in ex)),
            segments        = torch.stack(tuple(e.words.segments for e in ex)),
            attention_mask  = torch.stack(tuple(e.words.attention_mask for e in ex)),
            N               = torch.tensor(tuple(e.words.N for e in ex)),
        ), Entities(
            ids             = torch.stack(tuple(e.entities.ids for e in ex)),
            segments        = torch.stack(tuple(e.entities.segments for e in ex)),
            attention_mask  = torch.stack(tuple(e.entities.attention_mask for e in ex)),
            pos             = torch.stack(tuple(e.entities.pos for e in ex)),
            N               = torch.tensor(tuple(e.entities.N for e in ex)),
        )

    @classmethod
    def build(cls, ex: list[Example]):
        return cls(*cls.stack(ex))

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