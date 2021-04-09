from __future__ import annotations
from dataclasses import dataclass

from transformers import AutoTokenizer

import torch

@dataclass
class Words:
    """
    Contains all data related to the "words", e.g. text tokens used for forward passing daLUKE.

    ids: Tensor of tokenizer ids for each token, size: (B x) M
    attention_mask: Mask showing where the actual text is and what is padding, size: (B x) M
    N: Number of tokens
    spans: Optional; M x 2 vector showing token positions corresponding to full words; necessary for full-word masking
    """
    ids: torch.Tensor
    segments: torch.Tensor
    attention_mask: torch.Tensor
    N: int
    spans: torch.Tensor

    @classmethod
    def build(cls, ids: torch.Tensor,
            spans: list[list[int]]=None,
            max_len: int=512,
            sep_id: int=3,
            cls_id: int=2,
            pad_id: int=0,
        ):
        """
        For creating a single example: Pads and add special tokens.
        """
        N = ids.shape[0]
        word_ids = torch.full((max_len,), pad_id, dtype=torch.long)
        word_ids[:N+2] = torch.cat((torch.LongTensor([cls_id]), ids, torch.LongTensor([sep_id])))

        # Don't pad the spans as they are not given to model, but used for masking
        if spans is not None:
            spans = torch.LongTensor(spans)

        return cls(
            ids            = word_ids,
            segments       = cls._build_segments(max_len),
            attention_mask = cls._build_att_mask(N+2, max_len),
            N              = N,
            spans          = spans,
        )

    @staticmethod
    def _build_att_mask(fill: int, max_size: int):
        att_mask = torch.zeros(max_size, dtype=torch.long)
        att_mask[:fill] = 1
        return att_mask

    @staticmethod
    def _build_segments(max_size: int):
        # TODO: Is this really correct? Seems stupid ...
        return torch.zeros(max_size, dtype=torch.long)

@dataclass
class Entities(Words):
    """
    ids: Tensor of entity vocabulary ids for each entity, size: (B x ) M
    attention_mask: Mask showing where the actual content is and what is padding, size: (B x) M
    N: Number of entities
    pos: Saves position spans in each row for each entity as these are used for positional embeddings, size: (B x) M x max mention size
    """
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
        for i, (start, end) in enumerate(spans):
            ent_pos[i, :end-start] = torch.LongTensor(list(range(start, end)))
        ent_pos[ent_pos != -1] += 1 #+1 for [cls]

        return cls(
            ids            = ent_ids,
            segments       = cls._build_segments(max_len),
            attention_mask = cls._build_att_mask(N, max_len),
            N              = N,
            spans          = None, # We do not need to save the spans for masking as we do for words
            pos            = ent_pos,
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
    def stack(ex: list[Example], device: torch.device) -> (Words, Entities):
        return Words(
            ids             = torch.stack(tuple(e.words.ids for e in ex)).to(device),
            segments        = torch.stack(tuple(e.words.segments for e in ex)).to(device),
            attention_mask  = torch.stack(tuple(e.words.attention_mask for e in ex)).to(device),
            N               = torch.tensor(tuple(e.words.N for e in ex)).to(device),
            # Assume that if one of the word examples (1st one) in the batch has a span vector, all of them do
            spans           = [e.words.spans for e in ex] if ex[0].words.spans is not None else None,
        ), Entities(
            ids             = torch.stack(tuple(e.entities.ids for e in ex)).to(device),
            segments        = torch.stack(tuple(e.entities.segments for e in ex)).to(device),
            attention_mask  = torch.stack(tuple(e.entities.attention_mask for e in ex)).to(device),
            pos             = torch.stack(tuple(e.entities.pos for e in ex)).to(device),
            spans           = None,
            N               = torch.tensor(tuple(e.entities.N for e in ex)).to(device),
        )

    @classmethod
    def build(cls, ex: list[Example], device: torch.device):
        return cls(*cls.stack(ex, device=device))

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
