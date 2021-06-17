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
    ids: torch.IntTensor
    attention_mask: torch.IntTensor
    N: int
    spans: torch.IntTensor

    @classmethod
    def build(
        cls,
        ids: torch.IntTensor,
        spans: list[list[int]]=None,
        max_len: int=512,
        sep_id:  int=3,
        cls_id:  int=2,
        pad_id:  int=0,
    ):
        """
        For creating a single example: Pads and add special tokens.
        """
        N = ids.shape[0]
        word_ids = torch.full((max_len,), pad_id, dtype=torch.int)
        word_ids[:N+2] = torch.cat((torch.IntTensor([cls_id]), ids, torch.IntTensor([sep_id])))

        # Don't pad the spans as they are not given to model, but used for masking
        if spans is not None:
            spans = torch.IntTensor(spans) + 1 # For [CLS]

        return cls(
            ids            = word_ids,
            attention_mask = cls._build_att_mask(N+2, max_len),
            N              = N,
            spans          = spans,
        )

    @staticmethod
    def _build_att_mask(fill: int, max_size: int) -> torch.IntTensor:
        att_mask = torch.zeros(max_size, dtype=torch.int)
        att_mask[:fill] = 1
        return att_mask

@dataclass
class Entities(Words):
    """
    ids: Tensor of entity vocabulary ids for each entity, size: (B x ) M
    attention_mask: Mask showing where the actual content is and what is padding, size: (B x) M
    N: Number of entities
    pos: Saves position spans in each row for each entity as these are used for positional embeddings, size: (B x) M x max mention size
    """
    pos: torch.IntTensor

    @classmethod
    def build(
        cls,
        ids: torch.IntTensor,
        spans: list[tuple[int, int]],
        max_entities: int,
        max_entity_span: int,
    ):
        """
        For creating a single example

        ids: N ids found from entity vocab used to train the model
        spans: N long list containing start and end of entities
        """
        N = ids.shape[0]
        ent_ids = torch.zeros(max_entities, dtype=torch.int)
        ent_ids[:N] = ids

        ent_pos = torch.full((max_entities, max_entity_span), -1, dtype=torch.int)
        spans = [(s+1, e+1) for s, e in spans] # +1 for [CLS] token added to words
        for i, (start, end) in enumerate(spans):
            ent_pos[i, :end-start] = torch.arange(start, end)

        return cls(
            ids            = ent_ids,
            attention_mask = cls._build_att_mask(N, max_entities),
            N              = N,
            spans          = spans,
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
    def collate(ex: list[Example], device: torch.device, cut: bool) -> (Words, Entities):
        # Stack the tensors in specific field for each example and send to device
        tensor_collate = lambda field, subfield, limit: torch.stack(tuple(getattr(getattr(e, field), subfield)[:limit] for e in ex)).to(device)

        word_N = torch.tensor(tuple(e.words.N for e in ex)).to(device) + 2 # +2 for CLS, SEP
        ent_N = torch.tensor(tuple(e.entities.N for e in ex)).to(device)
        # It is here assumed that all word and entity ids already have been padded to exactly same length (as they have in their build methods)
        word_limit = max(word_N) if cut else len(ex[0].words.ids)
        ent_limit = max(ent_N) if cut else len(ex[0].entities.ids)
        return Words(
            ids             = tensor_collate("words", "ids", word_limit),
            attention_mask  = tensor_collate("words", "attention_mask", word_limit),
            N               = word_N,
            # Assume that if one of the word examples (1st one) in the batch has a span vector, all of them do
            spans           = [e.words.spans for e in ex] if ex[0].words.spans is not None else None,
        ), Entities(
            ids             = tensor_collate("entities", "ids", ent_limit),
            attention_mask  = tensor_collate("entities", "attention_mask", ent_limit),
            pos             = tensor_collate("entities", "pos", ent_limit),
            N               = ent_N,
            spans           = [e.entities.spans for e in ex] if ex[0].entities.spans is not None else None,
        )

    @classmethod
    def build(cls, ex: list[Example], device: torch.device, cut_extra_padding: bool=True):
        return cls(*cls.collate(ex, device=device, cut=cut_extra_padding))

def get_special_ids(tokenizer: AutoTokenizer) -> (int, int, int):
    """ Returns seperator id, close id and pad id """
    return tuple(tokenizer.convert_tokens_to_ids(t) for t in (tokenizer.sep_token, tokenizer.cls_token, tokenizer.pad_token))
