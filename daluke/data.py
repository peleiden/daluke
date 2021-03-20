from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from daluke import daBERT


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
        word_ids[:s+2] = torch.cat((torch.LongTensor([sep_id]), ids, torch.LongTensor([sep_id])))
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
    def build(cls, ids: torch.LongTensor, pos: torch.LongTensor,
            max_len: int=128,
            max_mention: int=30,
        ):
        """
        For creating a single example
        """
        s = ids.shape[0]
        ent_ids = torch.zeros(max_len, dtype=torch.long)
        ent_ids[:s] = ids

        pos[pos != -1] += 1
        ent_pos = torch.LongTensor(max_len, max_mention).fill_(-1)
        ent_pos[:pos.shape[0]] = pos

        return cls(
            ids            = ent_ids,
            segments       = cls._build_segments(max_len),
            attention_mask = cls._build_att_mask(s, max_len),
            pos            = ent_pos
        )

@dataclass
class Features:
    """
    Data to be forward passed to daLUKE
    """
    words: Words
    entities: Entities

if __name__ == '__main__':
    # TODO: Make this part of the eventual batchgenerator/dataloader/dataset classes
    tokenizer = AutoTokenizer.from_pretrained(daBERT)
    sep, cls, pad = (tokenizer.convert_tokens_to_ids(t) for t in (tokenizer.sep_token, tokenizer.cls_token, tokenizer.pad_token))
    print(sep, cls, pad)
