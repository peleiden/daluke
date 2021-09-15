from __future__ import annotations
import warnings
from itertools import chain

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from daluke.data import Words, Entities, Example, BatchedExamples, get_special_ids
from daluke.pretrain.data import MaskedBatchedExamples, ENTITY_UNK_TOKEN
from daluke.ner.data import NERBatchedExamples, NERDataset, Split, Sequences
from daluke.api.automodels import AutoDaLUKE

DEFAULT_NER_BATCHSIZE = 8

def get_subword_ids(text: str, tokenizer: PreTrainedTokenizerFast) -> list[list[int]]:
    """
    Helper function, passing each crude word (space separated) through tokenizer, returning nested ids
    """
    return tokenizer(text.split(), add_special_tokens=False)["input_ids"]

def get_word_id_tensor(subword_ids: list[list[int]]) -> torch.IntTensor:
    return torch.IntTensor(list(chain(*subword_ids)))

def get_entity_id_tensor(
    text: str,
    entity_spans: list[tuple[int, int]],
    entity_vocab: dict[str, dict[str, int]],
) -> torch.IntTensor:
    """ Maps the given entities to an id in the entity vocab """
    words = text.split()
    ids = list()
    for start, end in entity_spans:
        try:
            entity = " ".join(words[start: end])
        except IndexError as e:
            raise IndexError(
                f"The given entity with the range ({start}; {end}) did not fit in sequence.\n"+
                "Zero-indexing and end-exclusive spans are assumed. The sequence indexing was:\n"+
                " ".join(f"{w} ({i})" for w, i in enumerate(words))
            ) from e
        # Add unknown id (1) if these words are not in the vocabulary
        ids.append(
            entity_vocab.get(entity, entity_vocab[ENTITY_UNK_TOKEN])["id"]
        )
        if ids[-1] == entity_vocab[ENTITY_UNK_TOKEN]["id"]:
            warnings.warn("Unknown entity '%s'. Was your span correct?" % entity)
    return torch.IntTensor(ids)

def get_entity_subword_spans(subword_ids: list[list[int]], entity_spans:list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Convert entity ranges in fullwords to entity span position in subword (tokenized) units
    """
    cumlength = np.cumsum([len(t) for t in subword_ids])
    return [(cumlength[start-1] if start else 0, cumlength[end-1]) for start, end in entity_spans]

def example_from_str(
        text:             str,
        entity_spans:     list[tuple[int, int]],
        daluke:           AutoDaLUKE
    ) -> BatchedExamples:
    subword_ids = get_subword_ids(text, daluke.tokenizer)
    sep, cls_, pad = get_special_ids(daluke.tokenizer)

    w = Words.build(
        ids     = get_word_id_tensor(subword_ids),
        max_len = daluke.metadata["max-seq-length"],
        sep_id  = sep,
        cls_id  = cls_,
        pad_id  = pad,
    )
    e = Entities.build(
        ids             = get_entity_id_tensor(text, entity_spans, daluke.entity_vocab),
        spans           = get_entity_subword_spans(subword_ids, entity_spans),
        max_entities    = daluke.metadata["max-entities"],
        max_entity_span = daluke.metadata["max-entity-span"],
    )
    return BatchedExamples.build([
            Example(
                words    = w,
                entities = e,
            ),
        ],
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

def masked_example_from_str(
        text:             str,
        entity_spans:     list[tuple[int, int]],
        daluke:           AutoDaLUKE
    ) -> MaskedBatchedExamples:
    batched_examples = example_from_str(
        text, entity_spans, daluke,
    )
    mask_token = daluke.tokenizer.convert_tokens_to_ids(daluke.tokenizer.mask_token)
    wordmask = torch.zeros_like(batched_examples.words.ids, dtype=torch.bool, device=batched_examples.words.ids.device)
    wordmask[batched_examples.words.ids == mask_token] = True

    return MaskedBatchedExamples(
        words               = batched_examples.words,
        entities            = batched_examples.entities,
        word_mask_labels    = torch.LongTensor([]),
        word_mask           = wordmask,
        ent_mask_labels     = torch.LongTensor([]),
        ent_mask            = torch.zeros_like(batched_examples.entities.ids, dtype=torch.bool),
    )

class SingletonNERData(NERDataset):
    """
    A dataset for a single example to be forward passed
    """
    null_label = "O"
    labels = ("LOC", "PER", "ORG", "MISC")
    all_labels = (null_label,) + labels

    def load(self, texts: list[str], **_):
        word_texts = [text.split() for text in texts]
        self.data[Split.TEST] = Sequences(
            texts               = word_texts,
            annotations         = [[self.null_label for _ in range(len(words))] for words in word_texts],
            sentence_boundaries = [[len(words)] for words in word_texts],
        )
        self.loaded = True

def ner_examples_from_str(
        texts:     list[str],
        daluke:    AutoDaLUKE,
        batch_size: int=DEFAULT_NER_BATCHSIZE,
    ) -> list[NERBatchedExamples]:
    data = SingletonNERData(
        base_model      = daluke.metadata["base-model"],
        max_seq_length  = daluke.metadata["max-seq-length"],
        max_entities    = daluke.metadata["max-entities"],
        max_entity_span = daluke.metadata["max-entity-span"],
        device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    data.load(texts)
    # Extract examples from dataloader
    return list(data.build(Split.TEST, batch_size))
