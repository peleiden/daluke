from __future__ import annotations
import json
import os
import tarfile
import tempfile

import numpy as np
import torch
from torch import nn
from transformers.models.bert.modeling_bert import (
        BertEncoder,
        BertPooler,
        BertConfig,
        BertEmbeddings
)

from pelutils import log

from daluke.collect_modelfile import VOCAB_FILE, METADATA_FILE, MODEL_OUT
from daluke.luke import EntityEmbeddings, EntityAwareEncoder
from daluke.data import NERDataset


class DaLukeNER(nn.Module):
    """
    The BERT based model LUKE using Entity Aware Attention
    """
    def __init__(self, config_dict: dict, output_shape: int):
        """
        Build the architecture and setup the config
        """
        super().__init__()
        self.output_shape = output_shape
        self.config = BertConfig(**config_dict)
        self.config.entity_emb_size   = config_dict["entity_emb_size"]
        self.config.entity_vocab_size = 2

        # The general architecture
        self.encoder    = EntityAwareEncoder(self.config)
        self.pooler     = BertPooler(self.config)
        self.embeddings = BertEmbeddings(self.config)
        self.entity_embeddings = EntityEmbeddings(self.config)

        # For NER
        self.drop = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size*3, self.output_shape)
        self.apply(self._init_weights)

    def forward(self,
        ent_start_pos: torch.tensor,
        ent_end_pos: torch.tensor,
        # All below arguments are given to encoder
        word_ids: torch.tensor,
        ent_ids:  torch.tensor,
        word_seg_ids: torch.tensor,
        ent_seg_ids: torch.tensor,
        ent_pos_ids: torch.tensor,
        word_att_mask: torch.tensor,
        ent_att_mask: torch.tensor,
    ):
        """
        Classify NER by passing the word and entity id's through the encoder
        and running the linear classifier on the output
        """
        # Forward pass through encoder, saving the embeddings of words and entitites
        encodings = self.encode(word_ids, ent_ids, word_seg_ids, ent_seg_ids, ent_pos_ids, word_att_mask, ent_att_mask)
        hidden_state_w, hidden_state_ent = encodings[:2]
        hid_w_size = hidden_state_w.size()[-1]

        ent_start_pos = ent_start_pos.unsqueeze(-1).expand(-1, -1, hid_w_size)
        ent_end_pos = ent_end_pos.unsqueeze(-1).expand(-1, -1, hid_w_size)

        all_starts = torch.gather(hidden_state_w, -2, ent_start_pos)
        all_ends = torch.gather(hidden_state_w, -2, ent_end_pos)

        features = torch.cat([all_starts, all_ends, hidden_state_ent], dim=2)
        features = self.drop(features)
        return self.classifier(features)

    def encode(self,
        word_ids: torch.tensor,
        ent_ids:  torch.tensor,
        word_seg_ids: torch.tensor,
        ent_seg_ids: torch.tensor,
        ent_pos_ids: torch.tensor,
        word_att_mask: torch.tensor,
        ent_att_mask: torch.tensor,
    ):
        """
        Encode the words and entities using the entity aware encoder
        """
        w_embeds = self.embeddings(word_ids, word_seg_ids)
        ent_embeds = self.entity_embeddings(ent_ids, ent_pos_ids, ent_seg_ids)

        # Compute the extended attention mask
        att_mask = torch.cat((word_att_mask, ent_att_mask), dim=1) if ent_att_mask is not None else word_att_mask
        att_mask = att_mask.unsqueeze(1).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        att_mask = 10_000.0 * (att_mask - 1.0) #TODO: Understand this

        return self.encoder(w_embeds, ent_embeds, att_mask)

    def _init_weights(self, module: nn.Module):
        #FIXME: Document and rewrite. Follows luke completely atm.
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim == 1:  # embedding for bias parameters
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def span_probs_to_preds(span_probs: dict[tuple[int], np.ndarray], seq_len: int, dataset: NERDataset) -> list[str]:
    positives = list()
    for span, probs in span_probs.items():
        max_idx = probs.argmax()
        if (max_label := dataset.all_labels[max_idx]) != dataset.null_label:
            positives.append((probs[max_idx], span, max_label))
    preds = [dataset.null_label for _ in range(seq_len)]
    # Sort after max probability
    for _, span, label in reversed(sorted(positives)):
        if all(l == dataset.null_label for l in preds[span[0]:span[1]]):
            # Follow IOUB2 scheme: Set all to "I-X" unless first which is "B-X"
            for i in range(*span):
                preds[i] = f"I-{label}"
            preds[span[0]] = f"B-{label}"
    return preds

def mutate_for_ner(state_dict: dict, entity_vocab: list[dict]) -> dict:
    """
    For NER, we don't need the entity vocabulary layer: Only entity and not entity are considered
    """
    ent_embed_key = "entity_embeddings.entity_embeddings.weight"
    ent_embed = state_dict[ent_embed_key]
    mask_id = next(e["id"] for e in entity_vocab if e["entities"][0][0] == "[MASK]")
    mask_embed = ent_embed[mask_id].unsqueeze(0)
    state_dict[ent_embed_key] = torch.cat((ent_embed[:1], mask_embed))
    return state_dict

def load_from_archive(model: str) -> (list[dict], dict, dict):
    """
    Reads the directory or luke.tar.gz archive containing the model, the entity vocabulary and metadata
    """
    if os.path.isdir(model):
        with open(os.path.join(model, VOCAB_FILE), "r") as vfile:
            entity_vocab = [json.loads(l) for l in vfile.readlines()]
        with open(os.path.join(model, METADATA_FILE), "r") as metafile:
            metadata = json.load(metafile)
        with open(os.path.join(model, MODEL_OUT), "rb") as mfile:
            state_dict = torch.load(mfile, map_location="cpu")
        return entity_vocab, metadata, state_dict
    with tarfile.open(model, "r:gz") as tar:
        log.debug(f"Extracting {VOCAB_FILE} and {METADATA_FILE} ...")
        with tar.extractfile(tar.getmember(VOCAB_FILE)) as vfile:
            entity_vocab = [json.loads(l) for l in vfile.read().splitlines()]
        with tar.extractfile(tar.getmember(METADATA_FILE)) as metafile:
            metadata = json.load(metafile)
        log.debug(f"Extracting {MODEL_OUT} ...")
        with tar.extractfile(tar.getmember(MODEL_OUT)) as mfile:
            state_dict = torch.load(mfile, map_location="cpu")
    return entity_vocab, metadata, state_dict

def save_to_archive(outfile: str, entity_vocab: list[dict], metadata: dict, model: nn.Module):
    log.debug(f"Compressing {VOCAB_FILE} and {METADATA_FILE} ...")
    with tarfile.open(outfile, "w:gz") as tar:
        with tempfile.NamedTemporaryFile("w+") as tmp:
            for l in entity_vocab:
                tmp.write(json.dumps(l) + "\n")
            tmp.seek(0)
            tar.add(tmp.name, arcname=VOCAB_FILE)
        with tempfile.NamedTemporaryFile("w+") as tmp:
            tmp.write(json.dumps(metadata))
            tmp.seek(0)
            tar.add(tmp.name, arcname=METADATA_FILE)
        log.debug(f"Compressing {MODEL_OUT} ...")
        with tempfile.NamedTemporaryFile("wb+") as tmp:
            model.to(torch.device("cpu"))
            torch.save(model.state_dict(), tmp)
            tar.add(tmp.name, arcname=MODEL_OUT)
