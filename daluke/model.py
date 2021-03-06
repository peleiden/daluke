from __future__ import annotations
import json
import tarfile

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


LABELS = ("NIL", "LOC", "PER", "ORG", "MISC") #TODO: Make dependant on dataset

class DaLukeNER(nn.Module):
    """
    The BERT based model LUKE using Entity Aware Attention
    """
    def __init__(self, config_dict: dict):
        """
        Build the architecture and setup the config
        """
        super().__init__()
        self.config = BertConfig(**config_dict)
        self.config.entity_vocab_size = config_dict["entity_vocab_size"]
        self.config.entity_emb_size   = config_dict["entity_emb_size"]

        # The general architecture
        self.encoder    = EntityAwareEncoder(self.config)
        self.pooler     = BertPooler(self.config)
        self.embeddings = BertEmbeddings(self.config)
        self.entity_embeddings = EntityEmbeddings(self.config)

        # For NER
        self.drop = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size*3, len(LABELS))
        self.apply(self._init_weights)

    def forward(self,
        ent_start_pos,
        ent_end_pos,
        # All below arguments are given to encoder
        word_ids,
        ent_ids,
        word_seg_ids,
        ent_seg_ids,
        ent_pos_ids,
        word_att_mask,
        ent_att_mask,
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
        word_ids,
        ent_ids,
        word_seg_ids,
        ent_seg_ids,
        ent_pos_ids,
        word_att_mask,
        ent_att_mask,
    ):
        """
        Encode the words and entities using the entity aware encoder
        """
        w_embeds = self.embeddings(word_ids, word_seg_ids)
        ent_embeds = self.EntityEmbeddings(ent_ids, ent_pos_ids, ent_seg_ids)

        # Compute the extended attention mask
        att_mask = torch.cat([word_att_mask, ent_att_mask], dim=1) if ent_att_mask is None else word_att_mask
        att_mask = att_mask.unsqueeze(1).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        att_mask = -10_000.0 * (1.0 - att_mask) #TODO: Understand this

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

def load_from_archive(modelfile: str) -> (list[dict], dict, dict):
    """
    Reads the luke.tar.gz archive containing the model, the entity vocabulary and metadata
    """
    with tarfile.open(modelfile, "r:*") as tar:
        log.debug(f"Extracting {VOCAB_FILE} and {METADATA_FILE} ...")
        with tar.extractfile(tar.getmember(VOCAB_FILE)) as vfile:
            entity_vocab = [json.loads(l) for l in vfile.read().splitlines()]
        with tar.extractfile(tar.getmember(METADATA_FILE)) as metafile:
            metadata = json.load(metafile)
        log.debug(f"Extracting {MODEL_OUT} ...")
        with tar.extractfile(tar.getmember(MODEL_OUT)) as mfile:
            state_dict = torch.load(mfile, map_location="cpu")
    return entity_vocab, metadata, state_dict


