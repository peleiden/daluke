import functools, operator
from transformers import AutoConfig, AutoTokenizer

import torch
from daluke.model import DaLUKE, EntityEmbeddings
from daluke.data import BatchedExamples
from daluke import daBERT

from . import features_from_str

def _create_cfg():
    return AutoConfig.from_pretrained(daBERT)

def test_daluke():
    model = DaLUKE(_create_cfg(), 100, 256)
    features = features_from_str("Jeg hedder Jens Nielsen og er fra Danmark".split(), [(2,4), (7, 8)], {"[UNK]": 1, "Danmark": 42}, AutoTokenizer.from_pretrained(daBERT))
    model(BatchedExamples.build([features], torch.device("cpu"), True))

def test_ent_embeds():
    # From original LUKE repository
    # https://github.com/studio-ousia/luke/blob/6feefe657d97d2f847ace87f61f23b705f75d2aa/tests/test_model.py#L29
    cfg = _create_cfg()
    cfg.hidden_dropout_prob = 0
    ent_embeds = EntityEmbeddings(cfg, 5, cfg.hidden_size)

    entity_ids = torch.LongTensor([2, 3, 0])
    position_ids = torch.LongTensor(
        [
            [0, 1] + [-1] * (cfg.max_position_embeddings - 2),
            [3] + [-1] * (cfg.max_position_embeddings - 1),
            [-1] * cfg.max_position_embeddings,
        ]
    )
    token_type_ids = torch.LongTensor([0, 1, 0])

    emb = ent_embeds(entity_ids, position_ids, token_type_ids)
    assert emb.size() == (3, cfg.hidden_size)

    for n, (entity_id, position_id_list, token_type_id) in enumerate(zip(entity_ids, position_ids, token_type_ids)):
        entity_emb        = ent_embeds.ent_embeds.weight[entity_id]
        token_type_emb    = ent_embeds.typ_embeds.weight[token_type_id]
        position_emb_list = [ent_embeds.pos_embeds.weight[p] for p in position_id_list if p != -1]
        if position_emb_list:
            position_emb = functools.reduce(operator.add, position_emb_list) / len(position_emb_list)
            target_emb = ent_embeds.lnorm((entity_emb + position_emb + token_type_emb))
        else:
            target_emb = ent_embeds.lnorm((entity_emb + token_type_emb))

        assert torch.equal(emb[n], target_emb)
