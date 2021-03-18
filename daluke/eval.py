from __future__ import annotations
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from seqeval.metrics import classification_report

from pelutils import DataStorage, log

from daluke.model import span_probs_to_preds
from daluke.data import NERDataset

@dataclass
class NER_Results(DataStorage):
    """
    statistics: The output of seqeval.metrics.classification_report on the data
    """
    preds:      list[list[str]]
    span_probs: list[dict[tuple[int], np.ndarray]]

    statistics:         dict[str, dict[str, float]]
    statistics_nomisc:  dict[str, dict[str, float]]

def evaluate_ner(model: nn.Module, dataloader: torch.utils.data.DataLoader, dataset: NERDataset, device: torch.device) -> NER_Results:
    model.eval()
    span_probs: list[dict[tuple[int], np.ndarray]] = list()
    log.debug(f"Forward passing {len(dataloader)} batches")
    for batch in tqdm(dataloader):
        feature_spans = batch.pop("spans")
        inputs = {key: val.to(device) for key, val in batch.items()}
        with torch.no_grad():
            probs = F.softmax(model(**inputs), dim=2)
        # We save probability distribution, for every possible span in the example
        for i, spans in enumerate(feature_spans):
            span_probs.append({
                span: probs[i, j].detach().cpu().numpy() for j, span in enumerate(spans) if span
            })
    preds = [span_probs_to_preds(p, len(t), dataset) for p, t in zip(span_probs, dataset.texts)]

    stats = _stats_to_py_nums(
        classification_report(dataset.annotations, preds, output_dict=True, zero_division=0)
    )
    stats_nomisc = _stats_to_py_nums(
        classification_report(_rm_misc(dataset.annotations, dataset.null_label), _rm_misc(preds, dataset.null_label), output_dict=True)
    )
    #FIXME: Do this manually instead of rerunning everything
    log(classification_report(dataset.annotations, preds, zero_division=0, digits=4))
    log(classification_report(_rm_misc(dataset.annotations, dataset.null_label), _rm_misc(preds, dataset.null_label), digits=4))

    return NER_Results(
        preds=preds,
        span_probs=span_probs,
        statistics=stats,
        statistics_nomisc=stats_nomisc,
    )

def _rm_misc(seqs: list[list[str]], null_class: str) -> list[list[str]]:
    """ Convert all "MISC"-entities to null-entities """
    return [[null_class if "MISC" in ent else ent for ent in seq] for seq in seqs]

def _stats_to_py_nums(stats: dict[dict[str, Union[np.int64, np.float64]]]) -> dict[dict[Union[int, float]]]:
    """ Convert each element in stats dict to python numerical instead of numpy data type """
    return {skey: {key: val.item() for key, val in sval.items()} for skey, sval in stats.items()}
