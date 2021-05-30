from __future__ import annotations
from dataclasses import dataclass
from typing import Union
from collections import defaultdict
import json

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from seqeval.metrics import classification_report

from pelutils import DataStorage, log

from daluke.ner.model import span_probs_to_preds
from daluke.ner.data import NERDataset, Split

@dataclass
class NER_Results(DataStorage):
    """
    statistics: The output of seqeval.metrics.classification_report on the data
    """
    preds:      list[list[str]]
    span_probs: list[dict[tuple[int], np.ndarray]]

    statistics:         dict[str, dict[str, float]]
    statistics_nomisc:  dict[str, dict[str, float]]

    subfolder = "eval-results"

def evaluate_ner(model: nn.Module, dataloader: torch.utils.data.DataLoader, dataset: NERDataset, device: torch.device, split: Split, also_no_misc=True) -> NER_Results:
    model.eval()
    annotations, texts = dataset.data[split].annotations, dataset.data[split].texts
    span_probs: list[dict[tuple[int, int], np.ndarray]] = list(dict() for _ in range(len(texts)))
    log.debug(f"Forward passing {len(dataloader)} batches")
    for batch in tqdm(dataloader):
        with torch.no_grad():
            scores = model(batch)
        probs = F.softmax(scores, dim=2)
        # We save probability distribution, for every possible span in the example
        for idx, (i, spans) in zip(batch.text_nums, enumerate(batch.entities.fullword_spans)):
            span_probs[idx].update({
                span: probs[i, j].detach().cpu().numpy() for j, span in enumerate(spans) if span
            })
    preds = [span_probs_to_preds(p, len(t), dataset) for p, t in zip(span_probs, texts)]

    stats = _stats_to_py_nums(
        classification_report(annotations, preds, output_dict=True, zero_division=0)
    )
    log(classification_report(annotations, preds, zero_division=0, digits=4))

    if also_no_misc:
        #FIXME: Do this manually instead of rerunning everything
        stats_nomisc = _stats_to_py_nums(
            classification_report(_rm_misc(annotations, dataset.null_label), _rm_misc(preds, dataset.null_label), output_dict=True)
        )
        log(classification_report(_rm_misc(annotations, dataset.null_label), _rm_misc(preds, dataset.null_label), digits=4))

    return NER_Results(
        preds=preds,
        span_probs=span_probs,
        statistics=stats,
        statistics_nomisc=stats_nomisc if also_no_misc else {},
    )

def type_distribution(seqs: list[list[str]]):
    dist = defaultdict(lambda: 0)
    for seq in seqs:
        for pred in seq:
            dist[pred if "-" not in pred else pred.split("-")[-1]] += 1
    log("Type distribution:", json.dumps(dist, indent=4))
    return dist

def _rm_misc(seqs: list[list[str]], null_class: str) -> list[list[str]]:
    """ Convert all "MISC"-entities to null-entities """
    return [[null_class if "MISC" in ent else ent for ent in seq] for seq in seqs]

def _stats_to_py_nums(stats: dict[dict[str, Union[np.int64, np.float64]]]) -> dict[dict[Union[int, float]]]:
    """ Convert each element in stats dict to python numerical instead of numpy data type """
    return {skey: {key: val.item() for key, val in sval.items()} for skey, sval in stats.items()}
