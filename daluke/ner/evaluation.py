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

from pelutils import DataStorage, Table, log
from pelutils.ds import no_grad

from daluke.ner.model import span_probs_to_preds
from daluke.ner.data import NERDataset, Split

@dataclass
class NER_Results(DataStorage):
    """
    statistics: The output of seqeval.metrics.classification_report on the data
    """
    preds:      list[list[str]]
    span_probs: list[dict[tuple[int], np.ndarray]]

    statistics:        dict[str, dict[str, float]]
    statistics_nomisc: dict[str, dict[str, float]]

    # Prediction distribution: confusion_matrix["MISC"]["PER"] is the amount of times PER was predicted when MISC was the correct label
    confusion_matrix: dict[str, dict[str, int]]
    confusion_matrix_nomisc: dict[str, dict[str, int]]

    subfolder = "eval-results"

def _remove_iob(label: str) -> str:
    return label[2:] if "-" in label else label

def _format_confmat(confmat: dict[str, dict[str, int]]) -> Table:
    t = Table()
    t.add_header([""] + list(confmat))
    for true_class, preds in confmat.items():
        t.add_row([
            true_class,
            *preds.values(),
        ])
    return t

def confusion_matrix(annotations, preds, classes: list[str]) -> dict[str, dict[str, int]]:
    """ Builds a confusion matrix with true and predicted labels """
    annotations = [x for y in annotations for x in y]
    preds = [x for y in preds for x in y]
    confmat = { c: { c_: 0 for c_ in classes } for c in classes }
    for ann, pred in zip(annotations, preds):
        confmat[_remove_iob(ann)][_remove_iob(pred)] += 1
    return confmat

@no_grad
def evaluate_ner(model: nn.Module, dataloader: torch.utils.data.DataLoader, dataset: NERDataset, device: torch.device, split: Split, also_no_misc=True) -> NER_Results:
    model.eval()
    annotations, texts = dataset.data[split].annotations, dataset.data[split].texts
    span_probs: list[dict[tuple[int, int], np.ndarray]] = list(dict() for _ in range(len(texts)))
    log.debug(f"Forward passing {len(dataloader)} batches")
    for batch in tqdm(dataloader):
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
    confmat = confusion_matrix(annotations, preds, dataset.all_labels)
    confmat_nomisc = dict()
    log("Prediction distribution", _format_confmat(confmat))

    if also_no_misc:
        #FIXME: Do this manually instead of rerunning everything
        stats_nomisc = _stats_to_py_nums(
            classification_report(_rm_misc(annotations, dataset.null_label), _rm_misc(preds, dataset.null_label), output_dict=True)
        )
        log(classification_report(_rm_misc(annotations, dataset.null_label), _rm_misc(preds, dataset.null_label), digits=4))
        confmat_nomisc = confusion_matrix(_rm_misc(annotations, dataset.null_label), _rm_misc(preds, dataset.null_label), dataset.all_labels)
        log("Prediction distribution", _format_confmat(confmat))

    return NER_Results(
        preds=preds,
        span_probs=span_probs,
        statistics=stats,
        statistics_nomisc=stats_nomisc if also_no_misc else {},
        confusion_matrix=confmat,
        confusion_matrix_nomisc=confmat_nomisc,
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
