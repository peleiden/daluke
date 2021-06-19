from __future__ import annotations
from dataclasses import dataclass
from pelutils import DataStorage

@dataclass
class NER_TestResults(DataStorage):
    """
    preds: all predictions of model with `modelname` on the dataset with `dataname`
    classes: list of names of relevant (either guessed by model or included in data) classes. Often PER, LOG, ORG
    statistics: The output of seqeval.metrics.classification_report on the data
    """
    modelname: str
    dataname : str
    predictions: list[list[str]]

    statistics:         dict[str, dict[str, float]]
    statistics_nomisc:  dict[str, dict[str, float]]
