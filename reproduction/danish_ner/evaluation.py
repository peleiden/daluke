from dataclasses import dataclass

import numpy as np
from seqeval.metrics import classification_report

from pelutils import log, DataStorage

from models import NER_TestModel
from data import TestDataset

@dataclass
class NER_TestResults(DataStorage):
    """
    preds: all predictions of model with `modelname` on the dataset with `dataname`
    classes: list of names of relevant (either guessed by model or included in data) classes. Often PER, LOG, ORG
    statistics: The output of seqeval.metrics.classification_report on the data
    """
    modelname: str
    dataname : str
    preds: list[list[str]]

    classes: list[str]
    statistics: dict[str, dict[str, float]]

class Evaluator:
    def __init__(self, model: NER_TestModel, dataset: TestDataset):
        self.model = model
        self.dataset = dataset

        self.result = None

    def run(self):
        log(f"Evaluating {self.model.name} on {self.dataset.name} ...")
        preds, truths = self._get_results()
        log.debug(f"Calculating statistics for {len(preds)} sentences")
        self.result = self._calculate_stats(preds, truths)

    def _get_results(self) -> (list[list[str]], list[list[str]]):
        preds, truths = list(), list()
        for text, truth in self.dataset.get_data():
            preds.append(self.model.predict(text))
            truths.append(truth)
        return preds, truths

    def _calculate_stats(self, preds: list[list[str]], truth: list[list[str]]) -> NER_TestResults:
        #TODO: Use this to calculate some global stats without the `MISC` category
        classes = self._calculate_relevant_classes(preds, truth)
        # Set divide by zero cases to 0 to avoid warning for models that can't see "MISC"
        stats = classification_report(truth, preds, output_dict=True, zero_division=0)
        # Run `.item()` on every number to avoid json serialization errors for numpy format
        stats = {skey: {key: val.item() for key, val in sval.items()} for skey, sval in stats.items()}
        # TODO: Format this myself instead of running classification report twice
        log(classification_report(truth, preds, zero_division=0))

        return NER_TestResults(
                modelname  = self.model.name,
                dataname   = self.dataset.name,
                preds      = preds,
                classes    = classes,
                statistics = stats,
        )

    def _calculate_relevant_classes(self, preds: list[list[str]], truth: list[list[str]]) -> list[str]:
        pred_classes, true_classes = set(), set()
        for p, t in zip(preds, truth):
            pred_classes.update(self._get_class(_p) for _p in p)
            true_classes.update(self._get_class(_t) for _t in t)
        # Take smallest class number
        classes = pred_classes if len(pred_classes) < len(true_classes) else true_classes
        return list(classes)

    @staticmethod
    def _get_class(label: str) -> str:
        """
        Converts IOB format to simple class label
        """
        if label == "O":
            return label
        return label.split("-")[1]
