from dataclasses import dataclass
from pelutils import log, DataStorage

from models import NER_TestModel
from data import TestDataset


@dataclass
class NER_TestResults(DataStorage):
    pass

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


    def _get_results(self) -> (list[str], list[str]):
        preds, truths = list(), list()
        for text, truth in self.dataset.get_data():
            preds.append(self.model.predict(" ".join(text)))
            truths.append(truth)
        return preds, truths

    def _calculate_stats(self, preds: list[str], truth: list[str]):
        raise NotImplementedError

