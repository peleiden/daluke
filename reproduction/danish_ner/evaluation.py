from __future__ import annotations
from typing import Union

import numpy as np
from seqeval.metrics import classification_report

from pelutils import log

from models import NER_TestModel
from data import TestDataset
from results import NER_TestResults

class Evaluator:
    def __init__(self, model: NER_TestModel, dataset: TestDataset):
        self.model = model
        self.dataset = dataset

    def run(self):
        log(f"Evaluating {self.model.name} on {self.dataset.name} ...")
        preds, truths = self._get_results()
        log.debug(f"Calculating statistics for {len(preds)} sentences")
        return  self._calculate_stats(preds, truths)

    def _get_results(self) -> (list[list[str]], list[list[str]]):
        text, truths = self.dataset.get_data()
        preds = self.model.predict(text)
        return preds, truths

    def _calculate_stats(self, preds: list[list[str]], truth: list[list[str]]) -> NER_TestResults:
        # Convert to python numericals to avoid json serialization problems
        # Set divide by zero cases to 0 to avoid warnings for models that can't see "MISC"
        stats = self._stats_to_py_nums(
                    classification_report(truth, preds, output_dict=True, zero_division=0)
                )
        # If the dataset includes the MISC category, a version of the result without this is computed
        stats_nomisc = self._stats_to_py_nums(
                        classification_report(self._rm_misc(truth), self._rm_misc(preds), output_dict=True)
                    ) if any(any("MISC" in ent for ent in sent) for sent in truth) else stats

        #FIXME: Do this manually instead of rerunning everything
        log(classification_report(truth, preds, zero_division=0, digits=4))
        if stats != stats_nomisc:
            log(classification_report(self._rm_misc(truth), self._rm_misc(preds), digits=4))

        return NER_TestResults(
                modelname   = self.model.name,
                dataname    = self.dataset.name,
                predictions = preds,
                statistics  = stats,
                statistics_nomisc = stats_nomisc,
        )

    @staticmethod
    def _rm_misc(seqs: list[list[str]]) -> list[list[str]]:
        """ Convert all "MISC"-entities to "O"-entities """
        return [["O" if "MISC" in ent else ent for ent in seq] for seq in seqs]

    @staticmethod
    def _stats_to_py_nums(stats: dict[dict[Union[np.int64, np.float64]]]) -> dict[dict[Union[int, float]]]:
        """ Convert each element in stats dict to python numerical instead of numpy data type """
        return {skey: {key: val.item() for key, val in sval.items()} for skey, sval in stats.items()}
