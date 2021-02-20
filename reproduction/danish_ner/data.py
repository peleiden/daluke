from __future__ import annotations
from typing import Generator
from abc import ABC, abstractmethod
import os

from danlp.datasets import DDT, WikiAnn

from pelutils import log

class TestDataset(ABC):
    def __init__(self, name: str):
        self.name = name
        self.data = None

    @abstractmethod
    def setup(self, **kwargs):
        """ Set up the dataset, possibly downloading data """

    @abstractmethod
    def get_data(self) -> tuple(list[str], list[str]):
        """ List of words and list of entities (IOUB coded) """

class Dane(TestDataset):
    def setup(self):
        # Third element is test data
        self.data = DDT().load_as_simple_ner(predefined_splits=True)[2]

    def get_data(self) -> tuple(list[str], list[str]):
        return self.data # data is already (text, truth)

class Plank(TestDataset):
    def setup(self, data_path: str):
        self.data: list[list[tuple[str]]] = list()
        with open(os.path.join(data_path, "da_ddt-ud-ner-test.conll"), "r") as f:
            sentence = []
            for l in f:
                if not l.split():
                    self.data.append(sentence)
                    sentence = list()
                else:
                    sentence.append(l.split())

    def get_data(self) -> tuple(list[str], list[str]):
        words, truth = list(), list()
        for s in self.data:
            w, t = map(list, zip(*s))
            words.append(w)
            truth.append(t)
        return words, truth

class Wikiann(TestDataset):
    def setup(self, data_path: str):
        self.data: list[list[tuple[str]]] = list()
        with open(os.path.join(data_path, "test"), "r") as f:
            sentence = []
            for l in f:
                if not l.split():
                    self.data.append(sentence)
                    sentence = []
                else:
                    sentence.append(l.replace("da:", "").split())

    def get_data(self) -> tuple(list[str], list[str]):
        words, truth = list(), list()
        for s in self.data:
            w, t = map(list, zip(*s))
            words.append(w)
            truth.append(t)
        return words, truth

ALL_DATASETS = (
    Dane("DaNE"),
    Plank("Plank"),
    Wikiann("WikiANN"),
)

def setup_datasets(names_to_setup: list[str], wikiann_path: str="wikiann", plank_path: str="plank") -> list[TestDataset]:
    datasets = []
    for name in names_to_setup:
        try:
            datasets.append(
                next(d for d in ALL_DATASETS if d.name == name)
            )
        except IndexError as ie:
            raise ValueError(f"Dataset with given name {name} not found, see --help for options") from ie
    for d in datasets:
        log(f"Setting up dataset \"{d.name}\" ...")
        kwargs = dict()
        if isinstance(d, Wikiann):
            kwargs["data_path"] = wikiann_path
        elif isinstance(d, Plank):
            kwargs["data_path"] = plank_path
        d.setup(**kwargs)
    return datasets
