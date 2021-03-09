from __future__ import annotations
import os
import sys
from typing import Generator
from abc import ABC, abstractmethod

from danlp.datasets import DDT, WikiAnn

from pelutils import log

class TestDataset(ABC):
    def __init__(self, name: str):
        self.name = name
        self.data = None

    @abstractmethod
    def setup(self, *args, **kwargs):
        """ Set up the dataset, possibly downloading data """

    @abstractmethod
    def get_data(self) -> tuple(list[str], list[str]):
        """ List of words and list of entities (IOUB coded) """

class Dane(TestDataset):
    def setup(self, split="test"):
        # Third element is test data
        self.data = DDT().load_as_simple_ner(predefined_splits=True)[("train", "dev", "test").index(split)]

    def get_data(self) -> tuple(list[str], list[str]):
        return self.data # data is already (text, truth)

class Plank(TestDataset):
    def setup(self, data_path: str, split="test"):
        self.data: list[list[tuple[str]]] = list()
        with open(os.path.join(data_path, f"da_ddt-ud-ner-{split}.conll"), "r") as f:
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
    def setup(self, data_path: str, split="test"):
        self.data: list[list[tuple[str]]] = list()
        with open(os.path.join(data_path, split), "r") as f:
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

def setup_datasets(names_to_setup: list[str], wikiann_path: str="wikiann", plank_path: str="plank", split="test") -> list[TestDataset]:
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
        d.setup(**kwargs, split=split)
    return datasets

if __name__ == '__main__':
    """ Shows some Data stats """
    localdata = "../../local_data"

    localdata = os.path.join(sys.path[0], localdata)
    wikiann_p, plank_p = os.path.join(localdata, "wikiann"), os.path.join(localdata, "plank")
    log.configure(os.path.join(localdata, "data.log"), "data")

    for split in ("train", "dev", "test"):
        ds = setup_datasets(("DaNE", "Plank", "WikiANN"), wikiann_path=wikiann_p, plank_path=plank_p, split=split)
        for d in ds:
            log(f"{d.name} {split} sentences:", len(d.get_data()[0]))
    # now for better test statistics
    for d in ds:
        for ann in ("ORG", "PER", "LOC", "MISC"):
            log(f"#{ann} in {d.name}", sum(len([w for w in s if ann in w]) for s in d.get_data()[1]))
