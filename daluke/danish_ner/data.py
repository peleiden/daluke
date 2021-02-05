from __future__ import annotations
from typing import Generator
from abc import ABC, abstractmethod

from danlp.datasets import DDT, WikiAnn

from pelutils import log

class TestDataset(ABC):
    def __init__(self, name: str):
        self.name = name
        self.text: list[str] = None
        self.ne:   list[str] = None

    @abstractmethod
    def setup(self):
        """ Set up the dataset, possibly downloading data """

    @abstractmethod
    def get_data(self) -> Generator(tuple(list[str], list[str])):
        """ Yield list of words and list of entities (LALALA coded) """

class Dane(TestDataset):
    def setup(self):
        # Third element is test data
        self.text, self.ne = DDT().load_as_simple_ner(predefined_splits=True)[2]

    def get_data(self) -> (list[str], list[str]):
        for text, entities in zip(self.text, self.ne):
            yield text, entities

class Plank(TestDataset):
    def setup(self):
        raise NotImplementedError

    def get_data(self) -> Generator(tuple(list[str], list[str])):
        raise NotImplementedError

class Wikiann(TestDataset):
    def setup(self):
        self.data = WikiAnn().load_with_flair()
        raise NotImplementedError

    def get_data(self) -> Generator(tuple(list[str], list[str])):
        raise NotImplementedError

ALL_DATASETS = (
    Dane("DaNE"),
    Plank("Plank"),
    Wikiann("WikiANN"),
)

def setup_datasets(names_to_setup: list[str]) -> list[TestDataset]:
    datasets = []
    for name in names_to_setup:
        try:
            datasets.append(
                [d for d in ALL_DATASETS if d.name == name][0]
            )
        except IndexError as ie:
            raise ValueError(f"Dataset with given name {name} not found, see --help for options") from ie
    for d in datasets:
        log.debug(f"Setting up dataset \"{d.name}\" ...")
        d.setup()
    return datasets
