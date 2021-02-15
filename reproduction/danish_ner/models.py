from __future__ import annotations
from abc import ABC, abstractmethod

import danlp.models as dm
from flair.data import Sentence
from polyglot.text import Text as polyglot_text

from pelutils import log

class NER_TestModel(ABC):
    """ Allow testing a wide range of models by using the same NER model API """
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.categories: tuple = None

    @abstractmethod
    def setup(self):
        """ Set up the model, possibly downloading model files """

    @abstractmethod
    def predict(self, words: list[str]) -> list[str]:
        """ Predict the named entities of the input string `words` """

class Bert(NER_TestModel):
    def setup(self):
        self.model = dm.load_bert_ner_model()

    def predict(self, words: list[str]) -> list[str]:
        #TODO: This is a temporary woraround until danlp fixes giving BERT custom tokens, see
        # https://github.com/alexandrainst/danlp/issues/113
        masks = list()
        for w in words:
            tokens = self.model.tokenizer.tokenize(w)
            masks.extend([1]+[0]*(len(tokens)-1))
        _, labels = self.model.predict(" ".join(words))
        return [label for label, mask in zip(labels, masks) if mask]

class Flair(NER_TestModel):
    #FIXME: Dies on my machine due to CUDA only
    def setup(self):
        self.model = dm.load_flair_ner_model()

    def predict(self, words: list[str]) -> list[str]:
        sentence = Sentence(words)
        self.model.predict(sentence)
        labels = sentence.to_tagged_string
        raise NotImplementedError #FIXME

class Spacy(NER_TestModel):
    def setup(self):
        self.model = dm.load_spacy_model()

    def predict(self, words: list[str]) -> list[str]:
        tokens = self.model(words)
        labels = [token.ent_type_ for token in tokens]
        raise NotImplementedError #FIXME

class Polyglot(NER_TestModel):
    # See: https://polyglot.readthedocs.io/en/latest/NamedEntityRecognition.html
    def setup(self):
        self.model = polyglot_text

    def predict(self, words: str[list]) -> list[str]:
        text = self.model(words)
        labels = text.entities
        raise NotImplementedError #FIXME

class Daner(NER_TestModel):
    # https://github.com/ITUnlp/daner
    def setup(self):
        # FIXME: How to get programmatical access to this? daner is in java.
        # The best solution is probably to clone the repo and us os.system calls to run daner
        raise NotImplementedError

    def predict(self, words: str[list]) -> list[str]:
        raise NotImplementedError

ALL_MODELS = (
    Bert("BERT"),
    # Flair("Flair"),
    # Spacy("spaCy"),
    # Polyglot("Polyglot"),
    # Daner("daner"),
)

def setup_models(names_to_setup: list[str]) -> list[NER_TestModel]:
    models = []
    for name in names_to_setup:
        try:
            models.append(
                [m for m in ALL_MODELS if m.name == name][0]
            )
        except IndexError as ie:
            raise ValueError(f"Model with given name {name} not found, see --help for options") from ie
    for m in models:
        log(f"Setting up model \"{m.name}\" ... ")
        m.setup()
    return models
