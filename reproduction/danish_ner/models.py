from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Generator
import subprocess
import shutil
from pathlib import Path

import danlp.models as dm
from danlp.download import DEFAULT_CACHE_DIR, download_model, _unzip_process_func
from flair.data import Sentence, Token
from polyglot.tag import NEChunker
from polyglot.text import WordList
from spacy.util import load_model_from_path as spacy_load
from NERDA.datasets import download_dane_data
from NERDA.precooked import DA_BERT_ML, DA_ELECTRA_DA
import torch

from pelutils import log


class NER_TestModel(ABC):
    """ Allow testing a wide range of models by using the same NER model API """
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.categories: tuple = None

    @abstractmethod
    def setup(self, **kwargs):
        """ Set up the model, possibly downloading model files """

    @abstractmethod
    def predict(self, text: Generator[list[str]]) -> list[list[str]]:
        """ Predict the named entities of the input string `words` """

class Bert(NER_TestModel):
    def setup(self):
        self.model = dm.load_bert_ner_model()

    def predict(self, text: Generator[list[str]]) -> list[list[str]]:
        preds = list()
        for words in text:
            #TODO: This is a temporary woraround until danlp fixes giving BERT custom tokens, see
            # https://github.com/alexandrainst/danlp/issues/113
            masks = list()
            for w in words:
                tokens = self.model.tokenizer.tokenize(w)
                masks.extend([1]+[0]*(len(tokens)-1))
            _, labels = self.model.predict(" ".join(words))
            preds.append([label for label, mask in zip(labels, masks) if mask])
        return preds

class Flair(NER_TestModel):
    def setup(self):
        # A horrible hack to fix a check in flair.models.language_model for pytorch version that fails on cpu
        if "+cpu" in torch.__version__:
            torch.__version__ = torch.__version__.replace("+cpu", "")
        self.model = dm.load_flair_ner_model()

    def predict(self, text: Generator[list[str]]) -> list[list[str]]:
        preds = list()
        flair_sents = list()
        for words in text:
            s = Sentence()
            for word in words:
                s.add_token(Token(word))
            flair_sents.append(s)
        self.model.predict(flair_sents)
        return [[tok.tags["ner"].value for tok in s] for s in flair_sents]

class Spacy(NER_TestModel):
    def setup(self):
        path = download_model("spacy", DEFAULT_CACHE_DIR, process_func=_unzip_process_func)
        self.model = spacy_load(Path(path))

    def predict(self, text: Generator[list[str]]) -> list[list[str]]:
        preds = list()
        for words in text:
            pred = list()
            tok = self.model.tokenizer.tokens_from_list(words)
            self.model.entity(tok)
            pred = ["O" if t.ent_iob_ == "O" else f"{t.ent_iob_}-{t.ent_type_}" for t in tok]
            preds.append(pred)
        return preds

class Polyglot(NER_TestModel):
    """ https://polyglot.readthedocs.io/en/latest/NamedEntityRecognition.html """
    def setup(self):
        os.system("polyglot download embeddings2.da")
        os.system("polyglot download ner2.da")
        self.model = NEChunker(lang='da')

    def predict(self, text: Generator[list[str]]) -> list[list[str]]:
        preds = list()
        for words in text:
            word_ents = list(self.model.annotate(WordList(words, language='da')))
            preds.append([ent for word, ent in word_ents])
        return preds

class Daner(NER_TestModel):
    """ https://github.com/ITUnlp/daner """
    def setup(self, repo_path: str, data_path: str):
        self.data_path = data_path
        self.exe, self.model = os.path.join(repo_path, "stanford-ner.jar"), os.path.join(repo_path, "da01.model.gz")
        if not os.path.exists(self.exe) or not os.path.exists(self.model):
            raise FileNotFoundError(f"Could not find daner model in given repo path {os.path.abspath(repo_path)}")

    def predict(self, text: Generator[list[str]]) -> list[list[str]]:
        # Save the data temporarily to tisk
        tmppath = os.path.join(self.data_path, "tmp[dont delete]")
        os.makedirs(tmppath, exist_ok=True)
        textfile = os.path.join(tmppath, "danerdata.txt")
        added_periods = set()
        with open(textfile, "w") as t:
            for i, words in enumerate(text):
                t.write(" ".join(words))
                # Period added to make sure that daner does not read multiple sentences
                if not "." in words:
                    t.write(" .")
                    added_periods.add(i)
                t.write("\n")
        # Read saved data using daner java executable
        cmd = ("java", "-cp", self.exe, "edu.stanford.nlp.ie.crf.CRFClassifier", "-loadClassifier", self.model, "-textFile", textfile)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        # Read output of java software from stdout while extracting labels
        preds = list()
        lines = list()
        for i, line in enumerate(iter(p.stdout.readline, b'')):
            labels = list()
            line_ = list()
            for word in line.decode("utf8").split():
                line_.append(word)
                labels.append(word.split("/")[-1])
            if i in added_periods:
                labels = labels[:-1]
                lines = lines[:-1]
            # Dont include the last manually added period
            preds.append(labels)
            lines.append(line_)

        from evaluation import Evaluator
        for p, w, l in zip(preds, text, lines):
            try:
                assert len(p) == len(w)
                e = Evaluator("lala", "lala")
                for p_ in p:
                    e._get_class(p_)
            except Exception as allah:
                print(allah)
                print(p)
                print(w)
                print(l)
                breakpoint()
        # Cleanup
        shutil.rmtree(tmppath, ignore_errors=True)
        return preds

class Mbert(NER_TestModel):
    def setup(self):
        download_dane_data()
        self.model = DA_BERT_ML()
        self.model.download_network()
        self.model.load_network()

    def predict(self, text: Generator[list[str]]) -> list[list[str]]:
        return self.model.predict(list(text))


class Ælæctra(NER_TestModel):
    def setup(self):
        download_dane_data()
        self.model = DA_ELECTRA_DA()
        self.model.download_network()
        self.model.load_network()

    def predict(self, text: Generator[list[str]]) -> list[list[str]]:
        return self.model.predict(list(text))

ALL_MODELS = (
    Bert("BERT"),
    Flair("Flair"),
    Spacy("spaCy"),
    Polyglot("Polyglot"),
    #Daner("daner"),
    Mbert("mBERT"),
    Ælæctra("Ælæctra"),
)

def setup_models(names_to_setup: list[str], location: str, daner_path: str="daner") -> list[NER_TestModel]:
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
        kwargs = dict()
        if isinstance(m, Daner):
            kwargs["repo_path"] = daner_path
            kwargs["data_path"] = location
        m.setup(**kwargs)
    return models
