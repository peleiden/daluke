import os
from danish_ner.evaluation import NER_TestResults

def get_errors(lang: str, path: str):
    if lang == "da":
        res = NER_TestResults.load(path)
        raise NotImplentedError("Currently, Danish feed-forward is not supported")

    elif lang == "en":
        with open(os.path.join(path, "CoNLL2003", "eng.testb")) as truthfile:
            pass
        with open(os.path.join(path, "test_predictions.txt")) as preds:
            pass
    else:
        raise ValueError(f"Unknown language {lang}")

