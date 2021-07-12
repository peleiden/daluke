#!/usr/bin/env python3
from __future__ import annotations
import click
from pelutils import log, Table, Levels
from pelutils.ds import no_grad

from daluke.api.automodels import AutoMLMDaLUKE, AutoNERDaLUKE
from daluke.api.predict import predict_mlm, predict_ner

@click.group()
def cli():
    pass

@cli.command("masked")
@click.option("--filepath", default="")
@click.option("--text", default="")
@click.option("--entity-spans", default="")
@no_grad
def masked(filepath: str, text: str, entity_spans: list[str]):
    """ Entities are given as 'start1,end1;start2,end2 ...'
    Ends are optional. If not given, they will be set to start+1
    Spans are 1-indexed with inclusive ends """
    if not filepath and not text:
        raise ValueError("Either filepath or text must be given")
    elif filepath and text:
        raise ValueError("Filepath and text cannot both be given")
    elif filepath:
        with open(filepath) as f:
            text = f.read()

    entity_spans = [(int(x.split(",")[0])-1, int(x.split(",")[1])) if "," in x else (int(x)-1, int(x)) for x in entity_spans.split(";") if x]

    log.debug("Loading model")
    daluke_mlm = AutoMLMDaLUKE()

    text, top_preds = predict_mlm(text, entity_spans, daluke_mlm)
    log("The top 5 predictions with likelihoods for each [MASK] were", top_preds)
    log("DaLUKE's best predictions were", text)

@cli.command("ner")
@click.option("--filepath", default="")
@click.option("--text", default="")
@no_grad
def ner(filepath: str, text: str):
    if not filepath and not text:
        raise ValueError("Either filepath or text must be given")
    elif filepath and text:
        raise ValueError("Filepath and text cannot both be given")
    elif filepath:
        with open(filepath) as f:
            text = f.read()

    log.debug("Loading model")
    daluke_ner = AutoNERDaLUKE()

    preds = predict_ner(text, daluke_ner)
    t = Table()
    t.add_header(["Word", "IOB NER Prediction"])
    for word, pred in zip(text.split(), preds):
        t.add_row([word, pred])
    log(t)

def main():
    log.configure(print_level=Levels.DEBUG)
    with log.log_errors:
        cli()

if __name__ == "__main__":
    main()
