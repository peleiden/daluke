#!/usr/bin/env python3
import click
from pelutils import log, Table, Levels
from pelutils.ds import no_grad

from daluke.api.predict import predict_mlm, predict_ner

@click.group()
def cli():
    pass

@cli.command("masked")
@click.option("--filepath", default="")
@click.option("--text", default="")
@no_grad
def masked(filepath: str, text: str):
    if not filepath and not text:
        raise ValueError("Either filepath or text must be given")
    elif filepath and text:
        raise ValueError("Filepath and text cannot both be given")
    elif filepath:
        with open(filepath) as f:
            text = f.read()

    text, top_preds = predict_mlm(text)
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

    preds = predict_ner(text)
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