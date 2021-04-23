#!/usr/bin/env python3
"""
A convenience script to take the output of the luke pretraining and tar-gz it
for easy loading
"""
import os
import re
import tarfile
from argparse import ArgumentParser

from pelutils import log, Levels

from daluke.serialize import OUT_FILE, MODEL_OUT, VOCAB_FILE, METADATA_FILE
from daluke.pretrain.train import MODEL_OUT as MODEL_FILE

def _natural_sort(L: list) -> list:
    """ Natural sorting from https://stackoverflow.com/a/37036428 """
    p_num = re.compile(r"(\d+)")
    return sorted(L, key=lambda l: [
        int(s) if s.isdigit() else s.lower() for s in re.split(p_num, l)
    ])

def _get_newest_model(path: str) -> str:
    """
    In folder of model_epoch1.bin, model_epoch10.bin, model_epoch2.bin,
    model_epoch10.bin is returned.
    """
    pattern = re.compile(MODEL_FILE.format(i=r"\d+"))
    models = list()
    for f in os.listdir(path):
        if pattern.match(f):
            models.append(f)
    if not models:
        raise FileNotFoundError(f"Could not find a model matching {MODEL_FILE.format(i='X')}")
    return _natural_sort(models)[-1]

def main():
    parser = ArgumentParser(description=\
        "Standalone convenience script used to collect the results from the pretraining of daLUKE "\
        "performed by the pretraining module")
    parser.add_argument("inpath", type=str,
            help= "Path to the output folder of the pretraining containing the model file, entity vocab. and metadata"
    )
    parser.add_argument("outpath", type=str, help="Folder in which the collected model is to be placed")
    args = parser.parse_args()
    log.configure(os.path.join(args.outpath, "collect.log"), "Collector", print_level=Levels.DEBUG)

    vocabfile, metafile = os.path.join(args.inpath, VOCAB_FILE), os.path.join(args.inpath, METADATA_FILE)
    modelfile = os.path.join(args.inpath, _get_newest_model(args.inpath))

    outfile = os.path.join(args.outpath, OUT_FILE)
    with tarfile.open(outfile, "w:gz") as tar:
        for f, n in zip((vocabfile, metafile), (VOCAB_FILE, METADATA_FILE)):
            log.debug(f"Compressing {vocabfile} ...")
            tar.add(f, arcname=n)
        log.debug(f"Compressing {modelfile} as {MODEL_OUT} ...")
        tar.add(modelfile, arcname=MODEL_OUT)
    log("Succesfully compressed file saved to", outfile)

if __name__ == '__main__':
    with log.log_errors:
        main()
