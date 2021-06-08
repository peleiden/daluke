#!/usr/bin/env python3
"""
A convenience script to take the output of the luke pretraining and tar-gz it
for easy loading
"""
import os
import re
import shutil
import subprocess
import tarfile
from argparse import ArgumentParser

from pelutils import log, Levels

from daluke.serialize import COLLECT_OUT, MODEL_OUT, VOCAB_FILE, METADATA_FILE, save_to_archive
from daluke.pretrain.train import MODEL_OUT as MODEL_FILE

def _natural_sort(L: list) -> list:
    """ Natural sorting from https://stackoverflow.com/a/37036428 """
    p_num = re.compile(r"(\d+)")
    return sorted(L, key=lambda l: [
        int(s) if s.isdigit() else s.lower() for s in re.split(p_num, l)
    ])

def _get_newest_model(path: str) -> str:
    """
    In folder with daluke_epoch0.pt, daluke_epoch10.pt, daluke_epoch2.pt,
    daluke_epoch10.pt is returned.
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
        help= "Path to the output folder of the pretraining containing the model file. "\
            "Entity vocab. and metadata are assumed to be in parent folder of this"
    )
    parser.add_argument("outpath", type=str, help="Folder in which the collected model is to be placed."\
            "Can also be file name for model.")
    args = parser.parse_args()
    log.configure(os.path.join(args.outpath if os.path.isdir(args.outpath) else os.path.dirname(args.outpath), "collect.log"), "Collector", print_level=Levels.DEBUG)

    vocabfile, metafile = os.path.join(args.inpath, "..", VOCAB_FILE), os.path.join(args.inpath, "..", METADATA_FILE)
    modelfile = os.path.join(args.inpath, _get_newest_model(args.inpath))

    outfile = os.path.join(args.outpath, COLLECT_OUT) if os.path.isdir(args.outpath) else args.outpath

    # FIXME: Use `save_to_archive` from serialize
    if shutil.which("tar"):
        log.debug(f"Compressing to {outfile} using system tar tool...")
        try:
            for f, n in zip((vocabfile, metafile, modelfile), (VOCAB_FILE, METADATA_FILE, MODEL_OUT)):
                shutil.copy2(f, n)
            p = subprocess.Popen(
                ["tar", "-czvf", outfile, VOCAB_FILE, METADATA_FILE, MODEL_OUT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            p.wait()
        finally:
            for n in VOCAB_FILE, METADATA_FILE, MODEL_OUT:
                try:
                    os.remove(n)
                except FileNotFoundError:
                    pass
    else:
        with tarfile.open(outfile, "w:gz") as tar:
            for f, n in zip((vocabfile, metafile, modelfile), (VOCAB_FILE, METADATA_FILE, MODEL_OUT)):
                log.debug(f"Compressing {f} as {n} using build-in tar module (may take a while)...")
                tar.add(f, arcname=n)
    log("Succesfully compressed file saved to", outfile)

if __name__ == '__main__':
    with log.log_errors:
        main()
