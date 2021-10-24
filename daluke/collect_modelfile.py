#!/usr/bin/env python3
"""
A convenience script to take the output of the luke pretraining and tar-gz it
for easy loading
"""
from daluke.pretrain.data.build import DatasetBuilder
import os
import re
import json
import shutil
import subprocess
import tarfile
from argparse import ArgumentParser

from pelutils import log, Levels

from daluke.pretrain.train import MODEL_OUT as MODEL_FILE
from daluke.serialize import MODEL_OUT, VOCAB_FILE, METADATA_FILE, TOKEN_MAP_FILE

def _natural_sort(L: list) -> list:
    """ Natural sorting from https://stackoverflow.com/a/37036428 """
    p_num = re.compile(r"(\d+)")
    return sorted(L, key=lambda l: [
        int(s) if s.isdigit() else s.lower() for s in re.split(p_num, l)
    ])

def _get_newest_model(path: str) -> str:
    """ In folder with daluke_pu_0.pt, daluke_pu_10.pt, daluke_pu_2.pt, daluke_pu_10.pt is returned """
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
            "Entity vocab. and metadata are assumed to be in parent folder of this."\
            "Can also be path to an exact model file, in which case this will be used instead of the newest."
    )
    parser.add_argument("outpath", type=str, help="File path to the compressed model")
    args = parser.parse_args()
    log.configure(os.path.join(args.outpath if os.path.isdir(args.outpath) else os.path.dirname(args.outpath), "collect.log"), "Collector", print_level=Levels.DEBUG)

    modelpath = args.inpath if os.path.isdir(args.inpath) else os.path.dirname(args.inpath)
    vocabfile, metafile = os.path.join(modelpath, "..", VOCAB_FILE), os.path.join(modelpath, "..", METADATA_FILE)
    modelfile = os.path.join(args.inpath, _get_newest_model(args.inpath)) if os.path.isdir(args.inpath) else args.inpath

    os.makedirs(os.path.split(args.outpath)[0], exist_ok=True)

    ins, outs = [vocabfile, metafile, modelfile], [VOCAB_FILE, METADATA_FILE, MODEL_OUT]
    # If reduction is used, also collect the token map
    with open(metafile, "r") as f:
        is_reduced = json.load(f).get("reduced-vocab")
    if is_reduced:
        ins.append(os.path.join(modelpath, "..", DatasetBuilder.token_map_file))
        outs.append(TOKEN_MAP_FILE)
    log.debug(f"Using:", *ins)

    # Operate directly on disk as opposed to serialize.save_to_archive which requires us to load the data into mem.
    if shutil.which("tar"):
        log.debug(f"Compressing to {args.outpath} using system tar tool...")
        try:
            for f, n in zip(ins, outs):
                shutil.copy2(f, n)
            p = subprocess.Popen(
                ["tar", "-czvf", args.outpath] + outs,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            p.wait()
        finally:
            for n in outs:
                try:
                    os.remove(n)
                except FileNotFoundError:
                    pass
    else:
        with tarfile.open(args.outpath, "w:gz") as tar:
            for f, n in zip(ins, outs):
                log.debug(f"Compressing {f} as {n} using build-in tar module (may take a while)...")
                tar.add(f, arcname=n)
    log("Succesfully compressed file saved to", args.outpath)

if __name__ == '__main__':
    with log.log_errors:
        main()
