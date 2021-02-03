#!/usr/bin/env python3
from __future__ import annotations
import os

from daluke.danish_ner import setup_models

from pelutils import log, Parser



def main():
    options = dict()
    parser = Parser(options, name="Danish NER Reproduction")
    parser.parse()

    log.configure(
        os.path.join(parser.location, "danish_ner.log"), "Benchmark danish NER models",
    )

    models = setup_models()
    log(f"Succesfully set up {len(models)} models")


if __name__ == '__main__':
    with log.log_errors:
        main()
