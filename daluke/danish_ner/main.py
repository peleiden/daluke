#!/usr/bin/env python3
from pelutils import log

from daluke.danish_ner import setup_models

def main():
    #TODO: Use actual argparsing here
    log.configure(
        "local_test/danish_ner.log", "Benchmark danish NER models",
    )
    models = setup_models()
    log(f"Succesfully set up {len(models)} models")

if __name__ == '__main__':
    with log.log_errors:
        main()
