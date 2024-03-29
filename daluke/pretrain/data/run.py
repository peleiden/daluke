import os
from argparse import ArgumentParser

from pelutils.logger import log, Levels

from daluke.pretrain.data.build import DatasetBuilder

def run_build_dataset():

    parser = ArgumentParser()
    parser.add_argument("dump_db_file", type=str)
    parser.add_argument("entity_vocab_file", type=str)
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--validation-prob", type=float, default=0.01)
    parser.add_argument("--max-entities", type=int, default=128)
    parser.add_argument("--max-entity-span", type=int, default=30)
    parser.add_argument("--min-sentence-length", type=int, default=5)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--max-vocab-size", type=int, default=-1)
    args = parser.parse_args()

    log.configure(os.path.join(args.out_dir, "build-dataset.log"), "Build dataset", log_commit=True, print_level=Levels.DEBUG)

    builder = DatasetBuilder(**vars(args))
    builder.build()

if __name__ == "__main__":
    with log.log_errors:
        run_build_dataset()
