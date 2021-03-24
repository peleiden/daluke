import os
import shutil
from argparse import ArgumentParser

from pelutils.logger import log, Levels

from daluke.pretrain.data.build import Builder

def run_build_dataset():

    parser = ArgumentParser()
    parser.add_argument("dump_db_file", type=str)
    parser.add_argument("entity_vocab_file", type=str)
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--max-seq-length", type=str, default=512)
    parser.add_argument("--max-entity-length", type=str, default=128)
    parser.add_argument("--min-sentence-length", type=str, default=5)
    args = parser.parse_args()

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir)
    log.configure(os.path.join(args.out_dir, "build-dataset.log"), "Build dataset", log_commit=True, print_level=Levels.DEBUG)

    builder = Builder(**args.__dict__)
    builder.build()


if __name__ == "__main__":
    with log.log_errors:
        run_build_dataset()
