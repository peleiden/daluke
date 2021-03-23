import os
import multiprocessing as mp
import re
import shutil

from transformers import AutoTokenizer, XLMRobertaTokenizer
from wikipedia2vec.dump_db import DumpDB

from . import ICUSentenceTokenizer, load_entity_vocab

from pelutils import log

SENTENCE_TOKENIZER = "da"

def build_dataset(
    dump_db_file:      str,  # Location of file build by build-dump-db
    tokenizer_name:    str,  # Tokenizer to use, e.g. Maltehb/danish-bert-botxo for Danish BERT
    entity_vocab_file: str,  # Build by build-entity-vocab
    out_dir:           str,  # Where to put finished dataset. All contents will be removed before saving dataset
    max_seq_length      = 512,
    max_entity_length   = 128,
    min_sentence_length = 5,
):
    log.section("Building dataset")
    log("Reading dump database at %s" % dump_db_file)
    dump_db = DumpDB(dump_db_file)
    log("Building tokeninizer: %s" % tokenizer_name)
    tokenizer = (XLMRobertaTokenizer if "xlm-roberta" in tokenizer_name else AutoTokenizer).from_pretrained(tokenizer_name)
    log("Building sentence tokenizer: %s" % SENTENCE_TOKENIZER)
    sentence_tokenizer = ICUSentenceTokenizer(SENTENCE_TOKENIZER)
    log("Loading entity vocab at %s" % entity_vocab_file)
    entity_vocab = load_entity_vocab(entity_vocab_file)

    log("Resetting output directory at %s" % out_dir)
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)




