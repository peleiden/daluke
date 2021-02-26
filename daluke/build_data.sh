# This was run from LUKE repo

DUMP_FILE=da-dump-db
# TODO: Figure out how to use danish bert tokenizer
TOKENIZER=xlm-roberta-base  # One of xlm-roberta-base, xlm-roberta-large, xlm-roberta-large-finetuned-conll02-dutch, xlm-roberta-large-finetuned-conll02-spanish, xlm-roberta-large-finetuned-conll03-english, xlm-roberta-large-finetuned-conll03-german

mkdir -p data
cd data
wget https://dumps.wikimedia.org/dawiki/latest/dawiki-latest-pages-articles.xml.bz2
cd ..

alias python=python3  # So it works on HPC

echo "BUILD DUMP DATABASE"
python -m luke.cli build-dump-db\
    data/dawiki-latest-pages-articles.xml.bz2\
    data/$DUMP_FILE

echo "BUILD ENTITY VOCAB"
python -m luke.cli build-entity-vocab\
    data/$DUMP_FILE\
    data/entity-vocab.jsonl

echo "BUILD PRETRAINING DATASET"
# If you get an error that Locale cannot be imported from icu, run the following:
# pip install icu pyicu pycld2 morfessor
python -m luke.cli build-wikipedia-pretraining-dataset\
    data/$DUMP_FILE\
    $TOKENIZER\
    data/entity-vocab.jsonl\
    data/da-pretrain-dataset
