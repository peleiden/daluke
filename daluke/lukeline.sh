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

echo "PRETRAIN"
# TODO: Danish BERT
python -m luke.cli pretrain\
    data/da-pretrain-dataset\
    data/\
    --bert-model-name "roberta-base"\
    --num-epochs 2\
    --fp16\
    --log-dir data/logs

# echo "BUILD INTERWIKI DATABASE"
# python -m luke.cli build-interwiki-db\
#     data/$DUMP_FILE\
#     data/interwiki-db\
#     --language da  # This one failed

## Questions ##
#1) Which of the steps are needed? interwiki failed and I did not run multilingual-entity-vocab
#2) Where do I get to choose my pretrained transformer? Is it in the buid-wiki-pretraining? This seemed to only be about the tokenizer.
#3) How do I get the danish tokenizer (e.g. https://huggingface.co/Maltehb/danish-bert-botxo)? This failed as I had too old transformers version, but with the newest transformers version, the luke code failed.
