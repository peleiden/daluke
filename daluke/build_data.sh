# Run from LUKE repo
DATA_PATH=data
DUMP_FILE=da-dump-db
TOKENIZER="Maltehb/danish-bert-botxo"
# ^ One of xlm-roberta-base, xlm-roberta-large, xlm-roberta-large-finetuned-conll02-dutch, xlm-roberta-large-finetuned-conll02-spanish, xlm-roberta-large-finetuned-conll03-english, xlm-roberta-large-finetuned-conll03-german

P=$(pwd)
mkdir -p $DATA_PATH
cd $DATA_PATH
wget https://dumps.wikimedia.org/dawiki/latest/dawiki-latest-pages-articles.xml.bz2
cd $P

echo "BUILD DUMP DATABASE"
python3 -m luke.cli build-dump-db\
    $DATA_PATH/dawiki-latest-pages-articles.xml.bz2\
    $DATA_PATH/$DUMP_FILE

echo "BUILD ENTITY VOCAB"
python3 -m luke.cli build-entity-vocab\
    $DATA_PATH/$DUMP_FILE\
    $DATA_PATH/entity-vocab.jsonl

echo "BUILD PRETRAINING DATASET"
python3 -m luke.cli build-wikipedia-pretraining-dataset\
    $DATA_PATH/$DUMP_FILE\
    $TOKENIZER\
    $DATA_PATH/entity-vocab.jsonl\
    $DATA_PATH/da-pretrain-dataset\
    --sentence-tokenizer da

# echo "BUILD INTERWIKI DATABASE"
# python -m luke.cli build-interwiki-db\
#     data/$DUMP_FILE\
#     data/interwiki-db\
#     --language da  # This one failed
