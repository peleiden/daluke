# Run from LUKE repo
DATA_PATH=/work3/$USER/pdata2
DUMP_FILE=da-dump-db.dump
TOKENIZER="Maltehb/danish-bert-botxo"
# ^ One of xlm-roberta-base, xlm-roberta-large, xlm-roberta-large-finetuned-conll02-dutch,
# xlm-roberta-large-finetuned-conll02-spanish, xlm-roberta-large-finetuned-conll03-english,
# xlm-roberta-large-finetuned-conll03-german, Maltehb/danish-bert-botxo, etc.
PREPROCESS=repeat-entities  # default or repeat-entities
DALUKE=$HOME/daluke
LUKE=$DALUKE/luke
export PYTHONPATH=$PYTHONPATH:$DALUKE:$LUKE
module load python3/3.8.4

mkdir -p $DATA_PATH
cd $DATA_PATH
# wget https://dumps.wikimedia.org/dawiki/latest/dawiki-latest-pages-articles.xml.bz2
cd $DALUKE

echo "PREPROCESSING WIKIDATA"
cd $DALUKE
python3 daluke/pretrain/data/preprocess.py $DATA_PATH/../dawiki-20210301-pages-articles.xml.bz2 --func $PREPROCESS

echo "BUILD DUMP DATABASE"
cd $LUKE
rm -f $DATA_PATH/../$DUMP_FILE
rm -f $DATA_PATH/../$DUMP_FILE-lock
python3 -m luke.cli build-dump-db\
    $DATA_PATH/../dawiki-20210301-pages-articles.xml.$PREPROCESS.bz2\
    $DATA_PATH/../$DUMP_FILE

echo "BUILD ENTITY VOCAB"
python3 -m luke.cli build-entity-vocab\
    $DATA_PATH/../$DUMP_FILE\
    $DATA_PATH/../entity-vocab.jsonl

echo "BUILD PRETRAINING DATASET"
cd $DALUKE
python3 daluke/pretrain/data/run.py\
    $DATA_PATH/../da-dump-db.dump\
    $DATA_PATH/../entity-vocab.jsonl\
    $TOKENIZER\
    $DATA_PATH

