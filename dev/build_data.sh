# Run from daluke repo
DATA_PATH=/work3/$USER/pdata2
DUMP_FILE=da-dump-db.dump
TOKENIZER="Maltehb/danish-bert-botxo"
# ^ One of xlm-roberta-base, xlm-roberta-large, xlm-roberta-large-finetuned-conll02-dutch,
# xlm-roberta-large-finetuned-conll02-spanish, xlm-roberta-large-finetuned-conll03-english,
# xlm-roberta-large-finetuned-conll03-german, Maltehb/danish-bert-botxo, etc.
PREPROCESS=repeat-entities  # default or repeat-entities
WIKIDATE=20210901

DALUKE="$(dirname "$0")"
LUKE=$DALUKE/luke
export PYTHONPATH=$PYTHONPATH:$DALUKE:$LUKE

echo "EMPTYING $DATA_PATH"
rm -rf $DATA_PATH
mkdir -p $DATA_PATH

echo "DOWNLOADING WIKIDUMP"
cd $DATA_PATH
wget https://dumps.wikimedia.org/dawiki/$WIKIDATE/dawiki-$WIKIDATE-pages-articles.xml.bz2
cd $DALUKE

echo "BUILD TEMPORARY DUMP DATABASE"
cd $LUKE
rm -f $DATA_PATH/$DUMP_FILE*
python3 -m luke.cli build-dump-db\
    $DATA_PATH/../dawiki-$WIKIDATE-pages-articles.xml.$PREPROCESS.bz2\
    $DATA_PATH/$DUMP_FILE

echo "BUILD ENTITY VOCAB"
python3 -m luke.cli build-entity-vocab\
    $DATA_PATH/$DUMP_FILE\
    $DATA_PATH/../entity-vocab.jsonl

echo "PREPROCESSING WIKIDUMP"
cd $DALUKE
python3 daluke/pretrain/data/preprocess.py\
    $DATA_PATH/../dawiki-20210901-pages-articles.xml.bz2\
    --function $PREPROCESS\
    --entity-vocab-file $DATA_PATH/../entity-vocab.jsonl\
    --dagw-sections $DATA_PATH/../dagw/sektioner

echo "BUILD DUMP DATABASE"
cd $LUKE
rm -f $DATA_PATH/$DUMP_FILE*
python3 -m luke.cli build-dump-db\
    $DATA_PATH/../dawiki-20210301-pages-articles.xml.$PREPROCESS.bz2\
    $DATA_PATH/$DUMP_FILE

echo "BUILD PRETRAINING DATASET"
cd $DALUKE
python3 daluke/pretrain/data/run.py\
    $DATA_PATH/$DUMP_FILE\
    $DATA_PATH/../entity-vocab.jsonl\
    $TOKENIZER\
    $DATA_PATH
