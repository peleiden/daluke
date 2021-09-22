# Run from daluke repo
DATA_PATH=/work3/$USER/pdata3
DUMP_FILE=dump.db
TOKENIZER=xlm-roberta-base
# ^ One of xlm-roberta-base, xlm-roberta-large, xlm-roberta-large-finetuned-conll02-dutch,
# xlm-roberta-large-finetuned-conll02-spanish, xlm-roberta-large-finetuned-conll03-english,
# xlm-roberta-large-finetuned-conll03-german, Maltehb/danish-bert-botxo, etc.
PREPROCESS=repeat-entities  # default or repeat-entities
WIKIDATE=20210901

DALUKE=$PWD
LUKE=$DALUKE/luke
export PYTHONPATH=$PYTHONPATH:$DALUKE:$LUKE

echo "DOWNLOADING WIKIDUMP"
mkdir -p $DATA_PATH
cd $DATA_PATH
if [ ! -f dawiki-$WIKIDATE-pages-articles.xml.bz2 ];
then
    wget https://dumps.wikimedia.org/dawiki/$WIKIDATE/dawiki-$WIKIDATE-pages-articles.xml.bz2
fi
if [ ! -d dagw ];
then
    wget https://bit.ly/danishgigaword10
    unzip dagw_v1.0-release.zip
    rm dagw_v1.0-release.zip
fi
cd $DALUKE

echo "BUILD TEMPORARY DUMP DATABASE"
cd $LUKE
python3 -m luke.cli build-dump-db\
    $DATA_PATH/dawiki-$WIKIDATE-pages-articles.xml.bz2\
    $DATA_PATH/$DUMP_FILE

echo "BUILD ENTITY VOCAB"
python3 -m luke.cli build-entity-vocab\
    $DATA_PATH/$DUMP_FILE\
    $DATA_PATH/entity-vocab.jsonl

echo "PREPROCESSING WIKIDUMP"
cd $DALUKE
python3 daluke/pretrain/data/preprocess.py\
    $DATA_PATH/dawiki-20210901-pages-articles.xml.bz2\
    --function $PREPROCESS\
    --entity-vocab-file $DATA_PATH/entity-vocab.jsonl\
    --dagw-sections $DATA_PATH/dagw/sektioner

echo "BUILD DUMP DATABASE"
cd $LUKE
python3 -m luke.cli build-dump-db\
    $DATA_PATH/dawiki-20210301-pages-articles.xml.$PREPROCESS.bz2\
    $DATA_PATH/$DUMP_FILE

echo "BUILD PRETRAINING DATASET"
cd $DALUKE
python3 daluke/pretrain/data/run.py\
    $DATA_PATH/$DUMP_FILE\
    $DATA_PATH/entity-vocab.jsonl\
    $TOKENIZER\
    $DATA_PATH
