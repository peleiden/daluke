# Run from daluke repo
DATA_PATH=/work3/$USER/pdata3
DUMP_FILE=dump.db
TOKENIZER=xlm-roberta-large  # Must exist on Huggingface
WIKIDATE=20210901  # YYYYMMDD
PREPROCESS=repeat-entities

DALUKE=$PWD
LUKE=$DALUKE/luke
export PYTHONPATH=$PYTHONPATH:$DALUKE:$LUKE

mkdir -p $DATA_PATH
cd $DATA_PATH
if [ ! -f dawiki-$WIKIDATE-pages-articles.xml.bz2 ];
then
    echo "DOWNLOADING WIKIDUMP"
    wget https://dumps.wikimedia.org/dawiki/$WIKIDATE/dawiki-$WIKIDATE-pages-articles.xml.bz2
fi
if [ ! -d dagw ];
then
    echo "DOWNLOADING GIGAWORD"
    wget https://bit.ly/danishgigaword10
    unzip danishgigaword10
    rm danishgigaword10
fi

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
   $DATA_PATH/dawiki-$WIKIDATE-pages-articles.xml.bz2\
   --function $PREPROCESS\
   --entity-vocab-file $DATA_PATH/entity-vocab.jsonl\
   --dagw-sections $DATA_PATH/dagw/sektioner

echo "BUILD DUMP DATABASE"
cd $LUKE
python3 -m luke.cli build-dump-db\
   $DATA_PATH/dawiki-$WIKIDATE-pages-articles.xml.$PREPROCESS.bz2\
   $DATA_PATH/$PREPROCESS\_\_$DUMP_FILE

echo "BUILD PRETRAINING DATASET"
cd $DALUKE
python3 daluke/pretrain/data/run.py\
    $DATA_PATH/$PREPROCESS\_\_$DUMP_FILE\
    $DATA_PATH/entity-vocab.jsonl\
    $TOKENIZER\
    $DATA_PATH\
    --max-vocab-size 60000\
    --validation-prob 0.0025
