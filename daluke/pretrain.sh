DATA_PATH=data

echo "PRETRAIN"
python3 -m luke.cli pretrain\
    $DATA_PATH/da-pretrain-dataset\
    $DATA_PATH\
    --bert-model-name "Maltehb/danish-bert-botxo"\
    --num-epochs 2\
    --log-dir $DATA_PATH/logs
#    --cpu

#BERT models "roberta-base"/"Maltehb/danish-bert-botxo"

## Questions ##
#1) Which of the steps are needed? interwiki failed and I did not run multilingual-entity-vocab
#2) How do I get the danish tokenizer (e.g. https://huggingface.co/Maltehb/danish-bert-botxo)? This failed as I had too old transformers version, but with the newest transformers version, the luke code failed.
