
alias python=python3

echo "PRETRAIN"
# TODO: Danish BERT
python -m luke.cli pretrain\
    data/da-pretrain-dataset\
    data/\
    --bert-model-name "roberta-base"\
    --num-epochs 2\
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
