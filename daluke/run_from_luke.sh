# This was run from LUKE repo
cd data
wget https://dumps.wikimedia.org/dawiki/latest/dawiki-latest-pages-articles.xml.bz2
cd ..
python -m luke.cli build-dump-db data/dawiki-latest-pages-articles.xml.bz2 data/dawiki-dump
python -m luke.cli build-entity-vocab data/dawiki-dump data/entity-vocab
python -m luke.cli build-interwiki-db data/dawiki-dump data/interwiki-db --language da # This one failed
python -m luke.cli build-wikipedia-pretraining-dataset data/dawiki-dump "xlm-roberta-base" data/entity-vocab
python -m luke.cli pretrain data/dawiki-pretrain data/daluke-out

## Questions ##
#1) Which of the steps are needed? interwiki failed and I did not run multilingual-entity-vocab
#2) Where do I get to choose my pretrained transformer? Is it in the buid-wiki-pretraining? This seemed to only be about the tokenizer.
#3) How do I get the danish tokenizer (e.g. https://huggingface.co/Maltehb/danish-bert-botxo)? This failed as I had too old transformers version, but with the newest transformers version, the luke code failed.
