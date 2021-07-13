# DaLUKE: The Entity-aware, Danish Language Model

<img src="https://raw.githubusercontent.com/peleiden/daluke/master/daluke-mascot.png" align="right"/>

[![pytest](https://github.com/peleiden/daLUKE/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/peleiden/daLUKE/actions/workflows/pytest.yml)

Implementation of the knowledge-enhanced transformer [LUKE](https://github.com/studio-ousia/luke) pretrained on the Danish Wikipedia and evaluated on named entity recognition.

## Installation

```
pip install daluke
```
For including optional requirements that are necessary for training and general analysis:
```
pip install daluke[full]
```
Python 3.8 or newer is required.

## Explanation
For an explanation of the model, see our [bachelor's thesis](https://peleiden.github.io/bug-free-guacamole/main.pdf) or the original [LUKE paper](https://www.aclweb.org/anthology/2020.emnlp-main.523/).

## Usage
### Inference on simple NER or MLM examples
```bash
daluke ner --text "Thomas Delaney fører Danmark til sejr ved EM i fodbold."
daluke masked --text "Slutresultatet af kampen mellem Danmark og Rusland bliver [MASK]-[MASK]."
```
For Windows, or systems where `#!/usr/bin/env python3` is not linked to the correct Python interpreter, the command `python -m daluke.api.cli` can be used instead of `daluke`.

### Training DaLUKE yourself

This part shows how to recreate the entire DaLUKE training pipeline from dataset preparation to fine-tuning.
This guide is designed to be run in a bash shell.
If you use Windows, you will probably have to make some modifications to the shell scripts used.

```bash
# Download forked luke submodule
git submodule update --init --recursive
# Install requirements
pip install -r requirements.txt
pip install -r optional-requirements.txt
pip install -r luke/requirements.txt

# Build dataset
# The script performs all the steps of building the dataset, including downloading the Danish Wikipedia
# You only need to modify DATA_PATH to where you want the data to be saved
# Be aware that this takes several hours
dev/build_data.sh

# Start pretraining using default hyperparameters
python daluke/pretrain/run.py <INSERT DATA_PATH HERE> -c configs/pretrain-main.ini --name $NAME --save-every 5 --epochs 150 --name daluke --fp16
# Optional: Make plots of pretraining
python daluke/plot/plot_pretraining.py <DATA_PATH>/daluke

# Fine-tune on DaNE
python daluke/collect_modelfile.py <DATA_PATH>/daluke <DATA_PATH>/ner/daluke.tar.gz
python daluke/ner/run.py <DATA_PATH>/ner/daluke -c configs/main-finetune.ini --model <DATA_PATH>/ner/daluke.tar.gz --name finetune --eval
# Evaluate on DaNE test set
python daluke/ner/run_eval.py <DATA_PATH>/ner/daluke/finetune --model <DATA_PATH>/ner/daluke/finetune/daluke_ner_best.tar.gz
# Optional: Fine-tuning plots
python daluke/plot/plot_finetune_ner.py <DATA_PATH>/ner/daluke/finetune/train-results
```
