# DaLUKE: The Entity-aware, Danish Language Model
<img src="https://raw.githubusercontent.com/peleiden/daluke/master/daluke-mascot.png" align="right"/>

[![pytest](https://github.com/peleiden/daLUKE/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/peleiden/daLUKE/actions/workflows/pytest.yml)

Implementation of the knowledge-enhanced transformer [LUKE](https://github.com/studio-ousia/luke) pretrained on the Danish Wikipedia and evaluated on named entity recognition.

## Installation

```
pip install daluke
```
For 
Including optional requirements that are necessary for interacting with datasets and performing analysis:
```
pip install daluke[full]
```
Python 3.8 or newer is required.

## Usage
### Inference on simple NER or MLM examples
```
daluke ner --text "Thomas Delaney fører Danmark til sejr ved EM i fodbold."
daluke masked --text "Slutresultatet af kampen mellem Danmark og Rusland bliver [MASK]-[MASK]." 
```
For Windows, or systems where `#!/usr/bin/env python3` is not linked to the correct Python interpreter, the command `python -m daluke.api.cli` can be used instead of `daluke`.

## Explanation
For explanation of the model, see our [bachelor's thesis](https://peleiden.github.io/bug-free-guacamole/main.pdf) or the original [LUKE paper](https://www.aclweb.org/anthology/2020.emnlp-main.523/).
