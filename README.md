# daLUKE
![](daluke-mascot.png)
[![pytest](https://github.com/peleiden/daLUKE/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/peleiden/daLUKE/actions/workflows/pytest.yml)

Implementation of [LUKE](https://github.com/studio-ousia/luke) on Danish text corpus with a focus on named entity recognition. 

## Installation
```
pip install daluke
```
Including optional requirements that are necessary for interacting with datasets and performing analysis:
```
pip install daluke[full]
```

## Usage
### Forward pass simple NER or MLM examples
```
daluke ner --text "Thomas Delaney fører Danmark til sejr ved EM i fodbold."
daluke masked --text "Slutresultatet af kampen mellem Danmark og Rusland bliver [MASK]-[MASK]." 
```
