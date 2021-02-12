#!/bin/sh
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J "LUKE reproduction"
#BSUB -R "rusage[mem=20GB]"
#BSUB -n 1
#BSUB -W 24:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u s183911@student.dtu.dk
#BSUB -N

echo "Running job"
mkdir -p ../lukout
python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=../lukout \
    ner run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
