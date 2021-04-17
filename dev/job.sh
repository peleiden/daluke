#!/bin/sh
#BSUB -q gpua100
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=2"
#BSUB -J "LUKE-pre"
#BSUB -R "rusage[mem=60GB]"
#BSUB -n 2
#BSUB -W 24:00
#BSUB -u s183911@student.dtu.dk
#BSUB -N

DATA_PATH=/work3/$USER/pdata

echo "PRETRAIN"
daluke/pretrain/run.py\
    $DATA_PATH\
    -c local_data/training.ini

echo "Finished job"
