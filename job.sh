#!/bin/sh
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1"
#BSUB -J "LUKE-finetune"
#BSUB -R "rusage[mem=20GB]"
#BSUB -n 1
#BSUB -W 12:00
#BSUB -o $HOME/joblogs/%J.out
#BSUB -e $HOME/joblogs/%J.err
#BSUB -u s183911@student.dtu.dk
#BSUB -N

echo "Running job"

python3 daluke/run_ner.py local_data/daluketest -m local_data/daluke.tar.gz

echo "Completed job"
