#!/bin/sh
#BSUB -q gpua100
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=1"
#BSUB -J "LUKE-fine"
#BSUB -R "rusage[mem=20GB]"
#BSUB -n 1
#BSUB -W 24:00
#BSUB -u s183911@student.dtu.dk
#BSUB -N
#BSUB -oo ~/joblogs/stdout_%J
#BSUB -eo ~/joblogs/stderr_%J

DATA_PATH=/work3/$USER/ner-finetune

echo "Finetune"
daluke/ner/run.py $DATA_PATH\
    -m $DATA_PATH/../daluke.tar.gz\
    -c scripts/finetune.ini

RESPATH=$DATA_PATH/RUN # This name is set in the config
python daluke/plot/plot_finetune_ner.py $RESPATH/train-results

python daluke/ner/run_eval.py $RESPATH\
    -m $RESPATH/daluke_ner.tar.gz

echo "Finished job"
