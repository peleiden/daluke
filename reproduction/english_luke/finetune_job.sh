#!/bin/sh
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1"
#BSUB -J "LUKE-bench"
#BSUB -R "rusage[mem=20GB]"
#BSUB -n 1
#BSUB -W 24:00
#BSUB -o $HOME/joblogs/%J.out
#BSUB -e $HOME/joblogs/%J.err
#BSUB -u s183911@student.dtu.dk
#BSUB -N

echo "Running job"

#python -m examples.cli \
#    --model-file=data/luke_large_500k.tar.gz \
#    --output-dir=data/lukout \
#    ner run \
#    --data-dir=data/CoNLL2003 \
#    --checkpoint-file=data/pytorch_model.bin \
#    --no-train

SIZE=base
for i in {1..5}
do
    OUTDIR="data/lukout$SIZE$i"
    echo $OUTDIR
    mkdir -p $OUTDIR

    LR=$([ "$SIZE" == "large" ] && echo "1e-5" || echo "5e-5")
    python -m examples.cli \
        --model-file=data/luke_${SIZE}_500k.tar.gz \
        --output-dir=$OUTDIR \
        ner run \
        --data-dir=data/CoNLL2003 \
        --train-batch-size=8 \
        --gradient-accumulation-steps=2 \
        --learning-rate=$LR \
        --num-train-epochs=5 \
        --fp16
done
