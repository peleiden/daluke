#!/usr/bin/sh
MODELPATH=/work3/s183912/pdata2/johnny-charlie
OUTPATH=/work3/s183911/progress

for i in -1 9 24
do
    P=$MODELPATH/daluke_epoch$i
    echo $P
    python -m daluke.collect_modelfile "${P}.pt" "${OUTPATH}${i}.tar.gz"
done
