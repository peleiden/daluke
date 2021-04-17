DATA_PATH=/work3/$USER/pdata

echo "PRETRAIN"
daluke/pretrain/run.py\
    $DATA_PATH\
    -c local_data/training.ini
