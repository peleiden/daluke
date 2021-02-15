# Steps taken to reproduce LUKE NER fine-tuning

1. [LUKE repository](https://github.com/studio-ousia/luke) was cloned (at 6feefe)
2. Nvidia mixed precision module was installed by running

    ```
    git clone https://github.com/NVIDIA/apex.git; cd apex
    git checkout c3fad1ad120b23055f6630da0b029c8b626db78f
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
    ```
3. Dependency installation was performed by running `poetry install` from repo folder
4. CoNLL2003 dataset was downloaded to `data`. It is available f.ex. from [here](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003)
5. The released pre-trained LUKE was downloaded from [link from LUKE readme](https://drive.google.com/file/d/1S7smSBELcZWV7-slfrb94BKcSCCoxGfL/view?usp=sharing) to `data`
6. Finetuning was performed using commands in `finetune_job.sh` from this folder

