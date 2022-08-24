CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    /path/to/model_config \
    /path/to/model_checkpoint \
     --eval 'mAP' \