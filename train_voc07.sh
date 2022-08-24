CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch  \
	--nproc_per_node=2  \
	--master_port=18500 \
    ./tools/train.py \
    ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc_oamil.py \
    --work-dir='./outputs/VOC07/fasterrcnn_VOC07_noise0.4_OAMIL' \
    --launcher pytorch