python masked_parameter_count.py \
    --arch deit_small \
    --data-set TINY \
    --data-path /data/imagenet_family/tiny_imagenet/tiny-imagenet-200 \
    --pretrained_dir /home/ueno/pruning/width-and-Depth-pruning-for-Vision-Transformer/checkpoint/deit_small_patch16_224-cd65a155.pth \
    --eval_batch_size 1024 \
    --classifiers 10 \
    --classifier_choose 10 \
    --eval