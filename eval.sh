python masked_parameter_count.py \
    --arch deit_small \
    --data-set TINY \
    --data-path /data/imagenet_family/tiny_imagenet/tiny-imagenet-200 \
    --pretrained_dir logs/tiny_05/checkpoint.pth \
    --eval_batch_size 1024 \
    --classifiers 10 \
    --classifier_choose 10 \
    --eval