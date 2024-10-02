python main_wdpruning.py \
    --data-set TINY \
    --batch-size 128 \
    --data-path /data/imagenet_family/tiny_imagenet/tiny-imagenet-200 \
    --output_dir logs/tiny_01 \
    --epochs 500 \
    --classifiers 10 \
    --R_threshold 0.1 \