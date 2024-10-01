python main_wdpruning.py \
    --data-set TINY \
    --batch-size 128 \
    --data-path /data/imagenet_family/tiny_imagenet/tiny-imagenet-200 \
    --output_dir logs/tiny08 \
    --epochs 500 \
    --classifiers 10 \
    --R_threshold 0.8 \