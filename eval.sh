python masked_parameter_count.py \
    --arch deit_small \
    --data-set CIFAR100 \
    --pretrained_dir logs/cifar100/checkpoint.pth \
    --eval_batch_size 1024 \
    --classifiers 10 \
    --classifier_choose 10 \
    --eval