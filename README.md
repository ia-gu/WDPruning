# WDPruning
Reproduction of [WDPruning](https://cdn.aaai.org/ojs/20222/20222-13-24235-1-2-20220628.pdf).


## Installation
```
conda create -n wdp python=3.10 -y
conda activate wdp
pip install -r requirements.txt
```

## Demo

### Load DeiT Models
```
bash download_pretrain.sh
```

### Training on CIFAR10
To train CIFAR10, run:
```
bash main.sh
```

### Training on ImageNet

To train DeiT models on ImageNet, run:

DeiT
```
 python main_wdpruning.py --arch deit_small --data-set IMNET --batch-size 128 --data-path ../data/ILSVRC2012/ --output_dir logs --classifier 10 --R_threshold 0.8
```


### Pruning and Evaluation
Test the amout of parameters, GPU throughput of pruned transformer. 
```
python masked_parameter_count.py --arch deit_small --pretrained_dir logs/checkpoint.pth --eval_batch_size 1024 --classifiers 10 --classifier_choose 10
```
Note that '--classifier_choose' means choose which classifier to prune. '--classifier_choose 12' means choose the last classifier. 

\
\
Test the amout of parameters, CPU latency of pruned transformer.
```
python masked_parameter_count.py --arch deit_small --pretrained_dir logs/checkpoint.pth --no_cuda --eval_batch_size 1  --classifiers 10
```

## Acknowledgement

### Usage and License Notices: 
Please follow the license agreement of [Width & Depth Pruning for Vision Transformers](https://github.com/andyrull/width-and-Depth-pruning-for-Vision-Transformer).