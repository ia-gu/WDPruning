import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import random

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.vision_transformer import VisionTransformer
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate_classifiers
from losses import LossWithClassifierAndPruning
from samplers import RASampler

import utils
from vit_wdpruning import VisionTransformerWithWDPruning,_cfg, checkpoint_filter_fn
import math
import torch.nn as nn

def get_args_parser():
    parser = argparse.ArgumentParser('Training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model-path', type=str, default='checkpoint/deit_small_patch16_224-cd65a155.pth')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996)
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=0.05)
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR')
    parser.add_argument('--lr_pruner', type=float, default=0.0025, metavar='LR', help='learning rate of pruner')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor ')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original".'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothin')
    parser.add_argument('--train-interpolation', type=str, default='bicubic', help='random, bilinear, bicubic')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set')
    parser.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL', help='Name of teacher model to train')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distill', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str)
    parser.add_argument('--distillation-alpha', default=0.5, type=float)
    parser.add_argument('--distillation-tau', default=1.0, type=float)
    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str)
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR100','CIFAR10', 'IMNET', 'TINY', 'INAT', 'INAT19'], type=str)
    parser.add_argument('--inat-category', default='name', choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'], type=str, help='semantic granularity')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--R_threshold", default=0.5, type=float, help="The predefined pruning ratio (for scheduling).")
    parser.add_argument("--classifiers", type=int, nargs='+', default=[], help="The classifiers.")
    return parser

def save_model(output_dir, model):
    checkpoint_paths = [output_dir / 'checkpoint-best.pth']
    for checkpoint_path in checkpoint_paths:
        utils.save_on_master({
            'model': model.state_dict(),
        }, checkpoint_path)
    print("Saved model checkpoint to [DIR: %s]" % args.output_dir)
def get_param_groups(model, weight_decay,lambda_1,lambda_2):
    decay = []
    no_decay = []
    pruner = []
    nn_lambda = []
    for name, param in model.named_parameters():
        if 'mask_score' in name or 'threshold' in name:
            pruner.append(param)
        elif not param.requires_grad:
            continue  # frozen weights
        elif 'cls_token' in name or 'pos_embed' in name:
            continue  # frozen weights
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    nn_lambda.append(lambda_1)
    nn_lambda.append(lambda_2)

    return [
        {'params': no_decay, 'weight_decay': 0., 'name': 'base_no_decay'},
        {'params': decay, 'weight_decay': weight_decay, 'name': 'base_decay'},
        {'params': pruner, 'weight_decay': 0., 'name': 'pruner'},
        {'params': nn_lambda, 'weight_decay': 0., 'name': 'lambda'}
        ]
def adjust_learning_rate(param_groups, init_lr, min_lr, step, max_step, warming_up_step=2, warmup_pruner=False, init_lr_p=0.01):
    cos_lr = (math.cos(step / max_step * math.pi) + 1) * 0.5
    pruner_lr = min_lr + cos_lr * (init_lr_p - min_lr)
    lambda_lr = min_lr + cos_lr * (10.0 - min_lr)
    cos_lr = min_lr + cos_lr * (init_lr - min_lr)

    if step < warming_up_step:
        backbone_lr = 0
    else:
        backbone_lr = max(min_lr, cos_lr)
    print('## Using lr  %.7f for BACKBONE, cosine lr = %.7f for PRUNER' % (backbone_lr, pruner_lr))
    for param_group in param_groups:
        if param_group['name'] == 'pruner':
            param_group['lr'] = pruner_lr
        elif param_group['name'] == 'lambda':
            param_group['lr'] = -lambda_lr
        else:
            param_group['lr'] = backbone_lr

def main(args):

    print(args)
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "train_log.txt").open("a") as f:
            f.write(str(args) + "\n")


    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if 'deit_tiny' in args.model_path:
        embed_dim = 192
        num_heads = 3
    else:
        embed_dim = 384
        num_heads = 6
    model_path = args.model_path
    model = VisionTransformerWithWDPruning(num_classes=args.nb_classes,
        patch_size=16, embed_dim=embed_dim, depth=12, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, distilled=args.distill,
        head_pruning=True, fc_pruning=True,classifiers=args.classifiers
    )

    if args.data_set == 'IMNET':
        checkpoint = torch.load(model_path, map_location="cpu")
        ckpt = checkpoint_filter_fn(checkpoint, model)
        model.default_cfg = _cfg()
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('# missing keys=', missing_keys)
        print('# unexpected keys=', unexpected_keys)
        print('sucessfully loaded from pre-trained weights:', model_path)

    elif 'CIFAR' in args.data_set:
        param_dict = torch.load(model_path, map_location="cpu")['model']
        for i in param_dict:
            if 'threshold' not in i and 'mask_scores' not in i and 'head' not in i:
                try:
                    model.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
                except:
                    # import pdb; pdb.set_trace()
                    model.state_dict()[i.replace('module.', '')][0][:-1].copy_(param_dict[i][0])

    teacher_model = None
    if args.distill and 'IMNET' in args.data_set:
        teacher_model = VisionTransformer(num_classes=args.nb_classes,
                                             patch_size=16, embed_dim=embed_dim, depth=12, num_heads=num_heads, mlp_ratio=4,
                                             qkv_bias=True)
        checkpoint = torch.load(model_path, map_location="cpu")
        ckpt = checkpoint_filter_fn(checkpoint, teacher_model)
        missing_keys, unexpected_keys = teacher_model.load_state_dict(ckpt, strict=False)
        print('# missing keys=', missing_keys)
        print('# unexpected keys=', unexpected_keys)
        print('sucessfully loaded from pre-trained weights:', model_path)
        teacher_model.to(device)
        teacher_model.eval()

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    # optimizer = create_optimizer(args, model_without_ddp)
    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    lambda_1 = nn.Parameter(torch.tensor(0.0).cuda())
    lambda_2 = nn.Parameter(torch.tensor(0.0).cuda())
    parameter_group = get_param_groups(model_without_ddp, args.weight_decay,lambda_1,lambda_2)

    optimizer = torch.optim.AdamW(parameter_group, **opt_args)

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = LossWithClassifierAndPruning(
        criterion, model, R_threshold=args.R_threshold
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        warmup_step = args.warmup_epochs
        adjust_learning_rate(optimizer.param_groups, args.lr, args.min_lr, epoch, args.epochs,
                             warmup_pruner=False, warming_up_step=warmup_step, init_lr_p=args.lr_pruner)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer,lambda_1,lambda_2,
            device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate_classifiers(data_loader_val, model, device,classifiers = args.classifiers)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if max_accuracy < test_stats['acc1'] and epoch >= args.warmup_epochs:
            save_model(output_dir, model_without_ddp)
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "train_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("[DIR: %s]" % args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WDPruning training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
