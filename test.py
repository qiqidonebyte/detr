import argparse
import datetime
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser(description="Test on a datasets.", add_help=False)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--fpn', default=True, type=bool, help='Use FPN in backbone')
    parser.add_argument('--local_attn', default=False, type=bool, help='Use LocalSelfAttention')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    # * Segmentation
    parser.add_argument('--masks', default=False, action='store_true',
                        help="Train segmentation head if the flag is provided")
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', type=str, default='', help='load from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def create_test_data_loader(args):
    # 构建数据集
    dataset_test = build_dataset(image_set='test', args=args)
    # dataset_test = datasets.coco.CocoDetection(root=args.coco_path,
    #                                            annFile='annotations/instances_val2017.json',
    #                                            transform=datasets.coco.CocoDetection.default_transforms())
    sampler_val = torch.utils.data.SequentialSampler(dataset_test)
    base_ds = get_coco_api_from_dataset(dataset_test)
    # 创建数据加载器
    data_loader = DataLoader(
        dataset_test,
        args.batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers
    )
    return data_loader, base_ds


def test_on_dataset():
    args = get_args_parser().parse_args()
    # 打印所有参数及其值
    for arg_name in vars(args):  # 使用 vars 来获取 args 的 __dict__ 属性
        arg_value = getattr(args, arg_name)
        # 格式化输出，只打印命令行参数，跳过 argparse 自动生成的属性
        if not arg_name.startswith('_'):
            print(f"--{arg_name}: {arg_value}")

    # setup logger
    data_loader_test, base_ds = create_test_data_loader(args)
    # 构建模型
    model, criterion, postprocessors = build_model(args)
    # 加载权重
    device = torch.device('cuda:0' if (torch.cuda.is_available() and args.device == 'cuda') else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    print("Start Test")
    start_time = time.time()
    # 评估模型
    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                          data_loader_test, base_ds, device, args.output_dir)

    # 保存结果
    print("Save to Json")
    save_to_json(coco_evaluator, args)

    # 总共时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Test time {}'.format(total_time_str))


def save_to_json(coco_evaluator, args):
    # 保存结果的路径
    output_path = os.path.join(args.output_dir, "detections.json")
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        # 尝试将预测结果序列化为 JSON 并保存到文件
        with open(output_path, "w") as f:
            # 这里假设 coco_evaluator.coco_eval['bbox'].cocoDt 包含了所有预测结果
            # 您可能需要根据实际的 CocoEvaluator 类实现来调整
            all_predictions = {img_id: [
                {
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'score': ann['score']
                }
                for ann in anns.values()
            ] for img_id, anns in coco_evaluator.coco_eval['bbox'].cocoDt.anns.items()}
            json.dump(all_predictions, f, indent=4)

        print(f"Detection results saved to {output_path}")

    except Exception as e:
        # 如果保存过程中出现错误，打印 coco_evaluator 中的预测结果信息
        print("Error occurred while saving predictions:", e)
        print("Printing prediction results from coco_evaluator:")

        # 打印所有预测结果
        for img_id, anns in coco_evaluator.coco_eval['bbox'].cocoDt.anns.items():
            print(f"Image ID: {img_id}")
            for ann in anns.values():
                print(ann)
            print("-" * 40)  # 打印分隔线以便区分不同的图像

        # 您也可以选择将预测结果输出到一个临时文件或打印到控制台
        # 这里只是一个示例，具体实现可能需要根据您的需求调整


if __name__ == '__main__':
    test_on_dataset()
