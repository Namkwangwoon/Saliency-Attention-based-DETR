# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
# import json
import random
from pathlib import Path

import numpy as np
import torch
# from torch.utils.data import DataLoader, DistributedSampler

# import datasets
import util.misc as utils
# from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
from models.matcher import build_matcher


from datasets.SOCdataloader import get_loader
import datasets.metrics as M
import os
from tqdm import tqdm
import cv2
from util.misc import interpolate
import gc
from imantics import Polygons
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
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

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

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

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='../datasets/coco')
    parser.add_argument('--coco_panoptic_path', type=str, default='../datasets/coco_panoptic')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser



CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        # 7
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'computer monitor',
        # 13
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        # 20
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
        # 28
        'umbrella', 'food', 'lamp', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        # 36
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        # 41
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'quilt', 'wine glass',
        # 47
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        # 55
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        # 62
        'chair', 'couch', 'potted plant', 'bed', 'pillow', 'dining table', 'camera',
        # 69
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        # 77
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        # 84
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        # 90
        'toothbrush'
        ]


def anno_transform(anno_path):
    
    label, box, mask = [], [], []
    
    with open(anno_path) as f:
        json_data = json.load(f)
        
    w, h = json_data['width'], json_data['height']
    name = json_data['file_name']
    
    for o in json_data['objects']:
        if o['class_name'] in CLASSES:
            label.append(CLASSES.index(o['class_name']))
        else:
            label.append(0)
        box.append([o['bbox'][0]/w, o['bbox'][1]/h, o['bbox'][2]/w, o['bbox'][3]/h])
        mask.append(Polygons(o['polygons']).mask(w, h).array)
    
    return torch.tensor(label, dtype=torch.long), torch.tensor(box, dtype=torch.float), torch.tensor(mask)



def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    matcher = build_matcher(args).to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    model_path = 'OUTPUT/DETR_SOD'
    model_dir = '1e-5(aug,0.02)'
    model_list = sorted(os.listdir(os.path.join(model_path, model_dir)))
    model_name = 'checkpoint0166.pth'
    
    # writer = SummaryWriter()
        
    print("\nStart Evaluate")
    
    # for i, model_name in enumerate(model_list):
    
    
    # checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    checkpoint = torch.load(os.path.join(model_path, model_dir, model_name), map_location='cpu')
    
    ### classification layer의 weight를 load하지 않음
    # for k in ['class_embed.weight', 'class_embed.bias']:
    #         #   'bbox_embed.layers.0.weight', 'bbox_embed.layers.0.bias',
    #         #   'bbox_embed.layers.1.weight', 'bbox_embed.layers.1.bias',
    #         #   'bbox_embed.layers.2.weight', 'bbox_embed.layers.2.bias']:
    #     checkpoint['model'].move_to_end(k)
    #     checkpoint['model'].popitem()
    # model_without_ddp.detr.load_state_dict(checkpoint['model'], strict=False)
    ###

    model_without_ddp.load_state_dict(checkpoint['model'])
    
    
    
    
    model.eval()
    criterion.eval()
    
    ### SOC eval ###
    
    
    
    # SM = M.Smeasure()
    # EM = M.Emeasure()
    # MAE = M.MAE()
    
    
    
    
    img_root = '../datasets/SOC/ValSet/Imgs'
    # img_root = '../datasets/SOC/ValSet/Imgs_annoted'
    gt_root = '../datasets/SOC/ValSet/gt'
    anno_gt_dir = '../datasets/SOC/ValSet/annos_list'
    img_list = sorted(os.listdir(img_root))
    
    
    mf = open(os.path.join('Prediction/ValSet', model_dir, model_name, 'mae.txt'), 'w')
    sf = open(os.path.join('Prediction/ValSet', model_dir, model_name, 'mae.txt'), 'w')
        
    
    
    for img_name in tqdm(img_list, total=len(img_list)):
        
        SM = M.Smeasure()
        EM = M.Emeasure()
        MAE = M.MAE()
        
        
        img_path = os.path.join(img_root, img_name)
        gt_path = os.path.join(gt_root, img_name[:-3]+'png')
        img = cv2.imread(img_path)
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(-1, 0, 1).type(torch.cuda.FloatTensor).cuda()/255.0
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        
        
        
        anno = os.path.join(anno_gt_dir, img_name[:-3]+'json')
        label, box, mask = anno_transform(anno)
        
        target = [{'mask':mask, 'labels':label, 'boxes':box}]
        target = [{k:v.to(device) for k, v in t.items()} for t in target]
        
        
        
        
        outputs = model([img])
        outputs_masks = interpolate(outputs['pred_masks'], size=img.shape[-2:], mode="bilinear", align_corners=False).sigmoid()
        # outputs_masks = outputs_masks.sigmoid().cpu().detach().numpy()
        # saliency = outputs['saliency_value'][0].squeeze(-1)
        # ind = torch.arange(0, 100)[saliency>0.5].cuda()
        # print("target :", target)
        # print()
        # print("outputs :", outputs)
        
        # ind = matcher(outputs, target)[0][0]
        
        saliency = outputs['saliency_value'][0].squeeze(-1)
        ind = torch.arange(0, 100)[saliency>0.5].cuda()
        
        # print("ind :", ind)
        
        if len(ind)==0:
            pred = torch.zeros_like(outputs_masks[0][0]).cuda()
            
        else:
            pred = outputs_masks[0, ind]
            if pred.ndim == 3:
                pred = pred.max(0).values
                
                
        pred = pred * 255.0
        
        pred = pred.cpu().detach().numpy()
        
        
        
        # plt.figure(figsize=(10, 30))
        
        # plt.subplot(2, 1, 1)
        # plt.axis('off')
        # plt.imshow(pred)
        
        
        # plt.subplot(2, 1, 2)
        # plt.axis('off')
        # plt.imshow(gt)
        
        # plt.show()
        
        
        
        # for i, t in zip(ind[0][0], ind[0][1]):
        #     plt.subplot(2, 1, 1)
        #     plt.axis('off')
        #     plt.imshow(mask[t])
            
            
        #     plt.subplot(2, 1, 2)
        #     plt.axis('off')
        #     plt.imshow((outputs_masks[0][i]>0.5).cpu().detach().numpy())
            
        #     plt.show()
        
        

        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)
        
        
        sm = SM.get_results()['sm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']
        
        
        

    # sm = SM.get_results()['sm']
    # em = EM.get_results()['em']
    # mae = MAE.get_results()['mae']
    
    # print()
    # print(model_name)
    # print()
    # print(
    #     'MAE : ', mae.round(3), '\n',
    #     'Smeasure : ', sm.round(3), '\n',
    #     'meanEm : ', '-' if em['curve'] is None else em['curve'].mean().round(3), '\n',
    #     'maxEm : ', '-' if em['curve'] is None else em['curve'].max().round(3), '\n',
    #     sep=''
    # )
    # print()
    
    # del SM
    # del EM
    # del MAE
    
    
    
    
    gc.collect()


        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)