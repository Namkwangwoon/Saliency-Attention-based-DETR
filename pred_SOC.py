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
import time


from PIL import Image
import torchvision.transforms as transforms
from util.misc import interpolate
import matplotlib.pyplot as plt
import os
import cv2
from models.matcher import build_matcher
import json
from imantics import Polygons, Mask, Annotation
import torch.nn.functional as F
from tqdm import tqdm


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




def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
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
    
    matcher = build_matcher(args)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    
    trans = transforms.ToTensor()
    img_dir = '../datasets/SOC/ValSet/Imgs'
    anno_img_dir = '../datasets/SOC/ValSet/Imgs_annoted'
    # anno_img_dir = '../datasets/SOC/TrainSet/Imgs_annoted'
    anno_gt_dir = '../datasets/SOC/ValSet/annos_list'
    # anno_gt_dir = '../datasets/SOC/TrainSet/annos_list'
    img_list = sorted(os.listdir(img_dir))
    anno_img_list = sorted(os.listdir(anno_img_dir))
    anno_gt_list = sorted(os.listdir(anno_gt_dir))
    
    dict_name = "1e-5(baseline)"
    
    # if not os.path.exists(os.path.join('Prediction/features', dict_name)):
    #     os.makedirs(os.path.join('Prediction/features', dict_name))
        
    # for n in [40, 70, 99]:
    
    # model.load_state_dict(torch.load(os.path.join("OUTPUT/DETR_SOD", "checkpoint0090.pth"), map_location='cpu')['model'])
    # print()
    # print("checkpoint00%d.pth" % n)
    # model.load_state_dict(torch.load(os.path.join("OUTPUT/DETR_SOD", dict_name, "checkpoint00%d.pth" % n), map_location='cpu')['model'])
    # model.load_state_dict(torch.load(os.path.join("OUTPUT/DETR_SOD", dict_name, "checkpoint0026.pth"), map_location='cpu')['model'])
    model.load_state_dict(torch.load(os.path.join("OUTPUT/DETR_SOD", dict_name, "checkpoint0055.pth"), map_location='cpu')['model'])
    # model.load_state_dict(torch.load(os.path.join("OUTPUT/DETR_SOD", "checkpoint.pth"), map_location='cpu')['model'])
    model.eval()
    model.to(device)
    criterion.eval()
            
    idx = 0
    font_size = 15
    
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for im in tqdm(img_list, total=len(img_list)):
    # for im, gt in zip(anno_img_list[idx:], anno_gt_list[idx:]):
    # for im, gt in zip(anno_img_list, anno_gt_list):
        
        im_num = im.split("_")[-1][:-4]
        img = trans(Image.open(os.path.join(img_dir, im)))
        
        # anno = os.path.join(anno_gt_dir, gt)
        # label, box, mask = anno_transform(anno)
        
        # target = [{'masks':mask, 'labels':label, 'boxes':box}]
        # target = [{k: v.to(device) for k, v in t.items()} for t in target]
        
        
        outputs = model([img.to(device)])
        
        image = img.permute(1, 2, 0).cpu().detach().numpy()
        attention_maps = outputs['attention_maps']
        attention_maps = F.interpolate(attention_maps[0].sum(1).unsqueeze(1), size=img.shape[-2:], mode="bilinear")
        attention_maps = attention_maps.cpu().detach().numpy()
        outputs_masks = F.interpolate(outputs['pred_masks'], size=img.shape[-2:], mode="bilinear", align_corners=False)
        outputs_masks = outputs_masks.sigmoid().cpu().detach().numpy()
        
        scores = outputs['pred_logits'].softmax(-1)[0,:,:-1].max(-1).values
        labels = outputs['pred_logits'].softmax(-1)[0,:,:-1].argmax(-1)
        
        # indices = matcher(outputs, target)
        
                
        
        '''Attention maps + Mask'''
        
        # final_mask = torch.zeros(img.shape[-2:]).unsqueeze(0).unsqueeze(0)
        
        # for i, t in zip(indices[0][0], indices[0][1]):
        # # for i, q in enumerate(attention_maps):
        # #     if scores[i]>0.95:
        #     plt.figure(figsize=(160, 15))
        #     for j, h in enumerate(attention_maps[i]):
        #         if h.max() == 0:
        #             heatmap = np.zeros_like(image)
        #         else:
        #             heatmap = cv2.applyColorMap(np.uint8(h / h.max() * 255), cv2.COLORMAP_JET)
        #             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255
                
                
        #         plt.subplots_adjust(hspace=0.0, wspace=0.1)
                
        #         plt.subplot(1, 11, j+1)
        #         plt.axis('off')
        #         # plt.imshow(heatmap)
        #         plt.imshow(image*0.5 + heatmap*0.5)
                
            
        #     plt.subplot(1, 11, 9)    
        #     plt.axis('off')
        #     plt.imshow(outputs_masks[0][i])
            
        #     plt.title("%d-th query | %s(%f)" % (i, CLASSES[labels[i]], scores[i]), fontsize=font_size)
            
        #     plt.subplot(1, 11, 10)    
        #     plt.axis('off')
        #     plt.imshow(outputs_masks[0][i] > 0.5, cmap='gray')
        #     final_mask += interpolate(torch.from_numpy(outputs_masks[:, i]).unsqueeze(0), size=img.shape[-2:], mode='bilinear', align_corners=False)
            
            
        #     plt.subplot(1, 11, 11)
        #     plt.axis('off')
        #     plt.imshow(mask[t], cmap='gray')
        
        #     plt.title("%s" % (CLASSES[label[t]]), fontsize=font_size)
            
        #     plt.show()
            # plt.savefig(os.path.join("Prediction/features", dict_name, "{}_{}".format(int(im_num), i)), bbox_inches='tight', pad_inches=0)
            
        
        # plt.imshow(final_mask[0][0]>0.5, cmap='gray')
        # plt.show()
        
        # # break
            
            
            
        '''Value Added'''
        
        # if not os.path.exists(os.path.join('Prediction/features', dict_name)):
        #     os.mkdir(os.path.join('Prediction/features', dict_name))
        
        # for i, t in zip(indices[0][0], indices[0][1]):
            
        #     plt.figure(figsize=(30, 11))
            
        #     h = attention_maps[i][0]
            
        #     heatmap = cv2.applyColorMap(np.uint8(h / h.max() * 255), cv2.COLORMAP_JET)
        #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255
            
        #     # print(h.min(), "~", h.max())
        #     # print(heatmap.min(), "~", heatmap.max())
        #     plt.subplot(1, 4, 1)
        #     plt.axis('off')
        #     plt.imshow(image*0.5 + heatmap*0.5)
        #     # plt.imshow(h)
            
        #     plt.subplot(1, 4, 2)    
        #     plt.axis('off')
        #     plt.imshow(outputs_masks[0][i])
            
        #     # plt.title("%d-th query | %s(%f)" % (i, CLASSES[labels[i]], scores[i]), fontsize=font_size)
            
        #     plt.subplot(1, 4, 3)    
        #     plt.axis('off')
        #     plt.imshow(outputs_masks[0][i] > 0.5, cmap='gray')
            
            
        #     plt.subplot(1, 4, 4)
        #     plt.axis('off')
        #     plt.imshow(mask[t], cmap='gray')
        
        #     # plt.title("%s" % (CLASSES[label[t]]), fontsize=font_size)
            
        #     # plt.show()
        #     plt.savefig(os.path.join("Prediction/features", dict_name, "{}_{}".format(int(im_num), i)), bbox_inches='tight', pad_inches=0)
            
        
        # # break
        
        
        
        '''Saliency Classification'''
        
        
    #     saliency = outputs['saliency_value']
    #     sal = saliency[0].squeeze(-1)
        
    #     t = indices[0][0].tolist()
    #     p = torch.arange(0, 100)[sal>0.5].tolist()
        
    #     fn = set(t)-set(p)
    #     fp = set(p)-set(t)
    #     tp = set(p)&set(t)
        
    #     TP += len(tp)
    #     FN += len(fn)
    #     FP += len(fp)
    #     TN += 100 - len(tp) - len(fn) - len(fp)
        
    # print("TP", TP, "TN", TN, "FP", FP, "FN", FN)
    # print(TP+TN+FP+FN, "= 60000 ??")
    # print()



        ### Final mask aggregation '''
    
    
        saliency = outputs['saliency_value']
        sal = saliency[0].squeeze(-1)
        ind = torch.arange(0, 100)[sal>0.5]
        
        if len(ind)==0:
            final_mask = np.zeros_like(outputs_masks[0][0])
        
        else:
            final_mask = outputs_masks[0,ind]
            
            if final_mask.ndim == 3:
                final_mask = final_mask.max(0)
            
        
        # plt.imshow(final_mask, cmap='gray')
        # plt.show()
        
        
        if not os.path.exists(os.path.join('Prediction/ValSet', dict_name)):
            os.mkdir(os.path.join('Prediction/ValSet', dict_name))
        
        # cv2.imwrite(os.path.join('Prediction/ValSet', dict_name, im[:-4]+'.png'), (final_mask>0.5)*255)
        cv2.imwrite(os.path.join('Prediction/ValSet', dict_name, im[:-4]+'.png'), (final_mask)*255)

        if not os.path.exists(os.path.join('Prediction/ValSet', dict_name)):
            os.mkdir(os.path.join('Prediction/ValSet', dict_name))
        
        # cv2.imwrite(os.path.join('Prediction/ValSet', dict_name, im[:-4]+'.png'), (final_mask>0.5)*255)
        cv2.imwrite(os.path.join('Prediction/ValSet', dict_name, im[:-4]+'.png'), (final_mask)*255)
    
    
    
    
    
    
    

    # for m in outputs['pred_masks'][0]:
    #     plt.figure(figsize=(128, 12))
    #     for j in range(10):
    #         plt.subplot(1, 10, j+1)
    #         plt.axis('off')
    #         plt.imshow(m.sigmoid().cpu().detach().numpy(), cmap='gray')
            
    #     plt.show()
        
    # break
    
    
    
        
        # for j, h in enumerate(q):
        #     if h.max()>=0.5:
        #         # heatmap = cv2.applyColorMap(np.uint8(255*attention_maps), cv2.COLORMAP_JET)
        #         heatmap = cv2.applyColorMap(np.uint8(255*h), cv2.COLORMAP_JET)
        #         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255

        #         plt.figure(figsize=(30, 45))
        #         plt.subplots_adjust(hspace=0.0)

        #         plt.subplot(2, 1, 1)
        #         plt.axis('off')
        #         # plt.title("Object query {}, head {}, (max : {})".format(i+1, j+1, h.max()))
        #         plt.imshow(image*0.5 + heatmap*0.5)

        #         # plt.subplot(4, 1, 2)
        #         # plt.axis('off')
        #         # plt.imshow(Image.open(os.path.join(anno_gt_dir, gt)), cmap='gray')

        #         # plt.subplot(4, 1, 3)
        #         # plt.axis('off')
        #         # plt.imshow(((src_masks.sigmoid())[0][0]>0.5).detach().cpu().numpy()*255, cmap='gray')
                
        #         plt.subplot(2, 1, 2)
        #         plt.axis('off')
        #         plt.imshow(outputs_masks[i])

        #         plt.show()
        #         # plt.savefig(os.path.join('Prediction/ValSet', dict_name+'+', im[:-4]+'.png'), bbox_inches='tight', pad_inches=0)
        #         # plt.savefig(os.path.join("Prediction", "query_{}_head_{}".format(i+1, j+1, h.max())), bbox_inches='tight', pad_inches=0)
        #         # plt.savefig(os.path.join("Prediction/masks", dict_name, "feature_{}_{}".format(i, j)), bbox_inches='tight', pad_inches=0)
                
                

        # sumed_map = attention_maps.sum(0).sum(0)
        
        # plt.figure(figsize=(30, 90))
        # plt.subplots_adjust(hspace=0.0)
        
        # plt.subplot(4, 1, 1)
        # plt.axis('off')
        # map1 = sumed_map / sumed_map.max()
        # map1 = cv2.applyColorMap(np.uint8(255*map1), cv2.COLORMAP_JET)
        # map1 = cv2.cvtColor(map1, cv2.COLOR_BGR2RGB) / 255
        # plt.imshow(image * 0.5 + map1 * 0.5)
        
        # plt.subplot(4, 1, 2)
        # plt.axis('off')
        # map2 = np.tanh(sumed_map)
        # map2 = cv2.applyColorMap(np.uint8(255*map2), cv2.COLORMAP_JET)
        # map2 = cv2.cvtColor(map2, cv2.COLOR_BGR2RGB) / 255
        # plt.imshow(image * 0.5 + map2 * 0.5)
        
        # plt.subplot(4, 1, 3)
        # plt.axis('off')
        # plt.imshow(Image.open(os.path.join(anno_gt_dir, gt)), cmap='gray')
        
        # plt.subplot(4, 1, 4)
        # plt.axis('off')
        # plt.imshow(((src_masks.sigmoid())[0][0]>0.5).detach().cpu().numpy()*255, cmap='gray')
        
        # # plt.show()
        # plt.savefig(os.path.join("Prediction/masks/add_3conv", im[:-4]+'.png'), bbox_inches='tight', pad_inches=0)
        
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
