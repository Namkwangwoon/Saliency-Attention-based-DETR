# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
# from ast import keyword
import math
# import os
import sys
from typing import Iterable

import torch

import util.misc as utils
# from datasets.coco_eval import CocoEvaluator
# from datasets.panoptic_eval import PanopticEvaluator


from util.misc import interpolate
from models.segmentation import dice_loss, sigmoid_focal_loss
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import copy
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

import os
import datasets.metrics as M
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.transforms as transforms
import gc

import json
from imantics import Polygons


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


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # print("\nLR : %f" % (optimizer.param_groups[0]['lr']))
    final_loss_value = 0
    
    # size = data_loader.batch_size * len(data_loader)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    # for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # print()
        # for sam, tar in zip(samples.tensors, targets):
        #     print("sample :", sam.shape)
        #     print("masks :", tar['masks'].shape)
        #     print("labels :", tar['labels'])
        #     print("boxes :", tar['boxes'].type(torch.FloatTensor))
        #     print("final mask :", tar['final_mask'].shape)
            
        #     plt.subplot(1, 7, 1)
        #     plt.imshow(sam.permute(-2, -1, 0).cpu().detach().numpy())
            
        #     for i, m in enumerate(tar['masks']):
        #         plt.subplot(1, 7, i+2)
        #         plt.imshow(m.cpu().detach().numpy())
                
        #     plt.subplot(1, 7, 7)
        #     plt.imshow(tar['final_mask'][0].cpu().detach().numpy())
            
        #     plt.show()
            
        #     print()
        # print()
        
        outputs = model(samples)
        # src_masks, _ = outputs['pred_masks']
        # target_masks = targets[0]['masks']
        # src_masks = interpolate(src_masks[:, None].type(torch.float), size=targets.shape[-2:],
        #                         mode="bilinear", align_corners=False)
            
        
        # for i, t in enumerate(targets):
        #     print("labels :", t['labels'])
        #     print("boxes :", t['boxes'])
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(samples.tensors[i].permute(-2, -1, 0).cpu().detach().numpy())
            
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(t['masks'][0].cpu().detach().numpy())
            
        #     plt.show()
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        final_loss_value += losses
        
        
        # pred, target = src_masks.flatten(1).to(device), targets.flatten(1).to(device)
        # num_boxes = pred.shape[0]
        # L_dice = dice_loss(pred, target, num_boxes)
        # L_focal = sigmoid_focal_loss(pred, target, num_boxes)
        # losses =  L_dice + L_focal
        # losses.requires_grad_(True)
        # loss_value += losses
        
        # step = i * data_loader.batch_size
        # if step % 100 ==0:
        #     print("[%d epoch (%d/%d)] Loss : %f,    L_dice : %f,    L_focal : %f" % (epoch, step, size, losses, L_dice, L_focal))


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            print()
            print("outputs : ", outputs)
            print()
            print("targets : ", targets)
            print()
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return final_loss_value / len(data_loader)


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
# def evaluate(model, b_model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    # b_model.load_state_dict(torch.load("OUTPUT/DETR_SOD/4th_1(1e-4)/checkpoint.pth", map_location='cpu')['model'])
    # b_model.eval()
    # b_model.to(device)
    model.load_state_dict(torch.load("OUTPUT/DETR_SOD/checkpoint.pth", map_location='cpu')['model'])
    model.eval()
    criterion.eval()
    model.to(device)
    
    loss_mask = 0
    loss_final_mask = 0
    loss_saliency = 0
    
    batch_size = data_loader.batch_size
    l = len(data_loader)
    size = batch_size * l
    
    for i, (samples, targets) in enumerate(data_loader):
        
        samples = samples.tensors.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # for b, t in enumerate(targets):
        #     for m in t['masks']:
        #         plt.subplot(1, 2, 1)
        #         plt.imshow(samples[b].permute(-2, -1, 0).cpu().detach().numpy())
                
        #         plt.subplot(1, 2, 2)
        #         plt.imshow(m.cpu().detach().numpy())
                
        #         plt.show()

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        
        # L_dice = loss_dict['loss_dice']
        # L_focal = loss_dict['loss_mask']
        # losses = L_dice + L_focal
        L_mask = loss_dict['loss_mask'] + loss_dict['loss_dice']
        # L_mask = loss_dict['loss_dice']
        # L_sal = loss_dict['loss_saliency']

        loss_mask += L_mask * batch_size
        # loss_saliency += L_sal * batch_size
        
        target_masks = []
        for t in targets:
            target_masks.append(t['final_mask'][0])
        target_masks = torch.stack(target_masks)
        
        
        # for fin, targ in zip(final_mask, target_masks):
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(fin.sigmoid().cpu().detach().numpy()>0.5)
            
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(targ.cpu().detach().numpy())
            
        #     plt.show()
        
        
        
        
        
        # sigmoid_focal_loss(outputs['final_mask'], target_masks)
        
        # src_masks = outputs['pred_masks']
        # src_masks = src_masks.max(1).values
        # src_masks = interpolate(src_masks[:, None].type(torch.float), size=targets.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        
        # b_outputs = b_model(samples)
        # b_src_masks, _ = b_outputs['pred_masks']
        # b_src_masks = interpolate(b_src_masks[:, None].type(torch.float), size=targets.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        
        
        # for sam, tar, out, b in zip(samples, targets, src_masks, b_src_masks):
        #     plt.figure(figsize=(10, 50))
            
        #     sam_ = sam.permute(-2, -1, 0).cpu().mul_(torch.Tensor([0.485, 0.456, 0.406])).add_(torch.Tensor([0.229, 0.224, 0.225]))
        #     plt.subplot(4, 1, 1)
        #     plt.axis('off')
        #     plt.imshow(sam_)
            
        #     plt.subplot(4, 1, 2)
        #     plt.axis('off')
        #     plt.imshow(tar[0].cpu(), cmap='gray')
            
        #     plt.subplot(4, 1, 3)
        #     plt.axis('off')
        #     plt.imshow(b[0].cpu() > 0.5, cmap='gray')
            
        #     plt.subplot(4, 1, 4)
        #     plt.axis('off')
        #     plt.imshow(out[0].cpu() > 0.5, cmap='gray')
            
        #     plt.savefig('masks/%d.png' % i)
        #     # plt.show()
        
        
        ## loss
        # pred, target = src_masks.flatten(1).to(device), targets.flatten(1).to(device)
        # num_boxes = pred.shape[0]
        # L_dice = dice_loss(pred, target, num_boxes)
        # L_focal = sigmoid_focal_loss(pred, target, num_boxes)
        # losses =  L_dice + L_focal
        # losses.requires_grad_(True)
        # loss_value += losses
        
        
        
        step = i*len(samples)
        if step % 100 == 0:
            # print("[(%d/%d)] L_mask : %f,    L_saliency : %f" % (step, size, L_mask, L_sal))
            print("[(%d/%d)] L_mask : %f" % (step, size, L_mask))

    # print("loss_mask : %f   loss_saliency : %f" % (loss_mask / size, loss_saliency / size))
    print("loss_mask : %f" % (loss_mask / size))
    print()



    ### SOC eval ###
    
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()
    

    
    ''''''
    
    '''saliency index'''
    
    # img_root = '../datasets/SOC/ValSet/Imgs'
    # gt_root = '../datasets/SOC/ValSet/gt'
    # img_list = os.listdir(img_root)
    
    # for img_name in tqdm(img_list, total=len(img_list)):
    #     img_path = os.path.join(img_root, img_name)
    #     gt_path = os.path.join(gt_root, img_name[:-3]+'png')
    #     img = cv2.imread(img_path)
    #     img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(-1, 0, 1).type(torch.cuda.FloatTensor).cuda()/255.0
    #     gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
    #     outputs = model([img])
    #     outputs_masks = F.interpolate(outputs['pred_masks'], size=img.shape[-2:], mode="bilinear", align_corners=False)
    #     # outputs_masks = outputs_masks.sigmoid().cpu().detach().numpy()
    #     saliency = outputs['saliency_value'][0].squeeze(-1).sigmoid()
    #     ind = torch.arange(0, 100)[saliency>0.5].cuda()
        
    #     if len(ind)==0:
    #         pred = torch.zeros_like(outputs_masks[0][0]).cuda()
            
    #     else:
    #         pred = outputs_masks[0, ind]
    #         if pred.ndim == 3:
    #             pred = pred.max(0).values
                
    #     pred = pred * 255.0
        
    #     pred = pred.sigmoid().cpu().detach().numpy()

    #     SM.step(pred=pred, gt=gt)
    #     EM.step(pred=pred, gt=gt)
    #     MAE.step(pred=pred, gt=gt)
        
        
    '''matching index'''
    
    img_root = '../datasets/SOC/ValSet/Imgs_annoted'
    gt_root = '../datasets/SOC/ValSet/gt'
    anno_gt_dir = '../datasets/SOC/ValSet/annos_list'
    img_list = sorted(os.listdir(img_root))
    
    for img_name in tqdm(img_list, total=len(img_list)):
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
        outputs_masks = interpolate(outputs['pred_masks'], size=img.shape[-2:], mode="bilinear", align_corners=False)
        
        ind = criterion.matcher(outputs, target)[0][0]
        
        if len(ind)==0:
            pred = torch.zeros_like(outputs_masks[0][0]).cuda()
            
        else:
            pred = outputs_masks[0, ind]
            if pred.ndim == 3:
                pred = pred.max(0).values
                
        pred = pred * 255.0
        
        pred = pred.sigmoid().cpu().detach().numpy()

    
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)
    

    ''''''




    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    
    print()
    print(
        'MAE : ', mae.round(3), '\n',
        'Smeasure : ', sm.round(3), '\n',
        'meanEm : ', '-' if em['curve'] is None else em['curve'].mean().round(3), '\n',
        'maxEm : ', '-' if em['curve'] is None else em['curve'].max().round(3), '\n',
        sep=''
    )
    print()
    
    del SM
    del EM
    del MAE
    
    gc.collect()
    
    # return loss_mask / size, loss_saliency / size
    return loss_mask / size, loss_saliency / size, mae, sm, em['curve'].mean(), em['curve'].max()
    
    
    # unnorm = UnNormalizer()

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # panoptic_evaluator = None
    # # if 'panoptic' in postprocessors.keys():
    # #     panoptic_evaluator = PanopticEvaluator(
    # #         data_loader.dataset.ann_file,
    # #         data_loader.dataset.ann_folder,
    # #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    # #     )

    # # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    # # for samples, targets, labels in data_loader:
    # for samples, targets in data_loader:
    
    #     samples = samples.to(device)
    #     # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #     outputs = model(samples)
        
    #     # loss_dict = criterion(outputs, targets)
    #     # weight_dict = criterion.weight_dict

    #     # # reduce losses over all GPUs for logging purposes
    #     # loss_dict_reduced = utils.reduce_dict(loss_dict)
    #     # loss_dict_reduced_scaled = {k: v * weight_dict[k]
    #     #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
    #     # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
    #     #                               for k, v in loss_dict_reduced.items()}
    #     # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
    #     #                      **loss_dict_reduced_scaled,
    #     #                      **loss_dict_reduced_unscaled)
    #     # metric_logger.update(class_error=loss_dict_reduced['class_error'])

    #     # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    #     orig_target_sizes = torch.stack([torch.Tensor(list(targets.shape[-2:]))], dim=0).to(torch.int)
    #     results = postprocessors['bbox'](outputs, orig_target_sizes)
    #     if 'segm' in postprocessors.keys():
    #         # target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    #         target_sizes = torch.stack([torch.Tensor(list(targets.shape[-2:]))], dim=0).to(torch.int)
    #         results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
    #     # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
    #     # if coco_evaluator is not None:
    #     #     coco_evaluator.update(res)

    #     # if panoptic_evaluator is not None:
    #     #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
    #     #     for i, target in enumerate(targets):
    #     #         image_id = target["image_id"].item()
    #     #         file_name = f"{image_id:012d}.png"
    #     #         res_pano[i]["image_id"] = image_id
    #     #         res_pano[i]["file_name"] = file_name

    #     #     panoptic_evaluator.update(res_pano)



class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor