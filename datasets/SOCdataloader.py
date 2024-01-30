from cProfile import label
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image, ImageEnhance
import util.misc as utils


import torch
import json

from util.misc import collate_fn
import matplotlib.pyplot as plt

from imantics import Polygons, Mask, Annotation
import datasets.transforms as T
import numpy as np


class Config():
    def __init__(self) -> None:
        self.image_root = 'datasets/SOC/TrainSet/Imgs/'
        self.gt_root = 'datasets/SOC/TrainSet/gt/'

        # self-supervision
        self.lambda_loss_ss = 0.3   # 0 means no self-supervision

        # label smoothing
        self.label_smooth = 0.001   # epsilon for smoothing, 0 means no label smoothing, 

        # # preproc
        # self.preproc_activated = True
        # self.hflip_prob = 0.5
        # self.crop_border = 30      # < 1 as percent of min(wid, hei), >=1 as pixel
        # self.rotate_prob = 0.2
        # self.rotate_angle = 15
        # self.enhance_activated = True
        # self.enhance_brightness = (5, 15)
        # self.enhance_contrast = (5, 15)
        # self.enhance_color = (0, 20)
        # self.enhance_sharpness = (0, 30)
        # self.gaussian_mean = 0.1
        # self.gaussian_sigma = 0.35
        # self.pepper_noise = 0.0015
        # self.pepper_turn = 0.5

        
# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        
        self.trainsize = trainsize
        
        self.anno_dir = os.path.join(gt_root, 'annos_list')
        self.final_mask_dir = os.path.join(gt_root, 'gt_annoted')
        
        # test
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [os.path.join(self.anno_dir, f) for f in os.listdir(self.anno_dir)]
        self.final_masks = [os.path.join(self.final_mask_dir, f[:-4]+'png') for f in os.listdir(self.anno_dir)]
        
        self.size = len(self.images)
        
        if self.size > 1200:
            self.image_set = 'train'
        else:
            self.image_set = 'test'
        
        # Instance gt
        # self.images = [image_root + f[:-3]+'jpg' for f in os.listdir(os.path.join(gt_root, 'Instance_name')) if f.endswith('.txt')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.final_masks = sorted(self.final_masks)
        
        # self.filter_files()
        # self.train_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     transforms.RandomHorizontalFlip()])
        # self.val_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize((self.trainsize, self.trainsize)),
            normalize,
        ])
        self.val_transform = T.Compose([
            T.Resize((self.trainsize, self.trainsize)),
            normalize,
        ])
        
        self.config = Config()
        
        ###
        # self.names = [os.path.join(gt_root, 'Instance_name/') + f for f in os.listdir(os.path.join(gt_root, 'Instance_name/')) if f.endswith('.txt')]
        # self.names = sorted(self.names)
        ###

            
        # self.CLASSES = [
        # 'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        # 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        # 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        # 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        # 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        # 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        # 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        # 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        # 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        # 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        # 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        # 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        # 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        # 'toothbrush'
        # ]
        
        self.CLASSES = [
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
        
        
        

    def __getitem__(self, index):
        pre_image = self.rgb_loader(self.images[index])
        # if self.config.preproc_activated:
        #     if self.config.hflip_prob:
        #         image, gt = cv_random_flip(image, gt, prob=self.config.hflip_prob)
        #     if self.config.crop_border:
        #         image, gt = randomCrop(image, gt, border=self.config.crop_border)
        #     if self.config.rotate_prob:
        #         image, gt = randomRotation(image, gt, prob=self.config.rotate_prob, angle=self.config.rotate_angle)
        #     if self.config.enhance_activated:
        #         image = colorEnhance(image, self.config.enhance_brightness, self.config.enhance_contrast, self.config.enhance_color, self.config.enhance_sharpness)
        #     if self.config.gaussian_sigma:
        #         gt = randomGaussian(gt)
        #     if self.config.pepper_noise and self.config.pepper_turn:
        #         gt = randomPeper(gt)
        
        
        # r, g, b = gt[0,:,:], gt[1,:,:], gt[2,:,:]
        # y, m = torch.logical_and(r, g), torch.logical_and(r, b)
        
        # plt.subplot(1, 6, 1)
        # plt.imshow(gt.permute(-2, -1, 0).cpu().detach().numpy(), cmap='gray')
        
        # plt.subplot(1, 6, 2)
        # plt.imshow(torch.logical_xor(r, torch.logical_or(y,m)), cmap='gray')
        
        # plt.subplot(1, 6, 3)
        # plt.imshow(torch.logical_xor(g, y), cmap='gray')
        
        # plt.subplot(1, 6, 4)
        # plt.imshow(torch.logical_xor(b, m), cmap='gray')
        
        # plt.subplot(1, 6, 5)
        # plt.imshow(y, cmap='gray')
        
        # plt.subplot(1, 6, 6)
        # plt.imshow(m, cmap='gray')
        
        # plt.show()
        
        # gt = self.gt_processing(gt)
        
        ### bbox & class annotation
        
        label, box = torch.tensor([]), torch.tensor([])
        
        label, box, mask = self.anno_transform(self.gts[index])
        target = {'masks': mask, 'labels': label, 'boxes': box,
                    'final_mask': transforms.ToTensor()(self.binary_loader(self.final_masks[index]))}
        if self.image_set=='train':
            image, target = self.train_transform(pre_image, target)
        else:
            image, target = self.val_transform(pre_image, target)
        
        return image, target
        
        # name = []
        # f = open(self.names[index])
        # lines = f.readlines()
        # for line in lines:
        #     line = line.strip()
        #     name.append(CLASSES.index(line.split(':')[-1]))
        ###

        # return image, gt
        # return image, gt, torch.Tensor(name)


    def filter_files(self):
        # print(len(self.images), "==", len(self.gts))
        # assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('I') 

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size
    
    def anno_transform(self, anno_path):
        
        label, box, mask = [], [], []
        
        with open(anno_path) as f:
            json_data = json.load(f)
            
        w, h = json_data['width'], json_data['height']
        name = json_data['file_name']
        
        for o in json_data['objects']:
            if o['class_name'] in self.CLASSES:
                label.append(self.CLASSES.index(o['class_name']))
            else:
                label.append(0)
            # box.append([o['bbox'][0]/w, o['bbox'][1]/h, o['bbox'][2]/w, o['bbox'][3]/h])
            box.append([o['bbox'][0], o['bbox'][1], o['bbox'][2], o['bbox'][3]])
            mask.append(Polygons(o['polygons']).mask(w, h).array)
            
        mask = np.array(mask)
        
        return torch.tensor(label, dtype=torch.long), torch.tensor(box, dtype=torch.float), torch.tensor(mask)
    
    # def gt_processing(self, gt):
        
    #     gt_masks = []

    #     for i in range(5):
    #         gt
        
        
    #     return
        
    

# dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  collate_fn=utils.collate_fn)
    return data_loader

# test dataset and loader
class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

