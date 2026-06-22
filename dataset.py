import os
import random
import numpy as np
import cv2
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import augment_and_mix
import utils
from depth_generator import create_depth_generator

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]
        
class DenoisingDataset(Dataset):
    def __init__(self, opt):                                   		    # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)
        print(self.imglist)
        self.rainaug = opt.rainaug
        # 初始化深度图生成器
        self.depth_generator = create_depth_generator(opt.depth_model_path, device='cuda' if not opt.no_gpu else 'cpu')
        '''
        for pair in self.imglist:
            print(pair[0] + ' | ' + pair[1])
        '''
    def getRainLayer2(self, rand_id1, rand_id2):
        path_img_rainlayer_src = "./rainmix/Streaks_Garg06/" + str(rand_id1) + "-" + str(rand_id2) + ".png"
        rainlayer_rand = cv2.imread(path_img_rainlayer_src).astype(np.float32) / 255.0
        rainlayer_rand = cv2.cvtColor(rainlayer_rand, cv2.COLOR_BGR2RGB)
        return rainlayer_rand

    def getRandRainLayer2(self):
        rand_id1 = random.randint(1, 165)
        rand_id2 = random.randint(4, 8)
        rainlayer_rand = self.getRainLayer2(rand_id1, rand_id2)
        return rainlayer_rand
        
    def rain_aug(self, img_rainy, img_gt):
        img_rainy = (img_rainy.astype(np.float32)) / 255.0
        img_gt = (img_gt.astype(np.float32)) / 255.0
        if random.randint(0, 10) > 3:
            img_rainy_ret = img_rainy
        else:
            img_rainy_ret = img_gt
        img_gt_ret = img_gt

        rainlayer_rand2 = self.getRandRainLayer2()
        rainlayer_aug2 = augment_and_mix.augment_and_mix(rainlayer_rand2, severity = 3, width = 3, depth = -1) * 1
        #rainlayer_rand2ex = self.getRandRainLayer2()
        #rainlayer_aug2ex = augment_and_mix.augment_and_mix(rainlayer_rand2ex, severity = 3, width = 3, depth = -1) * 1

        height = min(img_gt.shape[0], rainlayer_aug2.shape[0])
        width = min(img_gt.shape[1], rainlayer_aug2.shape[1])
        #height = min(img_gt.shape[0], min(rainlayer_aug2.shape[0], rainlayer_aug2ex.shape[0]))
        #width = min(img_gt.shape[1], min(rainlayer_aug2.shape[1], rainlayer_aug2ex.shape[1]))
        
        cropper = RandomCrop(rainlayer_aug2.shape[:2], (height, width))
        rainlayer_aug2_crop = cropper(rainlayer_aug2)
        #cropper = RandomCrop(rainlayer_aug2ex.shape[:2], (height, width))
        #rainlayer_aug2ex_crop = cropper(rainlayer_aug2ex)
        #print(height, width, rainlayer_aug2_crop.shape, rainlayer_aug2ex_crop.shape)
        #rainlayer_aug2_crop = rainlayer_aug2_crop + rainlayer_aug2ex_crop
        cropper = RandomCrop(img_gt_ret.shape[:2], (height, width))
        img_rainy_ret = cropper(img_rainy_ret)
        img_gt_ret = cropper(img_gt_ret)
        img_rainy_ret = img_rainy_ret + rainlayer_aug2_crop - img_rainy_ret*rainlayer_aug2_crop
        np.clip(img_rainy_ret, 0.0, 1.0)
        
        img_rainy_ret = img_rainy_ret * 255
        img_gt_ret = img_gt_ret * 255
        
        #cv2.imwrite("./temp/temp.jpg", cv2.cvtColor(img_rainy_ret, cv2.COLOR_RGB2BGR))
        
        return img_rainy_ret, img_gt_ret
        
    def __getitem__(self, index):
        ## read an image
        img_rainy = cv2.imread(self.imglist[index][0])
        img_gt = cv2.imread(self.imglist[index][1])
        img_rainy = cv2.cvtColor(img_rainy, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        
        if self.rainaug:
            img_rainy, img_gt = self.rain_aug(img_rainy, img_gt)
            
        # random crop
        cropper = RandomCrop(img_gt.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        img_rainy = cropper(img_rainy)
        img_gt = cropper(img_gt)
        
        # random rotate and horizontal flip
        if self.opt.angle_aug:
            rotate = random.randint(0, 3)
            if rotate != 0:
                img_rainy = np.rot90(img_rainy, rotate)
                img_gt = np.rot90(img_gt, rotate)
            if np.random.random() >= 0.5:
                img_rainy = cv2.flip(img_rainy, flipCode = 0)
                img_gt = cv2.flip(img_gt, flipCode = 0)
                
        '''        
        # add noise
        img = img.astype(np.float32) # RGB image in range [0, 255]
        noise = np.random.normal(self.opt.mu, self.opt.sigma, img.shape).astype(np.float32)
        noisy_img = img + noise
        '''

        # normalization
        img_rainy = img_rainy.astype(np.float32) # RGB image in range [0, 255]
        img_gt = img_gt.astype(np.float32) # RGB image in range [0, 255]
        img_rainy = img_rainy / 255.0
        img_rainy = torch.from_numpy(img_rainy.transpose(2, 0, 1)).contiguous()
        img_gt = img_gt / 255.0
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).contiguous()
        
        # 生成深度图
        with torch.no_grad():
            # 为雾图和原图生成深度图
            img_rainy_depth = self.depth_generator(img_rainy.unsqueeze(0).cuda())
            img_gt_depth = self.depth_generator(img_gt.unsqueeze(0).cuda())
            
            # 移除batch维度
            img_rainy_depth = img_rainy_depth.squeeze(0)
            img_gt_depth = img_gt_depth.squeeze(0)

        return img_rainy, img_gt, img_rainy_depth, img_gt_depth
    
    def __len__(self):
        return len(self.imglist)
    
#  未添加depth_img之前的验证方法
class DenoisingValDataset(Dataset):
    def __init__(self, opt):                                   		    # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)
        # 初始化深度图生成器
        self.depth_generator = create_depth_generator(opt.depth_model_path, device='cuda' if not opt.no_gpu else 'cpu')

    def __getitem__(self, index):
        ## read an image
        img_rainy = cv2.imread(self.imglist[index][0])
        img_gt = cv2.imread(self.imglist[index][1])

        height = img_rainy.shape[0]
        width = img_rainy.shape[1]
        height_origin = height
        width_origin = width
        if height % 16 != 0:
            height = ((height // 16) + 1) * 16
        if width % 16 !=0:
            width = ((width // 16) + 1) * 16
        img_rainy = cv2.resize(img_rainy, (width, height))
        img_gt = cv2.resize(img_gt, (width, height))

        # print(img_gt.shape)
        # print(img_rainy.shape)

        img_rainy = cv2.cvtColor(img_rainy, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        # random crop
        if self.opt.crop:
            cropper = RandomCrop(img_rainy.shape[:2], (self.opt.crop_size, self.opt.crop_size))
            img_rainy = cropper(img_rainy)
            img_gt = cropper(img_gt)
        # random rotate and horizontal flip
        # according to paper, these two data augmentation methods are recommended
        if self.opt.angle_aug:
            rotate = random.randint(0, 3)
            if rotate != 0:
                img_rainy = np.rot90(img_rainy, rotate)
                img_gt = np.rot90(img_gt, rotate)
            if np.random.random() >= 0.5:
                img_rainy = cv2.flip(img_rainy, flipCode = 0)
                img_gt = cv2.flip(img_gt, flipCode = 0)


        # normalization
        img_rainy = img_rainy.astype(np.float32) # RGB image in range [0, 255]
        img_gt = img_gt.astype(np.float32) # RGB image in range [0, 255]
        img_rainy = img_rainy / 255.0
        img_rainy = torch.from_numpy(img_rainy.transpose(2, 0, 1)).contiguous()
        img_gt = img_gt / 255.0
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).contiguous()
        
        # 生成深度图
        with torch.no_grad():
            # 为雾图和原图生成深度图
            img_rainy_depth = self.depth_generator(img_rainy.unsqueeze(0).cuda())
            img_gt_depth = self.depth_generator(img_gt.unsqueeze(0).cuda())
            
            # 移除batch维度
            img_rainy_depth = img_rainy_depth.squeeze(0)
            img_gt_depth = img_gt_depth.squeeze(0)

        return img_rainy, img_gt, img_rainy_depth, img_gt_depth, height_origin, width_origin

    def __len__(self):
        return len(self.imglist)


"""
重写读取函数，增加depth的读取
"""
#  =================添加depth_img之后的验证方法，为了方便验证方法，不再融合几个模型中的代码。仅仅通过导入的方式直接读取图像=========================
class DenoisingValDataset(Dataset):
    def __init__(self, opt):  # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files_depth(opt.baseroot)

    def __getitem__(self, index):
        ## read an image
        img_rainy = cv2.imread(self.imglist[index][0])
        img_gt = cv2.imread(self.imglist[index][1])
        img_depth = cv2.imread(self.imglist[index][2])

        height = img_rainy.shape[0]
        width = img_rainy.shape[1]
        height_origin = height
        width_origin = width
        if height % 16 != 0:
            height = ((height // 16) + 1) * 16
        if width % 16 != 0:
            width = ((width // 16) + 1) * 16
        img_rainy = cv2.resize(img_rainy, (width, height))
        img_gt = cv2.resize(img_gt, (width, height))
        img_depth = cv2.resize(img_depth, (width, height))

        # print(img_gt.shape)
        # print(img_rainy.shape)

        img_rainy = cv2.cvtColor(img_rainy, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2RGB)

        # random crop
        if self.opt.crop:
            cropper = RandomCrop(img_rainy.shape[:2], (self.opt.crop_size, self.opt.crop_size))
            img_rainy = cropper(img_rainy)
            img_gt = cropper(img_gt)
            img_depth = cropper(img_depth)
        # random rotate and horizontal flip
        # according to paper, these two data augmentation methods are recommended
        if self.opt.angle_aug:
            rotate = random.randint(0, 3)
            if rotate != 0:
                img_rainy = np.rot90(img_rainy, rotate)
                img_gt = np.rot90(img_gt, rotate)
                img_depth = np.rot90(img_depth, rotate)
            if np.random.random() >= 0.5:
                img_rainy = cv2.flip(img_rainy, flipCode=0)
                img_gt = cv2.flip(img_gt, flipCode=0)
                img_depth = cv2.flip(img_depth, flipCode=0)

        # normalization
        img_rainy = img_rainy.astype(np.float32)  # RGB image in range [0, 255]
        img_gt = img_gt.astype(np.float32)  # RGB image in range [0, 255]
        img_depth = img_depth.astype(np.float32)  # RGB image in range [0, 255]
        img_rainy = img_rainy / 255.0
        img_rainy = torch.from_numpy(img_rainy.transpose(2, 0, 1)).contiguous()
        img_gt = img_gt / 255.0
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).contiguous()
        img_depth = img_depth / 255.0
        img_depth = torch.from_numpy(img_depth.transpose(2, 0, 1)).contiguous()

        return img_rainy, img_gt, img_depth,height_origin, width_origin

    def __len__(self):
        return len(self.imglist)