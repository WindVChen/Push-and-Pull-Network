import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.functional_pil as F_pil
import math
import random
from PIL import Image, ImageFilter
import glob
import os
import utils.DatasetSplit

class resize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        img = customResize(img)
        return img

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 1.5]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class customRandomCrop(nn.Module):
    def __init__(self, size, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.)):
        super().__init__()

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        return 0, 0, height, width

    def forward(self, img):
        if (len(np.asarray(img).shape) == 2):
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_GRAY2BGR)
            img = Image.fromarray(cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB))
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F_pil.crop(img, i, j, h, w)
        return img

class customRandomResizeCrop(customRandomCrop):
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F_pil.crop(img, i, j, h, w)
        return customResize(img)

def customResize(img, new_shape=(224, 224), color=(0, 0, 0), scaleup=True):
    if(len(np.asarray(img).shape) == 2):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    img = Image.fromarray(cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB))
    return img

def customAugmentation(scale = 0.7):
    augmentation = [
        # customRandomResizeCrop(224, scale=(scale, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomApply([GaussianBlur()], p=0.5),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        resize(),
        transforms.ToTensor(),
    ]
    augmentation = transforms.Compose(augmentation)
    return augmentation

def customGeneration():
    generation = [
        customRandomCrop(224, scale=(0.7, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomApply([GaussianBlur()], p=0.5),
        transforms.RandomRotation(180, expand=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ]
    generation = transforms.Compose(generation)
    return generation

if __name__ == '__main__':
    totalPath = r"G:\Dataset\FGSC-23\train"
    utils.DatasetSplit.removeAug(r"G:\Dataset\FGSC-23")
    allClassDir = os.listdir(totalPath)
    for dir in allClassDir:
        print("Start Augment {} ******".format(dir))
        cal = 0
        augNum = 0
        dir = os.path.join(totalPath, dir)
        images = glob.glob(dir + "/*.jpg")
        if(len(images)<200):
            imgList = []
            nameList = []
            ratio = math.ceil(200 / len(images))
            needAugNum = 200 - len(images)
            for times in range(ratio - 1):
                for imgFile in images:
                    img = Image.open(imgFile)
                    transform = customGeneration()
                    outImg = np.transpose(transform(img).numpy()*255, (1,2,0))
                    outImg = Image.fromarray(outImg.astype(np.uint8))
                    imgList.append(outImg)
                    nameList.append(imgFile.replace(".jpg", "_aug_{}.jpg".format(times)))
            sample = random.sample(range(0, len(imgList)), needAugNum)
            for i in sample:
                imgList[i].save(nameList[i])

            print("Augment Done! \n")
    pass