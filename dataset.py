from torch.utils.data import Dataset 
import os
try:
    from . import transform
except:
    import LMFNet.lib.transform as transform
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from PIL import Image
import random
from PIL import ImageEnhance


mean_rgb = np.array([[[0.391 * 255, 0.363 * 255, 0.338 * 255]]])
mean_t = np.array([[[0.170 * 255, 0.403 * 255, 0.556 * 255]]])
mean_d = np.array([[[0.034 * 255, 0.034 * 255, 0.034 * 255]]])
std_rgb = np.array([[[0.224 * 255, 0.217 * 255, 0.206 * 255]]])
std_t = np.array([[[0.160 * 255, 0.188 * 255, 0.238 * 255]]])
std_d = np.array([[[0.007 * 255, 0.007 * 255, 0.007 * 255]]])

# several data augumentation strategies
def cv_random_flip(img, label, ti):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        ti = ti.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     ti = ti.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, ti


def randomCrop(image, label, ti):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), ti.crop(random_region)


def randomRotation(image, label, ti):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        ti = ti.rotate(random_angle, mode)
    return image, label, ti


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)

def getRandomSample(rgb, t, d):
    n = np.random.randint(10)
    zero = np.random.randint(2)
    if n == 1:
        if zero:
            rgb = torch.from_numpy(np.zeros_like(rgb))
        else:
            rgb = torch.from_numpy(np.random.randn(*rgb.shape))
    elif n == 2:
        if zero:
            t = torch.from_numpy(np.zeros_like(t))
            d = torch.from_numpy(np.zeros_like(d))

        else:
            t = torch.from_numpy(np.random.randn(*t.shape))
            d = torch.from_numpy(np.random.randn(*d.shape))
    return rgb, t, d

class Data(Dataset):
    def __init__(self,root,mode='train'):
        lines = os.listdir(os.path.join(root,'GT'))
        self.samples=[]
        self.mode = mode
        self.trainsize=224
        self.testsize=224
        for line in lines:
            rgbpath = os.path.join(root,'V',line[:-4]+'.jpg')
            tpath = os.path.join(root, 'T', line[:-4] + '.jpg')
            dpath = os.path.join(root, 'D', line[:-4] + '.jpg')
            maskpath = os.path.join(root, 'GT', line)
           

            self.samples.append([rgbpath, tpath, dpath, maskpath])

        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def __getitem__(self, idx):
        rgbpath, tpath, dpath, maskpath = self.samples[idx]
        # rgb = cv2.imread(rgbpath).astype(np.float32)
        # 读取图像为array
        # rgb= np.array(Image.open(rgbpath).convert('RGB')).astype(np.float32)
        # t = np.array(Image.open(tpath).convert('RGB')).astype(np.float32)
       # d = np.array(Image.open(dpath).convert('RGB')).astype(np.float32)
        # mask = np.array(Image.open(maskpath).convert('RGB')).astype(np.float32)

        rgb = self.rgb_loader(rgbpath)
        mask = self.binary_loader(maskpath)
        t = self.rgb_loader(tpath)
        d = self.rgb_loader(dpath)
        
        H, W = mask.height, mask.width
        if self.mode == 'train':
            rgb, mask, t = cv_random_flip(rgb, mask, t)
            rgb, mask, t = randomCrop(rgb, mask, t)
            rgb, mask, t = randomRotation(rgb, mask, t)
            rgb = colorEnhance(rgb)
            # gt=randomGaussian(gt)
            mask = randomPeper(mask)
          

            # rgb, t, d = getRandomSample(rgb, t, d)
        rgb = self.img_transform(rgb)
        mask = self.gt_transform(mask)
        t = self.tis_transform(t)
        d = self.tis_transform(d)
        return rgb, t, d, mask, (H,W), maskpath.split('/')[-1]

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.samples)
