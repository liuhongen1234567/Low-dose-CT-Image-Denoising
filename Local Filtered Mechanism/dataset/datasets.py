import os
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def random_crop(image, gt, size):
    w, h = image.size
    new_h, new_w = size, size
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    image = image.crop((x, y, x + new_w, y + new_h))
    gt = gt.crop((x, y, x + new_w, y + new_h))
    return image, gt


class CT_Dataset(data.Dataset):
    def __init__(self, img_path, gt_path, is_train):
        super(CT_Dataset, self).__init__()
        img_files = []
        gt_files = []
        for name in os.listdir(img_path):
            if name.endswith('png'):
                img_files.append(os.path.join(img_path, name))
                str1 = name
                str2 = str1.replace("L", "H")
                gt_files.append(os.path.join(gt_path, str2))
        self.img_files = img_files
        self.gt_files = gt_files
        self.training = is_train

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        gt = Image.open(self.gt_files[index])
        if self.training:
            img, gt = random_crop(img, gt, 192)
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(img), transform(gt)
