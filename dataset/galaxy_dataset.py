import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
from astropy.io import fits
from prefetch_generator import BackgroundGenerator


def load_img(file):
    """
    加载图像，dat和fits均支持，不过仅支持CxHxW
    :param filename: 传入文件名，应当为CHW
    :return: 返回CHW的ndarray
    """
    if ".fits" in file:
        with fits.open(file) as hdul:
            return hdul[0].data.astype(np.float32)
    elif ".dat" in file:
        with open(file, "rb") as f:
            return pickle.load(f)
    else:
        raise TypeError

def arcsinh_rgb(imgs, bands, scales=None,m = 0.03):
    import numpy as np
    rgbscales =dict(g=(2,6.0), r=(1,3.4), z=(0,2.2))
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    Q = 50
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    # fI = MTF(I)
    I += (I == 0.) * 1e-9
    H,W = I.shape
    rgb = np.zeros((3,H,W), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[plane,:,:] = (img * scale + m) * fI / I
    rgb = np.clip(rgb, 0, 1)
    return rgb
class GalaxyDataset(Dataset):
    def __init__(self, annotations_file, transform):
        with open(annotations_file, "r") as file:
            imgs = []
            for line in file:
                line = line.strip("\n")
                line = line.rstrip("\n")
                words = line.split()
                label = str(line)[:-1].split("label:")
                hidden = list(label[-1][:].split(" "))
                hidden = hidden[:-1]
                votes = []
                for vote in hidden:
                    votes.append(float(vote))
                imgs.append((words[0], votes))
        self.imgs = imgs
        self.transform = transform
        self.pseudo_labels = None  # Initialize pseudo_labels as None

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = fits.getdata(path)
        if img.dtype.byteorder not in ('=', '|'):
            img = img.byteswap().newbyteorder()
        if self.pseudo_labels is not None:
            pseudo_label = self.pseudo_labels[index]
            return np.array(img), path, np.array(pseudo_label)
        else:
            return np.array(img), np.array(label)

    def set_pseudo_labels(self, pseudo_labels):
        self.pseudo_labels = pseudo_labels
    def __len__(self):
        return len(self.imgs)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class TestDataset(Dataset):
    def __init__(self, annotations_file, transform):
        with open(annotations_file, "r") as file:
            imgs = []
            for line in file:
                line = line.strip("\n")
                line = line.rstrip("\n")
                # line = line.replace("out_decals", "in_decals")
                words = line.split()
                label = str(line)[:-1].split("label:")
                hidden = list(label[-1][:].split(" "))
                hidden = hidden[:-1]
                votes = []
                for vote in hidden:
                    votes.append(float(vote))
                imgs.append((words[0], votes))
                path = words[0]
        self.path = path
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = load_img(path)
        return img, np.array(label), path

    def __len__(self):
        return len(self.imgs)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class PredictDataset(Dataset):
    def __init__(self, annotations_file, transform):
        with open(annotations_file, "r") as file:
            imgs = []
            for line in file:
                line = line.strip("\n")
                line = line.rstrip("\n")
                words = line.split()
                imgs.append((words[0]))
                path = words[0]
        self.path = path
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        img = load_img(path)
        return img, path

    def __len__(self):
        return len(self.imgs)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())