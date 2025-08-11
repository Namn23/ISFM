
import glob
import logging

import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from .Get_Patch_gray import Get_Random_Patch

logger = logging.getLogger()


def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = torch.clamp(Y, 0, 1)
    Cr = torch.clamp(Cr, 0, 1)
    Cb = torch.clamp(Cb, 0, 1)
    return Y, Cr, Cb



def YCrCb2RGB(Y, Cr, Cb):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = out.clamp(0, 1)
    return out


class MSRS_train(Dataset):
    def __init__(self, root_dir, img_size, config_task):
        super().__init__()

        self.img_items = []
        self.num_vif = 0

        train_dir = os.path.join(root_dir, "train")

        self.process_dir(train_dir, config_task=config_task)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ])


    def _process_VIF_dir(self, root_dir, task_config='VIF'):

        img_paths_vi = sorted(glob.glob(os.path.join(root_dir, 'vi', '*.png')))
        for img_path_v in img_paths_vi:
            jpg_name = img_path_v.split('/')[-1]

            vi = img_path_v
            ir = os.path.join(root_dir, "ir", jpg_name)

            name = "VIF_" + jpg_name
            vi_src = vi
            ir_src = ir
            item = {'vi': vi, 'ir': ir, "jpg_name": name, "vi_src": vi_src, "ir_src": ir_src}
            items = [(key, value) for key, value in item.items()]
            self.num_vif += 1
            self.img_items.append(items)

    def process_dir(self, root_dir, config_task):

        self._process_VIF_dir(root_dir, "VIF")

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        items = self.img_items[index]
        img_path_s1 = items[0]
        img_path_s2 = items[1]
        jpg_names = items[2]

        if random.random() > 0.5:
            patch_vi, patch_ir, patch_ir_rgb = Get_Random_Patch(img1_path=img_path_s1[1], img2_path=img_path_s2[1])
        else:
            with open(img_path_s1[1], 'rb') as f:
                patch_vi = Image.open(f).convert('RGB')
            with open(img_path_s2[1], 'rb') as f:
                patch_ir = Image.open(f).convert('L')
            with open(img_path_s2[1], 'rb') as f:
                patch_ir_rgb = Image.open(f).convert('RGB')
        vi = self.transform(patch_vi)
        ir = self.transform(patch_ir)
        vi_rgb = self.transform(patch_vi)
        ir_rgb = self.transform(patch_ir_rgb)

        vi_y, vi_cr, vi_cb = RGB2YCrCb(vi)

        return_dict = {
            "vi_y": vi_y,
            "ir": ir,
            "jpg_names": jpg_names[1],
            "vi_src": vi_rgb,
            "ir_src": ir_rgb,
            "cr": vi_cr,
            "cb": vi_cb,
        }

        return return_dict


class MSRS_test(Dataset):
    def __init__(self, root_dir, img_size, config_task):
        super().__init__()

        test_dir = os.path.join(root_dir, "test")

        self.img_items = []
        self.process_dir(test_dir, config_task)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _process_VIF_dir(self, root_dir, task_type="VIF"):

        vi_dir = os.path.join(root_dir, 'vi')
        for filename in os.listdir(vi_dir):
            if filename.endswith((".png", ".jpg", ".JPG", ".jpeg")):  # 可根据需要添加更多格式
                img_path_v = os.path.join(vi_dir, filename)
                jpg_name = img_path_v.split('/')[-1]
                vi = img_path_v
                ir = os.path.join(root_dir, "ir", jpg_name)
                name = "VIF_" + jpg_name

                item = {'vi': vi, 'ir': ir, "jpg_name": name}
                items = [(key, value) for key, value in item.items()]
                self.img_items.append(items)

    def process_dir(self, root_dir, config_task):

        self._process_VIF_dir(root_dir)

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        items = self.img_items[index]
        img_path_s1 = items[0]
        img_path_s2 = items[1]
        jpg_names = items[2]

        with open(img_path_s1[1], 'rb') as f:
            vi = Image.open(f).convert('RGB')
        with open(img_path_s2[1], 'rb') as f:
            ir = Image.open(f).convert('L')
            ir_rgb = Image.open(f).convert('RGB')
        vi_src = self.transform(vi)
        ir_src = self.transform(ir_rgb)
        vi = self.transform(vi)
        vi_y, vi_cr, vi_cb = RGB2YCrCb(vi)
        ir = self.transform(ir)


        return_dict = {
            "vi_y": vi_y,
            "ir": ir,
            "jpg_names": jpg_names[1],
            "cr": vi_cr,
            "cb": vi_cb,
            "vi_src": vi_src,
            "ir_src": ir_src,
        }

        return return_dict


