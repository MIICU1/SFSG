# -*- coding:utf-8 -*-
import json
from utils.conditions import index_conditions
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import cv2 as cv
class trainDatamodule(pl.LightningDataModule):
    def __init__(self, opts, train_data_root, train_json_path, train_cond_json_path, batch_size=1, num_workers=32,
                 **kwargs):
        super().__init__()
        self.opts = opts
        self.batch_size = opts.batch_size
        self.num_workers = num_workers
        self.train_data_root = train_data_root
        with open(train_json_path, "r") as f:
            self.train_pathes = json.load(f)

        with open(train_cond_json_path, "r") as f:
            self.cond_dict = index_conditions(json.load(f))
        print(len(self.train_pathes))


    def setup(self, stage=None):
        self.train_set = cond_Dataset(img_root=self.train_data_root, pathes=self.train_pathes, cond_dict=self.cond_dict)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


def process_masked_tensor(input_tensor: torch.Tensor) -> torch.Tensor:

    # 严格校验输入维度
    assert input_tensor.shape == (1, 8, 400), f"输入维度必须为(1,8,400)，当前为{input_tensor.shape}"

    # 复制Tensor避免修改原数据，降维为8×400便于操作
    masked_tensor = input_tensor.clone()
    arr_2d = masked_tensor.squeeze(0)  # 维度变为(8, 400)

    # prob = np.random.random()
    # if prob <= 0.5:
    arr_2d[2:7, :] = -2
    # else:
    #     arr_2d[2:4, :] = -2
    #     rand_rows = np.random.choice([4, 5, 6], size=2, replace=False)  # 随机选索引5（第6行）或6（第7行）
    #     for row in rand_rows:
    #         arr_2d[row, :] = -2

    # 恢复(1,8,400)维度并返回
    masked_tensor = arr_2d.unsqueeze(0)
    return masked_tensor

class cond_Dataset(Dataset):
    def __init__(self, img_root, pathes, cond_dict):
        assert os.path.exists(img_root)

        self.img_root = img_root
        self.pathes = pathes
        self.cond_dict = cond_dict

    def __getitem__(self, index):
        # img = cv.imread(os.path.join(self.img_root, self.pathes[index]), cv.IMREAD_GRAYSCALE)
        img = np.load(os.path.join(self.img_root, self.pathes[index][:-4]+'.npy'))
        img_mask = np.load(os.path.join(self.img_root, self.pathes[index][:-4] + '.npy'))
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        img /= 640
        img -= 0.5
        img /= 0.5 #值的范围从-1到1
        img_2 = img.numpy()

        img_mask = torch.from_numpy(img_mask).float()
        img_mask = img_mask.unsqueeze(0)
        img_mask /= 640
        img_mask -= 0.5
        img_mask /= 0.5  # 值的范围从-1到1
        img_mask = process_masked_tensor(img_mask)
        img_mask_2 = img_mask.numpy()

        class_cond = self.cond_dict[self.pathes[index][:-4]]['class']
        shift_cond = self.cond_dict[self.pathes[index][:-4]].get('mdice', 0.0)
        class_cond = torch.tensor(class_cond).long()
        shift_cond = torch.tensor(shift_cond).float()

        return {'image': img, 'path': self.pathes[index], 'class_cond':class_cond, 'shift_cond':shift_cond, 'mask':img_mask}

    def __len__(self):
        return len(self.pathes)
