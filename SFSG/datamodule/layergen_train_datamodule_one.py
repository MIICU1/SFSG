# -*- coding:utf-8 -*-
import json
from utils.conditions import index_conditions
import os
import copy
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import cv2 as cv
from utils.protrusion_detection import detect_proper_protrusions
from utils.idea_module import process_finite_element_modeling
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


import numpy as np


def extract_specific_rows(data: np.ndarray, extra_row: int) -> np.ndarray:
    """
    提取指定行数据：必提取第1、7、8行，额外提取指定的一行
    特殊逻辑：若extra_row≠2，取第2行和额外行第一个元素的差值，额外行整体减去该差值（平移至第2行第一个元素的高度）
    Args:
        data: 输入的numpy数组，要求维度为(8, 400)
        extra_row: 需要额外提取的行号（1-8，对应数组索引0-7）
    Returns:
        np.ndarray: 提取后的数组，维度为(4, 400)（必提3行+额外1行）
    Raises:
        AssertionError: 输入数组维度不符合要求
        ValueError: 额外行号超出1-8的范围，或额外行号与必提行重复
    """
    # 1. 严格校验输入维度（必须是8行400列）
    assert data.shape == (8, 400), f"输入维度必须为(8,400)，当前为{data.shape}"

    # 必提取的行索引（对应业务行1、7、8）
    mandatory_indices = [0, 7]
    # 第2行的数组索引（业务行2 → 索引1）
    row_2_index = 1
    # 第一个元素的列索引
    first_col_index = 0

    # 2. 校验额外行号的合法性
    if not isinstance(extra_row, int) or extra_row < 1 or extra_row > 8:
        raise ValueError(f"额外行号必须是1-8之间的整数，当前输入为{extra_row}")

    # 3. 转换额外行号为数组索引，并校验是否与必提行重复
    extra_index = extra_row - 1
    if extra_index in mandatory_indices:
        raise ValueError(f"额外行号不能是必提取行（1、7、8），当前输入为{extra_row}")

    # 4. 复制原数据（避免修改原数组），处理额外行的平移逻辑
    data_copy = data.copy()
    if extra_row != 2:  # 仅当extra_row≠2时执行平移
        # 仅取第一个元素计算差值：额外行第一个元素 - 第2行第一个元素
        fixed_diff = data_copy[extra_index, first_col_index] - data_copy[row_2_index, first_col_index]
        # 额外行整体减去该固定差值（所有元素都减这个数）
        data_copy[extra_index, :] = data_copy[extra_index, :] - fixed_diff

    # 5. 合并索引并排序（必提行+额外行）
    target_indices = sorted(mandatory_indices + [extra_index])

    # 6. 提取指定行并返回
    extracted_data = data_copy[target_indices, :]

    return extracted_data

class cond_Dataset(Dataset):
    def __init__(self, img_root, pathes, cond_dict):
        assert os.path.exists(img_root)

        self.img_root = img_root
        self.pathes = pathes
        self.cond_dict = cond_dict

    def __getitem__(self, index):
        img = np.load(os.path.join(self.img_root, self.pathes[index][:-4]+'.npy'))
        class_cond = self.cond_dict[self.pathes[index][:-4]]['class']
        rise_cond = self.cond_dict[self.pathes[index][:-4]]['rise_range']
        a = rise_cond[0]  # a已为两位小数，直接保留
        b = round(rise_cond[1], 2)  # 将b强制保留两位小数
        rise_cond = [a, b]  # 重构为新的[a,b]，b已标准化为两位小数
        row_cond = self.cond_dict[self.pathes[index][:-4]].get('row', 2)
        if row_cond == 0 or row_cond == 2:
            img = extract_specific_rows(img, 2)
        else:
            img = extract_specific_rows(img, row_cond)
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        img /= 640
        img -= 0.5
        img /= 0.5  # 值的范围从-1到1

        class_cond = torch.tensor(class_cond).long()
        rise_cond = torch.tensor(rise_cond).float()

        return {'image': img, 'path': self.pathes[index], 'class_cond':class_cond, 'rise_cond':rise_cond}

    def __len__(self):
        return len(self.pathes)
