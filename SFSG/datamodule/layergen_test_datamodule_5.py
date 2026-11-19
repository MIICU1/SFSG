# -*- coding:utf-8 -*-
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import json
import numpy as np
import os
class_to_name = {
    0: "NORMAL",
    1: "AMD",
    2: "DR",
    3: "CNV",
    4: "RVO",
    5: "CSC",
    6: "OTHERS"
}


class test_layergen_cond_Datamodule(pl.LightningDataModule):
    def __init__(self, test_data_root, test_json_path, cond_json_path,
                 selected_degree=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 改为直接筛选0.1-0.9的梯度值
                 batch_size=1, num_workers=8, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_data_root = test_data_root
        self.selected_degree = selected_degree  # 要筛选的梯度值（如[0.1,0.2,...0.9]）

        # 加载测试集路径
        with open(test_json_path, "r") as f:
            self.test_pathes = json.load(f)

        # 加载条件列表（新格式：path/class/shift）
        with open(cond_json_path, "r") as f:
            self.cond_list = json.load(f)

        # print(f"原始cond_list长度: {len(self.cond_list)}")
        # print(f"前5条cond_list示例: {self.cond_list[:5]}")
        #
        # self.select_list = []
        # save_dir = "./result_6mm/5/12"
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # 核心筛选逻辑：按路径中的梯度目录（如0.1/0.2）或shift值筛选
        # for item in self.cond_list:
        #     try:
        #         # 1. 解析路径中的梯度目录（如"AMD/0.1/AMD_0.16.png" → 提取0.1）
        #         path = item['path']
        #         # 分割路径获取梯度目录（第二个层级）
        #         path_parts = path.split('/')
        #         if len(path_parts) >= 2:
        #             degree_dir = path_parts[1]  # 取"0.1"/"0.2"等
        #             degree = float(degree_dir)
        #         else:
        #             # 若路径无梯度目录，用shift值的整数十位近似（如0.16→0.1）
        #             shift = item['shift']
        #             degree = round(shift * 10) / 10  # 0.16→1→0.1, 0.28→3→0.3（四舍五入到0.1梯度）
        #
        #         # 2. 筛选符合selected_degree的item
        #         if degree in self.selected_degree:
        #             self.select_list.append(item)
        #
        #     except Exception as e:
        #         print(f"处理item {item} 出错: {e}，跳过该item")
        #         continue
        #
        # print(f'筛选后select_list长度: {len(self.select_list)}')
        # print(f'筛选后前5条示例: {self.select_list[:5]}')

    def setup(self, stage=None):
        self.test_set = test_cond_Dataset(cond_list=self.cond_list,img_root=self.test_data_root, pathes=self.test_pathes)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

import torch
import numpy as np

import torch
import numpy as np


def process_array(original_arr):
    """
    处理数组：
    1. 原始形状(4,400) → 在第1/2行之间插入4行0 → (8,400)
    2. 第一行全部设为0，最后一行全部设为640
    3. 所有数值舍四进五取整（0.4及以下舍，0.5及以上进）
    """
    # 校验原始数组形状
    if original_arr.shape != (3, 400):
        raise ValueError(f"输入数组形状必须为(4,400)，当前为{original_arr.shape}")

    # 步骤1：在第1行和第2行之间插入4行0（形状变为8,400）
    insert_rows = np.zeros((5, 400), dtype=original_arr.dtype)  # 填充4行0
    new_arr = np.concatenate([
        original_arr[:2, :],  # 取前2行（索引0、1）
        insert_rows,  # 插入4行0
        original_arr[2:, :]  # 取剩余行（索引2、3）
    ], axis=0)

    # 步骤2：设置第一行为0，最后一行为640
    new_arr[0, :] = 0  # 第一行全部设为0
    new_arr[-1, :] = 640  # 最后一行全部设为640

    # 步骤3：舍四进五取整（0.4舍，0.5进，等价于np.round）
    # np.round默认就是舍四进五，例如：2.4→2，2.5→3，-1.4→-1，-1.5→-2
    new_arr_int = np.round(new_arr).astype(np.int32)

    return new_arr_int
def process_masked_tensor(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    极简掩码处理模块：输入(1,8,400)的Tensor，返回掩码后的(1,8,400)Tensor
    掩码规则：
    1. 第3、4、5、6行（索引2、3、4、5）整行全部赋值-2（无列区间限制）；
    2. 原range1/range2列区间内，第7行（索引6）赋值-2（保留原列区间逻辑）；
    3. 原range1_end~range2_start列区间内，第7行（索引6）赋值-2（保留原列区间逻辑）。
    （若需第7行也整行赋值-2，可删除随机区间逻辑，直接 arr_2d[6, :] = -2）
    Args:
        input_tensor: 输入Tensor，维度必须为(1,8,400)
    Returns:
        masked_tensor: 处理后的(1,8,400)Tensor
    """
    # 校验输入维度
    assert input_tensor.shape == (1, 8, 400), f"输入维度必须为(1,8,400)，当前为{input_tensor.shape}"

    # 复制Tensor避免修改原数据，降维为8×400便于操作
    masked_tensor = input_tensor.clone()
    arr_2d = masked_tensor.squeeze(0)

    # 随机生成中间段长度和起点（仅用于第7行的列掩码区间计算）
    middle_length = np.random.randint(low=100, high=241)
    max_start = 400 - middle_length
    start_point = np.random.randint(low=0, high=max_start + 1)

    # 计算第7行的掩码列区间（仅用于第7行）


    # 固定选择第7行（索引6）
    rand_row = 6  # 6=第7行

    # 核心修改：第3、4、5、6行（索引2-5）整行赋值-2（无列区间限制）
    arr_2d[2:7, :] = -2  # ":" 表示所有列，整行赋值

    # 第7行：仅在原列区间内赋值-2（若需整行赋值，替换为 arr_2d[6, :] = -2）

    # 恢复(1,8,400)维度并返回
    masked_tensor = arr_2d.unsqueeze(0)
    return masked_tensor
class test_cond_Dataset(Dataset):
    def __init__(self, cond_list,img_root,pathes):
        self.cond_list = cond_list
        self.img_root = img_root
        self.pathes = pathes
    def __getitem__(self, index):
        a = os.path.join(self.img_root, self.pathes[index])
        img_mask = np.load(a)

        img_mask = process_array(img_mask)
        path =  self.cond_list[index]['path']
        class_cond = self.cond_list[index]['class']
        shift_cond = self.cond_list[index]['rise_range']

        class_cond = torch.tensor(class_cond).long()
        shift_cond = torch.tensor(shift_cond).float()
        img_mask = torch.from_numpy(img_mask).float()
        img_mask = img_mask.unsqueeze(0)
        img_mask /= 640
        img_mask -= 0.5
        img_mask /= 0.5  # 值的范围从-1到1
        img_mask = process_masked_tensor(img_mask)
        img_mask_2 = img_mask.numpy()
        return {'path':path, 'class_cond':class_cond, 'shift_cond':shift_cond, 'mask':img_mask}

    def __len__(self):
        return len(self.cond_list)