# -*- coding:utf-8 -*-
import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from argparse import ArgumentParser
import pytorch_lightning as pl
import numpy as np
from ldm.models.diffusion.ddim import DDIMSampler
from base.init_experiment import initExperiment
from train_stage2 import DDPM
from utils.conditions import index_conditions, path_key
from datamodule.layergen_test_datamodule_5 import process_array, process_masked_tensor

class Train210Dataset(Dataset):
    def __init__(self, data_root, paths_json, conditions_json):
        self.root = Path(data_root)
        with open(paths_json, encoding='utf-8-sig') as handle:
            self.paths = json.load(handle)
        with open(conditions_json, encoding='utf-8-sig') as handle:
            conditions = json.load(handle)
        if not isinstance(self.paths, list) or not all(isinstance(p, str) for p in self.paths):
            raise ValueError('Path JSON must be a list of relative path strings.')
        self.conditions = index_conditions(conditions)
        for name in self.paths:
            key = path_key(name)
            if key not in self.conditions:
                raise KeyError('Condition missing for: ' + name)
            category = self.conditions[key]['class']
            if not isinstance(category, int) or not 0 <= category <= 6:
                raise ValueError('Invalid class for: ' + name)
        print('Matched input slices:', len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        name = self.paths[index]
        key = path_key(name)
        array = np.load(self.root / (key + '.npy'), allow_pickle=False)
        if array.shape == (3, 400):
            array = process_array(array)
        elif array.shape != (8, 400):
            raise ValueError(name + ': expected (3,400) or (8,400), got ' + str(array.shape))
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite boundary coordinates: ' + name)
        mask = torch.from_numpy(array.copy()).float().unsqueeze(0)
        mask = (mask / 640.0 - 0.5) / 0.5
        mask = process_masked_tensor(mask)
        category = self.conditions[key]['class']
        return {'path': key + '.npy',
                'class_cond': torch.tensor(category, dtype=torch.long), 'mask': mask}


class Train210Datamodule(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

    def setup(self, stage=None):
        self.test_set = Train210Dataset(self.opts.test_data_root, self.opts.test_json_path,
                                       self.opts.test_json_cond_path)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.opts.batch_size, shuffle=False,
                          num_workers=self.opts.num_workers)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--result_save_dir", type=str, default='runs/stage2_inference') # path/to/generated/label
    parser.add_argument('--ckpt_path', type=str, required=True)  # path/to/TDPM/checkpoint
    parser.add_argument('--test_json_cond_path', type=str, default='jsons/rise_5_train210.json')
    parser.add_argument('--test_data_root', required=True)
    parser.add_argument('--test_json_path', default='jsons/oct6mm_val/train_pathes.json')
    parser.add_argument('--use_ddim', type=bool, default=True)
    parser.add_argument('--ddim_steps', type=int, default=100)
    # lightning args
    parser.add_argument("--command", default="test")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--reproduce', default=False)
    parser.add_argument('--logger', default=False)
    return parser

def layer_code2img(code):
    """
    将网络输出的特征图保存为 .npy 文件
    Args:
        code: 输入特征图，形状为 (H, W) 或 (C, H, W)
        filename: 保存的文件名 (例如 "feature_map.npy")
    """
    # 如果是多通道，取第一个通道
    if len(code.shape) == 3:
        code = code[0]

    # 确保是numpy数组
    if not isinstance(code, np.ndarray):
        code = code.detach().cpu().numpy()

    # 直接保存为.npy文件，不进行归一化或转换为图像
    return code

def main(opts):
    datamodule = Train210Datamodule(opts)
    # exit()
    model = DDPM_test(opts)
    trainer = pl.Trainer.from_argparse_args(opts)
    trainer.test(model=model, datamodule=datamodule)


class DDPM_test(DDPM):
    def __init__(self, opts):
        super().__init__(opts)
        self.init_from_ckpt(opts.ckpt_path)
        os.makedirs(opts.result_save_dir, exist_ok=True)

    def test_step(self, batch, batch_idx):
        pathes = batch['path']
        class_cond = batch['class_cond']
        mask = batch['mask']
        # The Stage 2 model uses the class and mask in its global denoiser.
        cond = [class_cond, torch.zeros_like(class_cond, dtype=torch.float32), mask]

        if os.path.exists(os.path.join(self.opts.result_save_dir, pathes[-1])): return
        if self.opts.use_ddim:
            samples = self.ddim_sample(cond=cond, batch_size=len(pathes))
        else:
            samples = self.sample(cond=cond, batch_size=len(pathes), return_intermediates=False, clip_denoised=True)
        samples = samples * 0.5 + 0.5
        samples = samples.to('cpu').numpy() * 640
        # for i in range(len(pathes_16)):
        #     img = layer_code2img(samples[i, 0, :, :])
        #     if '/' in pathes_16[i]:
        #         names = pathes_16[i].split('/')
        #         save_dir = '/'.join([self.opts.result_save_dir] + names[:-1])
        #         os.makedirs(save_dir, exist_ok=True)
        #         print(save_dir)
        #     cv.imwrite(os.path.join(self.opts.result_save_dir, pathes_16[i]), img)
        for i in range(len(pathes)):
            img = layer_code2img(samples[i, 0, :, :])
            current_path = pathes[i]

            # 1. 获取当前样本真实的 rise 和 range（直接用原值，不四舍五入）
            dir_name, file_name = os.path.split(current_path)
            base_name, ext = os.path.splitext(file_name)  # 去掉 .png
            #
            # # 3. 分割名字，例如 RVO_0.72_0.555 → ["RVO", "0.72", "0.555"]
            # category = base_name.split("_")[0]
            # new_base = f"{category}_{rise_str}_{range_str}"

            # 5. 拼接最终路径（强制 .npy）
            final_name = base_name + ".npy"
            save_path = os.path.join(self.opts.result_save_dir, dir_name, final_name)

            # 6. 创建目录
            save_dir = os.path.join(self.opts.result_save_dir, dir_name)
            os.makedirs(save_dir, exist_ok=True)

            # 7. 保存
            np.save(save_path, img)
    def ddim_sample(self, cond, batch_size, return_intermediates=False):
        ddim_sampler = DDIMSampler(self)
        shape = [self.channels] + list(self.image_size)
        # samples, intermediates = ddim_sampler.sample(self.opts.ddim_steps, batch_size,
        #                                              shape, cond, verbose=False)
        samples= ddim_sampler.sample(self.opts.ddim_steps, batch_size,
                                                     shape, cond, verbose=False)
        # if return_intermediates:
        #     return samples, intermediates
        # else:
        #     return samples
        return samples


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    initExperiment(opts)
    main(opts)
