# -*- coding:utf-8 -*-
import os
import cv2 as cv
from argparse import ArgumentParser
import pytorch_lightning as pl
import numpy as np
from ldm.models.diffusion.ddim import DDIMSampler

from base.init_experiment import initExperiment
from train_stage1 import DDPM
from networks.modified_monai_vit_w_cond_one import ViTAutoEnc as LegacyViTAutoEnc

from datamodule.layergen_test_datamodule_one import test_layergen_cond_Datamodule
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--result_save_dir", type=str, default='runs/stage1_inference') # path/to/generated/label
    parser.add_argument('--ckpt_path', type=str, required=True) # path/to/TDPM/checkpoint
    parser.add_argument('--test_json_path', type=str, default='jsons/gene/gene_40000.json')
    parser.add_argument('--use_ddim', type=bool, default=True)
    parser.add_argument('--ddim_steps', type=int, default=100)
    # lightning args
    parser.add_argument("--command", default="test")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--reproduce', default=False)
    parser.add_argument('--logger', default=False)
    parser.add_argument('--architecture', choices=('legacy', 'stage1'), default='legacy',
                        help='Select the architecture used to train the checkpoint.')
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
    datamodule = test_layergen_cond_Datamodule(cond_json_path=opts.test_json_path, batch_size=opts.batch_size,
                                               num_workers=opts.num_workers)
    # exit()
    model_class = LegacyDDPMTest if opts.architecture == 'legacy' else DDPM_test
    model = model_class(opts)
    trainer = pl.Trainer.from_argparse_args(opts)
    trainer.test(model=model, datamodule=datamodule)


class DDPM_test(DDPM):
    def __init__(self, opts):
        super().__init__(opts)
        self.init_from_ckpt(opts.ckpt_path)
        os.makedirs(opts.result_save_dir, exist_ok=True)

    def test_step(self, batch, batch_idx):
        pathes = batch['path']
        copied_pathes = []
        # for path in pathes:
        #     # 分离文件名和扩展名（处理 .png 等后缀）
        #     import os
        #     dir_name, file_name = os.path.split(path)
        #     base_name, ext = os.path.splitext(file_name)  # base_name是文件名主体，ext是扩展名（如.png）
        #     # 拼接新文件名：原主体 + _2 + 扩展名
        #     new_file_name = f"{base_name}_2{ext}"
        #     # 组合新路径
        #     new_path = os.path.join(dir_name, new_file_name)
        #     copied_pathes.append(new_path)
        #
        # # 拼接原始列表和修改后的复制列表，形成16个元素
        # pathes_16 = pathes + copied_pathes


        class_cond = batch['class_cond']
        shift_cond = batch['shift_cond']
        cond = [class_cond, shift_cond]

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
            img = layer_code2img(samples[i, 0, :, :])  # 获取特征图
            if '/' in pathes[i]:
                names = pathes[i].split('/')
                save_dir = '/'.join([self.opts.result_save_dir] + names[:-1])
                os.makedirs(save_dir, exist_ok=True)
                print(save_dir)
            save_path = os.path.join(self.opts.result_save_dir, pathes[i][:-4])  # 保持原路径
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


class LegacyDDPMTest(DDPM_test):
    backbone = LegacyViTAutoEnc


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    initExperiment(opts)
    main(opts)
