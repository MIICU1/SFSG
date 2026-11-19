# -*- coding:utf-8 -*-
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torchvision.utils import save_image

from base.init_experiment import initExperiment
from base.metrics_calculation import cal_metrics_monai_v3
from networks.model_acrrossnet.unet import UNet
from utils.util import instantiate_from_config
from utils.util_for_metric import save_metrics_to_json_v2, get_mean_metrics

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='unet_real')
    parser.add_argument('--result_root', type=str, default='runs/unet')
    parser.add_argument('--data_config', default='configs/layersegseg/octa300.yaml')
    parser.add_argument("--command", default="fit")
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--limit_train_batches", type=int, default=4000)
    parser.add_argument("--base_learning_rate", type=float, default=4.5e-6)
    parser.add_argument('--accumulate_grad_batches', type=int, default=8)
    parser.add_argument('--scale_lr', type=bool, default=True)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--reproduce', type=int, default=False)
    return parser


def main(opts, cfg):
    datamodule = instantiate_from_config(cfg.data)
    model = SegSP(opts)
    if opts.command == "fit":
        ckpt_callback = ModelCheckpoint(save_last=True, save_top_k=4, monitor='dice', mode='max',
                                        filename="model-{epoch}-{dice:.3f}")
        trainer = pl.Trainer.from_argparse_args(opts, callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=50)])
        trainer.fit(model=model, datamodule=datamodule)


class SegSP(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.n_classes = 7
        self.segmenter = UNet(n_channels=1, n_classes=self.n_classes)

        self.seg_loss = DiceCELoss(lambda_dice=0.5, lambda_ce=0.5, softmax=True, to_onehot_y=True)
        self.save_hyperparameters()

    def forward(self, x):
        return self.segmenter(x)

    def training_step(self, batch, batch_idx):
        img = batch['image']
        label = batch['label']

        pred = self.segmenter(img)
        loss_seg = self.seg_loss(pred, label)

        self.log('train_loss/seg', loss_seg)
        return loss_seg

    def on_validation_epoch_start(self):
        self.eval_metrics_list = []
        self.visual_images = []

    def validation_step(self, batch, batch_idx):
        image, label = batch['image'], batch['label']
        path = batch['path'][0]

        pred = self(image)
        prb_out = torch.softmax(pred, dim=1)
        bin_out = torch.argmax(prb_out, dim=1, keepdim=True)

        cal_metrics_monai_v3(bin_out, label,self.n_classes, path, eval_metrics_list=self.eval_metrics_list, suffix='val')
        if batch_idx == 0:
            visuals = torch.cat(
                [image * 0.5 + 0.5, label.float() / (self.n_classes - 1), bin_out.float() / (self.n_classes - 1)],
                dim=0).to('cpu')
            save_name = path.split('/')
            save_name = '_'.join(save_name)
            img_save_dir = os.path.join(self.opts.default_root_dir, 'val_visual', str(self.current_epoch))
            os.makedirs(img_save_dir, exist_ok=True)
            save_image(visuals, os.path.join(img_save_dir, save_name))

    def validation_epoch_end(self, outputs):
        save_metrics_to_json_v2(self.eval_metrics_list, self.current_epoch, self.opts.default_root_dir)
        # wandb log metrics
        eval_metrics = get_mean_metrics(self.eval_metrics_list)
        self.logger.log_metrics(eval_metrics, self.current_epoch)
        self.log('dice', eval_metrics['val/dice'], on_epoch=True)

    def configure_optimizers(self):
        lr = self.opts.learning_rate
        params = list(self.segmenter.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=self.device)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    cfg = OmegaConf.load(opts.data_config)
    initExperiment(opts, cfg)
    main(opts, cfg)
