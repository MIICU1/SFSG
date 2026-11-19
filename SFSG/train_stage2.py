import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from datamodule.layergen_train_datamodule_5 import trainDatamodule
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from ldm.modules.ema import LitEma
from ldm.util import default
from ldm.models.diffusion.ddpm import DDPM_base
# from utils.layercode2mask import layer_code2img
from networks.modified_monai_vit_w_cond_PSM_5_2 import ViTAutoEnc
import cv2 as cv
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='OCTA_6mm_1') # save name
    parser.add_argument('--result_root', type=str, default='runs/stage2') # path/to/results
    parser.add_argument('--data_config',default=None) # path/to/config
    parser.add_argument('--first_stage_ckpt', default=None)  # path/to/pretrained/checkpoint
    parser.add_argument("--command", default="fit")
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument("--max_epochs", type=int, default=800)
    parser.add_argument("--limit_train_batches", type=int, default=10000)# 10000
    parser.add_argument("--base_learning_rate", type=float, default=4.5e-6)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--scale_lr', type=bool, default=True)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--reproduce', type=int, default=False)
    parser.add_argument('--train_data_root', required=True, help='Directory containing the training inputs.')
    parser.add_argument('--train_json_path', default='jsons/oct6mm_val/train_pathes.json')
    parser.add_argument('--train_cond_json_path', required=True, help='Condition JSON for the training samples.')
    return parser


def main(opts, data_cfg):
    opts.default_root_dir = opts.result_root
    datamodule = trainDatamodule(opts, train_data_root=opts.train_data_root,
                                 train_json_path=opts.train_json_path,
                                 train_cond_json_path=opts.train_cond_json_path)  # 数据处理
    model = DDPM(opts)
    if opts.command == "fit":
        ckpt_callback = ModelCheckpoint(save_last=True, filename="model-{epoch}")
        trainer = pl.Trainer.from_argparse_args(opts, callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=50)])
        # ckpt_path = opts.first_stage_ckpt  # 预训练权重的位置
        # model.init_from_ckpt(ckpt_path)  # 加载权重
        trainer.fit(model=model, datamodule=datamodule)

def layer_code2img(code):
    """
    将网络输出的特征图转换为可视化图像

    Args:
        code: 输入特征图，形状为 (H, W) 或 (C, H, W)

    Returns:
        img: 可视化图像，形状为 (H, W) 或 (H, W, 3)
    """
    # 如果是多通道，取第一个通道
    if len(code.shape) == 3:
        code = code[0]

    # 确保是numpy数组
    if not isinstance(code, np.ndarray):
        code = code.detach().cpu().numpy()

    # 归一化到0-255范围
    code = code.astype(np.float32)
    min_val = np.min(code)
    max_val = np.max(code)

    if max_val > min_val:
        code = (code - min_val) / (max_val - min_val) * 255.0
    else:
        code = np.zeros_like(code)

    # 转换为8位无符号整数
    img = code.astype(np.uint8)

    # 如果是单通道，转换为三通道用于显示
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    return img


class DDPM(DDPM_base):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.save_hyperparameters()
        self.model = ViTAutoEnc(in_channels=1, patch_size=(8, 1), img_size=(8, 400), spatial_dims=2)
        self.batch_size=16
        self.image_size = (8, 400)
        self.channels = 1

        self.parameterization = "eps"  # all assuming fixed variance schedules
        self.loss_type = "l2"
        self.use_ema = True
        self.use_positional_encodings = False
        self.v_posterior = 0.  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        self.register_schedule()

        # self.local_len = 32

        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def q_sample(self, x_start, t, noise=None):#4 加噪
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t, cond):  # 3

        noise = torch.randn_like(x_start)
        x_noisy_local = self.q_sample(x_start=x_start, t=t, noise=noise)  # 加噪到时间步t

        model_out = self.apply_model(x_noisy_local, t, cond)

        if self.parameterization == "eps": #预测噪声
            # target = noise
            target = noise
        elif self.parameterization == "x0": #直接预测x0
            # target = x_start
            target =x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        # loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'
        # loss_simple = loss.mean() * self.l_simple_weight
        # loss_vlb = (self.lvlb_weights[t] * loss).mean()

        loss = loss.mean()
        loss_dict = {}
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        # loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})
        loss_dict.update({f'{log_prefix}/loss': loss})
        return loss, loss_dict

    def forward(self, x, cond): #2
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, cond)

    def apply_model(self, x, t, cond): #5
        class_cond, shift_cond, mask = cond
        return self.model(x=x, timesteps=t, class_cond=class_cond, mask_prompt = mask)

    def apply_model_local(self, x, t, cond, pos_start): #5
        class_cond, shift_cond = cond
        return self.model.local_forward(x=x, timesteps=t, class_cond=class_cond, shift_cond=shift_cond, pos_start=pos_start)

    def training_step(self, batch, batch_idx): #1
        x = batch['image']
        class_cond = batch['class_cond']
        shift_cond = batch['shift_cond']
        mask = batch['mask']
        # print(x.shape, class_cond.shape, shift_cond.shape)
        loss, loss_dict = self(x, [class_cond, shift_cond,mask])

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True,batch_size=self.batch_size)
        if batch_idx == 0:
            self.batch_sample = batch
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def on_train_epoch_end(self):
        # print(-1)
        with self.ema_scope("Plotting"): #使用ema模型进行推理，可以提高生成图像的质量和稳定性
            img_save_dir = os.path.join(self.opts.result_root, 'val_results', str(self.current_epoch))
            os.makedirs(img_save_dir, exist_ok=True)
            x = self.batch_sample['image'][:8] #8,1,8,400

            class_cond = self.batch_sample['class_cond'][:8]
            shift_cond = self.batch_sample['shift_cond'][:8]
            mask = self.batch_sample['mask'][:8]
            samples = self.sample(cond=[class_cond, shift_cond, mask], batch_size=x.shape[0], #先sample 再p_sample_loop 再p_sample。或者是再p_sample_loop_local 再p_sample_local。
                                               return_intermediates=False, clip_denoised=True)
            # decoding
            x = x.to('cpu').numpy()
            samples = samples.to('cpu').numpy()
            for i in range(x.shape[0]):
                visual = []
                visual.append(layer_code2img(x[i, 0, :, :]))
                visual.append(layer_code2img(samples[i, 0, :, :]))
                visual = cv.hconcat(visual)
                save_name = str(i) + '_' + str(class_cond[i].item()) + '_' + str(shift_cond[i].item()) + '.png'
                # print(save_name)
                cv.imwrite(os.path.join(img_save_dir, save_name), visual)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised):
        model_out = self.apply_model(x, t, cond)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        # no noise when t == 0
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, return_intermediates=False, log_every_t=100, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        if return_intermediates:
            intermediates = [img]
        with tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps,
                  mininterval=5) as pbar:
            for i in pbar:
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond,
                                    clip_denoised=clip_denoised)
                if return_intermediates and (i % log_every_t == 0 or i == self.num_timesteps - 1):
                    intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=1, return_intermediates=False, clip_denoised=True): #clip_denoised的含义是去噪过程可能会生成超过-1或1的值，会影响结果，这个参数会把超过的值变成-1或1
        return self.p_sample_loop([batch_size, self.channels] + list(self.image_size), cond,
                                  return_intermediates=return_intermediates, clip_denoised=clip_denoised)
    def p_mean_variance_local(self, x, t, cond, pos_start, clip_denoised):
        model_out = self.apply_model_local(x, t, cond, pos_start)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_local(self, x, t, cond, pos_start, clip_denoised, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance_local(x=x, t=t, cond=cond, pos_start=pos_start, clip_denoised=clip_denoised)
        # no noise when t == 0
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    @torch.no_grad()
    def p_sample_loop_local(self, shape, cond, pos_start, return_intermediates=False, log_every_t=100, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        if return_intermediates:
            intermediates = [img]
        with tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps,
                  mininterval=5) as pbar:
            for i in pbar:
                img = self.p_sample_local(img, torch.full((b,), i, device=device, dtype=torch.long), cond, pos_start,
                                    clip_denoised=clip_denoised)
                if return_intermediates and (i % log_every_t == 0 or i == self.num_timesteps - 1):
                    intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample_local(self, cond, local_len, pos_start, batch_size=1, return_intermediates=False, clip_denoised=True):
        return self.p_sample_loop_local([batch_size, self.channels, self.image_size[0], local_len], cond, pos_start,
                                  return_intermediates=return_intermediates, clip_denoised=clip_denoised)

    def configure_optimizers(self):
        lr = self.opts.base_learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    data_cfg = OmegaConf.load(opts.data_config) if opts.data_config else OmegaConf.create({"params": {}})
    main(opts, data_cfg)
