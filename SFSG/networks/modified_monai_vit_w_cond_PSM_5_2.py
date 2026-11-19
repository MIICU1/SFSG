
from typing import Sequence, Union
import numpy as np
import torch
import torch.nn as nn

# from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep
from ldm.modules.diffusionmodules.util import timestep_embedding, linear

__all__ = ["ViTAutoEnc"]


from monai.networks.layers import Conv, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}

# Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4, pos_embed="conv")

    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int,
        num_heads: int,
        pos_embed: str,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = look_up_option(pos_embed, SUPPORTED_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # print(x.shape)
        x = self.patch_embeddings(x)
        # print(x.shape)
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        # print(x.shape, self.position_embeddings.shape)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

    def local_forward(self, x, subseg_positions):
        x = self.patch_embeddings(x)
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        # start_pos = torch.randint(0, 400-local_len+1, (1,), device=x.device)
        # print(x.shape, self.position_embeddings[:, start_pos:start_pos+local_len, :].shape)
        pos_embed_list = []
        for (s, e) in subseg_positions:
            # 截取当前子段在原始图像中对应的位置嵌入（s到e）
            # self.position_embeddings形状假设为 [1, total_patches, embed_dim]
            # 其中total_patches对应原始图像400长度的patch数量
            pos_embed_sub = self.position_embeddings[:, s:e, :]  # 子段的位置嵌入
            pos_embed_list.append(pos_embed_sub)

        # 拼接3个子段的位置嵌入，与输入x的patch嵌入长度匹配
        pos_embeddings = torch.cat(pos_embed_list, dim=1)  # 形状 [1, num_patches, embed_dim]

        # 3. 相加并返回（与原逻辑一致）
        embeddings = x + pos_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class ViTAutoEnc(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Modified to also give same dimension outputs as the input size of the image
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        out_channels: int = 1,
        deconv_chns: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        num_classes: int = 7,
        fusion_mode: str = "concat"  # 掩码融合方式：add（相加）/ concat（拼接）/ gate（门控）
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            out_channels: number of output channels.
            deconv_chns: number of channels for the deconvolution layers.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            # It will provide an output of same size as that of the input
            >>> net = ViTAutoEnc(in_channels=1, patch_size=(16,16,16), img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), output will be same size as of input
            >>> net = ViTAutoEnc(in_channels=3, patch_size=(16,16,16), img_size=(128,128,128), pos_embed='conv')

        """

        super().__init__()

        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.spatial_dims = spatial_dims
        self.fusion_mode = fusion_mode

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )
        if fusion_mode == "gate":
            # 门控融合：学习掩码对图像Patch的权重
            self.gate = nn.Sequential(
                linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        elif fusion_mode == "concat":
            # 拼接后降维回768（保证后续Transformer输入维度不变）
            self.concat_proj = linear(hidden_size * 2, hidden_size)

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

        new_patch_size = [4] * self.spatial_dims
        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        # self.conv3d_transpose* is to be compatible with existing 3d model weights.
        self.conv3d_transpose = conv_trans(hidden_size, deconv_chns, kernel_size=patch_size, stride=patch_size)
        self.conv3d_transpose_1 = conv_trans(in_channels=deconv_chns, out_channels=out_channels, kernel_size=1, stride=1)

        self.model_channels = hidden_size

        self.embed_dim = self.model_channels//4
        self.time_embed = nn.Sequential(
            linear(self.embed_dim,  self.model_channels),
            nn.SiLU(),
            linear( self.model_channels, self.model_channels),
        )
        self.class_embed = nn.Embedding(num_classes, self.model_channels)
        self.shift_embed = nn.Sequential(
            linear(1, self.embed_dim),
            nn.SiLU(),
            linear(self.embed_dim, self.model_channels),
        )
        # self.shift_emb = nn.Embedding(1, self.model_channels)

    def forward(self, x, timesteps, class_cond, mask_prompt):
        """
        Args:
            x: 原始图像 [batch_size, 1, 8, 400]
            timesteps: 时间步 [batch_size]
            class_cond: 类别条件 [batch_size]
            shift_cond: 偏移条件 [batch_size]
            mask_prompt: 掩码提示 [batch_size, 1, 8, 400]（和图像同维度）
        """
        # ========== 第一步：图像和掩码提示都做Patch嵌入 ==========
        # 图像嵌入：[16,1,8,400] → [16,400,768]
        x_embed = self.patch_embedding(x)
        # 掩码提示嵌入：[16,1,8,400] → [16,400,768]
        mask_embed = self.patch_embedding(mask_prompt)

        # ========== 第二步：融合图像嵌入和掩码嵌入 ==========

        if self.fusion_mode == "concat":
            fused_embed = torch.cat([x_embed, mask_embed], dim=-1)
            fused_embed = self.concat_proj(fused_embed)
        elif self.fusion_mode == "gate":
            concat_feat = torch.cat([x_embed, mask_embed], dim=-1)
            gate_weight = self.gate(concat_feat)  # [16,400,768]
            fused_embed = x_embed * gate_weight + mask_embed * (1 - gate_weight)
        else:
            raise ValueError(f"不支持的融合方式：{self.fusion_mode}")

        # ========== 第三步：处理原有time/class/shift嵌入 ==========
        t_emb = timestep_embedding(timesteps, self.embed_dim, repeat_only=False)
        t_emb = self.time_embed(t_emb)  # [16,768]


        c_emb = self.class_embed(class_cond)  # [16] → [16,768]


        # 扩维为[16,1,768]，适配序列拼接
        t_emb = t_emb.unsqueeze(1)
        c_emb = c_emb.unsqueeze(1)
        spatial_size = x.shape[2:]
        # ========== 第四步：拼接所有条件嵌入 + 融合后的Patch嵌入 ==========
        # 拼接后维度：[16, 1(t)+1(c)+1(s)+400(fused), 768] → [16,403,768]（和你原代码一致）
        x = torch.cat((t_emb, c_emb, fused_embed), dim=1)
        patch_num = fused_embed.shape[1]
        # print(x.shape)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = x[:, -patch_num:, :]
        x = self.norm(x)
        x = x.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        # print(x.shape)

        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)
        return x

    def local_forward(self, x, timesteps, class_cond, shift_cond, pos_start):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        t_emb = timestep_embedding(timesteps, self.embed_dim, repeat_only=False)
        t_emb = self.time_embed(t_emb)

        shift_cond = shift_cond.unsqueeze(1)
        c_emb = self.class_embed(class_cond)
        s_emb = self.shift_embed(shift_cond)

        t_emb = t_emb.unsqueeze(1)
        c_emb = c_emb.unsqueeze(1)
        s_emb = s_emb.unsqueeze(1)

        spatial_size = x.shape[2:]
        x = self.patch_embedding.local_forward(x, pos_start)
        patch_num = x.shape[1]
        # print(x.shape, t_emb.shape, c_emb.shape, s_emb.shape)
        x = torch.cat((t_emb, c_emb, s_emb, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = x[:, -patch_num:, :]
        x = self.norm(x)
        x = x.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        # print(x.shape)

        x = self.conv3d_transpose(x) #（16，16，8，74）
        x = self.conv3d_transpose_1(x) #（16，1，8，74）
        return x