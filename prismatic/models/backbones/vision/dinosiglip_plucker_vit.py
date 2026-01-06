"""
dinosiglip_vit.py

Vision backbone that returns concatenated features from both DINOv2 and SigLIP.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import timm
import torch
from PIL import Image
from timm.models.vision_transformer import Block, VisionTransformer

from torch import nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.base_vision import (
    ImageTransform,
    LetterboxPad,
    VisionBackbone,
    compute_sequence_patches,
    unpack_tuple,
)


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone module for injecting Plucker Embeddings.
"""
class FrozenBatchNorm2d(torch.nn.Module):

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias



# Registry =>> Supported DinoSigLIP Pairs (as TIMM identifiers)
DINOSigLIP_VISION_BACKBONES = {
    "dinosiglip-plucker-vit-so-224px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_224",
    }
}

    # "dinosiglip-vit-so-384px": {
    #     "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
    #     "siglip": "vit_so400m_patch14_siglip_384",
    # },


@dataclass
class DinoSigLIPImageTransform:
    dino_image_transform: ImageTransform
    siglip_image_transform: ImageTransform
    is_prismatic: bool = True

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {"dino": self.dino_image_transform(img, **kwargs), "siglip": self.siglip_image_transform(img, **kwargs)}


class DinoSigLIPPluckerViTBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        image_sequence_len: int = 1,
    ) -> None:
        super().__init__(
            vision_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
            image_sequence_len=image_sequence_len,
        )
        self.dino_timm_path_or_url = DINOSigLIP_VISION_BACKBONES[vision_backbone_id]["dino"]
        self.siglip_timm_path_or_url = DINOSigLIP_VISION_BACKBONES[vision_backbone_id]["siglip"]

        # Initialize both Featurizers (ViTs) by downloading from HF / TIMM Hub if necessary
        self.dino_featurizer: VisionTransformer = timm.create_model(
            self.dino_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.dino_featurizer.eval()
        self.dino_featurizer.requires_grad_(False)

        self.siglip_featurizer: VisionTransformer = timm.create_model(
            self.siglip_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.siglip_featurizer.eval()
        self.siglip_featurizer.requires_grad_(False)

        # Monkey-Patch the `forward()` function of the featurizers to ensure FSDP-compatibility
        #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        self.dino_featurizer.forward = unpack_tuple(
            partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - 2})
        )
        self.siglip_featurizer.forward = unpack_tuple(
            partial(self.siglip_featurizer.get_intermediate_layers, n={len(self.siglip_featurizer.blocks) - 2})
        )

        # Get Configs for _both_ Featurizers =>> Note :: Override default image size for larger resolution models
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        self.siglip_data_cfg = timm.data.resolve_model_data_config(self.siglip_featurizer)
        self.siglip_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize *both* Transforms
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        default_siglip_transform = timm.data.create_transform(**self.siglip_data_cfg, is_training=False)

        # Fix =>> SigLIP default transform resizes to *larger* than `self.default_image_size` (crops image)!!
        assert isinstance(default_siglip_transform, Compose), "Unexpected `default_image_transform`!"
        assert isinstance(default_siglip_transform.transforms[0], Resize)
        default_siglip_transform = Compose(
            [
                Resize(self.default_image_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                *default_siglip_transform.transforms[1:],
            ]
        )

        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_image_transform`!"
            assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_image_transform`!"
            assert isinstance(default_dino_transform.transforms[0], Resize)
            assert isinstance(default_siglip_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            dino_transform = Compose(
                [
                    Resize(target_size, interpolation=default_dino_transform.transforms[0].interpolation),
                    *default_dino_transform.transforms[1:],
                ]
            )
            siglip_transform = Compose(
                [
                    Resize(target_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                    *default_siglip_transform.transforms[1:],
                ]
            )

            self.image_transform = DinoSigLIPImageTransform(dino_transform, siglip_transform)

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = DinoSigLIPImageTransform(default_dino_transform, default_siglip_transform)

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_transform`!"
            assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_transform`!"
            assert (
                "mean" in self.dino_data_cfg and "mean" in self.siglip_data_cfg
            ), "DinoSigLIP `data_cfg` missing `mean`!"

            # Compute Padding Fill Value(s) (rescaled normalization mean if applicable)
            dino_fill = tuple([int(x * 255) for x in self.dino_data_cfg["mean"]])
            siglip_fill = tuple([int(x * 255) for x in self.siglip_data_cfg["mean"]])

            # Build New Transform
            self.image_transform = DinoSigLIPImageTransform(
                Compose([LetterboxPad(dino_fill), *default_dino_transform.transforms]),
                Compose([LetterboxPad(siglip_fill), *default_siglip_transform.transforms]),
            )

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

        # PLUCKER
        self._init_plucker()

    def _init_plucker(self):
        # Plucker fusion modules (before connector)
        # Encode 6-channel Plücker map to a 512-d feature grid
        self.plucker_encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FrozenBatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            FrozenBatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            FrozenBatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            FrozenBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            FrozenBatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Project to vision hidden size and fuse with SigLIP tokens
        vision_hidden = self.embed_dim
        # vision_hidden = int(self.siglip_featurizer.embed_dim)
        self.plucker_out_proj = nn.Conv2d(512, vision_hidden, kernel_size=1)
        self.vision_fusion_proj = nn.Linear(vision_hidden * 2, vision_hidden)
        # Normalize streams and align dtypes with vision encoder
        self.vision_ln = nn.LayerNorm(vision_hidden, elementwise_affine=False)
        self.plucker_ln = nn.LayerNorm(vision_hidden, elementwise_affine=False)
        vf_dtype = next(self.siglip_featurizer.parameters()).dtype
        self.plucker_out_proj = self.plucker_out_proj.to(dtype=vf_dtype)
        self.vision_fusion_proj = self.vision_fusion_proj.to(dtype=vf_dtype)


    def train(self, mode: bool = True) -> "DinoSigLIPPluckerViTBackbone":
        """Override train to keep DINO/SigLIP frozen."""
        super().train(mode)
        # Always keep base vision encoders in eval mode
        self.dino_featurizer.eval()
        self.siglip_featurizer.eval()
        return self


    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""
        if self.image_sequence_len == 1:
            dino_patches = self.dino_featurizer(pixel_values["dino"])
            siglip_patches = self.siglip_featurizer(pixel_values["siglip"])
            image_hidden_states = torch.cat([dino_patches, siglip_patches], dim=2)

            # Determine grid size from number of tokens L = s*s
            b, l, dv = image_hidden_states.shape
            s = int(l ** 0.5)
            if s * s != l:
                raise ValueError("Non-square token grid from vision encoder; cannot align Plücker features")

            # Encode Plücker and pool to s x s grid
            p_feat = self.plucker_encoder(pixel_values["plucker"])
            p_feat = F.adaptive_avg_pool2d(p_feat, output_size=(s, s))  # [B, 512, s, s]
            p_feat = self.plucker_out_proj(p_feat)  # [B, Dv, s, s]
            plucker_patches = p_feat.flatten(2).transpose(1, 2)  # [B, L, Dv]

            # Concatenate and fuse back to Dv (normalize per token first)
            if plucker_patches.dtype != image_hidden_states.dtype:
                plucker_patches = plucker_patches.to(dtype=image_hidden_states.dtype)
            image_hidden_states = self.vision_ln(image_hidden_states)
            plucker_patches = self.plucker_ln(plucker_patches)
            fused = torch.cat([image_hidden_states, plucker_patches], dim=-1)  # [B, L, 2*Dv]
            image_hidden_states = self.vision_fusion_proj(fused)  # [B, L, Dv]


        else:
            featurizers = {
                "dino": self.dino_featurizer,
                "siglip": self.siglip_featurizer,
            }

            patches = compute_sequence_patches(pixel_values, featurizers, self.image_sequence_len)
            dino_patches, siglip_patches = patches["dino"], patches["siglip"]
            # TODO (mung3477): Forward process for plucker embeddings

            image_hidden_states = torch.cat([dino_patches, siglip_patches], dim=2)

        return image_hidden_states

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        assert self.dino_featurizer.patch_embed.num_patches == self.siglip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches * self.image_sequence_len

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
