from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import einops
import numba
import numpy as np

import torch
import torch.nn as nn

from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer
from transformers.utils import ModelOutput

from .configuration_embodiedmae import EmbodiedMAEConfig

try:
    import pytorch3d.ops as torch3d_ops
    from pytorch3d.loss import chamfer_distance
except ImportError:
    torch3d_ops = None
    chamfer_distance = None

def concat_tensor(
    tensors: List[torch.Tensor | None], dim: int = -1, **kwargs
) -> Tuple[torch.Tensor, list]:
    filtered_tensors = [t for t in tensors if t is not None]
    mask = [(1.0 if t is not None else 0.0) for t in tensors]
    return torch.cat(filtered_tensors, dim=dim, **kwargs), mask


def concat_sequence_with_dummy(
    tensors: List[torch.Tensor | None], seq_lens: List[int]
) -> torch.Tensor:
    """Concatenate a sequence of tensors. If a tensor is `None`, it will be replaced by a dummy tensor of zeros.

    Args:
        tensors (List[torch.Tensor  |  None]):
            Tensors to concatenate. If a tensor is `None`, it will be replaced by a dummy tensor of zeros.
        seq_lens (List[int]):
            Expected sequence length of each tensor.
    """
    assert len(tensors) == len(seq_lens)
    for t in tensors:
        if t is not None:
            b, d = t.shape[0], t.shape[2]
            device, dtype = t.device, t.dtype
    x = []
    for t, seq_len in zip(tensors, seq_lens):
        if t is None:
            x.append(torch.zeros((b, seq_len, d), dtype=dtype, device=device))
        else:
            x.append(t)
    return torch.cat(x, dim=1)


def patchify(
    pixel_values, patch_size, num_channels, interpolate_pos_encoding: bool = False
):
    """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            interpolation flag passed during the forward pass.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
            Patchified pixel values.
    """
    # sanity checks
    if not interpolate_pos_encoding and (
        pixel_values.shape[2] != pixel_values.shape[3]
        or pixel_values.shape[2] % patch_size != 0
    ):
        raise ValueError(
            "Make sure the pixel values have a squared size that is divisible by the patch size"
        )
    if pixel_values.shape[1] != num_channels:
        raise ValueError(
            "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
        )

    # patchify
    batch_size = pixel_values.shape[0]
    num_patches_h = pixel_values.shape[2] // patch_size
    num_patches_w = pixel_values.shape[3] // patch_size
    patchified_pixel_values = pixel_values.reshape(
        batch_size, num_channels, num_patches_h, patch_size, num_patches_w, patch_size
    )
    patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
    patchified_pixel_values = patchified_pixel_values.reshape(
        batch_size, num_patches_h * num_patches_w, patch_size**2 * num_channels
    )
    return patchified_pixel_values


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        q = self.q(x)
        q = einops.rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        kv = self.kv(context)
        kv = einops.rearrange(kv, "b t (h d) -> b h t d", h=self.num_heads)
        k, v = torch.chunk(kv, 2, dim=-1)

        attn_drop = self.attn_drop if self.training else 0.0
        x = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = einops.rearrange(x, "b h t d -> b t (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def unpatchify(patchified_pixel_values, patch_size, num_channels, original_image_size):
    """
    Args:
        patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
            Patchified pixel values.
        original_image_size (`Tuple[int, int]`, *optional*):
            Original image size.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
            Pixel values.
    """
    original_height, original_width = original_image_size
    num_patches_h = original_height // patch_size
    num_patches_w = original_width // patch_size
    # sanity check
    if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
        raise ValueError(
            f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
        )

    # unpatchify
    batch_size = patchified_pixel_values.shape[0]
    patchified_pixel_values = patchified_pixel_values.reshape(
        batch_size,
        num_patches_h,
        num_patches_w,
        patch_size,
        patch_size,
        num_channels,
    )
    patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
    pixel_values = patchified_pixel_values.reshape(
        batch_size,
        num_channels,
        num_patches_h * patch_size,
        num_patches_w * patch_size,
    )
    return pixel_values


@numba.jit(nopython=True)
def get_mm_shuffle_indices(p, embedding_sz, unmask_sz=128):
    b = p.shape[0]
    n_modals = len(embedding_sz)
    embedding_sz = np.array(embedding_sz)
    indices = np.empty((b, embedding_sz.sum()), dtype=np.int64)

    for i in numba.prange(b):
        um_sz = np.round(p[i] * unmask_sz).astype(np.int64)
        um_sz[-1] = unmask_sz - um_sz[:-1].sum()
        m_sz = embedding_sz - um_sz
        cm_um_sz = np.cumsum(um_sz)
        cm_m_sz = np.cumsum(m_sz)

        for j in range(n_modals):
            shuffle_idx = (
                np.argsort(np.random.random(embedding_sz[j])) + embedding_sz[:j].sum()
            )
            um = shuffle_idx[: um_sz[j]]
            m = shuffle_idx[um_sz[j] :]

            if j == 0:
                indices[i, : cm_um_sz[j]] = um
                indices[i, unmask_sz : cm_m_sz[j] + unmask_sz] = m
            else:
                indices[i, cm_um_sz[j - 1] : cm_um_sz[j]] = um
                indices[i, cm_m_sz[j - 1] + unmask_sz : cm_m_sz[j] + unmask_sz] = m
    return indices


def prepare_shuffle_idx(
    has_rgb: bool,
    has_depth: bool,
    has_pc: bool,
    batch_size: int,
    unmask_sz: int,
    dirichlet: torch.distributions.Dirichlet,
    embedding_sz: Tuple[int, int, int],
    # rgb: Optional[torch.Tensor],
    # depth: Optional[torch.Tensor],
    # pc: Optional[torch.Tensor],
    add_mask: bool = True,
    shuffle_idx: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = "cuda",
):
    """Prepare shuffle indices for the input embeddings.

    Args:
        rgb (Optional[torch.Tensor]):
            RGB image from [-1, 1] range, shape (B, C, H, W).
        depth (Optional[torch.Tensor]):
            Depth map from [0, 2] range, shape (B, C, H, W).
        pc (Optional[torch.Tensor]):
            Point cloud data, shape (B, N, 3), where N is the number of points.
        add_mask (bool, optional):
            Whether to add a mask for masked autoencoding. Defaults to True.
        unmask_sz (Optional[int], optional):
            Size of the unmasked tokens. If None, it will be set to self.unmask_sz. Defaults to None.
        shuffle_idx (Optional[torch.Tensor], optional):
            Shuffle indices for the input embeddings. If provided, it will be used to restore the original order.

    Returns:
        _type_: _description_
    """
    # provide at least one modality
    if not any([has_rgb, has_depth, has_pc]):
        raise ValueError("provide at least one modality")

    b = batch_size

    if add_mask:
        if shuffle_idx is not None:
            restore_idx = torch.argsort(shuffle_idx, 1)
        else:
            mask = [float(each) for each in [has_rgb, has_depth, has_pc]]
            # multi-modal shuffle
            if sum(mask) > 1:
                p = dirichlet.sample((b,)).numpy()
                p = p * np.array(mask)[None]
                p = p / p.sum(-1, keepdims=True)
                shuffle_idx = get_mm_shuffle_indices(p, embedding_sz, unmask_sz)
            # uni-modal shuffle
            else:
                shuffle_idx = get_shuffle_indices(embedding_sz[mask.index(1.0)])
            restore_idx = np.argsort(shuffle_idx, 1)
            shuffle_idx = torch.tensor(shuffle_idx, device=device)
            restore_idx = torch.tensor(restore_idx, device=device)
    else:
        # the missing modality is regarded as masked
        unmask_parts, mask_parts = [], []
        cumsum_emb_sz = np.cumsum(embedding_sz)
        for i, has_modal in enumerate([has_rgb, has_depth, has_pc]):
            indices = torch.arange(
                cumsum_emb_sz[i - 1] if i > 0 else 0,
                cumsum_emb_sz[i],
                device=device,
            )
            if has_modal:
                unmask_parts.append(indices)
            else:
                mask_parts.append(indices)
        shuffle_idx = torch.cat(unmask_parts + mask_parts, dim=0)[None].repeat(b, 1)
        restore_idx = torch.argsort(shuffle_idx, 1)
        unmask_sz = sum([len(part) for part in unmask_parts])

    return shuffle_idx, restore_idx, unmask_sz


@numba.jit(nopython=True)
def get_shuffle_indices(embedding_sz):
    shuffle_idx = np.argsort(np.random.random(embedding_sz))
    return shuffle_idx


def torch_int(x):
    import torch

    return (
        x.to(torch.int64)
        if torch.jit.is_tracing() and isinstance(x, torch.Tensor)
        else int(x)
    )


def fps_and_knn(x: torch.Tensor, num_centers: int, num_knn: int):
    dtype = x.dtype
    x = x.to(torch.float32)
    centers, _ = torch3d_ops.sample_farthest_points(
        x, K=num_centers
    )  # (b, num_centers, 3)
    knn_points = torch3d_ops.knn_points(
        centers, x, K=num_knn, return_nn=True
    ).knn  # (b, num_centers, knn, 3)
    return centers.to(dtype), knn_points.to(dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@dataclass
class EncoderModelOutput(ModelOutput):
    embedding: torch.Tensor = None
    pc_centers: torch.Tensor = None
    pc_knn: torch.Tensor = None
    shuffle_idx: torch.Tensor = None
    restore_idx: torch.Tensor = None
    last_hidden_states: Optional[torch.Tensor] = None
    add_mask: bool = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    unmask_sz: int = None


@dataclass
class DecoderInput(ModelOutput):
    rgb_embedding: torch.Tensor = None
    depth_embedding: torch.Tensor = None
    pc_embedding: torch.Tensor = None
    unmasked_emb: torch.Tensor = None
    shuffle_idx: torch.Tensor = None
    pc_centers: torch.Tensor = None
    pc_knn: torch.Tensor = None
    add_mask: bool = None
    unmask_sz: int = None


class SharedMlp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(approximate="tanh"),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MaxPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.max(self.dim)[0]


class PointGroupEmbedding(nn.Module):
    def __init__(self, point_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            SharedMlp(point_dim, 64),
            SharedMlp(64, 128),
            SharedMlp(128, 256),
            MaxPool(-2),
            nn.Linear(256, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Conv2dPatchify(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        hidden_size: int = 768,
        num_channels: int = 3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.patchify = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[-3]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.patchify(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class PatchEmbeddings(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        hidden_size: int = 768,
        num_channels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.embeddings = Conv2dPatchify(patch_size, hidden_size, num_channels)
        # Use learnable positional embeddings initialized at sin-cos
        pos_emb = get_2d_sincos_pos_embed(hidden_size, image_size // patch_size)
        pos_emb = torch.tensor(pos_emb, dtype=torch.float32)[None]
        self.position_embeddings = nn.Parameter(pos_emb)
        self.dropout = nn.Dropout(dropout)

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and interpolation at torch.float32 precision.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if (
            not torch.jit.is_tracing()
            and num_patches == num_positions
            and height == width
        ):
            return self.position_embeddings

        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return patch_pos_embed

    def forward(self, pixel_values: Optional[torch.Tensor]) -> torch.Tensor:
        if pixel_values is None:
            return None
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.embeddings.patchify.weight.dtype
        embeddings = self.embeddings(pixel_values.to(dtype=target_dtype))
        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(
            embeddings, height, width
        )
        embeddings = self.dropout(embeddings)
        return embeddings


class EmbodiedMAERGBEmbeddings(PatchEmbeddings):
    def __init__(self, config: EmbodiedMAEConfig):
        super().__init__(
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            num_channels=3,
            dropout=0.0,
        )


class EmbodiedMAEDepthEmbeddings(PatchEmbeddings):
    def __init__(self, config: EmbodiedMAEConfig):
        super().__init__(
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            num_channels=1,
            dropout=0.0,
        )


class EmbodiedMAEPointCloudEmbeddings(nn.Module):
    def __init__(self, config: EmbodiedMAEConfig):
        super().__init__()
        self.config = config
        self.num_centers, self.num_knn = config.num_pc_centers, config.num_pc_knn
        self.knn_embeddings = PointGroupEmbedding(3, config.hidden_size)
        self.center_embeddings = nn.Sequential(
            nn.Linear(3, config.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, point_cloud: Optional[torch.Tensor]) -> torch.Tensor:
        if point_cloud is None or self.config.enable_point_cloud is False:
            return None, None, None
        centers, knn_points = fps_and_knn(
            point_cloud, num_centers=self.num_centers, num_knn=self.num_knn
        )
        normed_knn_points = knn_points - centers.unsqueeze(-2)
        center_emb = self.center_embeddings(centers)
        knn_emb = self.knn_embeddings(normed_knn_points)
        return center_emb + knn_emb, centers, normed_knn_points


class EmbodiedMAEDecoder(nn.Module):
    def __init__(self, config: EmbodiedMAEConfig):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        self.config = config

        pos_emb = get_2d_sincos_pos_embed(
            config.decoder_hidden_size, image_size // patch_size
        )
        self.rgb_pos_embed = nn.Parameter(torch.tensor(pos_emb)[None])
        self.depth_pos_embed = nn.Parameter(torch.tensor(pos_emb)[None])
        self.pc_pos_embed = nn.Sequential(
            nn.Linear(3, config.decoder_hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size),
        )

        num_patches = (config.image_size // config.patch_size) ** 2
        self.embedding_sz = (num_patches, num_patches, config.num_pc_centers)
        self.unmask_sz = config.unmask_sz
        self.context_pos_emb = nn.Parameter(
            torch.randn(sum(self.embedding_sz), config.decoder_hidden_size)
        )
        nn.init.trunc_normal_(self.context_pos_emb, std=config.initializer_range)

        self.rgb_query_proj = nn.Linear(config.hidden_size, config.decoder_hidden_size)
        self.depth_query_proj = nn.Linear(
            config.hidden_size, config.decoder_hidden_size
        )
        self.pc_query_proj = nn.Linear(config.hidden_size, config.decoder_hidden_size)
        self.rgb_query_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        self.depth_query_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        self.pc_query_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )

        self.context_proj = nn.Linear(config.hidden_size, config.decoder_hidden_size)
        self.context_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )

        self.rgb_cross_attn = CrossAttention(config.decoder_hidden_size)
        self.depth_cross_attn = CrossAttention(config.decoder_hidden_size)
        self.pc_cross_attn = CrossAttention(config.decoder_hidden_size)

        dec_config = deepcopy(config)
        dec_config.hidden_size = config.decoder_hidden_size
        dec_config.num_hidden_layers = config.decoder_num_hidden_layers
        dec_config.num_attention_heads = config.decoder_num_attention_heads

        self.rgb_layer = nn.ModuleList(
            [Dinov2Layer(dec_config) for _ in range(dec_config.num_hidden_layers)]
        )
        self.depth_layer = nn.ModuleList(
            [Dinov2Layer(dec_config) for _ in range(dec_config.num_hidden_layers)]
        )
        self.pc_layer = nn.ModuleList(
            [Dinov2Layer(dec_config) for _ in range(dec_config.num_hidden_layers)]
        )

        self.rgb_out_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        self.depth_out_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        self.pc_out_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )

        self.rgb_out_proj = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * 3
        )
        self.depth_out_proj = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2
        )
        self.pc_out_proj = nn.Linear(config.decoder_hidden_size, config.num_pc_knn * 3)

        self.norm_pix_loss = config.norm_pix_loss

    def get_decoder_input(self, encoder_output: EncoderModelOutput):
        """Convert the encoder output to decoder input."""
        unmasked_emb = encoder_output.last_hidden_states
        unmask_sz = encoder_output.unmask_sz

        masked_emb = torch.zeros(
            (
                unmasked_emb.shape[0],
                sum(self.embedding_sz) - unmask_sz,
                unmasked_emb.shape[-1],
            ),
            device=unmasked_emb.device,
            dtype=unmasked_emb.dtype,
        )
        all_emb = torch.cat([unmasked_emb, masked_emb], dim=1)
        all_emb = torch.gather(
            all_emb,
            1,
            encoder_output.restore_idx.unsqueeze(-1).repeat(1, 1, all_emb.shape[-1]),
        )
        rgb_emb, depth_emb, pc_emb = torch.split(all_emb, self.embedding_sz, dim=1)

        return DecoderInput(
            rgb_embedding=rgb_emb,
            depth_embedding=depth_emb,
            pc_embedding=pc_emb,
            unmasked_emb=unmasked_emb,
            shuffle_idx=encoder_output.shuffle_idx,
            pc_centers=encoder_output.pc_centers,
            pc_knn=encoder_output.pc_knn,
            add_mask=encoder_output.add_mask,
            unmask_sz=unmask_sz,
        )

    def forward(self, decoder_input: DecoderInput):
        unmask_sz = (
            decoder_input.unmask_sz if decoder_input.unmask_sz else self.unmask_sz
        )
        rgb_query = self.rgb_query_proj(decoder_input.rgb_embedding)
        depth_query = self.depth_query_proj(decoder_input.depth_embedding)
        pc_query = self.pc_query_proj(decoder_input.pc_embedding)
        rgb_query = self.rgb_query_norm(rgb_query + self.rgb_pos_embed)
        depth_query = self.depth_query_norm(depth_query + self.depth_pos_embed)
        if decoder_input.pc_centers is not None:
            pc_pos_embed = self.pc_pos_embed(decoder_input.pc_centers)
        else:
            pc_pos_embed = 0
        pc_query = self.pc_query_norm(pc_query + pc_pos_embed)

        context = self.context_proj(decoder_input.unmasked_emb)
        shuffle_idx = decoder_input.shuffle_idx[:, :unmask_sz]
        context_pos_emb = self.context_pos_emb[shuffle_idx]
        context = self.context_norm(context + context_pos_emb)

        rgb_emb = self.rgb_cross_attn(rgb_query, context)
        depth_emb = self.depth_cross_attn(depth_query, context)
        pc_emb = self.pc_cross_attn(pc_query, context)

        for layers in self.rgb_layer:
            rgb_emb = layers(rgb_emb)[0]
        for layers in self.depth_layer:
            depth_emb = layers(depth_emb)[0]
        for layers in self.pc_layer:
            pc_emb = layers(pc_emb)[0]

        rgb_emb = self.rgb_out_norm(rgb_emb)
        depth_emb = self.depth_out_norm(depth_emb)
        pc_emb = self.pc_out_norm(pc_emb)

        rgb_out = self.rgb_out_proj(rgb_emb)
        depth_out = self.depth_out_proj(depth_emb)
        pc_out = self.pc_out_proj(pc_emb)

        return rgb_out, depth_out, pc_out

    def get_loss(self, decoder_input: DecoderInput, rgb, depth, pc):
        unmask_sz = decoder_input.unmask_sz
        b = rgb.shape[0]
        rgb_out, depth_out, pc_out = self(decoder_input)

        target_rgb, target_depth = (
            patchify(rgb, self.config.patch_size, 3),
            patchify(depth, self.config.patch_size, 1),
        )
        target_pc = decoder_input.pc_knn * 10.0  # meters to centimeters

        if self.norm_pix_loss:
            rgb_mean, rgb_std = (
                target_rgb.mean(-1, keepdim=True),
                target_rgb.std(-1, keepdim=True),
            )
            depth_mean, depth_std = (
                target_depth.mean(-1, keepdim=True),
                target_depth.std(-1, keepdim=True),
            )
        else:
            rgb_mean, rgb_std = 0.0, 1.0
            depth_mean, depth_std = 0.0, 1.0

        target_rgb = (target_rgb - rgb_mean) / (rgb_std + 1e-8)
        target_depth = (target_depth - depth_mean) / (depth_std + 1e-8)

        mask = torch.ones((b, sum(self.embedding_sz)), device=rgb.device)
        mask[
            torch.arange(b, device=rgb.device)[:, None],
            decoder_input.shuffle_idx[:, :unmask_sz],
        ] = 0
        rgb_mask, depth_mask, pc_mask = torch.split(mask, self.embedding_sz, dim=1)

        rgb_loss = (
            (rgb_out - target_rgb).pow(2).mean(-1) * rgb_mask
        ).sum() / rgb_mask.sum()
        depth_loss = (
            (depth_out - target_depth).abs().mean(-1) * depth_mask
        ).sum() / depth_mask.sum()

        pred_pc = einops.rearrange(pc_out[pc_mask.bool()], "b (k n) -> b k n", n=3)
        target_pc = target_pc[pc_mask.bool()]
        
        if chamfer_distance is None:
            raise ImportError(
                "Please install `pytorch3d` to use the Chamfer Distance loss. "
                "Check https://github.com/facebookresearch/pytorch3d for installation instructions."
            )
        pc_loss = chamfer_distance(pred_pc.float(), target_pc.float(), norm=1)[0]

        return rgb_loss, depth_loss, pc_loss

    @torch.no_grad()
    def visualize(
        self,
        decoder_input: DecoderInput,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pc: torch.Tensor,
    ):
        """Visualize the predictions of the decoder.

        Args:
            decoder_input (DecoderInput):
                `decoder_input` from `get_decoder_input`.
            rgb (torch.Tensor):
                RGB image with shape (B, 3, H, W) in [-1, 1] range.
            depth (torch.Tensor):
                Depth map with shape (B, 1, H, W) in [0, inf] range. Unit is meters.
            pc (torch.Tensor):
                Point cloud with shape (B, N, 3), where N=8192 is the number of points. Unit is meters.

        Returns:
            _type_: _description_
        """
        rgb_out, depth_out, pc_out = self(decoder_input)
        pc_centers = decoder_input.pc_centers
        pc_out = einops.rearrange(pc_out, "... (k n) -> ... k n", n=3)
        plt_pc = pc_out / 10.0 + pc_centers.unsqueeze(-2)
        b = rgb_out.shape[0]
        unmask_sz = decoder_input.unmask_sz

        target_rgb, target_depth = (
            patchify(rgb, self.config.patch_size, 3),
            patchify(depth, self.config.patch_size, 1),
        )

        if self.norm_pix_loss:
            rgb_mean, rgb_std = (
                target_rgb.mean(-1, keepdim=True),
                target_rgb.std(-1, keepdim=True),
            )
            depth_mean, depth_std = (
                target_depth.mean(-1, keepdim=True),
                target_depth.std(-1, keepdim=True),
            )
        else:
            rgb_mean, rgb_std = 0.0, 1.0
            depth_mean, depth_std = 0.0, 1.0

        pred_rgb = rgb_out * (rgb_std + 1e-8) + rgb_mean
        pred_depth = depth_out * (depth_std + 1e-8) + depth_mean

        mask = torch.ones((b, sum(self.embedding_sz)), device=rgb.device)
        if decoder_input.add_mask:
            mask[
                torch.arange(b, device=rgb.device)[:, None],
                decoder_input.shuffle_idx[:, :unmask_sz],
            ] = 0
        rgb_mask, depth_mask, _ = torch.split(mask, self.embedding_sz, dim=1)

        masked_rgb = torch.ones_like(target_rgb) - 2.0
        masked_rgb[~rgb_mask.bool()] = target_rgb[~rgb_mask.bool()].to(masked_rgb.dtype)
        masked_rgb = unpatchify(
            masked_rgb,
            self.config.patch_size,
            3,
            (self.config.image_size, self.config.image_size),
        )
        pred_rgb[~rgb_mask.bool()] = target_rgb[~rgb_mask.bool()].to(pred_rgb.dtype)
        pred_rgb = unpatchify(
            pred_rgb,
            self.config.patch_size,
            3,
            (self.config.image_size, self.config.image_size),
        )

        masked_depth = torch.zeros_like(pred_depth)
        masked_depth[~depth_mask.bool()] = target_depth[~depth_mask.bool()].to(
            masked_depth.dtype
        )
        masked_depth = unpatchify(
            masked_depth,
            self.config.patch_size,
            1,
            (self.config.image_size, self.config.image_size),
        )
        pred_depth[~depth_mask.bool()] = target_depth[~depth_mask.bool()].to(
            pred_depth.dtype
        )
        pred_depth = unpatchify(
            pred_depth,
            self.config.patch_size,
            1,
            (self.config.image_size, self.config.image_size),
        )

        plt_rgb = (
            torch.cat([rgb.float(), masked_rgb.float(), pred_rgb.float()], 2) * 0.5
            + 0.5
        ).clip(0, 1)
        plt_depth = (
            torch.cat([depth.float(), masked_depth.float(), pred_depth.float()], 2)
            / 2.0
        ).clip(0, 1)

        return (
            plt_rgb.permute(0, 2, 3, 1).cpu(),
            plt_depth.permute(0, 2, 3, 1).cpu(),
            plt_pc.cpu(),
        )
