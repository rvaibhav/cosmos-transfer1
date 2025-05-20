# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from contextlib import nullcontext
from typing import List, Optional, Union

import numpy as np
import torch
import transformer_engine as te
from einops import rearrange

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
from packaging import version
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformer_engine.pytorch.attention import DotProductAttention, apply_rotary_pos_emb

from imaginaire.utils import log

# ---------------------- Feed Forward Network -----------------------


class FeedForward(nn.Module):
    """
    Transformer FFN with optional gating

    Parameters:
        d_model (int): Dimensionality of input features.
        d_ff (int): Dimensionality of the hidden layer.
        dropout (float, optional): Dropout rate applied after the activation function. Defaults to 0.1.
        activation (callable, optional): The activation function applied after the first linear layer.
                                         Defaults to nn.ReLU().
        is_gated (bool, optional): If set to True, incorporates gating mechanism to the feed-forward layer.
                                   Defaults to False.
        bias (bool, optional): If set to True, adds a bias to the linear layers. Defaults to True.

    Example:
        >>> ff = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(64, 10, 512)  # Example input tensor
        >>> output = ff(x)
        >>> print(output.shape)  # Expected shape: (64, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()

        tp_group = parallel_state.get_tensor_model_parallel_group(check_initialized=False)
        sequence_parallel = getattr(parallel_state, "sequence_parallel", False)
        if tp_group is None:
            tp_size = 1  # TP is not initialized.
        else:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()

        if tp_size == 1:
            self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
            self.layer2 = nn.Linear(d_ff, d_model, bias=bias)
        else:
            assert is_gated is False, "is_gated needs to be False to support Tensor Parallelism"
            assert dropout == 0.0, "dropout needs to be 0.0 to support Tensor Parallelism"
            self.layer1 = te.pytorch.Linear(
                d_model,
                d_ff,
                bias=bias,
                tp_size=tp_size,
                tp_group=tp_group,
                parallel_mode="column",
                sequence_parallel=sequence_parallel,
            )
            self.layer2 = te.pytorch.Linear(
                d_ff,
                d_model,
                bias=bias,
                tp_size=tp_size,
                tp_group=tp_group,
                parallel_mode="row",
                sequence_parallel=sequence_parallel,
            )

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_gate(x)
        else:
            x = g
        # Apply dropout
        # TODO: (qsh 2024-08-27) We skip dropout to save memory, maybe we can add it back later
        assert self.dropout.p == 0.0, "we skip dropout to save memory"
        # x = self.dropout(x)
        return self.layer2(x)


class SwiGLUFeedForward(FeedForward):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=nn.SiLU(),
            is_gated=True,
            bias=False,
        )


class GPT2FeedForward(FeedForward):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=nn.GELU(),
            is_gated=False,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        # TODO: (qsh 2024-08-27) We skip dropout to save memory, maybe we can add it back later
        assert self.dropout.p == 0.0, "we skip dropout to save memory"

        x = self.layer1(x)

        def activation_layer2_forward(x):
            x = self.activation(x)
            x = self.layer2(x)
            return x

        # (qsh 2024-10-09) the impl will recomputed the activation, which is useful for saving memory
        # we only use for video project now!
        x = checkpoint(activation_layer2_forward, x, use_reentrant=False)
        return x


# ---------------------- Normalization Layer -----------------------


def normalize(x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 0) -> torch.Tensor:
    """
    Normalizes the input tensor along specified dimensions such that the average square norm of elements is adjusted.

    Args:
        x (torch.Tensor): The input tensor to normalize.
        dim (list, optional): The dimensions over which to normalize. If None, normalizes over all dimensions except the first.
        eps (float, optional): A small constant to ensure numerical stability during division.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class STDNorm(nn.Module):
    """
    A normalization layer that adjusts the input tensor such that the average square norm of elements
    along the specified dimension(s) approximates a standard deviation of 1

    Attributes:
        dim (int): The dimension over which normalization is performed. Default is -1, which typically represents the last dimension.
        eps (float): A small constant added for numerical stability during normalization.

    Example:
        >>> layer = STDNorm(dim=1)
        >>> x = torch.randn(10, 20)
        >>> y = layer(x)
    """

    def __init__(self, dim: int = -1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return normalize(x, self.dim, self.eps)


def get_normalization(name: str, channels: int, dim=-1, elementwise_affine: bool = True):
    if name == "I":
        return nn.Identity()
    elif name == "S":
        return STDNorm(dim)
    elif name == "R":
        return te.pytorch.RMSNorm(channels, eps=1e-6)
    elif name == "L":
        return nn.LayerNorm(channels, elementwise_affine=elementwise_affine, bias=False)
    else:
        raise ValueError(f"Normalization {name} not found")


# ---------------------- Attention Op -----------------------
# A list of attention ops
if version.parse(torch.__version__) >= version.parse("2.3.0"):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    sdpa_context = sdpa_kernel
    USE_SDPA = True
elif version.parse(torch.__version__) >= version.parse("2.0.0"):
    from torch.backends.cuda import SDPBackend, sdp_kernel

    sdpa_context = sdp_kernel
    USE_SDPA = False
else:
    sdpa_context = nullcontext
    USE_SDPA = False
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:  # noqa
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

OCEAN_IS_AVAILABLE = False

import torch.nn.functional as F
from torch import Tensor


class BaseAttentionOp(nn.Module):
    def __init__(self):
        super().__init__()


class FusedAttentionOp(BaseAttentionOp):
    def __init__(self):
        super().__init__()


class VisualizationLayer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def forward(self, x):
        return x


class EinSumAttentionOp(BaseAttentionOp):
    def __init__(self, force_fp32: bool = False):
        super().__init__()
        self.attn_viz = VisualizationLayer(type="attn_w")
        self.force_fp32 = force_fp32

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Use einsum to compute the scaled dot-product attention over the input tensors.
        Mainly for debug and visualization purposes, not efficient!
        B is the batch size, M the sequence length, H the number of heads, and K the embeding size per head

        check F.scaled_dot_product_attention
        Args:
            q (Tensor): The query tensor of shape [B, Mq, H, K]
            k (Tensor): The key tensor of shape [B, Mk, H, K]
            v (Tensor): The value tensor of shape [B, Mk, H, V]

        Returns:
            Tensor: [B, Mq, H, V]
        """
        in_dtype = q.dtype
        if self.force_fp32:
            q, k = q.float(), k.float()
        w = torch.einsum("nqhd,nkhd->nhqk", q, k / np.sqrt(q.shape[-1]))
        if mask is not None:
            w = w.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        w = self.attn_viz(w)
        w = w.softmax(dim=3).to(in_dtype)
        return torch.einsum("nhqk,nkhd->nqhd", w, v)


class TorchAttentionOp(FusedAttentionOp):
    def __init__(self, backend: Optional[Union[List[SDPBackend], SDPBackend]] = SDPBackend.EFFICIENT_ATTENTION):
        super().__init__()
        self.backend = backend
        self.sdpa_context = sdpa_context if self.backend is not None else nullcontext
        if self.backend is not None:
            log.warning(
                "SDPA context manager is not working well with torch.compile, causing graph breaks and "
                "significant slowdowns. If you are using torch.compile you'll most likely want to turn off "
                "this context manager."
            )

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Computes the scaled dot-product attention over the input tensors using the specified backend.
        B is the batch size, M the sequence length, H the number of heads, and K the embeding size per head

        check F.scaled_dot_product_attention
        Args:
            q (Tensor): The query tensor of shape [B, Mq, H, K] / [B, ..., H, K]
            k (Tensor): The key tensor of shape [B, Mk, H, V] / [B, ..., H, K]
            v (Tensor): The value tensor of shape [B, Mk, H, V] / [B, ..., H, V]

            mask (Optional[Tensor]): An optional mask tensor. Follow scaled_dot_product_attention API, mask should be a boolean tensor with shape [B, H, Mq, Mk]

        Returns:
            Tensor: [B, Mq, H, V] / [B, ..., H, V]
        """
        in_q_shape = q.shape
        in_k_shape = k.shape
        q = rearrange(q, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
        k = rearrange(k, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
        v = rearrange(v, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
        if mask is not None:
            assert mask.dtype == torch.bool, "Mask should be a boolean tensor"
        with self.sdpa_context(self.backend):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale is dim_head ** -0.5 per default
        return rearrange(out, "b h ... l -> b ... h l").view(*in_q_shape[:-1], in_k_shape[-1])


class XformerAttentionOp(FusedAttentionOp):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Performs memory-efficient attention operation using the provided query, key, and value tensors.

        B is the batch size, M the sequence length, H the number of heads, and K the embeding size per head

        Args:
            q (Tensor): The query tensor of shape [B, Mq, H, K]
            k (Tensor): The key tensor of shape [B, Mk, H, K]
            v (Tensor): The value tensor of shape [B, Mk, H, V]
            mask (Optional[Tensor]): An optional mask tensor. Currently, masking is not supported, and this argument should be None.

        Returns:
            Tensor: The output tensor after applying the attention mechanism to the input tensors.

        Raises:
            AssertionError: If xformers is not available or if a mask is provided since masking is not yet supported.
        """
        assert XFORMERS_IS_AVAILABLE, "xformers is not available. Please install it."
        # assert mask is None, "Masking is not supported yet."
        in_q_shape = q.shape
        in_k_shape = k.shape
        q = q.view(in_q_shape[0], -1, in_q_shape[-2], in_q_shape[-1])
        k = k.view(in_k_shape[0], -1, in_k_shape[-2], in_k_shape[-1])
        v = v.view(in_k_shape[0], -1, in_k_shape[-2], in_k_shape[-1])
        B, Mq, H, K = q.shape
        _, Mk, _, V = v.shape
        if mask is not None:
            mask = mask.to(dtype=q.dtype).expand(B, H, Mq, Mk).contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
        return out.view(*in_q_shape[:-1], V)


class RegionalAttentionOp(BaseAttentionOp):
    def __init__(
        self,
        heads,
        dim_head,
        num_gqa_groups=None,
        attention_dropout=0,
        qkv_format="bshd",
        attn_mask_type="no_mask",
        tp_size=1,
        tp_group=None,
        sequence_parallel=False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.qkv_format = qkv_format
        self.tp_size = tp_size
        self.scale = dim_head**-0.5
        self.attention_dropout = attention_dropout
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        self.dot_product_attention = DotProductAttention(
            self.heads,
            self.dim_head,
            num_gqa_groups=num_gqa_groups,
            attention_dropout=attention_dropout,
            qkv_format=qkv_format,
            attn_mask_type=attn_mask_type,
            tp_size=tp_size,
            tp_group=tp_group,
            sequence_parallel=sequence_parallel,
        )

    def forward(
        self,
        q,
        k,
        v,
        regional_k=None,
        regional_v=None,
        region_masks=None,
        core_attention_bias_type="no_bias",
        core_attention_bias=None,
    ):
        # Early return for non-regional case
        if regional_k is None or regional_v is None or region_masks is None:
            return self.dot_product_attention(
                q,
                k,
                v,
                attention_mask=None,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
            )
        # Get dimensions
        is_bshd = self.qkv_format == "bshd"
        if is_bshd:
            batch_size, seq_len, num_heads, head_dim = q.shape
        else:
            seq_len, batch_size, num_heads, head_dim = q.shape

        # Process region masks
        processed_masks = []
        prompt_len = k.shape[1] if is_bshd else k.shape[0]
        num_regions = len(regional_k)

        def preprocess_mask(mask: Tensor) -> Tensor:
            mask = mask.permute(3, 0, 1, 2)
            B, T, H, W = mask.shape
            mask = mask.unsqueeze(1)  # dummy unsqueeze since trilinear interpolation expects 5D

            mask_i = [
                torch.nn.functional.interpolate(
                    mask[:, :, :1, :, :],
                    size=(1, 44, 80),
                    mode="trilinear",
                    align_corners=False,
                )
            ]
            for wi in range(1, T, 8):
                mask_i += [
                    torch.nn.functional.interpolate(
                        mask[:, :, wi : wi + 8, :, :],
                        size=(1, 44, 80),
                        mode="trilinear",
                        align_corners=False,
                    )
                ]
            assert len(mask_i) == 16
            mask = torch.cat(mask_i, dim=2)
            mask = mask.squeeze(1)
            return (mask > 0.5).float()

        for i in range(num_regions):
            mask = region_masks[i]
            mask = mask.to(q.device)
            if mask.shape[0] != seq_len:
                mask = preprocess_mask(mask)
                mask = rearrange(mask, "b t h w ->  b (t h w)")
            processed_masks.append(mask)

        hidden_seq_len = seq_len
        regional_attention_mask = torch.zeros(
            (batch_size, hidden_seq_len, (num_regions + 1) * prompt_len), device=q.device, dtype=torch.bool
        )
        start_idx = 0
        for i, mask in enumerate(processed_masks):
            regional_attention_mask[:, :, (i + 1) * prompt_len : (i + 2) * prompt_len] = mask.unsqueeze(-1).bool()
        regional_masks_tensor = torch.stack(processed_masks, dim=-1).bool()  # [B, S, R]
        global_mask = (regional_masks_tensor.sum(dim=-1) == 0).unsqueeze(-1).bool()  # [B, S, 1]
        regional_attention_mask[:, :, :prompt_len] = global_mask
        combined_k = torch.cat([k] + regional_k, dim=0)
        combined_v = torch.cat([v] + regional_v, dim=0)

        attn_bias = torch.zeros_like(regional_attention_mask, dtype=torch.float32)
        attn_bias = attn_bias.masked_fill(~regional_attention_mask, float("-inf"))
        attn_bias = attn_bias.unsqueeze(1).expand(-1, num_heads, -1, -1)
        output = self.dot_product_attention(
            q,
            combined_k,
            combined_v,
            attention_mask=None,
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=attn_bias,
        )

        # TODO(@ticai): make this a configurable parameter
        base_ratio = 0.5  # signifies the weight of the global prompt
        if base_ratio is not None:
            base_output = self.dot_product_attention(
                q,
                k,
                v,
                attention_mask=None,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
            )
            output = output * (1 - base_ratio) + base_output * base_ratio

        if self.tp_size > 1 and not self.sequence_parallel:
            torch.distributed.all_reduce(output, group=self.tp_group)

        return output


# ---------------------- Attention Impl -----------------------


class Attention(nn.Module):
    """
    Generalized attention impl.

    Allowing for both self-attention and cross-attention configurations depending on whether a `context_dim` is provided.
    If `context_dim` is None, self-attention is assumed.

    Parameters:
        query_dim (int): Dimension of each query vector.
        context_dim (int, optional): Dimension of each context vector. If None, self-attention is assumed.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimension of each head. Defaults to 64.
        dropout (float, optional): Dropout rate applied to the output of the attention block. Defaults to 0.0.
        attn_op (BaseAttentionOp, optional): Custom attention operation to be used instead of the default.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, and value projections. Defaults to False.
        out_bias (bool, optional): If True, adds a learnable bias to the output projection. Defaults to False.
        qkv_norm (str, optional): A string representing normalization strategies for query, key, and value projections.
                                  Defaults to "SSI".
        norm_args (dict, optional): Arguments to pass to the normalization function. Defaults to an empty dict.

    Examples:
        >>> attn = Attention(query_dim=128, context_dim=256, heads=4, dim_head=32, dropout=0.1)
        >>> query = torch.randn(10, 128)  # Batch size of 10
        >>> context = torch.randn(10, 256)  # Batch size of 10
        >>> output = attn(query, context)  # Perform the attention operation

    Note:
        https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attn_op: Optional[BaseAttentionOp] = None,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qkv_norm: str = "SSI",
        norm_args: dict = {},
        backend: str = "transformer_engine",
        qkv_format: str = "bshd",
    ) -> None:
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}. Norm options are {qkv_norm} and norm args are {norm_args}."
        )
        self.is_selfattn = context_dim is None  # self attention

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_format = qkv_format
        norm_dim = dim_head
        self.backend = backend
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.inner_dim = inner_dim
        tp_group = parallel_state.get_tensor_model_parallel_group(check_initialized=False) if USE_MEGATRON else None
        if tp_group is None:
            self.tp_size = 1  # TP is not initialized.
        else:
            self.tp_size = parallel_state.get_tensor_model_parallel_world_size()

        if self.backend == "torch":
            assert (
                self.tp_size == 1
            ), f"Attention backend {self.backend} cannot use TP size > 1. Attempted: {self.tp_size}"

        assert self.heads % self.tp_size == 0, "the number of heads should be divisible by TP size"
        if self.tp_size == 1:
            self.to_q = nn.Sequential(
                nn.Linear(query_dim, inner_dim, bias=qkv_bias),
                get_normalization(qkv_norm[0], norm_dim, **norm_args),
            )
            self.to_k = nn.Sequential(
                nn.Linear(context_dim, inner_dim, bias=qkv_bias),
                get_normalization(qkv_norm[1], norm_dim, **norm_args),
            )
            self.to_v = nn.Sequential(
                nn.Linear(context_dim, inner_dim, bias=qkv_bias),
                get_normalization(qkv_norm[2], norm_dim, **norm_args),
            )

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, query_dim, bias=out_bias),
                nn.Dropout(dropout),
            )
        else:  # TP enabled.
            sequence_parallel = getattr(parallel_state, "sequence_parallel", False)
            if sequence_parallel:
                assert qkv_format == "sbhd", "sequence parallel only supports sbhd format"

            self.to_q = nn.Sequential(
                te.pytorch.Linear(
                    query_dim,
                    inner_dim,
                    bias=qkv_bias,
                    tp_size=self.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="column",
                ),
                get_normalization(qkv_norm[0], norm_dim, **norm_args),
            )
            self.to_k = nn.Sequential(
                te.pytorch.Linear(
                    context_dim,
                    inner_dim,
                    bias=qkv_bias,
                    tp_size=self.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="column",
                ),
                get_normalization(qkv_norm[1], norm_dim, **norm_args),
            )
            self.to_v = nn.Sequential(
                te.pytorch.Linear(
                    context_dim,
                    inner_dim,
                    bias=qkv_bias,
                    tp_size=self.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="column",
                ),
                get_normalization(qkv_norm[2], norm_dim, **norm_args),
            )

            self.to_out = nn.Sequential(
                te.pytorch.Linear(
                    inner_dim,
                    query_dim,
                    bias=out_bias,
                    tp_size=self.tp_size,
                    tp_group=tp_group,
                    parallel_mode="row",
                    sequence_parallel=sequence_parallel,
                ),
                nn.Dropout(dropout),
            )

        if attn_op:  # use what is given
            self.attn_op = attn_op
        elif self.backend == "transformer_engine":
            sequence_parallel = getattr(parallel_state, "sequence_parallel", False) if USE_MEGATRON else False
            self.attn_op: BaseAttentionOp = DotProductAttention(
                self.heads,
                self.dim_head,
                num_gqa_groups=self.heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
                tp_size=self.tp_size,
                tp_group=tp_group,
                sequence_parallel=sequence_parallel,
            )
            self.regional_attn_op = RegionalAttentionOp(
                self.heads,
                self.dim_head,
                num_gqa_groups=self.heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="arbitrary",
                tp_size=self.tp_size,
                tp_group=tp_group,
                sequence_parallel=sequence_parallel,
            )
        elif self.backend == "torch":
            self.attn_op = TorchAttentionOp(None)
        else:
            raise ValueError(f"Backend {backend} not found")

    def cal_qkv(
        self, x, context=None, mask=None, rope_emb=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del kwargs

        """
        self.to_q, self.to_k, self.to_v are nn.Sequential with projection + normalization layers.
        Before 07/24/2024, these modules normalize across all heads.
        After 07/24/2024, to support tensor parallelism and follow the common practice in the community,
        we support to normalize per head.
        To keep the checkpoint copatibility with the previous code,
        we keep the nn.Sequential but call the projection and the normalization layers separately.
        """

        q = self.to_q[0](x)
        context = x if context is None else context
        k = self.to_k[0](context)
        v = self.to_v[0](context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (n c) -> b ... n c", n=self.heads // self.tp_size, c=self.dim_head),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(q, k, v, rope_emb):
            q = self.to_q[1](q)
            k = self.to_k[1](k)
            v = self.to_v[1](v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            return q, k, v

        q, k, v = checkpoint(apply_norm_and_rotary_pos_emb, q, k, v, rope_emb, use_reentrant=False)

        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        if self.backend == "transformer_engine":
            seq_dim = self.qkv_format.index("s")
            assert (
                q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1
            ), "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
            out = self.attn_op(q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None)  # [B, Mq, H, V]
            return self.to_out(out)
        elif self.backend == "torch":
            out = self.attn_op(q, k, v, mask=mask)  # [B, Mq, H, V]
            return self.to_out(rearrange(out, " b ... n c -> b ... (n c)"))
        else:
            raise ValueError(f"Backend {self.backend} not found")

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rope_emb=None,
        regional_contexts=None,
        region_masks=None,
        **kwargs,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
            regional_contexts (Optional[Tensor]): Stacked regional context tensors [B, R, M, D] or [R, M, B, D] if THWBD format
            region_masks (Optional[Tensor]): Region masks [B, R, S] or [R, S, B] if THWBD format
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)

        # Early return if no regional contexts
        if regional_contexts is None or region_masks is None:
            return self.cal_attn(q, k, v, mask)

        # Process regional contexts
        regional_k = []
        regional_v = []

        # Determine format based on qkv_format
        is_bshd = self.qkv_format == "bshd"

        # Get number of regions
        num_regions = regional_contexts.shape[1] if is_bshd else regional_contexts.shape[0]

        # Process each region
        for i in range(num_regions):
            # Extract regional context
            reg_context = regional_contexts[:, i] if is_bshd else regional_contexts[i]

            # Ensure correct dtype
            if reg_context.dtype != context.dtype:
                reg_context = reg_context.to(dtype=context.dtype)

            _, k_regional, v_regional = self.cal_qkv(x, reg_context, mask, rope_emb=rope_emb, **kwargs)

            regional_k.append(k_regional)
            regional_v.append(v_regional)

        # Apply regional attention
        combined_attn = self.regional_attn_op(
            q,
            k,  # from global prompt
            v,  # from global prompt
            regional_k=regional_k,
            regional_v=regional_v,
            region_masks=region_masks,
            core_attention_bias_type="no_bias",
            core_attention_bias=None,
        )

        # Apply output projection
        return self.to_out(combined_attn)
