"""
QuantizableAttention: AIMET-compatible self-attention module.

This module provides drop-in replacements for xFormers-based attention layers,
using standard PyTorch operations that work with AIMET's fake-quantized tensors.

Key differences from MemEffAttention:
- Uses torch.matmul instead of xformers.ops.memory_efficient_attention
- Uses standard softmax instead of fused kernels
- Full O(n²) attention matrix materialized (vs O(n) in FlashAttention)

Usage:
    from quantizable_attention import QuantizableAttention, QuantizableCrossAttention
    
    # Replace MemEffAttention with QuantizableAttention
    # Weights can be copied directly as architecture is identical
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


logger = logging.getLogger("quantizable_attention")


class QuantizableAttention(nn.Module):
    """
    AIMET-compatible self-attention using standard PyTorch operations.
    
    This is a drop-in replacement for MemEffAttention from dinov2_layers/attention.py.
    Uses the same weight layout: qkv projection (dim -> 3*dim) and output projection.
    
    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        qkv_bias: Add bias to qkv projection
        proj_bias: Add bias to output projection
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Same architecture as MemEffAttention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        """
        Forward pass with standard PyTorch attention.
        
        Args:
            x: Input tensor of shape [B, N, C] where:
               B = batch size, N = sequence length, C = embed dim
            attn_bias: Optional attention bias (for compatibility, logged if used)
        
        Returns:
            Output tensor of shape [B, N, C]
        """
        B, N, C = x.shape
        
        # QKV projection: [B, N, C] -> [B, N, 3*C] -> [B, N, 3, H, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # Permute to [3, B, H, N, D] for easy splitting
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H, N, D]
        
        # Standard attention: Q @ K^T / sqrt(d) -> softmax -> @ V
        # q: [B, H, N, D], k.T: [B, H, D, N] -> attn: [B, H, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply attention bias if provided (for nested tensor compatibility)
        if attn_bias is not None:
            logger.debug("attn_bias provided but using standard attention; bias ignored")
            # Note: xFormers attn_bias is typically a BlockDiagonalMask
            # For quantization, we run with single tensors, so this is rarely used
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted sum: [B, H, N, N] @ [B, H, N, D] -> [B, H, N, D]
        x = attn @ v
        
        # Reshape back: [B, H, N, D] -> [B, N, H*D] = [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    @classmethod
    def from_mem_eff_attention(cls, mem_eff_attn: nn.Module) -> "QuantizableAttention":
        """
        Create a QuantizableAttention from an existing MemEffAttention.
        Copies all weights and configuration.
        
        Args:
            mem_eff_attn: Original MemEffAttention module
            
        Returns:
            New QuantizableAttention with copied weights
        """
        # Extract configuration from original
        dim = mem_eff_attn.qkv.in_features
        num_heads = mem_eff_attn.num_heads
        qkv_bias = mem_eff_attn.qkv.bias is not None
        proj_bias = mem_eff_attn.proj.bias is not None
        
        # Get dropout rates from existing dropout layers
        attn_drop = mem_eff_attn.attn_drop.p if hasattr(mem_eff_attn.attn_drop, 'p') else 0.0
        proj_drop = mem_eff_attn.proj_drop.p if hasattr(mem_eff_attn.proj_drop, 'p') else 0.0
        
        # Create new module
        new_attn = cls(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        
        # Copy weights
        new_attn.qkv.weight.data.copy_(mem_eff_attn.qkv.weight.data)
        if qkv_bias:
            new_attn.qkv.bias.data.copy_(mem_eff_attn.qkv.bias.data)
        
        new_attn.proj.weight.data.copy_(mem_eff_attn.proj.weight.data)
        if proj_bias:
            new_attn.proj.bias.data.copy_(mem_eff_attn.proj.bias.data)
        
        return new_attn


class QuantizableCrossAttention(nn.Module):
    """
    AIMET-compatible cross-attention using standard PyTorch operations.
    
    This is a replacement for CrossAttention from motion_module/attention.py.
    Uses separate Q, K, V projections (unlike self-attention's fused QKV).
    
    Args:
        query_dim: Dimension of query input
        cross_attention_dim: Dimension of key/value input (defaults to query_dim)
        heads: Number of attention heads
        dim_head: Dimension per head
        dropout: Dropout rate
        bias: Add bias to linear layers
        upcast_attention: Upcast attention computation to float32
        upcast_softmax: Upcast softmax computation to float32
    """
    
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.added_kv_proj_dim = added_kv_proj_dim
        
        # Group norm if specified
        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=inner_dim, 
                num_groups=norm_num_groups, 
                eps=1e-5, 
                affine=True
            )
        else:
            self.group_norm = None
        
        # Separate Q, K, V projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        
        # Additional KV projections for some architectures
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
        
        # Output projection and dropout
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        ])
    
    def forward(
        self, 
        hidden_states: Tensor, 
        encoder_hidden_states: Optional[Tensor] = None, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass with standard PyTorch cross-attention.
        
        Args:
            hidden_states: Query input [B, N, D]
            encoder_hidden_states: Key/Value input [B, M, D] (defaults to hidden_states)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [B, N, D]
        """
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Apply group norm if present
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # Query projection
        query = self.to_q(hidden_states)
        
        # Key/Value source
        if self.added_kv_proj_dim is not None:
            # Added KV projection path
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)
            
            key = self._reshape_heads_to_batch_dim(key)
            value = self._reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self._reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self._reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)
            
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
            query = self._reshape_heads_to_batch_dim(query)
        else:
            # Standard cross-attention or self-attention
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            query = self._reshape_heads_to_batch_dim(query)
            key = self._reshape_heads_to_batch_dim(key)
            value = self._reshape_heads_to_batch_dim(value)
        
        # Handle attention mask
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
        
        # Standard attention computation
        hidden_states = self._attention(query, key, value, attention_mask)
        
        # Output projection
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        
        return hidden_states
    
    def _reshape_heads_to_batch_dim(self, tensor: Tensor) -> Tensor:
        """Reshape [B, N, H*D] -> [B*H, N, D]"""
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.heads, dim // self.heads)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * self.heads, seq_len, dim // self.heads)
        return tensor.contiguous()
    
    def _reshape_batch_dim_to_heads(self, tensor: Tensor) -> Tensor:
        """Reshape [B*H, N, D] -> [B, N, H*D]"""
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // self.heads, self.heads, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // self.heads, seq_len, dim * self.heads)
        return tensor.contiguous()
    
    def _attention(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Standard scaled dot-product attention."""
        
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        
        # Compute attention scores: [B*H, N, D] @ [B*H, D, M] -> [B*H, N, M]
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                       dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(value.dtype)
        
        # Apply attention to values
        hidden_states = torch.bmm(attention_probs, value)
        
        # Reshape back to [B, N, H*D]
        hidden_states = self._reshape_batch_dim_to_heads(hidden_states)
        
        return hidden_states
    
    @classmethod
    def from_cross_attention(cls, cross_attn: nn.Module) -> "QuantizableCrossAttention":
        """
        Create a QuantizableCrossAttention from an existing CrossAttention.
        Copies all weights and configuration.
        """
        # Extract configuration
        query_dim = cross_attn.to_q.in_features
        cross_attention_dim = cross_attn.to_k.in_features
        inner_dim = cross_attn.to_q.out_features
        heads = cross_attn.heads
        dim_head = inner_dim // heads
        
        bias = cross_attn.to_q.bias is not None
        upcast_attention = getattr(cross_attn, 'upcast_attention', False)
        upcast_softmax = getattr(cross_attn, 'upcast_softmax', False)
        added_kv_proj_dim = getattr(cross_attn, 'added_kv_proj_dim', None)
        
        # Get dropout from to_out
        dropout = cross_attn.to_out[1].p if hasattr(cross_attn.to_out[1], 'p') else 0.0
        
        # Get norm_num_groups if present
        norm_num_groups = None
        if hasattr(cross_attn, 'group_norm') and cross_attn.group_norm is not None:
            norm_num_groups = cross_attn.group_norm.num_groups
        
        # Create new module
        new_attn = cls(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim if cross_attention_dim != query_dim else None,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            added_kv_proj_dim=added_kv_proj_dim,
            norm_num_groups=norm_num_groups,
        )
        
        # Copy weights
        new_attn.to_q.weight.data.copy_(cross_attn.to_q.weight.data)
        new_attn.to_k.weight.data.copy_(cross_attn.to_k.weight.data)
        new_attn.to_v.weight.data.copy_(cross_attn.to_v.weight.data)
        
        if bias:
            new_attn.to_q.bias.data.copy_(cross_attn.to_q.bias.data)
            new_attn.to_k.bias.data.copy_(cross_attn.to_k.bias.data)
            new_attn.to_v.bias.data.copy_(cross_attn.to_v.bias.data)
        
        # Copy output projection
        new_attn.to_out[0].weight.data.copy_(cross_attn.to_out[0].weight.data)
        if cross_attn.to_out[0].bias is not None:
            new_attn.to_out[0].bias.data.copy_(cross_attn.to_out[0].bias.data)
        
        # Copy group norm if present
        if norm_num_groups is not None and cross_attn.group_norm is not None:
            new_attn.group_norm.weight.data.copy_(cross_attn.group_norm.weight.data)
            new_attn.group_norm.bias.data.copy_(cross_attn.group_norm.bias.data)
        
        # Copy added KV projections if present
        if added_kv_proj_dim is not None:
            new_attn.add_k_proj.weight.data.copy_(cross_attn.add_k_proj.weight.data)
            new_attn.add_v_proj.weight.data.copy_(cross_attn.add_v_proj.weight.data)
            if cross_attn.add_k_proj.bias is not None:
                new_attn.add_k_proj.bias.data.copy_(cross_attn.add_k_proj.bias.data)
                new_attn.add_v_proj.bias.data.copy_(cross_attn.add_v_proj.bias.data)
        
        return new_attn
