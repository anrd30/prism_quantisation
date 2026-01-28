"""
Test suite for QuantizableAttention modules.

Tests:
1. Numerical equivalence between original and quantizable attention
2. Weight copying correctness
3. Gradient flow through quantizable attention
4. AIMET compatibility (can create QuantizationSimModel)

Run: python test_quantizable_attention.py
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import numpy as np

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VDA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)
sys.path.append(SCRIPT_DIR)

from quantizable_attention import QuantizableAttention, QuantizableCrossAttention
from video_depth_anything.dinov2_layers.attention import Attention, MemEffAttention


class TestQuantizableAttention(unittest.TestCase):
    """Test QuantizableAttention against original Attention class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dim = 384  # vits embedding dim
        self.num_heads = 6
        self.batch_size = 2
        self.seq_len = 100
        
        torch.manual_seed(42)
        
    def test_output_shape(self):
        """Test that output shape matches input shape [B, N, C]."""
        attn = QuantizableAttention(
            dim=self.dim, 
            num_heads=self.num_heads,
            qkv_bias=True
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
        out = attn(x)
        
        self.assertEqual(out.shape, x.shape)
        
    def test_equivalence_with_base_attention(self):
        """Test numerical equivalence with base Attention class (non-xFormers)."""
        # Create both attention modules with same weights
        original = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            proj_bias=True
        ).to(self.device)
        
        quantizable = QuantizableAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            proj_bias=True
        ).to(self.device)
        
        # Copy weights from original to quantizable
        quantizable.qkv.weight.data.copy_(original.qkv.weight.data)
        quantizable.qkv.bias.data.copy_(original.qkv.bias.data)
        quantizable.proj.weight.data.copy_(original.proj.weight.data)
        quantizable.proj.bias.data.copy_(original.proj.bias.data)
        
        # Ensure eval mode (no dropout randomness)
        original.eval()
        quantizable.eval()
        
        # Test with same input
        x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
        
        with torch.no_grad():
            out_original = original(x)
            out_quantizable = quantizable(x)
        
        # Check numerical equivalence
        max_diff = (out_original - out_quantizable).abs().max().item()
        self.assertLess(max_diff, 1e-5, f"Max difference: {max_diff}")
        
        print(f"✓ Numerical equivalence test passed (max diff: {max_diff:.2e})")
        
    def test_weight_copy_from_mem_eff_attention(self):
        """Test weight copying from MemEffAttention."""
        original = MemEffAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            proj_bias=True
        ).to(self.device)
        
        # Create quantizable version
        quantizable = QuantizableAttention.from_mem_eff_attention(original)
        quantizable = quantizable.to(self.device)
        
        # Check weights are identical
        self.assertTrue(torch.allclose(original.qkv.weight, quantizable.qkv.weight))
        self.assertTrue(torch.allclose(original.qkv.bias, quantizable.qkv.bias))
        self.assertTrue(torch.allclose(original.proj.weight, quantizable.proj.weight))
        self.assertTrue(torch.allclose(original.proj.bias, quantizable.proj.bias))
        
        print("✓ Weight copy test passed")
        
    def test_gradient_flow(self):
        """Test that gradients flow through quantizable attention."""
        attn = QuantizableAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=True
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim, 
                       device=self.device, requires_grad=True)
        
        out = attn(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(attn.qkv.weight.grad)
        self.assertIsNotNone(attn.proj.weight.grad)
        
        # Check gradients are non-zero
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertGreater(attn.qkv.weight.grad.abs().sum().item(), 0)
        
        print("✓ Gradient flow test passed")
        
    def test_deterministic_output(self):
        """Test that eval mode produces deterministic outputs."""
        attn = QuantizableAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            attn_drop=0.1,  # Non-zero dropout
            proj_drop=0.1
        ).to(self.device).eval()
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
        
        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x)
        
        self.assertTrue(torch.allclose(out1, out2))
        print("✓ Deterministic output test passed")


class TestQuantizableCrossAttention(unittest.TestCase):
    """Test QuantizableCrossAttention."""
    
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.query_dim = 320
        self.heads = 8
        self.dim_head = 40
        self.batch_size = 2
        self.seq_len = 64
        
        torch.manual_seed(42)
        
    def test_output_shape_self_attention(self):
        """Test output shape for self-attention (no encoder_hidden_states)."""
        attn = QuantizableCrossAttention(
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.query_dim, device=self.device)
        out = attn(x)
        
        self.assertEqual(out.shape, x.shape)
        print("✓ Self-attention output shape test passed")
        
    def test_output_shape_cross_attention(self):
        """Test output shape for cross-attention."""
        cross_dim = 768
        attn = QuantizableCrossAttention(
            query_dim=self.query_dim,
            cross_attention_dim=cross_dim,
            heads=self.heads,
            dim_head=self.dim_head
        ).to(self.device)
        
        query = torch.randn(self.batch_size, self.seq_len, self.query_dim, device=self.device)
        encoder_hidden = torch.randn(self.batch_size, 32, cross_dim, device=self.device)
        
        out = attn(query, encoder_hidden_states=encoder_hidden)
        
        self.assertEqual(out.shape, query.shape)
        print("✓ Cross-attention output shape test passed")


class TestAIMETCompatibility(unittest.TestCase):
    """Test AIMET compatibility of quantizable attention."""
    
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def test_aimet_quantsim_creation(self):
        """Test that AIMET QuantizationSimModel can be created with QuantizableAttention."""
        try:
            from aimet_torch.quantsim import QuantizationSimModel
            from aimet_common.defs import QuantScheme
        except ImportError:
            self.skipTest("AIMET not installed, skipping compatibility test")
            return
        
        # Create a simple model with quantizable attention
        class SimpleViTBlock(nn.Module):
            def __init__(self, dim=384, num_heads=6):
                super().__init__()
                self.norm = nn.LayerNorm(dim)
                self.attn = QuantizableAttention(dim, num_heads, qkv_bias=True)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
                
            def forward(self, x):
                x = x + self.attn(self.norm(x))
                x = x + self.mlp(self.norm(x))
                return x
        
        model = SimpleViTBlock().to(self.device).eval()
        dummy_input = torch.randn(1, 100, 384, device=self.device)
        
        # Try to create QuantizationSimModel
        try:
            sim = QuantizationSimModel(
                model=model,
                dummy_input=dummy_input,
                quant_scheme=QuantScheme.post_training_tf,
                default_param_bw=8,
                default_output_bw=8
            )
            print("✓ AIMET QuantizationSimModel created successfully!")
            
            # Try a forward pass
            with torch.no_grad():
                out = sim.model(dummy_input)
            
            self.assertEqual(out.shape, dummy_input.shape)
            print("✓ AIMET quantized forward pass successful!")
            
        except Exception as e:
            self.fail(f"AIMET QuantizationSimModel creation failed: {e}")


class TestFullModelReplacement(unittest.TestCase):
    """Test full VDA model attention replacement."""
    
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def test_vda_model_preparation(self):
        """Test preparing full VDA model for quantization."""
        try:
            from video_depth_anything.video_depth import VideoDepthAnything
            from prepare_model_for_quantization import (
                prepare_vda_for_quantization, 
                verify_no_xformers,
                count_attention_layers
            )
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
            return
        
        # Load a small VDA model
        configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        
        model = VideoDepthAnything(**configs['vits'], metric=False)
        
        # Check for pretrained weights
        checkpoint_path = os.path.join(VDA_PATH, "checkpoints", "video_depth_anything_vits.pth")
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state)
            print("✓ Loaded pretrained weights")
        else:
            print("⚠ No pretrained weights found, using random init")
        
        model = model.to(self.device).eval()
        
        # Count attention layers before
        before_counts = count_attention_layers(model)
        print(f"\nBefore: {before_counts}")
        
        # Prepare model
        model = prepare_vda_for_quantization(model, replace_cross_attn=False, verbose=False)
        
        # Count attention layers after
        after_counts = count_attention_layers(model)
        print(f"After: {after_counts}")
        
        # Verify
        is_ready = verify_no_xformers(model)
        self.assertTrue(is_ready, "Model still has xFormers dependencies")
        
        # Test forward pass
        dummy_input = torch.randn(1, 2, 3, 140, 140, device=self.device)  # [B, T, C, H, W]
        
        with torch.no_grad():
            try:
                output = model(dummy_input)
                print(f"✓ Forward pass successful, output shape: {output.shape}")
            except Exception as e:
                self.fail(f"Forward pass failed: {e}")


def run_quick_test():
    """Run a quick sanity check without unittest framework."""
    print("=" * 60)
    print("Quick Sanity Check for QuantizableAttention")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test 1: Basic forward pass
    print("\n1. Testing basic forward pass...")
    attn = QuantizableAttention(dim=384, num_heads=6, qkv_bias=True).to(device)
    x = torch.randn(2, 100, 384, device=device)
    out = attn(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"   ✓ Output shape: {out.shape}")
    
    # Test 2: Equivalence with base Attention
    print("\n2. Testing equivalence with base Attention...")
    original = Attention(dim=384, num_heads=6, qkv_bias=True).to(device).eval()
    quantizable = QuantizableAttention(dim=384, num_heads=6, qkv_bias=True).to(device).eval()
    
    # Copy weights
    quantizable.qkv.weight.data.copy_(original.qkv.weight.data)
    quantizable.qkv.bias.data.copy_(original.qkv.bias.data)
    quantizable.proj.weight.data.copy_(original.proj.weight.data)
    quantizable.proj.bias.data.copy_(original.proj.bias.data)
    
    with torch.no_grad():
        out_orig = original(x)
        out_quant = quantizable(x)
    
    max_diff = (out_orig - out_quant).abs().max().item()
    assert max_diff < 1e-5, f"Max diff too large: {max_diff}"
    print(f"   ✓ Max difference: {max_diff:.2e}")
    
    # Test 3: Weight copy from MemEffAttention
    print("\n3. Testing weight copy from MemEffAttention...")
    mem_eff = MemEffAttention(dim=384, num_heads=6, qkv_bias=True).to(device)
    copied = QuantizableAttention.from_mem_eff_attention(mem_eff).to(device)
    
    assert torch.allclose(mem_eff.qkv.weight, copied.qkv.weight)
    assert torch.allclose(mem_eff.proj.weight, copied.proj.weight)
    print("   ✓ Weights copied correctly")
    
    print("\n" + "=" * 60)
    print("All quick tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick sanity check only")
    parser.add_argument("--full", action="store_true", help="Run full test suite including VDA model")
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    elif args.full:
        # Run all tests
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        # Default: run quick test
        run_quick_test()
