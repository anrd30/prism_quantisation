"""
Test script for INT8 attention CUDA kernel.

Usage:
    cd ops/
    pip install -e .
    python test_kernel.py
"""

import torch
import time

def test_int8_attention():
    """Test the INT8 attention kernel."""
    try:
        import vda_int8_ops
    except ImportError:
        print("ERROR: vda_int8_ops not installed.")
        print("Run: cd ops && pip install -e .")
        return False
    
    print("=" * 60)
    print("INT8 Attention Kernel Test")
    print("=" * 60)
    
    # Test parameters
    B, H, N, D = 2, 6, 100, 64  # batch, heads, seq_len, head_dim
    
    print(f"Shape: B={B}, H={H}, N={N}, D={D}")
    
    # Create random INT8 tensors
    torch.manual_seed(42)
    device = "cuda"
    
    # Simulate quantized Q, K, V (INT8)
    q = torch.randint(-128, 127, (B, H, N, D), dtype=torch.int8, device=device)
    k = torch.randint(-128, 127, (B, H, N, D), dtype=torch.int8, device=device)
    v = torch.randint(-128, 127, (B, H, N, D), dtype=torch.int8, device=device)
    
    scale = (D ** -0.5)  # 1/sqrt(d)
    
    # Quantization scales (from AIMET encodings)
    q_scale = 0.05
    k_scale = 0.05
    v_scale = 0.05
    out_scale = 1.0
    
    print(f"\nInput tensors:")
    print(f"  Q: {q.shape}, dtype={q.dtype}")
    print(f"  K: {k.shape}, dtype={k.dtype}")
    print(f"  V: {v.shape}, dtype={v.dtype}")
    
    # Warm up
    for _ in range(3):
        output = vda_int8_ops.int8_attention(q, k, v, scale, q_scale, k_scale, v_scale, out_scale)
    
    torch.cuda.synchronize()
    
    # Benchmark
    num_runs = 100
    start = time.time()
    for _ in range(num_runs):
        output = vda_int8_ops.int8_attention(q, k, v, scale, q_scale, k_scale, v_scale, out_scale)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\nOutput: {output.shape}, dtype={output.dtype}")
    print(f"\nPerformance:")
    print(f"  {num_runs} runs in {elapsed*1000:.1f}ms")
    print(f"  {elapsed/num_runs*1000:.3f}ms per attention")
    
    # Compare with FP32 reference
    print("\n" + "=" * 60)
    print("Comparing with FP32 reference...")
    
    q_fp = q.float() * q_scale
    k_fp = k.float() * k_scale
    v_fp = v.float() * v_scale
    
    attn = (q_fp @ k_fp.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    ref_output = attn @ v_fp
    
    # Compare
    # Note: INT8 will have quantization error, so we check relative error
    abs_diff = (output - ref_output).abs()
    rel_diff = abs_diff / (ref_output.abs() + 1e-6)
    
    print(f"  Max absolute diff: {abs_diff.max().item():.4f}")
    print(f"  Mean absolute diff: {abs_diff.mean().item():.4f}")
    print(f"  Max relative diff: {rel_diff.max().item():.4f}")
    print(f"  Mean relative diff: {rel_diff.mean().item():.4f}")
    
    if rel_diff.mean().item() < 0.1:  # 10% average error is acceptable for INT8
        print("\n✓ Test PASSED!")
        return True
    else:
        print("\n✗ Test FAILED - error too high")
        return False


def benchmark_vs_fp32():
    """Benchmark INT8 vs FP32 attention."""
    try:
        import vda_int8_ops
    except ImportError:
        print("vda_int8_ops not installed")
        return
    
    print("\n" + "=" * 60)
    print("Benchmark: INT8 vs FP32")
    print("=" * 60)
    
    device = "cuda"
    
    for N in [100, 256, 512, 1024]:
        B, H, D = 1, 6, 64
        
        # INT8
        q_int8 = torch.randint(-128, 127, (B, H, N, D), dtype=torch.int8, device=device)
        k_int8 = torch.randint(-128, 127, (B, H, N, D), dtype=torch.int8, device=device)
        v_int8 = torch.randint(-128, 127, (B, H, N, D), dtype=torch.int8, device=device)
        
        # FP32
        q_fp32 = torch.randn(B, H, N, D, device=device)
        k_fp32 = torch.randn(B, H, N, D, device=device)
        v_fp32 = torch.randn(B, H, N, D, device=device)
        
        scale = D ** -0.5
        
        # Warmup
        for _ in range(5):
            _ = vda_int8_ops.int8_attention(q_int8, k_int8, v_int8, scale)
            attn = (q_fp32 @ k_fp32.transpose(-2, -1)) * scale
            _ = attn.softmax(dim=-1) @ v_fp32
        
        torch.cuda.synchronize()
        
        # INT8 timing
        num_runs = 50
        start = time.time()
        for _ in range(num_runs):
            _ = vda_int8_ops.int8_attention(q_int8, k_int8, v_int8, scale)
        torch.cuda.synchronize()
        int8_time = (time.time() - start) / num_runs * 1000
        
        # FP32 timing
        start = time.time()
        for _ in range(num_runs):
            attn = (q_fp32 @ k_fp32.transpose(-2, -1)) * scale
            _ = attn.softmax(dim=-1) @ v_fp32
        torch.cuda.synchronize()
        fp32_time = (time.time() - start) / num_runs * 1000
        
        speedup = fp32_time / int8_time
        
        print(f"N={N:4d}: INT8={int8_time:.3f}ms, FP32={fp32_time:.3f}ms, Speedup={speedup:.2f}x")


if __name__ == "__main__":
    success = test_int8_attention()
    if success:
        benchmark_vs_fp32()
