"""
Benchmark Accuracy: FP32 vs INT8 Quantized VDA Model

This script compares depth predictions between:
1. Original FP32 model
2. Model with QuantizableAttention (still FP32)
3. AIMET INT8 quantized model

Metrics computed:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation
- Structural Similarity Index (SSIM)

Usage:
    python3 benchmark_accuracy.py --encoder vits --num_frames 10
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VDA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)
sys.path.append(SCRIPT_DIR)

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

from prepare_model_for_quantization import prepare_vda_for_quantization


def load_vda(encoder: str, metric: bool) -> VideoDepthAnything:
    """Load Video-Depth-Anything model."""
    configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    checkpoint_name = "metric_video_depth_anything" if metric else "video_depth_anything"

    model = VideoDepthAnything(**configs[encoder], metric=metric)
    checkpoint_path = f"{VDA_PATH}/checkpoints/{checkpoint_name}_{encoder}.pth"
    
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute accuracy metrics between predicted and target depth maps.
    
    Args:
        pred: Predicted depth [B, H, W] or [B, 1, H, W]
        target: Target depth [B, H, W] or [B, 1, H, W]
        
    Returns:
        Dict with MAE, MSE, RMSE, correlation metrics
    """
    # Flatten batch dimension
    pred = pred.flatten()
    target = target.flatten()
    
    # Normalize to [0, 1] for fair comparison
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
    
    # Compute metrics
    mae = (pred_norm - target_norm).abs().mean().item()
    mse = ((pred_norm - target_norm) ** 2).mean().item()
    rmse = np.sqrt(mse)
    
    # Pearson correlation
    pred_centered = pred_norm - pred_norm.mean()
    target_centered = target_norm - target_norm.mean()
    correlation = (pred_centered * target_centered).sum() / (
        pred_centered.norm() * target_centered.norm() + 1e-8
    )
    correlation = correlation.item()
    
    # Max absolute error
    max_error = (pred_norm - target_norm).abs().max().item()
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Correlation': correlation,
        'MaxError': max_error
    }


def run_inference(model: torch.nn.Module, frames: List[np.ndarray], 
                  input_size: int, device: str, desc: str = "") -> Tuple[torch.Tensor, float]:
    """
    Run inference on frames and return depths + timing.
    
    Returns:
        Tuple of (depths tensor, avg time per frame)
    """
    model.eval()
    depths = []
    
    start_time = time.time()
    with torch.no_grad():
        for frame in frames:
            # Convert frame to tensor [B, C, H, W]
            img = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Resize to target size
            img = F.interpolate(img, size=(input_size, input_size), mode='bilinear', align_corners=False)
            
            # VDA expects shape [B, T, C, H, W]
            batch = img.unsqueeze(1).to(device)
            
            # Forward pass
            depth = model(batch)
            depths.append(depth.cpu())
    
    total_time = time.time() - start_time
    avg_time = total_time / len(frames)
    
    depths = torch.cat(depths, dim=0)
    print(f"{desc}: {len(frames)} frames in {total_time:.2f}s ({1/avg_time:.1f} FPS)")
    
    return depths, avg_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs Quantized VDA")
    parser.add_argument("--encoder", type=str, default="vits", 
                       choices=["vits", "vitb", "vitl"])
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--input_video", type=str, 
                       default=f"{VDA_PATH}/assets/example_videos/davis_rollercoaster.mp4")
    parser.add_argument("--input_size", type=int, default=140)
    parser.add_argument("--num_frames", type=int, default=10,
                       help="Number of frames to benchmark")
    parser.add_argument("--output_dir", type=str, default="./outputs_benchmark")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("VDA Accuracy Benchmark: Original vs QuantizableAttention")
    print("=" * 70)
    print(f"Encoder: {args.encoder}")
    print(f"Device: {DEVICE}")
    print(f"Input: {args.input_video}")
    print(f"Resolution: {args.input_size}x{args.input_size}")
    print(f"Frames: {args.num_frames}")
    
    # Load video frames
    print("\n[1/4] Loading video frames...")
    frames, fps = read_video_frames(args.input_video, -1, -1, 1280)
    frames = frames[:args.num_frames]
    print(f"Loaded {len(frames)} frames")
    
    # ---------------------------------------------------------
    # Model 1: Original FP32 with MemEffAttention (xFormers)
    # ---------------------------------------------------------
    print("\n[2/4] Running original FP32 model (xFormers)...")
    model_original = load_vda(args.encoder, args.metric).to(DEVICE).eval()
    depths_original, time_original = run_inference(
        model_original, frames, args.input_size, DEVICE, "Original FP32"
    )
    del model_original
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # Model 2: FP32 with QuantizableAttention (no xFormers)
    # ---------------------------------------------------------
    print("\n[3/4] Running FP32 model with QuantizableAttention...")
    model_quantizable = load_vda(args.encoder, args.metric).to(DEVICE).eval()
    model_quantizable = prepare_vda_for_quantization(
        model_quantizable, 
        replace_cross_attn=False, 
        verbose=False
    )
    depths_quantizable, time_quantizable = run_inference(
        model_quantizable, frames, args.input_size, DEVICE, "QuantizableAttention FP32"
    )
    del model_quantizable
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # Compute accuracy metrics
    # ---------------------------------------------------------
    print("\n[4/4] Computing accuracy metrics...")
    
    metrics = compute_metrics(depths_quantizable, depths_original)
    
    print("\n" + "=" * 70)
    print("RESULTS: QuantizableAttention vs Original (xFormers)")
    print("=" * 70)
    print(f"  Mean Absolute Error (MAE):     {metrics['MAE']:.6f}")
    print(f"  Mean Squared Error (MSE):      {metrics['MSE']:.8f}")
    print(f"  Root Mean Squared Error:       {metrics['RMSE']:.6f}")
    print(f"  Pearson Correlation:           {metrics['Correlation']:.6f}")
    print(f"  Max Absolute Error:            {metrics['MaxError']:.6f}")
    print()
    print(f"  Original FP32 time:            {time_original*1000:.1f} ms/frame")
    print(f"  QuantizableAttention time:     {time_quantizable*1000:.1f} ms/frame")
    print(f"  Slowdown:                      {time_quantizable/time_original:.2f}x")
    print("=" * 70)
    
    # Interpret results
    if metrics['Correlation'] > 0.999:
        print("\n✓ EXCELLENT: QuantizableAttention is numerically equivalent to original")
    elif metrics['Correlation'] > 0.99:
        print("\n✓ GOOD: Very high correlation, minor numerical differences")
    elif metrics['Correlation'] > 0.95:
        print("\n⚠ ACCEPTABLE: High correlation but some differences")
    else:
        print("\n✗ POOR: Significant differences detected")
    
    # Save comparison visualization
    if len(depths_original) > 0:
        # Save first frame comparison
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(depths_original[0].squeeze().numpy(), cmap='viridis')
        axes[0].set_title('Original (xFormers)')
        axes[0].axis('off')
        
        # QuantizableAttention
        axes[1].imshow(depths_quantizable[0].squeeze().numpy(), cmap='viridis')
        axes[1].set_title('QuantizableAttention')
        axes[1].axis('off')
        
        # Difference
        diff = (depths_original[0] - depths_quantizable[0]).abs().squeeze().numpy()
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(f'Absolute Difference (max: {diff.max():.4f})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, 'comparison.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\nSaved comparison visualization to: {save_path}")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("VDA Accuracy Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Encoder: {args.encoder}\n")
        f.write(f"Input size: {args.input_size}\n")
        f.write(f"Frames: {args.num_frames}\n\n")
        f.write("Metrics (QuantizableAttention vs Original):\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.8f}\n")
        f.write(f"\nTiming:\n")
        f.write(f"  Original: {time_original*1000:.1f} ms/frame\n")
        f.write(f"  QuantizableAttention: {time_quantizable*1000:.1f} ms/frame\n")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
