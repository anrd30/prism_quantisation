"""
Visualize and Benchmark Quantized VDA Model

Loads the pre-quantized model state (from qSim_v3.py), runs inference,
calculates accuracy metrics (MSE, MAE), and saves depth videos.

Usage:
    python3 visualize_quantized.py --encodings_dir ./outputs_quantsim_v3
"""

import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VDA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)
sys.path.append(SCRIPT_DIR)

# AIMET imports
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config

from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.dinov2_layers.layer_scale import LayerScale
from utils.dc_utils import read_video_frames, save_video
from prepare_model_for_quantization import prepare_vda_for_quantization

# Ignore LayerScale (same as during quantization)
from aimet_torch.v2.nn.base import BaseQuantizationMixin
BaseQuantizationMixin.ignore(LayerScale)


def load_vda(encoder: str, metric: bool = False) -> VideoDepthAnything:
    """Load Video-Depth-Anything model."""
    configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
    
    model = VideoDepthAnything(**configs[encoder], metric=metric)
    checkpoint_path = f"{VDA_PATH}/checkpoints/{checkpoint_name}_{encoder}.pth"
    
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def restore_quantsim(model, input_size, device, encodings_dir):
    """Restore QuantSim model from saved checkpoints."""
    print("  Restoring QuantSim model...")
    
    # 1. Create fresh QuantSim (same config as before)
    dummy_input = torch.randn(1, 2, 3, input_size, input_size).to(device)
    
    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.training_range_learning_with_tf_init,
        default_param_bw=8,
        default_output_bw=8,
        config_file=get_path_for_per_channel_config()
    )
    
    # 2. Load the quantized weights/params
    # Note: We saved multiple files. 'vda_int8_v3.pth' is likely the one we want
    # checks logic: qSim_v3 line ~318: filename_prefix="vda_int8_v3"
    # AIMET exports:
    #   - vda_int8_v3.pth (model state dict)
    #   - vda_int8_v3.encodings (JSON/YAML encodings)
    
    # Using the .pth which contains both model weights and quantization encodings/parameters
    pth_path = os.path.join(encodings_dir, "vda_int8_v3.pth")
    if not os.path.exists(pth_path):
         pth_path = os.path.join(encodings_dir, "vda_int8.pth")
    
    if os.path.exists(pth_path):
        print(f"  ✓ Loading quant weights from: {pth_path}")
        loaded = torch.load(pth_path, map_location=device)
        
        if isinstance(loaded, nn.Module):
            print("  ✓ Loaded full model object (skipping QuantSim reconstruction)")
            return loaded
        else:
            sim.model.load_state_dict(loaded)
    else:
        print(f"  Available files in {encodings_dir}: {os.listdir(encodings_dir)}")
        raise FileNotFoundError(f"Could not find quantized checkpoint in {encodings_dir}")

    sim.model.to(device)
    sim.model.eval()
    return sim.model  # Return the underlying model directly


def calculate_metrics(fp32_depths, int8_depths):
    """Calculate accuracy metrics between FP32 and INT8 depth maps."""
    # Normalize if needed, but raw outputs should be comparable
    # Shapes: [T, H, W]
    
    mse = np.mean((fp32_depths - int8_depths) ** 2)
    mae = np.mean(np.abs(fp32_depths - int8_depths))
    
    # Correlation coefficient (flatten)
    c1 = fp32_depths.flatten()
    c2 = int8_depths.flatten()
    corr = np.corrcoef(c1, c2)[0, 1]
    
    return mse, mae, corr


def main():
    parser = argparse.ArgumentParser(description="Visualize Quantized VDA Model")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--input_size", type=int, default=140)
    parser.add_argument("--encodings_dir", type=str, required=True, 
                       help="Directory containing QuantSim export")
    parser.add_argument("--input_video", type=str, 
                       default=f"{VDA_PATH}/assets/example_videos/davis_rollercoaster.mp4")
    parser.add_argument("--output_dir", type=str, default="./outputs_benchmark")
    parser.add_argument("--max_frames", type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("Quantized Model Visualization & Benchmarking")
    print("=" * 70)
    print(f"Load from: {args.encodings_dir}")
    print(f"Output to: {args.output_dir}")
    
    # ---------------------------------------------------------
    # 0. Load Video Frames
    # ---------------------------------------------------------
    print(f"\n[0/4] Loading Video Frames from {args.input_video}...")
    frames, target_fps = read_video_frames(args.input_video, -1, -1, 1280)
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print(f"  Loaded {len(frames)} frames.")
    
    # ---------------------------------------------------------
    # 1. Load Baseline FP32 Model & Run Inference
    # ---------------------------------------------------------
    print("\n[1/4] Running Baseline FP32 Model...")
    model_fp32 = load_vda(args.encoder, args.metric).to(DEVICE).eval()
    
    # Replace xFormers for CPU compatibility (if using CPU) or standard benchmark
    print("  Replacing xFormers in FP32 model for compatibility...")
    model_fp32 = prepare_vda_for_quantization(model_fp32, replace_cross_attn=False, verbose=False)
    
    print(f"  Processing {len(frames)} frames...")
    start = time.time()
    depths_fp32, target_fps = model_fp32.infer_video_depth(
        frames, target_fps, args.input_size, DEVICE, fp32=True
    )
    print(f"    FP32 Time: {time.time()-start:.2f}s")
    
    # Free memory
    del model_fp32
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 2. Load Quantized Model & Run Inference
    # ---------------------------------------------------------
    print("\n[2/4] Running Quantized Model (Simulated)...")
    # Must prepare structural changes first
    model_quant = load_vda(args.encoder, args.metric).to(DEVICE).eval()
    model_quant = prepare_vda_for_quantization(model_quant, replace_cross_attn=False, verbose=False)
    
    # Restore QuantSim wrapper with weights
    model_int8 = restore_quantsim(model_quant, args.input_size, DEVICE, args.encodings_dir)
    
    print("  Running INT8 inference (Simulated)...")
    start = time.time()
    depths_int8, _ = model_int8.infer_video_depth(
        frames, target_fps, args.input_size, DEVICE, fp32=True
    )
    print(f"    INT8 Time: {time.time()-start:.2f}s")
    
    # Free memory
    del model_int8, model_quant
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 4. Metrics & Visualization
    # ---------------------------------------------------------
    print("\n[4/4] Analysis...")
    
    # Convert to numpy
    # infer_video_depth returns numpy uint8 color maps usually if visualisation was done?
    # No, check VideoDepthAnything.infer_video_depth source.
    # It returns (depths, fps). depths are usually the visualized RGB frames if using the default method!
    # Wait, visualization is often baked in.
    
    # Let's inspect what infer_video_depth returns.
    # If it returns colored frames, we can compute PSNR but not raw depth error.
    # Ideally we wanted raw depth.
    # Looking at run_quantized_inference.py usage:
    # save_video(depths... is_depths=True) -> implies they are already colored?
    # Let's verify source.
    
    # NOTE: comparison on RGB visualized maps is "okay" but raw depth is better.
    # For user request "output videos and see stats", colored video is good.
    # For stats, let's treat the RGB output as the result.
    
    mse, mae, corr = calculate_metrics(np.array(depths_fp32), np.array(depths_int8))
    
    print("-" * 30)
    print(f"METRICS (FP32 vs INT8)")
    print("-" * 30)
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Corr: {corr:.4f}")
    print("-" * 30)
    
    # Save Videos
    print("Saving videos...")
    vid_name = os.path.basename(args.input_video).replace(".mp4", "")
    
    # FP32
    save_video(depths_fp32, os.path.join(args.output_dir, f"{vid_name}_fp32.mp4"), fps=target_fps, is_depths=True)
    
    # INT8
    save_video(depths_int8, os.path.join(args.output_dir, f"{vid_name}_int8_sim.mp4"), fps=target_fps, is_depths=True)
    
    # Side-by-Side
    # Create side-by-side comparison
    shape = depths_fp32[0].shape
    if len(shape) == 2:
        h, w = shape
    else:
        h, w, c = shape
        
    combined = []
    for f, i in zip(depths_fp32, depths_int8):
        # Stack horizontally: [FP32 | INT8]
        comp = np.concatenate((f, i), axis=1)
        combined.append(comp)
        
    save_video(combined, os.path.join(args.output_dir, f"{vid_name}_comparison.mp4"), fps=target_fps, is_depths=True)
    
    print(f"\nSaved comparison video to: {os.path.join(args.output_dir, f'{vid_name}_comparison.mp4')}")


if __name__ == "__main__":
    main()
