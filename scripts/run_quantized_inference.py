"""
Run VDA inference with quantized model and save video output.

Usage:
    python3 run_quantized_inference.py --encoder vits --input_size 140
"""

import sys
import os
import argparse
import time
import torch

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


def main():
    parser = argparse.ArgumentParser(description="Run quantized VDA inference")
    parser.add_argument("--encoder", type=str, default="vits", 
                       choices=["vits", "vitb", "vitl"])
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--input_video", type=str, 
                       default=f"{VDA_PATH}/assets/example_videos/davis_rollercoaster.mp4")
    parser.add_argument("--input_size", type=int, default=140)
    parser.add_argument("--output_dir", type=str, default="./outputs_quantized_video")
    parser.add_argument("--max_frames", type=int, default=-1,
                       help="Max frames to process (-1 for all)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("VDA Quantized Video Inference")
    print("=" * 70)
    print(f"Encoder: {args.encoder}")
    print(f"Device: {DEVICE}")
    print(f"Input: {args.input_video}")
    print(f"Resolution: {args.input_size}x{args.input_size}")
    
    # Load video frames
    print("\n[1/3] Loading video frames...")
    frames, target_fps = read_video_frames(args.input_video, -1, -1, 1280)
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print(f"Loaded {len(frames)} frames at {target_fps} FPS")
    
    # Load and prepare model
    print("\n[2/3] Loading and preparing model...")
    model = load_vda(args.encoder, args.metric).to(DEVICE).eval()
    model = prepare_vda_for_quantization(model, replace_cross_attn=False, verbose=False)
    print(f"Model ready with QuantizableAttention")
    
    # Run inference
    print("\n[3/3] Running inference...")
    start_time = time.time()
    
    depths, fps = model.infer_video_depth(
        frames,
        target_fps,
        input_size=args.input_size,
        device=DEVICE,
        fp32=True  # Use FP32 for this demo
    )
    
    total_time = time.time() - start_time
    print(f"Inference complete: {len(frames)} frames in {total_time:.2f}s ({len(frames)/total_time:.1f} FPS)")
    
    # Save outputs
    video_name = os.path.basename(args.input_video).replace(".mp4", "")
    
    # Save source video
    src_path = os.path.join(args.output_dir, f"{video_name}_src.mp4")
    save_video(frames, src_path, fps=fps)
    print(f"Saved source video: {src_path}")
    
    # Save depth video
    depth_path = os.path.join(args.output_dir, f"{video_name}_depth_quantizable.mp4")
    save_video(depths, depth_path, fps=fps, is_depths=True)
    print(f"Saved depth video: {depth_path}")
    
    print("\n" + "=" * 70)
    print("✓ Video inference complete!")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
