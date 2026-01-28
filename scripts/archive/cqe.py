import sys
import os
import argparse
import torch
import numpy as np

# Add VDA repo to PYTHONPATH
VDA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)

from aimet_torch.cross_layer_equalization import equalize_model
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video


def load_vda(encoder, metric):
    configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'

    model = VideoDepthAnything(**configs[encoder], metric=metric)
    model.load_state_dict(torch.load(
        f'{VDA_PATH}/checkpoints/{checkpoint_name}_{encoder}.pth', map_location='cpu'
    ))
    return model


def apply_cqe_vda(model, input_size):
    print("\n=== Running Cross-Layer Equalization ===")

    # CQE only uses 4D dummy inputs (N, C, H, W)
    dummy_shape = (1, 3, input_size, input_size)

    # Apply CQE on patch embedding conv
    try:
        equalize_model(model.pretrained.patch_embed, input_shapes=dummy_shape)
        print("✓ CQE applied to patch_embed")
    except Exception as e:
        print("✗ patch_embed skipped:", e)

    # Apply CQE on decoder convolutions
    try:
        equalize_model(model.decoder, input_shapes=dummy_shape)
        print("✓ CQE applied to decoder")
    except Exception as e:
        print("✗ decoder skipped:", e)

    print("=== CQE Complete ===")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, default=f"{VDA_PATH}/assets/example_videos/davis_rollercoaster.mp4")
    parser.add_argument("--output_dir", type=str, default="./outputs_cqe")
    parser.add_argument("--input_size", type=int, default=140)
    parser.add_argument("--encoder", type=str, default="vits")
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_vda(args.encoder, args.metric).to(DEVICE).eval()
    print("Original params:", sum(p.numel() for p in model.parameters()))

    # Apply CQE only to conv-based modules
    apply_cqe_vda(model, args.input_size)

    print("After CQE params:", sum(p.numel() for p in model.parameters()))

    # Run inference
    frames, target_fps = read_video_frames(args.input_video, -1, -1, 1280)

    import time
    start = time.time()
    depths, fps = model.infer_video_depth(
        frames,
        target_fps,
        input_size=args.input_size,
        device=DEVICE,
        fp32=args.fp32
    )
    latency = (time.time() - start) / len(frames)

    print("\nFPS after CQE:", fps)
    print("Latency per frame:", latency)

    # Save videos
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.basename(args.input_video)
    save_video(frames, os.path.join(args.output_dir, video_name.replace(".mp4", "_src.mp4")), fps=fps)
    save_video(depths, os.path.join(args.output_dir, video_name.replace(".mp4", "_vis.mp4")), fps=fps, is_depths=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "vda_cqe.pth"))
    print("Saved quant model at outputs_cqe/vda_cqe.pth")
