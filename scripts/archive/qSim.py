import sys
import os
import torch
import argparse

# Add VDA repo to path
VDA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames
from video_depth_anything.dinov2_layers.layer_scale import LayerScale
from aimet_torch.v2.nn.base import BaseQuantizationMixin
from video_depth_anything.dinov2_layers.attention import Attention

BaseQuantizationMixin.ignore(LayerScale)
BaseQuantizationMixin.ignore(torch.nn.MultiheadAttention)
BaseQuantizationMixin.ignore(Attention)

# AIMET imports
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config


# ---------------------------------------------------------
# Load VDA model
# ---------------------------------------------------------
def load_vda(encoder, metric):
    configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    checkpoint_name = "metric_video_depth_anything" if metric else "video_depth_anything"

    model = VideoDepthAnything(**configs[encoder], metric=metric)
    state = torch.load(f"{VDA_PATH}/checkpoints/{checkpoint_name}_{encoder}.pth", map_location="cpu")
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------
# Calibration function required by compute_encodings
# ---------------------------------------------------------
def calibration_forward_pass(model, frames, input_size, device):
    model.eval()
    with torch.no_grad():
        for frame in frames[:16]:   # 16 representative frames
            img = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.

            img = torch.nn.functional.interpolate(
                img, size=(input_size, input_size), mode='bilinear'
            )

            # VDA expects shape [B, T, C, H, W]
            batch = img.unsqueeze(1).to(device)

            # Just call forward once to let quantizers observe ranges
            _ = model.forward(batch)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="vits")
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--input_video", type=str, default=f"{VDA_PATH}/assets/example_videos/davis_rollercoaster.mp4")
    parser.add_argument("--input_size", type=int, default=140)
    parser.add_argument("--output_dir", type=str, default="./outputs_quantsim_latest")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load FP32 model
    model = load_vda(args.encoder, args.metric).to(DEVICE).eval()
    print("Loaded VDA model with params:", sum(p.numel() for p in model.parameters()))

    # ---------------------------------------------------------
    # Step 1 — Create QuantSim
    # ---------------------------------------------------------
    # VDA expects shape [B, T, C, H, W] for dummy input
    dummy_input = torch.randn(1, 32, 3, args.input_size, args.input_size).to(DEVICE)

    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.training_range_learning_with_tf_init,
        default_param_bw=8,
        default_output_bw=8,
        config_file=get_path_for_per_channel_config()
    )

    print("QuantSim created successfully!")

    # ---------------------------------------------------------
    # Step 2 — Load calibration video
    # ---------------------------------------------------------
    frames, _ = read_video_frames(args.input_video, -1, -1, 1280)

    # ---------------------------------------------------------
    # Step 3 — Compute Encodings
    # ---------------------------------------------------------
    print("Running calibration...")
    sim.compute_encodings(
        lambda m: calibration_forward_pass(m, frames, args.input_size, DEVICE)
    )
    print("✓ Encodings computed successfully!")

    # ---------------------------------------------------------
    # Step 4 — Export INT8 model
    # ---------------------------------------------------------
    export_path = args.output_dir
    sim.export(
        path=export_path,
        filename_prefix="vda_int8",
        dummy_input=dummy_input.cpu()
    )

    print("\n=== INT8 Export Complete ===")
    print(f"Model + encodings saved in: {export_path}")
