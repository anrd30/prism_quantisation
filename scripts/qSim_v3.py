"""
AIMET Quantization for Video-Depth-Anything (VDA) Model

This script:
1. Loads the VDA model
2. Replaces xFormers attention layers with AIMET-compatible attention
3. Creates QuantizationSimModel
4. Runs calibration
5. Exports INT8 model

Usage:
    python3 qSim_v2.py --encoder vits --input_size 140

Requirements:
    - AIMET installed
    - VDA model checkpoint in Video-Depth-Anything/checkpoints/
"""

import sys
import os
import time
import argparse
import torch

# Add VDA repo to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VDA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)
sys.path.append(SCRIPT_DIR)

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames

# Import our quantization utilities
from prepare_model_for_quantization import (
    prepare_vda_for_quantization,
    verify_no_xformers,
    count_attention_layers
)

# AIMET imports
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

# Note: We no longer need to ignore Attention classes since we replaced them!
# The old workaround was:
#   from aimet_torch.v2.nn.base import BaseQuantizationMixin
#   BaseQuantizationMixin.ignore(LayerScale)
#   BaseQuantizationMixin.ignore(Attention)
# Now we only ignore LayerScale which uses element-wise multiplication

from video_depth_anything.dinov2_layers.layer_scale import LayerScale
from aimet_torch.v2.nn.base import BaseQuantizationMixin
BaseQuantizationMixin.ignore(LayerScale)


# ---------------------------------------------------------
# Load VDA model
# ---------------------------------------------------------
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
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


# ---------------------------------------------------------
# Calibration function required by compute_encodings
# ---------------------------------------------------------
def calibration_forward_pass(model, frames, input_size, device, num_frames=16):
    """
    Run calibration frames through the model to collect activation ranges.
    
    Args:
        model: The quantized model
        frames: Video frames for calibration
        input_size: Input resolution
        device: CUDA device
        num_frames: Number of frames to use for calibration
    """
    model.eval()
    with torch.no_grad():
        for i, frame in enumerate(frames[:num_frames]):
            # Convert frame to tensor [B, C, H, W]
            img = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Resize to target size
            img = torch.nn.functional.interpolate(
                img, size=(input_size, input_size), mode='bilinear', align_corners=False
            )
            
            # VDA expects shape [B, T, C, H, W]
            batch = img.unsqueeze(1).to(device)
            
            # Forward pass to let quantizers observe activation ranges
            _ = model.forward(batch)
            
            if (i + 1) % 4 == 0:
                print(f"  Calibration progress: {i + 1}/{min(len(frames), num_frames)} frames")



# ---------------------------------------------------------
# AdaRound Helpers
# ---------------------------------------------------------
def create_calibration_dataloader(video_path, num_frames, input_size, device):
    """Create a dataloader for AdaRound."""
    frames, _ = read_video_frames(video_path, num_frames, -1, 1280)
    
    # Preprocess frames to tensors
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensors = []
    for frame in frames:
        t = transform(frame).unsqueeze(0)  # [1, 3, H, W]
        tensors.append(t)
    
    # Stack into batches
    dataset = torch.cat(tensors, dim=0)  # [N, 3, H, W]
    
    class SimpleDataLoader:
        def __init__(self, data, batch_size=4):
            self.data = data
            self.batch_size = batch_size
            self.device = device
        
        def __iter__(self):
            for i in range(0, len(self.data), self.batch_size):
                batch = self.data[i:i+self.batch_size].to(self.device)
                # VDA expects [B, T, C, H, W] for video
                # For per-frame model, T=1 usually works, or match dummy input T
                yield batch.unsqueeze(1)  # [B, 1, C, H, W]
        
        def __len__(self):
            return (len(self.data) + self.batch_size - 1) // self.batch_size
            
    return SimpleDataLoader(dataset)


def apply_adaround(model, dummy_input, data_loader, output_dir):
    """Apply AdaRound weight optimization."""
    print("\n  Applying AdaRound...")
    
    params = AdaroundParameters(
        data_loader=data_loader,
        num_batches=len(data_loader),
        default_num_iterations=500,
        default_reg_param=0.01,
        default_beta_range=(20, 2),
        default_warm_start=0.2,
    )
    
    try:
        model = Adaround.apply_adaround(
            model=model,
            dummy_input=dummy_input,
            params=params,
            path=output_dir,
            filename_prefix="vda_adaround",
            default_param_bw=8,
            default_quant_scheme=QuantScheme.training_range_learning_with_tf_init,
        )
        print("    ✓ AdaRound complete")
    except Exception as e:
        print(f"    ✗ AdaRound failed: {e}")
        print("      Continuing without AdaRound...")
    
    return model


def apply_cle(model: torch.nn.Module, input_size: int) -> torch.nn.Module:
    """Apply Cross-Layer Equalization to conv layers."""
    print("\n  Applying Cross-Layer Equalization (CLE)...")
    
    dummy_shape = (1, 3, input_size, input_size)
    
    # 1. Apply to patch embedding
    try:
        equalize_model(model.pretrained.patch_embed, input_shapes=dummy_shape)
        print("    ✓ CLE applied to patch_embed")
    except Exception as e:
        print(f"    ✗ patch_embed skipped: {e}")
    
    # 2. Apply to head convolutions
    try:
        # The head has conv layers in scratch module
        # DINOv2 DPT head structure: head.scratch.{layerX_rn, refinenetX}
        for name in ['layer1_rn', 'layer2_rn', 'layer3_rn', 'layer4_rn']:
            if hasattr(model.head.scratch, name):
                module = getattr(model.head.scratch, name)
                try:
                    equalize_model(module, input_shapes=dummy_shape)
                    print(f"    ✓ CLE applied to head.scratch.{name}")
                except:
                    pass
    except Exception as e:
        print(f"    ✗ head.scratch skipped: {e}")
    
    return model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AIMET Quantization for VDA")
    parser.add_argument("--encoder", type=str, default="vits", 
                       choices=["vits", "vitb", "vitl"])
    parser.add_argument("--metric", action="store_true",
                       help="Use metric depth model")
    parser.add_argument("--input_video", type=str, 
                       default=f"{VDA_PATH}/assets/example_videos/davis_rollercoaster.mp4")
    parser.add_argument("--input_size", type=int, default=140,
                       help="Input resolution (will be square)")
    parser.add_argument("--output_dir", type=str, default="./outputs_quantsim_v2",
                       help="Output directory for exported model")
    parser.add_argument("--num_calib_frames", type=int, default=16,
                       help="Number of frames for calibration")
    parser.add_argument("--skip_cle", action="store_true", help="Skip Cross-Layer Equalization")
    parser.add_argument("--skip_adaround", action="store_true", help="Skip AdaRound")
    args = parser.parse_args()
    
    # Optional arguments for advanced features
    SKIP_CLE = getattr(args, 'skip_cle', False)
    SKIP_ADAROUND = getattr(args, 'skip_adaround', False)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"AIMET Quantization for Video-Depth-Anything (v3)")
    print(f"{'='*60}")
    print(f"Encoder: {args.encoder}")
    print(f"Device: {DEVICE}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    
    # ---------------------------------------------------------
    # Step 1 — Load FP32 model (same as qSim_v2)
    # ---------------------------------------------------------
    print(f"\n[Step 1/4] Loading FP32 model...")
    model = load_vda(args.encoder, args.metric).to(DEVICE).eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Count attention layers before
    print("\nAttention layers BEFORE preparation:")
    for attn_type, count in count_attention_layers(model).items():
        print(f"  {attn_type}: {count}")
    
    # ---------------------------------------------------------
    # Step 2 — Replace xFormers attention (same as qSim_v2)
    # ---------------------------------------------------------
    print(f"\n[Step 2/4] Replacing xFormers attention layers...")
    model = prepare_vda_for_quantization(
        model, 
        replace_cross_attn=False,
        verbose=True
    )
    
    # Verify no xFormers remain
    is_ready = verify_no_xformers(model)
    if not is_ready:
        raise RuntimeError("Model still has xFormers dependencies!")
        
    # ---------------------------------------------------------
    # Step 2.5 — Cross-Layer Equalization (Optional)
    # ---------------------------------------------------------
    if not SKIP_CLE:
        print(f"\n[Step 2.5] Applying CLE...")
        model = apply_cle(model, args.input_size)
        model = model.to(DEVICE)  # IMPORTANT: CLE may move model to CPU
    else:
        print(f"\n[Step 2.5] Skipping CLE (--skip_cle)")
        
    # ---------------------------------------------------------
    # Step 2.6 — AdaRound (Optional)
    # ---------------------------------------------------------
    if not SKIP_ADAROUND:
        print(f"\n[Step 2.6] Applying AdaRound...")
        # Create dataloader for AdaRound
        data_loader = create_calibration_dataloader(
            args.input_video, args.num_calib_frames, args.input_size, DEVICE
        )
        
        # Create dummy input matching dataloader shape
        # AdaRound typically works with single frame batches or small sequences
        dummy_input_adaround = torch.randn(1, 1, 3, args.input_size, args.input_size).to(DEVICE)
        
        model = apply_adaround(model, dummy_input_adaround, data_loader, args.output_dir)
        model = model.to(DEVICE)
    else:
        print(f"\n[Step 2.6] Skipping AdaRound (--skip_adaround)")
    
    print("\nAttention layers AFTER preparation:")
    for attn_type, count in count_attention_layers(model).items():
        print(f"  {attn_type}: {count}")
    
    # ---------------------------------------------------------
    # Step 3 — Create QuantSim
    # ---------------------------------------------------------
    print(f"\n[Step 3/5] Creating QuantizationSimModel...")
    
    # VDA expects shape [B, T, C, H, W] for dummy input
    dummy_input = torch.randn(1, 2, 3, args.input_size, args.input_size).to(DEVICE)
    
    start_time = time.time()
    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.training_range_learning_with_tf_init,
        default_param_bw=8,
        default_output_bw=8,
        config_file=get_path_for_per_channel_config()
    )
    print(f"✓ QuantSim created in {time.time() - start_time:.1f}s")
    
    # ---------------------------------------------------------
    # Step 4 — Load calibration video and compute encodings
    # ---------------------------------------------------------
    print(f"\n[Step 4/5] Running calibration...")
    print(f"Calibration video: {args.input_video}")
    
    frames, _ = read_video_frames(args.input_video, -1, -1, 1280)
    print(f"Loaded {len(frames)} frames")
    
    start_time = time.time()
    sim.compute_encodings(
        lambda m: calibration_forward_pass(
            m, frames, args.input_size, DEVICE, args.num_calib_frames
        )
    )
    print(f"✓ Encodings computed in {time.time() - start_time:.1f}s")
    
    # ---------------------------------------------------------
    # Step 5 — Export INT8 model
    # ---------------------------------------------------------
    print(f"\n[Step 5/5] Exporting INT8 model...")
    
    export_path = args.output_dir
    sim.export(
        path=export_path,
        filename_prefix="vda_int8",
        dummy_input=dummy_input.cpu()
    )
    
    print(f"\n{'='*60}")
    print(f"✓ INT8 Export Complete!")
    print(f"{'='*60}")
    print(f"Model + encodings saved in: {export_path}")
    print(f"Files:")
    for f in os.listdir(export_path):
        fpath = os.path.join(export_path, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
