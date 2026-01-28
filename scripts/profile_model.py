import torch
import torch.nn as nn
import argparse
import os
import sys
import time

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VDA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)
sys.path.append(SCRIPT_DIR)

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames

def load_model(encoder, checkpoint_path=None):
    print(f"Loading {encoder}...")
    configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    model = VideoDepthAnything(**configs[encoder])
    
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location='cpu')
        # If it's a quantized checkpoint (whole model), use directly
        if isinstance(state, nn.Module):
            model = state
        else:
            model.load_state_dict(state)
            
    return model

def profile_inference(model, input_size, device, output_file="profile_trace.json"):
    print(f"Profiling inference (Device: {device})...")
    model.to(device).eval()
    
    # Dummy input
    x = torch.randn(1, 1, 3, input_size, input_size).to(device)
    
    # Warmup
    for _ in range(5):
        _ = model(x)
        
    print("Running Profiler...")
    try:
        from torch.profiler import profile, record_function, ProfilerActivity
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                     record_shapes=True,
                     profile_memory=True,
                     with_stack=True) as prof:
            with record_function("model_inference"):
                model(x)
                
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        # Export
        prof.export_chrome_trace(output_file)
        print(f"Trace saved to {output_file}")
        print("Tip: Open chrome://tracing in Chrome and load this file.")
        
    except ImportError:
        print("torch.profiler not found. Basic timing:")
        start = time.time()
        for _ in range(50):
            model(x)
        end = time.time()
        print(f"Avg Latency: {(end-start)/50*1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="vits")
    parser.add_argument("--model_path", help="Path to .pth checkpoint (optional)")
    parser.add_argument("--output", default="profile_trace.json")
    args = parser.parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model_path:
        model = load_model(args.encoder, args.model_path)
    else:
        # Load default VDA
        from qSim_v3 import load_vda
        model = load_vda(args.encoder, False)
        
    profile_inference(model, 266, DEVICE, args.output)
