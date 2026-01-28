import torch
import os
import sys
import numpy as np
import json
import argparse

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VDA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)
sys.path.append(os.path.join(SCRIPT_DIR, ".."))

def inspect_pytorch_checkpoint(pth_path):
    print(f"\nScanning PyTorch Checkpoint: {pth_path}")
    state = torch.load(pth_path, map_location='cpu')
    
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    
    # Check if AIMET wrappers exist
    quantized_layers = 0
    total_params = 0
    unique_counts = []
    
    for key, val in state.items():
        # Look for quantization encoding parameters usually stored in state
        # AIMET QuantSim wrappers store 'encoding_min', 'encoding_max' buffers 
        # inside the module usually.
        if "encoding_min" in key or "encoding_max" in key:
            quantized_layers += 1
            
        # Check weight discretization
        if "weight" in key and "quantizer" not in key and val.ndim > 1:
            total_params += 1
            # Heuristic: If it's simulated quantization, the float weights usually
            # align to a grid.
            # But AdaRound *updates* the weights to be optimal for quant.
            pass

    print(f"  Found {quantized_layers} encoding parameters (indicates QuantSim wrappers).")
    return quantized_layers > 0

def inspect_encodings(encodings_path):
    print(f"\nScanning Encodings JSON: {encodings_path}")
    if not os.path.exists(encodings_path):
        print("  File not found.")
        return False
        
    with open(encodings_path, 'r') as f:
        data = json.load(f)
        
    if "activation_encodings" in data:
        print(f"  Activation Encodings: {len(data['activation_encodings'])}")
    if "param_encodings" in data:
        print(f"  Param Encodings: {len(data['param_encodings'])}")
        
    return True

def inspect_onnx(onnx_path):
    print(f"\nScanning ONNX Model: {onnx_path}")
    if not os.path.exists(onnx_path):
        print("  File not found.")
        return False
        
    try:
        import onnx
        model = onnx.load(onnx_path)
        
        # Count QuantizeLinear / DequantizeLinear nodes
        q_nodes = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
        dq_nodes = [n for n in model.graph.node if n.op_type == "DequantizeLinear"]
        
        print(f"  QuantizeLinear Nodes: {len(q_nodes)}")
        print(f"  DequantizeLinear Nodes: {len(dq_nodes)}")
        
        if len(q_nodes) > 0:
            print("  ✓ Model contains Explicit Quantization nodes (QDQ format).")
            return True
        else:
            print("  ⚠ No QDQ nodes found. Might be FP32 or improperly exported.")
            return False
            
    except ImportError:
        print("  onnx library not installed. Skipping ONNX check.")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./scripts/outputs_quantsim_v2")
    args = parser.parse_args()
    
    print("=== Quantization Verification Tool ===")
    
    # 1. Check Pytorch
    pth = os.path.join(args.output_dir, "vda_int8.pth")
    inspect_pytorch_checkpoint(pth)
    
    # 2. Check Encodings
    enc = os.path.join(args.output_dir, "vda_int8.encodings")
    inspect_encodings(enc)
    
    # 3. Check ONNX
    onnx_path = os.path.join(args.output_dir, "vda_int8.onnx")
    inspect_onnx(onnx_path)

if __name__ == "__main__":
    main()
