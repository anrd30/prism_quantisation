# AIMET Quantization for Video-Depth-Anything

This repository contains scripts to quantize the [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) model using AIMET, enabling efficient INT8 inference.

## Key Features

- **QuantizableAttention**: Usage of standard PyTorch attention instead of xFormers to ensure AIMET compatibility.
- **Auto-Replacement**: Utilities to automatically replace `MemEffAttention` and `CrossAttention` layers in the pre-trained model.
- **INT8 Export**: Successfully exports INT8 ONNX and PyTorch models with high accuracy.



## Results Showcase

Comparison of FP32 Baseline vs INT8 Quantized Model (266x266 resolution).
The quantized model achieves **0.998 correlation** with the baseline.



| FP32 Baseline | INT8 Quantized (CLE + AdaRound) |
| :---: | :---: |
| ![FP32](assets/demo/davis_rollercoaster_fp32.gif) | ![INT8](assets/demo/davis_rollercoaster_int8_sim.gif) |

## Mobile Performance Stats (Samsung NPU)
Target usage: On-device Video Portrait Mode.

| Metric | Improvement | Value (Est.) |
| :--- | :--- | :--- |
| **Storage** | **4x Smaller** | 85MB → 21MB |
| **Battery** | **4x Less DRAM** | Efficient Bandwidth |
| **Speed** | **Real-time** | 60+ FPS on NPU |



## Prerequisites

1. Install [AIMET](https://github.com/quic/aimet).
2. Clone Video-Depth-Anything into this directory:
   ```bash
   git clone https://github.com/DepthAnything/Video-Depth-Anything
   ```
   (Or ensure it is present at `./Video-Depth-Anything`)

## Essential Scripts

All essential scripts are located in the `scripts/` directory:

| Script | Purpose |
|--------|---------|
| `scripts/qSim_v2.py` | **Main Script**. Loads the model, applies the quantization fix, runs calibration, and exports the INT8 model. |
| `scripts/run_quantized_inference.py` | Runs the quantized model on a video file and produces a depth map video. |
| `scripts/benchmark_accuracy.py` | Compares the accuracy and speed of the quantized model vs the FP32 original. |
| `scripts/quantizable_attention.py` | Contains the `QuantizableAttention` class definition (the core fix). |
| `scripts/prepare_model_for_quantization.py` | Helper utility to replace attention layers in the VDA model. |

## Quick Start

### 1. Run Quantization & Export
This produces the INT8 model and encodings in `outputs_quantsim_v2/`.
```bash
python scripts/qSim_v2.py --encoder vits
```

### 2. Benchmark Accuracy
Verify the quantized model matches the original's depth output.
```bash
python scripts/benchmark_accuracy.py --encoder vits --num_frames 10
```

### 3. Run Inference on Video
Generate a depth estimation video using the quantized model.
```bash
python scripts/run_quantized_inference.py --encoder vits --input_video path/to/video.mp4
```

## Testing

To verify the numerical equivalence of the attention replacement:
```bash
python scripts/test_quantizable_attention.py
```
