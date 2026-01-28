"""
Prepare VDA Model for AIMET Quantization

This script provides utilities to replace xFormers-based attention layers
with AIMET-compatible PyTorch attention layers.

Usage:
    from prepare_model_for_quantization import prepare_vda_for_quantization
    
    model = load_vda("vits", metric=False)
    model = prepare_vda_for_quantization(model)
    # Now model is ready for AIMET QuantizationSimModel
"""

import sys
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn as nn

# Add VDA repo to path
VDA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)

from quantizable_attention import QuantizableAttention, QuantizableCrossAttention


logger = logging.getLogger("prepare_model_for_quantization")
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """
    Get a submodule by its dot-separated name path.
    
    Args:
        model: Root module
        name: Dot-separated path like "pretrained.blocks.0.attn"
    
    Returns:
        The submodule at that path
    """
    parts = name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """
    Replace a submodule at the given dot-separated name path.
    
    Args:
        model: Root module
        name: Dot-separated path like "pretrained.blocks.0.attn"
        new_module: New module to put at that location
    """
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    
    last_part = parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)


def find_attention_modules(model: nn.Module) -> Dict[str, Tuple[str, nn.Module]]:
    """
    Find all attention modules that need to be replaced in the model.
    
    Returns a dict mapping module names to (module_type, module) tuples where:
    - module_type is 'MemEffAttention' or 'CrossAttention'
    - module is the actual module instance
    
    Args:
        model: Model to search
        
    Returns:
        Dict mapping name -> (type_name, module)
    """
    attention_modules = OrderedDict()
    
    for name, module in model.named_modules():
        module_class_name = module.__class__.__name__
        
        if module_class_name == 'MemEffAttention':
            attention_modules[name] = ('MemEffAttention', module)
            
        elif module_class_name == 'CrossAttention':
            # Check if it's using xFormers
            if getattr(module, '_use_memory_efficient_attention_xformers', False):
                attention_modules[name] = ('CrossAttention', module)
    
    return attention_modules


def replace_mem_eff_attention(model: nn.Module, verbose: bool = True) -> Tuple[nn.Module, int]:
    """
    Replace all MemEffAttention modules with QuantizableAttention.
    
    Args:
        model: Model to modify (modified in-place)
        verbose: Print replacement progress
        
    Returns:
        Tuple of (modified model, number of replacements)
    """
    count = 0
    
    # Find all MemEffAttention modules first (can't modify during iteration)
    replacements = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'MemEffAttention':
            replacements.append((name, module))
    
    # Perform replacements
    for name, original_module in replacements:
        # Determine device of original module
        device = next(original_module.parameters()).device
        
        # Create quantizable replacement with copied weights
        new_attention = QuantizableAttention.from_mem_eff_attention(original_module)
        
        # Move to same device as original
        new_attention = new_attention.to(device)
        
        # Replace in model
        set_module_by_name(model, name, new_attention)
        count += 1
        
        if verbose:
            logger.info(f"Replaced MemEffAttention: {name}")
            logger.info(f"  Config: dim={new_attention.qkv.in_features}, heads={new_attention.num_heads}")
    
    return model, count



def replace_cross_attention(model: nn.Module, verbose: bool = True) -> Tuple[nn.Module, int]:
    """
    Replace all xFormers-enabled CrossAttention modules with QuantizableCrossAttention.
    
    Args:
        model: Model to modify (modified in-place)
        verbose: Print replacement progress
        
    Returns:
        Tuple of (modified model, number of replacements)
    """
    count = 0
    
    # Find all CrossAttention modules using xFormers
    replacements = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'CrossAttention':
            if getattr(module, '_use_memory_efficient_attention_xformers', False):
                replacements.append((name, module))
    
    # Perform replacements
    for name, original_module in replacements:
        # Determine device of original module
        device = next(original_module.parameters()).device
        
        # Create quantizable replacement with copied weights
        new_attention = QuantizableCrossAttention.from_cross_attention(original_module)
        
        # Move to same device as original
        new_attention = new_attention.to(device)
        
        # Replace in model
        set_module_by_name(model, name, new_attention)
        count += 1
        
        if verbose:
            logger.info(f"Replaced CrossAttention (xFormers): {name}")
            logger.info(f"  Config: query_dim={new_attention.to_q.in_features}, heads={new_attention.heads}")
    
    return model, count



def disable_cross_attention_xformers(model: nn.Module, verbose: bool = True) -> Tuple[nn.Module, int]:
    """
    Disable xFormers mode in attention modules that have the flag.
    
    This affects:
    - CrossAttention
    - TemporalAttention (inherits from CrossAttention)
    - Any other module with _use_memory_efficient_attention_xformers flag
    
    These modules have fallback to standard attention when xFormers is disabled.
    
    Args:
        model: Model to modify (modified in-place)
        verbose: Print modification progress
        
    Returns:
        Tuple of (modified model, number of modifications)
    """
    count = 0
    
    for name, module in model.named_modules():
        # Check if module has the xFormers flag and it's enabled
        if hasattr(module, '_use_memory_efficient_attention_xformers'):
            if getattr(module, '_use_memory_efficient_attention_xformers', False):
                module._use_memory_efficient_attention_xformers = False
                count += 1
                
                class_name = module.__class__.__name__
                if verbose:
                    logger.info(f"Disabled xFormers in {class_name}: {name}")
    
    return model, count



def prepare_vda_for_quantization(
    model: nn.Module, 
    replace_cross_attn: bool = False,
    verbose: bool = True
) -> nn.Module:
    """
    Prepare a Video-Depth-Anything model for AIMET quantization.
    
    This function:
    1. Replaces all MemEffAttention with QuantizableAttention
    2. Either replaces CrossAttention or disables its xFormers mode
    
    Args:
        model: VDA model to prepare
        replace_cross_attn: If True, replace CrossAttention modules.
                           If False, just disable xFormers mode in them.
        verbose: Print replacement progress
        
    Returns:
        Modified model ready for AIMET
    """
    logger.info("=" * 60)
    logger.info("Preparing VDA model for AIMET quantization")
    logger.info("=" * 60)
    
    # Step 1: Replace MemEffAttention (DINOv2 encoder)
    model, mem_eff_count = replace_mem_eff_attention(model, verbose=verbose)
    logger.info(f"\nReplaced {mem_eff_count} MemEffAttention modules")
    
    # Step 2: Handle CrossAttention (motion module)
    if replace_cross_attn:
        model, cross_count = replace_cross_attention(model, verbose=verbose)
        logger.info(f"Replaced {cross_count} CrossAttention modules")
    else:
        model, cross_count = disable_cross_attention_xformers(model, verbose=verbose)
        logger.info(f"Disabled xFormers in {cross_count} CrossAttention modules")
    
    logger.info("=" * 60)
    logger.info("Model preparation complete!")
    logger.info("=" * 60)
    
    return model


def verify_no_xformers(model: nn.Module) -> bool:
    """
    Verify that the model no longer contains any xFormers-dependent layers.
    
    Args:
        model: Model to check
        
    Returns:
        True if model is xFormers-free, False otherwise
    """
    issues = []
    
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        
        # Check for MemEffAttention
        if class_name == 'MemEffAttention':
            issues.append(f"Found MemEffAttention: {name}")
        
        # Check for CrossAttention with xFormers enabled
        if class_name == 'CrossAttention':
            if getattr(module, '_use_memory_efficient_attention_xformers', False):
                issues.append(f"Found CrossAttention with xFormers enabled: {name}")
    
    if issues:
        logger.warning("Model still has xFormers dependencies:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("✓ Model is xFormers-free and ready for AIMET")
        return True


def count_attention_layers(model: nn.Module) -> Dict[str, int]:
    """
    Count different types of attention layers in the model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dict mapping attention type names to counts
    """
    counts = {}
    
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        
        if 'Attention' in class_name:
            counts[class_name] = counts.get(class_name, 0) + 1
    
    return counts


# -----------------------------------------------------------------------------
# CLI for testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare VDA model for AIMET quantization")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--metric", action="store_true", help="Use metric depth model")
    parser.add_argument("--replace_cross_attn", action="store_true", 
                       help="Replace CrossAttention (vs just disabling xFormers)")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save prepared model state dict")
    args = parser.parse_args()
    
    # Load VDA model
    from video_depth_anything.video_depth import VideoDepthAnything
    
    configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    checkpoint_name = "metric_video_depth_anything" if args.metric else "video_depth_anything"
    
    model = VideoDepthAnything(**configs[args.encoder], metric=args.metric)
    state = torch.load(f"{VDA_PATH}/checkpoints/{checkpoint_name}_{args.encoder}.pth", map_location="cpu")
    model.load_state_dict(state)
    
    print(f"\nLoaded VDA model: {args.encoder}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Count attention layers before
    print("\nAttention layers BEFORE preparation:")
    for attn_type, count in count_attention_layers(model).items():
        print(f"  {attn_type}: {count}")
    
    # Prepare model
    model = prepare_vda_for_quantization(
        model, 
        replace_cross_attn=args.replace_cross_attn,
        verbose=True
    )
    
    # Count attention layers after
    print("\nAttention layers AFTER preparation:")
    for attn_type, count in count_attention_layers(model).items():
        print(f"  {attn_type}: {count}")
    
    # Verify
    is_ready = verify_no_xformers(model)
    
    # Save if requested
    if args.save and is_ready:
        torch.save(model.state_dict(), args.save)
        print(f"\nSaved prepared model to: {args.save}")
