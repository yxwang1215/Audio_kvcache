"""
Visualize Key-Value Cache Similarity Heatmaps

This script generates cosine similarity heatmaps for k and v caches across different layers and heads.
For each layer, it creates separate heatmaps for key cache and value cache.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.insert(0, current_dir)

# Import model and utilities
from modeling_qwen2_5_omni_sparsekv import Qwen2_5OmniForConditionalGeneration
from processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from sparsemm_utils import DynamicCacheSplitHeadFlatten
from transformers.cache_utils import DynamicCache

# Import audio head utilities for SparseMM (if needed)
try:
    from audio_head_utils import (
        load_audio_head_scores,
        get_audio_head_score_path
    )
    AUDIO_HEAD_UTILS_AVAILABLE = True
except ImportError:
    AUDIO_HEAD_UTILS_AVAILABLE = False
    print("Warning: audio_head_utils not available, SparseMM head scores cannot be loaded")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def compute_cosine_similarity_matrix(cache_tensor: torch.Tensor) -> np.ndarray:
    """
    Compute cosine similarity matrix for a cache tensor.
    
    Args:
        cache_tensor: Tensor of shape [seq_len, head_dim] or [num_heads * seq_len, head_dim]
    
    Returns:
        Similarity matrix of shape [seq_len, seq_len]
    """
    # Ensure tensor is on CPU and convert to numpy
    # Convert to float32 first to handle bfloat16 and other unsupported types
    if isinstance(cache_tensor, torch.Tensor):
        # Convert to float32 to avoid issues with bfloat16 (numpy doesn't support it)
        cache_tensor = cache_tensor.detach().cpu().float()
        cache_np = cache_tensor.numpy()
    else:
        cache_np = cache_tensor
    
    # Normalize to unit vectors for cosine similarity
    norms = np.linalg.norm(cache_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    cache_normalized = cache_np / norms
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(cache_normalized, cache_normalized.T)
    
    return similarity_matrix


def extract_cache_from_past_key_values(
    past_key_values, 
    num_layers: int,
    num_heads: int,
    num_key_value_heads: int,
    model=None,
    total_length: Optional[int] = None
) -> Dict[str, Dict[int, List[torch.Tensor]]]:
    """
    Extract k and v caches from past_key_values.
    
    Args:
        past_key_values: Cache object (DynamicCacheSplitHeadFlatten or DynamicCache)
        num_layers: Number of layers
        num_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        model: Model object (optional, used to get head_lens for SparseMM)
        total_length: Total sequence length (input + generated tokens). If cache length exceeds this, it will be trimmed.
    
    Returns:
        Dictionary with structure:
        {
            'key': {layer_idx: [head0_cache, head1_cache, ...]},  # num_key_value_heads caches
            'value': {layer_idx: [head0_cache, head1_cache, ...]}  # num_key_value_heads caches
        }
    """
    cache_data = {
        'key': {},
        'value': {}
    }
    
    if isinstance(past_key_values, DynamicCacheSplitHeadFlatten):
        # For SparseMM cache: key_cache and value_cache are List[torch.Tensor]
        # Each layer has a flattened tensor [total_tokens, head_dim]
        # We need to split by heads using head_lens from kv_cluster
        for layer_idx in range(len(past_key_values.key_cache)):
            if layer_idx >= len(past_key_values.key_cache):
                continue
                
            key_cache_layer = past_key_values.key_cache[layer_idx]
            value_cache_layer = past_key_values.value_cache[layer_idx]
            
            if key_cache_layer is None or value_cache_layer is None:
                continue
            
            # Try to get head_lens from model's kv_cluster
            head_lens = None
            if model is not None:
                try:
                    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                        base_model = model.thinker.model
                        if hasattr(base_model, 'layers') and layer_idx < len(base_model.layers):
                            layer = base_model.layers[layer_idx]
                            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'kv_cluster'):
                                kv_cluster = layer.self_attn.kv_cluster
                                if hasattr(kv_cluster, 'head_lens') and kv_cluster.head_lens is not None:
                                    head_lens = kv_cluster.head_lens.cpu().numpy()
                except Exception as e:
                    print(f"Warning: Could not get head_lens for layer {layer_idx}: {e}")
            
            # Split cache by heads
            if isinstance(key_cache_layer, torch.Tensor):
                if key_cache_layer.dim() == 2:
                    # Flatten format: [total_tokens, head_dim]
                    total_tokens, head_dim = key_cache_layer.shape
                    
                    # Use num_key_value_heads instead of num_heads for GQA
                    if head_lens is not None and len(head_lens) == num_key_value_heads:
                        # Use head_lens to split (each head may have different length)
                        key_heads = []
                        value_heads = []
                        offset = 0
                        for head_idx in range(num_key_value_heads):
                            head_len = int(head_lens[head_idx])
                            if offset + head_len <= total_tokens:
                                key_head = key_cache_layer[offset:offset+head_len, :]
                                value_head = value_cache_layer[offset:offset+head_len, :]
                                key_heads.append(key_head)
                                value_heads.append(value_head)
                                offset += head_len
                            else:
                                # If we run out of tokens, break
                                print(f"Warning: Layer {layer_idx}, head {head_idx}: Not enough tokens (offset={offset}, head_len={head_len}, total={total_tokens})")
                                break
                        
                        if len(key_heads) == num_key_value_heads:
                            cache_data['key'][layer_idx] = key_heads
                            cache_data['value'][layer_idx] = value_heads
                        else:
                            # Fallback: try to split equally
                            print(f"Warning: Layer {layer_idx}: Could not split using head_lens, using equal split")
                            tokens_per_head = total_tokens // num_key_value_heads
                            if tokens_per_head > 0:
                                key_heads = [key_cache_layer[i*tokens_per_head:(i+1)*tokens_per_head, :] 
                                           for i in range(num_key_value_heads)]
                                value_heads = [value_cache_layer[i*tokens_per_head:(i+1)*tokens_per_head, :] 
                                             for i in range(num_key_value_heads)]
                                cache_data['key'][layer_idx] = key_heads
                                cache_data['value'][layer_idx] = value_heads
                            else:
                                cache_data['key'][layer_idx] = [key_cache_layer]
                                cache_data['value'][layer_idx] = [value_cache_layer]
                    else:
                        # Equal split: assume each head has same number of tokens
                        tokens_per_head = total_tokens // num_key_value_heads
                        if tokens_per_head > 0:
                            key_heads = [key_cache_layer[i*tokens_per_head:(i+1)*tokens_per_head, :] 
                                       for i in range(num_key_value_heads)]
                            value_heads = [value_cache_layer[i*tokens_per_head:(i+1)*tokens_per_head, :] 
                                         for i in range(num_key_value_heads)]
                            cache_data['key'][layer_idx] = key_heads
                            cache_data['value'][layer_idx] = value_heads
                        else:
                            # Single head or too few tokens
                            cache_data['key'][layer_idx] = [key_cache_layer]
                            cache_data['value'][layer_idx] = [value_cache_layer]
                elif key_cache_layer.dim() == 4:
                    # 4D format: [batch, num_heads, seq_len, head_dim]
                    batch, heads, seq_len, head_dim = key_cache_layer.shape
                    key_heads = [key_cache_layer[0, h, :, :] for h in range(heads)]
                    value_heads = [value_cache_layer[0, h, :, :] for h in range(heads)]
                    cache_data['key'][layer_idx] = key_heads
                    cache_data['value'][layer_idx] = value_heads
                else:
                    # Unknown format, treat as single head
                    cache_data['key'][layer_idx] = [key_cache_layer]
                    cache_data['value'][layer_idx] = [value_cache_layer]
    
    elif isinstance(past_key_values, DynamicCache):
        # For standard DynamicCache: each layer has tuple of (key, value)
        # key and value are [batch, num_heads, seq_len, head_dim]
        for layer_idx in range(len(past_key_values)):
            if layer_idx >= len(past_key_values):
                continue
                
            layer_cache = past_key_values[layer_idx]
            if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                key_cache, value_cache = layer_cache
                # key_cache shape: [batch, num_heads, seq_len, head_dim]
                if key_cache.dim() == 4:
                    batch, heads, seq_len, head_dim = key_cache.shape
                    # For GQA, heads should be num_key_value_heads
                    # Extract each head: [seq_len, head_dim]
                    num_kv_heads = min(heads, num_key_value_heads)
                    key_heads = [key_cache[0, h, :, :] for h in range(num_kv_heads)]
                    value_heads = [value_cache[0, h, :, :] for h in range(num_kv_heads)]
                    cache_data['key'][layer_idx] = key_heads
                    cache_data['value'][layer_idx] = value_heads
                elif key_cache.dim() == 3:
                    # Shape: [num_heads, seq_len, head_dim]
                    heads, seq_len, head_dim = key_cache.shape
                    num_kv_heads = min(heads, num_key_value_heads)
                    key_heads = [key_cache[h, :, :] for h in range(num_kv_heads)]
                    value_heads = [value_cache[h, :, :] for h in range(num_kv_heads)]
                    cache_data['key'][layer_idx] = key_heads
                    cache_data['value'][layer_idx] = value_heads
    
    # Trim cache to total_length if specified and cache is longer
    if total_length is not None:
        for cache_type in ['key', 'value']:
            for layer_idx in list(cache_data[cache_type].keys()):
                layer_caches = cache_data[cache_type][layer_idx]
                trimmed_caches = []
                for head_idx, head_cache in enumerate(layer_caches):
                    if head_cache is not None and head_cache.dim() >= 2:
                        cache_len = head_cache.shape[0]
                        if cache_len > total_length:
                            # Trim to total_length
                            trimmed_cache = head_cache[:total_length, :]
                            trimmed_caches.append(trimmed_cache)
                            print(f"Warning: Layer {layer_idx}, {cache_type} cache, Head {head_idx}: "
                                  f"Cache length ({cache_len}) > total length ({total_length}), trimmed to {total_length}")
                        else:
                            trimmed_caches.append(head_cache)
                    else:
                        trimmed_caches.append(head_cache)
                cache_data[cache_type][layer_idx] = trimmed_caches
    
    return cache_data


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    title: str,
    output_path: str,
    layer_idx: int,
    head_idx: Optional[int] = None,
    cache_type: str = "key"
):
    """
    Plot and save a similarity heatmap.
    
    Args:
        similarity_matrix: NxN similarity matrix
        title: Plot title
        output_path: Path to save the figure
        layer_idx: Layer index
        head_idx: Head index (optional)
        cache_type: "key" or "value"
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use colormap similar to the reference image (purple/blue to yellow)
    # Using 'plasma' or 'viridis' colormap for better visualization
    # vmin and vmax based on actual similarity range
    vmin = float(np.min(similarity_matrix))
    vmax = float(np.max(similarity_matrix))
    
    # Create heatmap with colormap that goes from dark purple/blue to bright yellow
    im = ax.imshow(similarity_matrix, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # Set labels
    n_tokens = similarity_matrix.shape[0]
    # For inference steps, we label from 0 to n_tokens-1 (actual token positions)
    ax.set_xlabel('Inference Step', fontsize=12)
    ax.set_ylabel('Inference Step', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set ticks (show every 10th token if too many)
    # Labels should show actual inference step numbers (0, 1, 2, ..., n_tokens-1)
    if n_tokens > 50:
        tick_step = max(1, n_tokens // 20)
        ticks = list(range(0, n_tokens, tick_step))
        if ticks[-1] != n_tokens - 1:
            ticks.append(n_tokens - 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticklabels([str(t) for t in ticks])
    else:
        # For smaller matrices, show all ticks
        ax.set_xticks(range(n_tokens))
        ax.set_yticks(range(n_tokens))
        if n_tokens <= 20:
            ax.set_xticklabels(range(n_tokens))
            ax.set_yticklabels(range(n_tokens))
    
    # Add grid for better readability
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved heatmap to: {output_path}")


def visualize_cache_similarity(
    model,
    processor,
    sample_data: Dict,
    output_dir: str,
    num_samples: int = 1,
    max_new_tokens: int = 100,
    layers_to_plot: Optional[List[int]] = None,
    heads_to_plot: Optional[List[int]] = None,
    average_heads: bool = False
):
    """
    Main function to visualize cache similarity.
    
    Args:
        model: The model
        processor: The processor
        sample_data: Dictionary with 'audio' and 'prompt' keys
        output_dir: Directory to save heatmaps
        num_samples: Number of samples to process (default: 1)
        max_new_tokens: Maximum number of tokens to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model config
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
        base_model = model.thinker.model
        num_layers = base_model.config.num_hidden_layers
        num_heads = base_model.config.num_attention_heads
        num_key_value_heads = getattr(base_model.config, 'num_key_value_heads', num_heads)
    else:
        print("Warning: Cannot determine num_layers and num_heads, using defaults")
        num_layers = 28  # Default for Qwen2.5-Omni-3B
        num_heads = 16
        num_key_value_heads = 4  # Default for Qwen2.5-Omni-3B (GQA)
    
    print(f"Model config: {num_layers} layers, {num_heads} attention heads, {num_key_value_heads} key-value heads")
    
    # Process sample
    audio_np = sample_data['audio']
    prompt_text = sample_data.get('prompt', 'Transcribe this audio accurately.')
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_np},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    # Process multimodal info
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    if isinstance(text, list):
        text = text[0] if len(text) > 0 else ""
    
    # Tokenize
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True
    )
    inputs = inputs.to(model.device).to(model.dtype)
    
    print(f"Input tokens: {inputs.input_ids.shape[1]}")
    
    # Check if SparseMM is enabled by checking if any layer has kv_cluster
    # Use the sparsemm_enabled flag from main() if available, otherwise auto-detect
    use_sparsemm = getattr(model, '_sparsemm_enabled', False)
    if not use_sparsemm:
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
            if hasattr(base_model, 'layers') and len(base_model.layers) > 0:
                first_layer = base_model.layers[0]
                if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'kv_cluster'):
                    kv_cluster = first_layer.self_attn.kv_cluster
                    if kv_cluster is not None:
                        use_sparsemm = True
                        print("[Cache] Detected SparseMM is enabled, using DynamicCacheSplitHeadFlatten")
    
    # Initialize cache based on whether SparseMM is enabled
    if use_sparsemm:
        try:
            past_key_values = DynamicCacheSplitHeadFlatten()
            print("[Cache] Using DynamicCacheSplitHeadFlatten for SparseMM")
        except Exception as e:
            print(f"Warning: Could not create DynamicCacheSplitHeadFlatten: {e}, falling back to DynamicCache")
            past_key_values = DynamicCache()
            use_sparsemm = False
    else:
        past_key_values = DynamicCache()
        print("[Cache] Using standard DynamicCache (SparseMM not enabled)")
    
    # Generate with cache
    print(f"Generating {max_new_tokens} tokens...")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            use_audio_in_video=True,
            return_audio=False,
            thinker_max_new_tokens=max_new_tokens,
            thinker_do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            past_key_values=past_key_values
        )
    
    generated_tokens = out_ids.shape[1] - inputs.input_ids.shape[1]
    total_length = out_ids.shape[1]  # Total tokens = input + generated
    print(f"Generated {generated_tokens} new tokens")
    print(f"Total sequence length: {total_length} tokens (input: {inputs.input_ids.shape[1]}, generated: {generated_tokens})")
    
    # Extract cache data
    print("Extracting cache data...")
    cache_data = extract_cache_from_past_key_values(past_key_values, num_layers, num_heads, num_key_value_heads, model=model, total_length=total_length)
    
    # Print summary of extracted cache
    print(f"\nExtracted cache summary:")
    for cache_type in ['key', 'value']:
        print(f"  {cache_type.upper()} cache: {len(cache_data[cache_type])} layers")
        for layer_idx in sorted(cache_data[cache_type].keys()):
            num_heads_in_layer = len(cache_data[cache_type][layer_idx])
            print(f"    Layer {layer_idx}: {num_heads_in_layer} head(s)")
    
    # Generate heatmaps for each layer
    print("Generating heatmaps...")
    
    # Filter layers if specified
    all_layer_indices = set()
    for cache_type in ['key', 'value']:
        all_layer_indices.update(cache_data[cache_type].keys())
    
    if layers_to_plot is not None:
        layer_indices_to_process = [idx for idx in sorted(all_layer_indices) if idx in layers_to_plot]
    else:
        layer_indices_to_process = sorted(all_layer_indices)
    
    print(f"Processing {len(layer_indices_to_process)} layers: {layer_indices_to_process}")
    
    # Print KV cache lengths for selected layers
    print(f"\n{'='*60}")
    print(f"KV Cache Lengths for Selected Layers:")
    print(f"Total sequence length: {total_length} tokens")
    print(f"{'='*60}")
    for layer_idx in layer_indices_to_process:
        print(f"\nLayer {layer_idx}:")
        for cache_type in ['key', 'value']:
            if layer_idx in cache_data[cache_type]:
                layer_caches = cache_data[cache_type][layer_idx]
                head_lengths = []
                for head_idx, head_cache in enumerate(layer_caches):
                    if head_cache is not None and head_cache.dim() >= 2:
                        seq_len = head_cache.shape[0]
                        head_lengths.append(f"Head{head_idx}:{seq_len}")
                    elif head_cache is not None:
                        seq_len = head_cache.numel()
                        head_lengths.append(f"Head{head_idx}:{seq_len}")
                    else:
                        head_lengths.append(f"Head{head_idx}:None")
                print(f"  {cache_type.upper():5s} cache: {', '.join(head_lengths)}")
            else:
                print(f"  {cache_type.upper():5s} cache: Not found")
    print(f"{'='*60}\n")
    
    # Statistics
    total_plots = 0
    skipped_plots = 0
    
    for cache_type in ['key', 'value']:
        print(f"\n=== Processing {cache_type.upper()} cache ===")
        for layer_idx in layer_indices_to_process:
            if layer_idx not in cache_data[cache_type]:
                print(f"  Layer {layer_idx}: No {cache_type} cache found, skipping")
                continue
                
            layer_caches = cache_data[cache_type][layer_idx]
            # Print head lengths (may be different for each head)
            head_lengths = [cache.shape[0] if cache is not None and cache.dim() == 2 else 0 
                          for cache in layer_caches]
            print(f"  Layer {layer_idx}: Found {len(layer_caches)} head(s) with lengths: {head_lengths}")
            
            # Process each head in this layer
            for head_idx, head_cache in enumerate(layer_caches):
                # Filter heads if specified
                if heads_to_plot is not None and head_idx not in heads_to_plot:
                    continue
                    
                if head_cache is None:
                    print(f"    Head {head_idx}: Cache is None, skipping")
                    skipped_plots += 1
                    continue
                    
                if head_cache.numel() == 0:
                    print(f"    Head {head_idx}: Cache is empty, skipping")
                    skipped_plots += 1
                    continue
                
                # Get sequence length
                if head_cache.dim() == 2:
                    seq_len, head_dim = head_cache.shape
                elif head_cache.dim() == 1:
                    print(f"    Head {head_idx}: Warning - Unexpected 1D cache shape {head_cache.shape}, attempting to reshape")
                    # Try to infer: if it's a single vector, create a 1x1 similarity matrix
                    if head_cache.numel() == head_dim:
                        seq_len = 1
                    else:
                        print(f"    Head {head_idx}: Cannot process 1D cache, skipping")
                        skipped_plots += 1
                        continue
                else:
                    print(f"    Head {head_idx}: Warning - Unexpected cache shape {head_cache.shape}, skipping")
                    skipped_plots += 1
                    continue
                
                if seq_len < 1:
                    print(f"    Head {head_idx}: seq_len={seq_len} is invalid, skipping")
                    skipped_plots += 1
                    continue
                
                if seq_len == 1:
                    print(f"    Head {head_idx}: seq_len=1 (single token), creating 1x1 similarity matrix")
                    # For single token, similarity is always 1.0
                    similarity_matrix = np.array([[1.0]])
                else:
                    print(f"    Head {head_idx}: Processing - seq_len={seq_len}, head_dim={head_dim}")
                    # Compute similarity matrix
                    similarity_matrix = compute_cosine_similarity_matrix(head_cache)
                
                # Create title based on cache type
                if cache_type == 'key':
                    title = f"Layer {layer_idx} Head {head_idx} - Key Cache Cosine Similarity Heatmap"
                else:
                    title = f"Layer {layer_idx} Head {head_idx} - Value Cache Cosine Similarity Heatmap"
                
                # Create output path
                output_path = os.path.join(
                    output_dir,
                    f"layer_{layer_idx}_head_{head_idx}_{cache_type}_similarity.png"
                )
                
                # Plot and save
                try:
                    plot_similarity_heatmap(
                        similarity_matrix,
                        title,
                        output_path,
                        layer_idx,
                        head_idx,
                        cache_type
                    )
                    total_plots += 1
                    print(f"    ✓ Saved: {output_path}")
                except Exception as e:
                    print(f"    ✗ Error plotting layer {layer_idx}, head {head_idx}, {cache_type}: {e}")
                    skipped_plots += 1
                    import traceback
                    traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total plots generated: {total_plots}")
    print(f"  Skipped: {skipped_plots}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Visualize KV cache similarity heatmaps')
    parser.add_argument('--model-path', type=str, 
                       default='/mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B',
                       help='Path to model')
    parser.add_argument('--audio-path', type=str, default=None,
                       help='Path to audio file (if None, will use first LibriSpeech sample)')
    parser.add_argument('--output-dir', type=str, default='./cache_similarity_heatmaps',
                       help='Output directory for heatmaps')
    parser.add_argument('--max-new-tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to process from LibriSpeech dataset (default: 1). Only applies when --audio-path is not specified.')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated list of layer indices to plot (e.g., "0,27" for first and last). If None, plot all layers.')
    parser.add_argument('--heads', type=str, default=None,
                       help='Comma-separated list of head indices to plot (e.g., "0,1,2"). If None, plot all heads.')
    parser.add_argument('--average-heads', action='store_true',
                       help='Average similarity across all heads for each layer (creates one plot per layer instead of per head)')
    parser.add_argument('--enable-sparsemm', action='store_true',
                       help='Enable SparseMM attention (requires head score file)')
    parser.add_argument('--sparsemm-budget', type=float, default=4096.0,
                       help='SparseMM budget (absolute value >= 1.0, or ratio < 1.0, e.g., 0.7 for 70%%)')
    parser.add_argument('--sparsemm-ratio', type=float, default=0.2,
                       help='SparseMM ratio parameter (default: 0.2)')
    parser.add_argument('--sparsemm-window-size', type=int, default=64,
                       help='SparseMM window size (default: 64)')
    parser.add_argument('--sparsemm-kernel-size', type=int, default=7,
                       help='SparseMM kernel size (default: 7)')
    parser.add_argument('--sparsemm-pooling', type=str, default='maxpool',
                       choices=['maxpool', 'avgpool'],
                       help='SparseMM pooling method (default: maxpool)')
    parser.add_argument('--head-score-file', type=str, default='aggregated',
                       help='Head score file type (default: aggregated)')
    parser.add_argument('--head-score-batch', type=str, default=None,
                       help='Head score batch identifier (optional)')
    args = parser.parse_args()
    
    # Parse layers and heads
    layers_to_plot = None
    if args.layers:
        try:
            layers_to_plot = [int(x.strip()) for x in args.layers.split(',')]
        except ValueError:
            print(f"Warning: Invalid layers format '{args.layers}', plotting all layers")
    
    heads_to_plot = None
    if args.heads:
        try:
            heads_to_plot = [int(x.strip()) for x in args.heads.split(',')]
        except ValueError:
            print(f"Warning: Invalid heads format '{args.heads}', plotting all heads")
    
    # Load model
    print("Loading model...")
    device_map = {"": 0}
    processor = Qwen2_5OmniProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.disable_talker()
    model.eval()
    
    # Enable SparseMM if requested
    sparsemm_enabled = False
    if args.enable_sparsemm:
        if not AUDIO_HEAD_UTILS_AVAILABLE:
            print("Error: --enable-sparsemm requires audio_head_utils, but it's not available")
            return
        
        if hasattr(model, 'enable_sparsemm_attention'):
            if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                base_model = model.thinker.model
                num_hidden_layers = base_model.config.num_hidden_layers
                num_attention_heads = base_model.config.num_attention_heads
                
                # Load head score file
                head_score_path = get_audio_head_score_path(
                    score_type=args.head_score_file,
                    batch=args.head_score_batch
                )
                
                print(f"[SparseMM] Loading audio head scores from: {head_score_path}")
                
                # Load head scores (simplified version)
                try:
                    head_scores = load_audio_head_scores(head_score_path, use_mean=True)
                    
                    # Convert to SparseMM format
                    if isinstance(head_scores, dict) and "__uniform__" in head_scores:
                        uniform_val = float(head_scores["__uniform__"])
                        head_score_list = [uniform_val] * (num_hidden_layers * num_attention_heads)
                    else:
                        head_score_list = []
                        for layer_idx in range(num_hidden_layers):
                            for head_idx in range(num_attention_heads):
                                key = f"{layer_idx}-{head_idx}"
                                if key in head_scores:
                                    score_value = head_scores[key]
                                    if isinstance(score_value, list):
                                        score = np.mean(score_value)
                                    else:
                                        score = float(score_value)
                                else:
                                    score = 0.0
                                head_score_list.append(score)
                    
                    # Normalize
                    head_score_array = np.array(head_score_list)
                    if head_score_array.sum() > 0:
                        head_score_array = head_score_array / head_score_array.sum()
                    else:
                        head_score_array = np.ones_like(head_score_array) / len(head_score_array)
                    
                    head_score_data = {
                        'scores': head_score_array.tolist(),
                        'tensor': torch.tensor(head_score_array, dtype=torch.float32)
                    }
                    
                    # Determine if budget is ratio or absolute
                    budget_is_ratio = (args.sparsemm_budget < 1.0)
                    if budget_is_ratio:
                        init_budget = 4096  # Placeholder, will be updated per sample
                        print(f"[SparseMM] Budget is a ratio ({args.sparsemm_budget:.2%}), using placeholder {init_budget} for initialization")
                    else:
                        init_budget = int(args.sparsemm_budget)
                        print(f"[SparseMM] Budget is absolute value: {init_budget}")
                    
                    sparsemm_config = {
                        'budget': init_budget,
                        'ratio': args.sparsemm_ratio,
                        'window_size': args.sparsemm_window_size,
                        'kernel_size': args.sparsemm_kernel_size,
                        'pooling': args.sparsemm_pooling
                    }
                    
                    print(f"[SparseMM] Enabling SparseMM with config: {sparsemm_config}")
                    model.enable_sparsemm_attention(head_score_data, sparsemm_config)
                    sparsemm_enabled = True
                    model._sparsemm_enabled = True  # Store flag for later use
                    print(f"[SparseMM] SparseMM enabled successfully")
                except Exception as e:
                    print(f"[SparseMM] Error enabling SparseMM: {e}")
                    import traceback
                    traceback.print_exc()
                    print("[SparseMM] Continuing without SparseMM...")
            else:
                print("[SparseMM] Warning: Cannot find thinker model, skipping SparseMM initialization")
        else:
            print("[SparseMM] Warning: Model does not support enable_sparsemm_attention")
    else:
        print("[SparseMM] SparseMM not enabled (use --enable-sparsemm to enable)")
    
    # Load sample data
    if args.audio_path:
        # Load from file
        import soundfile as sf
        import librosa
        audio_np, sr = librosa.load(args.audio_path, sr=16000, mono=True)
        sample_data = {
            'audio': audio_np,
            'prompt': 'Transcribe this audio accurately.',
            'sample_id': os.path.basename(args.audio_path).replace('.', '_'),
            'sample_idx': 0
        }
        
        # Single sample case (audio_path provided)
        print(f"\n{'='*60}")
        print(f"Processing custom audio file: {args.audio_path}")
        print(f"{'='*60}")
        
        # Visualize
        visualize_cache_similarity(
            model,
            processor,
            sample_data,
            args.output_dir,
            num_samples=1,
            max_new_tokens=args.max_new_tokens,
            layers_to_plot=layers_to_plot,
            heads_to_plot=heads_to_plot,
            average_heads=args.average_heads
        )
    else:
        # Use LibriSpeech samples
        # Import from the actual file name (with dot, not underscore)
        import importlib.util
        librispeech_module_path = os.path.join(current_dir, 'LibriSpeech_qwen2.5.py')
        if os.path.exists(librispeech_module_path):
            spec = importlib.util.spec_from_file_location("LibriSpeech_qwen2_5", librispeech_module_path)
            librispeech_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(librispeech_module)
            load_librispeech_long_dataset = librispeech_module.load_librispeech_long_dataset
            librispeech_doc_to_audio = librispeech_module.librispeech_doc_to_audio
            asr_doc_to_text = librispeech_module.asr_doc_to_text
        else:
            print(f"Error: Cannot find LibriSpeech_qwen2.5.py at {librispeech_module_path}")
            return
        librispeech_path = '/mnt/users/wangyuxuan-20250915/Datasets/AudioMarathon/librispeech-long'
        dataset = load_librispeech_long_dataset(librispeech_path, "test-clean")
        if not dataset:
            print("Error: Failed to load dataset")
            return
        
        # Limit number of samples
        num_samples_to_process = min(args.num_samples, len(dataset))
        print(f"Processing {num_samples_to_process} sample(s) from dataset (total: {len(dataset)})")
        
        # Process each sample
        for sample_idx in range(num_samples_to_process):
            doc = dataset[sample_idx]
            audio_result = librispeech_doc_to_audio(doc)
            if audio_result is None:
                print(f"Warning: Failed to load audio for sample {sample_idx}, skipping")
                continue
            
            audio_np, sr = audio_result
            prompt_text = asr_doc_to_text(doc)
            sample_data = {
                'audio': audio_np,
                'prompt': prompt_text,
                'sample_id': doc.get('id', f'sample_{sample_idx}'),
                'sample_idx': sample_idx
            }
            
            # Create output directory for this sample
            if num_samples_to_process > 1:
                sample_output_dir = os.path.join(args.output_dir, f"sample_{sample_idx}_{sample_data['sample_id']}")
            else:
                sample_output_dir = args.output_dir
            
            print(f"\n{'='*60}")
            print(f"Processing sample {sample_idx + 1}/{num_samples_to_process}: {sample_data['sample_id']}")
            print(f"{'='*60}")
            
            # Visualize
            visualize_cache_similarity(
                model,
                processor,
                sample_data,
                sample_output_dir,
                num_samples=1,  # Already processing one at a time
                max_new_tokens=args.max_new_tokens,
                layers_to_plot=layers_to_plot,
                heads_to_plot=heads_to_plot,
                average_heads=args.average_heads
            )
        
        print(f"\n{'='*60}")
        print(f"Completed processing {num_samples_to_process} sample(s)")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

'''
# 生成所有layer和head的k和v cache图（单个样本，不使用SparseMM）
conda activate Qwen3_2 &&
python visualize_cache_similarity.py --output-dir ./cache_similarity_heatmaps --max-new-tokens 1000 --num-samples 3 

# 启用SparseMM并自定义配置
conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=1 \
python visualize_cache_similarity.py --enable-sparsemm \
--sparsemm-budget 0.7 \
--sparsemm-ratio 0.5 \
--sparsemm-window-size 32 \
--sparsemm-kernel-size 5 \
--sparsemm-pooling maxpool \
--head-score-file aggregated \
--output-dir ./cache_similarity_heatmaps --max-new-tokens 1000 \
--layers "0,6,12,18,24,30,35" \
--num-samples 3 \

# 组合使用：处理3个样本，只生成前3层，前5个head的图（启用SparseMM）
conda activate Qwen3_2 &&
python visualize_cache_similarity.py --enable-sparsemm \
    --num-samples 3 \
    --layers "0,1,2" \
    --heads "0,1,2,3,4" \
    --output-dir ./cache_similarity_heatmaps
'''