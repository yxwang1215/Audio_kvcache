"""
Audio Head Score Utilities
Load and process attention head scores from audio_head/head_score directory
"""
import json
import os
import numpy as np
import torch
from typing import Dict, List, Optional


def load_audio_head_scores(
    score_file: str,
    use_mean: bool = True
) -> Dict[str, List[float]]:
    """
    Load audio head scores from JSON file
    
    Args:
        score_file: Path to the JSON file containing head scores
        use_mean: If True, use mean of the score list; otherwise use first value
    
    Returns:
        Dictionary with keys "layer-head" and values as score lists
    """
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"Head score file not found: {score_file}")
    
    with open(score_file, 'r') as f:
        head_scores = json.load(f)
    
    print(f"Loaded {len(head_scores)} head scores from {score_file}")
    return head_scores


def process_audio_head_scores(
    head_scores: Dict[str, List[float]],
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_groups: int = 1,
    use_mean: bool = True
) -> torch.Tensor:
    """
    Process audio head scores into a tensor format compatible with SparseMM
    
    Args:
        head_scores: Dictionary with "layer-head" keys and score lists as values
        num_hidden_layers: Number of hidden layers in the model
        num_attention_heads: Number of attention heads per layer
        num_key_value_groups: Number of key-value groups (for GQA)
        use_mean: If True, use mean of score list; otherwise use first value
    
    Returns:
        Normalized tensor of shape [num_hidden_layers * num_attention_heads]
    """
    # Handle special compact uniform specification: {"__uniform__": value}
    if isinstance(head_scores, dict) and "__uniform__" in head_scores:
        try:
            uniform_val = float(head_scores["__uniform__"])
        except Exception:
            uniform_val = 1.0
        head_score_list = [uniform_val] * (num_hidden_layers * num_attention_heads)
    else:
        # Initialize score list
        head_score_list = []
        # Extract scores in layer-head order
        for layer_idx in range(num_hidden_layers):
            for head_idx in range(num_attention_heads):
                key = f"{layer_idx}-{head_idx}"
                if key in head_scores:
                    score_value = head_scores[key]
                    if isinstance(score_value, list):
                        # Use mean or first value
                        if use_mean:
                            score = np.mean(score_value)
                        else:
                            score = score_value[0]
                    else:
                        score = float(score_value)
                else:
                    # Default score if not found
                    print(f"Warning: Score not found for {key}, using default 0.0")
                    score = 0.0
                head_score_list.append(score)
    
    # Convert to tensor and normalize
    head_score_tensor = torch.tensor(head_score_list, dtype=torch.float32)
    
    # Normalize to sum to 1.0
    score_sum = head_score_tensor.sum()
    if score_sum > 0:
        head_score_tensor = head_score_tensor / score_sum
    else:
        # If all scores are zero, use uniform distribution
        head_score_tensor = torch.ones_like(head_score_tensor) / len(head_score_tensor)
        print("Warning: All scores are zero, using uniform distribution")
    
    print(f"Processed {len(head_score_list)} head scores")
    print(f"Score range: [{head_score_tensor.min():.6f}, {head_score_tensor.max():.6f}]")
    print(f"Score sum: {head_score_tensor.sum():.6f}")
    
    return head_score_tensor


def get_audio_head_score_path(
    base_dir: str = "/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score",
    score_type: str = "aggregated",
    batch: Optional[str] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None
) -> str:
    """
    Get the path to audio head score file
    
    Args:
        base_dir: Base directory for head scores
        score_type: Type of score file (aggregated, mean, from_ranking, topk32, etc.)
        batch: Batch identifier (e.g., "batch1"), if None uses non-batch version
        alpha: Alpha parameter value (if provided, will be included in filename)
        beta: Beta parameter value (if provided, will be included in filename)
        gamma: Gamma parameter value (if provided, will be included in filename)
    
    Returns:
        Path to the score file
    """
    # If alpha, beta, gamma are provided, construct filename with these parameters
    if alpha is not None and beta is not None and gamma is not None:
        # Determine naming format based on score_type
        if score_type == "topk32" or score_type.startswith("topk"):
            # Format: audio_head_scores_topk32_alpha0_beta0_gamma1.json
            # Use integer format for topk32 files (round to nearest integer)
            alpha_str = str(int(round(alpha)))
            beta_str = str(int(round(beta)))
            gamma_str = str(int(round(gamma)))
            filename = f"audio_head_scores_{score_type}_alpha{alpha_str}_beta{beta_str}_gamma{gamma_str}.json"
        else:
            # Format: audio_head_scores_aggregated_alpha_1_00_beta_1_00_gamma_1_00.json
            # Format alpha, beta, gamma as floats with 2 decimal places
            alpha_str = f"{alpha:.2f}".replace(".", "_")
            beta_str = f"{beta:.2f}".replace(".", "_")
            gamma_str = f"{gamma:.2f}".replace(".", "_")
            filename = f"audio_head_scores_{score_type}_alpha_{alpha_str}_beta_{beta_str}_gamma_{gamma_str}.json"
    elif batch:
        filename = f"audio_head_scores_{score_type}_{batch}.json"
    else:
        filename = f"audio_head_scores_{score_type}.json"
    
    filepath = os.path.join(base_dir, filename)
    return filepath

