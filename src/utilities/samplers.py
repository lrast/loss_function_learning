# Functions for sampling different CLS token distributions
import torch


def normal_sampler(patch_embeddings: torch.Tensor) -> torch.Tensor:
    """Unit normal sampler"""
    dtype = patch_embeddings.dtype
    device = patch_embeddings.device
    batch_size, _, hidden_size = patch_embeddings.shape

    cls_token = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)

    return cls_token


def constant_sampler(default_token, patch_embeddings: torch.Tensor) -> torch.Tensor:
    """Samples a constant value: for debugging"""
    batch_size, _, hidden_size = patch_embeddings.shape
    return default_token.repeat((batch_size, 1, 1))
