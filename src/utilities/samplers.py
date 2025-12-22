# Functions for sampling different CLS token distributions
import torch


def normal_sampler(patch_embeddings: torch.Tensor) -> torch.Tensor:
    """Unit normal sampler"""
    dtype = patch_embeddings.dtype
    device = patch_embeddings.device
    batch_size, _, hidden_size = patch_embeddings.shape

    cls_token = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)

    return cls_token


def wide_normal_sampler(patch_embeddings: torch.Tensor) -> torch.Tensor:
    """Unit normal sampler"""
    dtype = patch_embeddings.dtype
    device = patch_embeddings.device
    batch_size, _, hidden_size = patch_embeddings.shape

    cls_token = 10*torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)

    return cls_token


def constant_sampler(default_token, patch_embeddings: torch.Tensor) -> torch.Tensor:
    """Samples a constant value: for debugging"""
    batch_size, _, hidden_size = patch_embeddings.shape
    return default_token.repeat((batch_size, 1, 1))


def previous_activity_sampler(activity_file, patch_embeddings: torch.Tensor) -> torch.Tensor:
    batch_size, _, hidden_size = patch_embeddings.shape
    dtype = patch_embeddings.dtype
    device = patch_embeddings.device

    activity = torch.load(activity_file)['activity'].view(-1, hidden_size)

    inds = torch.randint(len(activity), (batch_size,))

    return activity[inds].reshape(batch_size, 1, hidden_size).to(dtype).to(device)


def resample_activity(activity, patch_embeddings):
    batch_size, _, hidden_size = patch_embeddings.shape
    dtype = patch_embeddings.dtype
    device = patch_embeddings.device

    return activity.clone().reshape(batch_size, 1, hidden_size).to(dtype).to(device)
