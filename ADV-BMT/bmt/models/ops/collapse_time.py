def collapse_time(tensor):
    if tensor.ndim == 4:
        B, T, N, D = tensor.shape
        tensor = tensor.swapaxes(1, 2).reshape(B, N, T * D)
    else:
        raise ValueError(f"Unknown tensor shape: {tensor.shape}")
    return tensor
