"""CUDA attention backends supported by variable ALIGN token shapes."""
from contextlib import nullcontext


def attention_backend(device):
    """Avoid cuDNN execution-plan failures; retain Flash, efficient and math SDPA.

    Restore caller settings after the operation, including on exceptions.
    """
    if device.type != 'cuda':
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
