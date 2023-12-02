import torch


def get_torch_device_type():
    if is_mps_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def is_mps_available():
    try:
        return torch.backends.mps.is_available()
    except:
        return False
