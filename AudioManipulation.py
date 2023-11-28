import torch
import numpy as np
import importlib
import subprocess
import sys


def hijack_import(importname, installname):
    try:
        importlib.import_module(importname)
    except ModuleNotFoundError:
        print(f"Import failed for {importname}, Installing {installname}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", installname])


hijack_import("librosa", "librosa")

import librosa.effects


# -----------------
# AUDIO ARRANGEMENT
# -----------------


class JoinAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor_1": ("AUDIO", ),
                "tensor_2": ("AUDIO", ),
                "gap": ("INT", {"default": 0, "min": -1e9, "max": 1e9, "step": 1}),
                "overlap_method": (("overwrite", "linear", "sigmoid"), {"default": "sigmoid"})
                },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("joined_tensor", "sample_rate")
    FUNCTION = "join_audio"

    CATEGORY = "Audio/Arrangement"

    def join_audio(self, tensor_1, tensor_2, gap, overlap_method, sample_rate):
        joined_length = tensor_1.size(2) + tensor_2.size(2) + gap
        joined_tensor = torch.zeros((tensor_1.size(0), tensor_1.size(1), joined_length), device=tensor_1.device)
        tensor_1_masked = tensor_1.clone()
        tensor_2_masked = tensor_2.clone()

        # Overlapping
        if gap < 0:
            gap_abs = abs(gap)
            mask = np.ones(gap_abs)
            if overlap_method == 'linear':
                mask = np.linspace(0.0, 1.0, num=gap_abs)
            elif overlap_method == 'sigmoid':
                k = 6
                mask = np.linspace(-1.0, 1.0, num=gap_abs)
                mask = 1 / (1 + np.exp(-mask * k))
            mask = torch.from_numpy(mask).to(device=tensor_1.device)
            tensor_1_masked[:, :, -gap_abs:] *= 1.0 - mask
            tensor_2_masked[:, :, :gap_abs] *= mask

        joined_tensor[:, :, :tensor_1.size(2)] += tensor_1_masked
        joined_tensor[:, :, tensor_1.size(2) + gap:] += tensor_2_masked

        return joined_tensor, sample_rate


class BatchJoinAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_tensor": ("AUDIO",),
                "gap": ("INT", {"default": 0, "min": -1e9, "max": 1e9, "step": 1}),
                "overlap_method": (("overwrite", "linear", "sigmoid"), {"default": "sigmoid"})
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("joined_tensor", "sample_rate")
    FUNCTION = "batch_join_audio"

    CATEGORY = "Audio/Arrangement"

    def batch_join_audio(self, batch_tensor, gap, overlap_method, sample_rate):
        joined_length = batch_tensor.size(2) * batch_tensor.size(0) + gap * (batch_tensor.size(0) - 1)
        joined_tensor = torch.zeros((1, batch_tensor.size(1), joined_length), device=batch_tensor.device)
        tensor_masked = batch_tensor.clone()

        # Overlapping
        if gap < 0:
            gap_abs = abs(gap)
            mask = np.ones(gap_abs)
            if overlap_method == 'linear':
                mask = np.linspace(0.0, 1.0, num=gap_abs)
            elif overlap_method == 'sigmoid':
                k = 6
                mask = np.linspace(-1.0, 1.0, num=gap_abs)
                mask = 1 / (1 + np.exp(-mask * k))
            mask = torch.from_numpy(mask).to(device=batch_tensor.device)
            tensor_masked[:-1, :, -gap_abs:] *= 1.0 - mask
            tensor_masked[1:, :, :gap_abs] *= mask

        for i, sample in enumerate(tensor_masked):
            sample_start = (batch_tensor.size(2) + gap) * i
            sample_end = sample_start + batch_tensor.size(2)
            joined_tensor[:, :, sample_start:sample_end] += sample

        return joined_tensor, sample_rate


class LayerAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor_1": ("AUDIO", ),
                "tensor_2": ("AUDIO", ),
                "weight_1": ("FLOAT", {"default": 1.0, "min": -1e9, "max": 1e9, "step": 1}),
                "weight_2": ("FLOAT", {"default": 1.0, "min": -1e9, "max": 1e9, "step": 1})
                },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("tensor", "sample_rate")
    FUNCTION = "layer_audio"

    CATEGORY = "Audio/Arrangement"

    def layer_audio(self, tensor_1, tensor_2, weight_1, weight_2, sample_rate):
        layered_length = max(tensor_1.size(2), tensor_2.size(2))
        layered_tensor = torch.zeros((tensor_1.size(0), tensor_1.size(1), layered_length), device=tensor_1.device)
        layered_tensor[:, :, :tensor_1.size(2)] += tensor_1 * weight_1
        layered_tensor[:, :, :tensor_2.size(2)] += tensor_2 * weight_2

        return layered_tensor, sample_rate
    

class CutAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO",),
                "start": ("INT",),
                "end": ("INT",),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("cut_tensor", "sample_rate")
    FUNCTION = "cut_audio"

    CATEGORY = "Audio/Arrangement"

    def cut_audio(self, tensor, start, end, sample_rate):
        return tensor.clone()[:, :, start:end], sample_rate


class DuplicateAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO",),
                "count": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("out_tensor", "sample_rate")
    FUNCTION = "duplicate_audio"

    CATEGORY = "Audio/Arrangement"

    def duplicate_audio(self, tensor, count, sample_rate):
        return tensor.repeat(count, 1, 1), sample_rate

# ------------------
# AUDIO MANIPULATION
# ------------------


class StretchAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO", ),
                "rate": ("FLOAT", {"default": 1.0, "min": 1e-9, "max": 1e9, "step": 0.1})
                },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("tensor", "sample_rate")
    FUNCTION = "stretch_audio"

    CATEGORY = "Audio/Manipulation"

    def stretch_audio(self, tensor, rate, sample_rate):
        y = tensor.cpu().numpy()
        y = librosa.effects.time_stretch(y, rate=rate)
        tensor_out = torch.from_numpy(y).to(device=tensor.device)

        return tensor_out, sample_rate


class ReverseAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO",),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("tensor", "sample_rate")
    FUNCTION = "reverse_audio"

    CATEGORY = "Audio/Manipulation"

    def reverse_audio(self, tensor, sample_rate):
        return torch.flip(tensor.clone(), (2,)), sample_rate


class ResampleAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO",),
                "sample_rate_target": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1}),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("out_tensor", "sample_rate")
    FUNCTION = "resample_audio"

    CATEGORY = "Audio/Manipulation"

    def resample_audio(self, tensor, sample_rate, sample_rate_target):
        y = tensor.cpu().numpy()
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=sample_rate_target)
        tensor_out = torch.from_numpy(y).to(device=tensor.device)

        return tensor_out, sample_rate_target


class SeparatePercussion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO",),
                "kernel_size": ("INT", {"default": 31, "min": 1, "max": 1e9, "step": 1}),
                "power": ("FLOAT", {"default": 2.0, "min": 0, "max": 1e9, "step": 0.1}),
                "margin": ("FLOAT", {"default": 1.0, "min": 0, "max": 1e9, "step": 0.1})

            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "INT")
    RETURN_NAMES = ("harmonic_tensor", "percussion_tensor", "sample_rate")
    FUNCTION = "separate_audio"

    CATEGORY = "Audio/Manipulation"

    def separate_audio(self, tensor, kernel_size, power, margin, sample_rate):
        y = tensor.cpu().numpy()
        h, p = librosa.effects.hpss(y, kernel_size=kernel_size, power=power, margin=margin)
        harmonic_tensor = torch.from_numpy(h).to(device=tensor.device)
        percussion_tensor = torch.from_numpy(p).to(device=tensor.device)

        return harmonic_tensor, percussion_tensor, sample_rate

# --------
# ENVELOPE
# --------


NODE_CLASS_MAPPINGS = {
    'JoinAudio': JoinAudio,
    'BatchJoinAudio': BatchJoinAudio,
    'LayerAudio': LayerAudio,
    'CutAudio': CutAudio,
    'DuplicateAudio': DuplicateAudio,
    'StretchAudio': StretchAudio,
    'ReverseAudio': ReverseAudio,
    'ResampleAudio': ResampleAudio,
    'SeparatePercussion': SeparatePercussion
}
