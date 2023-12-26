import torch
import numpy as np
import importlib
import subprocess
import sys


def check_import(importname, installname=None):
    try:
        importlib.import_module(importname)
    except ModuleNotFoundError:
        installname = installname if installname else importname
        print(f"Required module '{importname}' not found. Please install it using 'pip install {installname}'.")


check_import("librosa", "librosa")

import librosa.effects


# -----------------
# AUDIO ARRANGEMENT
# -----------------


class JoinAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO", ),
                "audio_2": ("AUDIO", ),
                "gap": ("INT", {"default": 0, "min": -1e9, "max": 1e9, "step": 1}),
                "overlap_method": (("overwrite", "linear", "sigmoid"), {"default": "sigmoid"})
                },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️joined_audio", "sample_rate")
    FUNCTION = "join_audio"

    CATEGORY = "🎙️Jags_Audio/Arrangement"

    def join_audio(self, audio_1, audio_2, gap, overlap_method, sample_rate):
        joined_length = audio_1.size(2) + audio_2.size(2) + gap
        joined_tensor = torch.zeros((audio_1.size(0), audio_1.size(1), joined_length), device=audio_1.device)
        tensor_1_masked = audio_1.clone()
        tensor_2_masked = audio_2.clone()

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
            mask = torch.from_numpy(mask).to(device=audio_1.device)
            tensor_1_masked[:, :, -gap_abs:] *= 1.0 - mask
            tensor_2_masked[:, :, :gap_abs] *= mask

        joined_tensor[:, :, :audio_1.size(2)] += tensor_1_masked
        joined_tensor[:, :, audio_1.size(2) + gap:] += tensor_2_masked

        return joined_tensor, sample_rate


class BatchJoinAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_audio": ("AUDIO",),
                "gap": ("INT", {"default": 0, "min": -1e9, "max": 1e9, "step": 1}),
                "overlap_method": (("overwrite", "linear", "sigmoid"), {"default": "sigmoid"})
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️joined_audio", "sample_rate")
    FUNCTION = "batch_join_audio"

    CATEGORY = "🎙️Jags_Audio/Arrangement"

    def batch_join_audio(self, batch_audio, gap, overlap_method, sample_rate):
        joined_length = batch_audio.size(2) * batch_audio.size(0) + gap * (batch_audio.size(0) - 1)
        joined_tensor = torch.zeros((1, batch_audio.size(1), joined_length), device=batch_audio.device)
        tensor_masked = batch_audio.clone()

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
            mask = torch.from_numpy(mask).to(device=batch_audio.device)
            tensor_masked[:-1, :, -gap_abs:] *= 1.0 - mask
            tensor_masked[1:, :, :gap_abs] *= mask

        for i, sample in enumerate(tensor_masked):
            sample_start = (batch_audio.size(2) + gap) * i
            sample_end = sample_start + batch_audio.size(2)
            joined_tensor[:, :, sample_start:sample_end] += sample

        return joined_tensor, sample_rate
    

class CutAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start": ("INT",),
                "end": ("INT",),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️cut_audio", "sample_rate")
    FUNCTION = "cut_audio"

    CATEGORY = "🎙️Jags_Audio/Arrangement"

    def cut_audio(self, audio, start, end, sample_rate):
        return audio.clone()[:, :, start:end], sample_rate


class DuplicateAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "count": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️out_audio", "sample_rate")
    FUNCTION = "duplicate_audio"

    CATEGORY = "🎙️Jags_Audio/Arrangement"

    def duplicate_audio(self, audio, count, sample_rate):
        return audio.repeat(count, 1, 1), sample_rate


# ------------------
# AUDIO MANIPULATION
# ------------------


class StretchAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "rate": ("FLOAT", {"default": 1.0, "min": 1e-9, "max": 1e9, "step": 0.1})
                },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "stretch_audio"

    CATEGORY = "🎙️Jags_Audio/Manipulation"

    def stretch_audio(self, audio, rate, sample_rate):
        tensor = tensor.cpu().numpy()
        #convert GPU tensor to CPU tensor for numpy else use a alternative method tensor.cuda().numpy ()
        y = tensor.cpu().numpy()
        y = librosa.effects.time_stretch(y, rate=rate)
        tensor_out = torch.from_numpy(y).to(device=tensor.device)

        return tensor_out, sample_rate


class ReverseAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "reverse_audio"

    CATEGORY = "🎙️Jags_Audio/Manipulation"

    def reverse_audio(self, audio, sample_rate):
        return torch.flip(audio.clone(), (2,)), sample_rate


class ResampleAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1}),
                "sample_rate_target": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️out_audio", "sample_rate")
    FUNCTION = "resample_audio"

    CATEGORY = "🎙️Jags_Audio/Manipulation"

    def resample_audio(self, audio, sample_rate, sample_rate_target):
        tensor = tensor.cpu().numpy()
        #convert GPU tensor to CPU tensor for numpy else use a alternative method tensor.cuda().numpy ()
        y = tensor.cpu().numpy()
        y = librosa.resample(y, sample_rate, sample_rate_target)
        tensor_out = torch.from_numpy(y).to(device=tensor.device)

        return tensor_out, sample_rate_target



# --------
# ENVELOPE
# --------


NODE_CLASS_MAPPINGS = {
    'JoinAudio': JoinAudio,
    'BatchJoinAudio': BatchJoinAudio,
    'CutAudio': CutAudio,
    'DuplicateAudio': DuplicateAudio,
    'StretchAudio': StretchAudio,
    'ReverseAudio': ReverseAudio,
    'ResampleAudio': ResampleAudio
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'JoinAudio': 'Jags_JoinAudio',
    'BatchJoinAudio': 'Jags_BatchJoinAudio',
    'CutAudio': 'Jags_CutAudio',
    'DuplicateAudio': 'Jags_DuplicateAudio',
    'StretchAudio': 'Jags_StretchAudio',
    'ReverseAudio': 'Jags_ReverseAudio',
    'ResampleAudio': 'Jags_ResampleAudio'
}
