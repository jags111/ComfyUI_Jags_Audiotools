import torch
import numpy as np

from custom_nodes.SampleDiffusion.EXT_SampleDiffusion import AudioInference
from diffusion_library.sampler import SamplerType
from diffusion_library.scheduler import SchedulerType


# Split up audio into a sequence of smaller clips (represented as a 4d tensor) to be processed individually
class SliceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO",),
                "clip_size": ("INT", {"default": 1, "min": 1, "max": 1e9, "step": 1}),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("tensor_list", "sample_rate")
    FUNCTION = "slice_audio"

    CATEGORY = "Audio/VariationUtils"

    def slice_audio(self, tensor, clip_size, sample_rate):
        return list(torch.split(tensor.clone(), clip_size, 2)), sample_rate


class BatchToList:
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

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("tensor_list", "sample_rate")
    FUNCTION = "batch_to_list"

    CATEGORY = "Audio/VariationUtils"

    def batch_to_list(self, tensor, sample_rate):
        return list(torch.split(tensor.clone(), 1, 0)), sample_rate


class ConcatAudioList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor_list": ("AUDIO_LIST",)
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("tensor", "sample_rate")
    FUNCTION = "concat_audio"

    CATEGORY = "Audio/VariationUtils"

    def concat_audio(self, tensor_list, sample_rate):
        return torch.cat(tensor_list, 2), sample_rate


# Perform variation on each audio tensor in a list
class SequenceVariation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "audio_model": ("DD_MODEL",),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1e9, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 1e9, "step": 1}),
                "sampler": (SamplerType._member_names_, {"default": "V_IPLMS"}),
                "sigma_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1280, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 50, "min": 0.0, "max": 1280, "step": 0.01}),
                "rho": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 128.0, "step": 0.01}),
                "scheduler": (SchedulerType._member_names_, {"default": "V_CRASH"}),
                "noise_level": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1}),
                "tensor_list": ("AUDIO_LIST",)
            },
            "optional": {},
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("tensor_list", "sample_rate")
    FUNCTION = "do_variation"

    CATEGORY = "Audio/VariationUtils"

    def do_variation(self, audio_model, batch_size, steps, sampler, sigma_min, sigma_max, rho, scheduler, tensor_list, noise_level=0.7, seed=-1):
        audio_inference = AudioInference()
        tensor_list_out = []
        sample_rate = 44100
        for tensor in tensor_list:
            _, tensor_out, sample_rate = audio_inference.do_sample(
                audio_model=audio_model,
                mode='Variation',
                batch_size=batch_size,
                steps=steps,
                sampler=sampler,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                scheduler=scheduler,
                input_tensor=tensor,
                noise_level=noise_level,
                seed=seed)
            tensor_list_out.append(tensor_out)
        return tensor_list_out, sample_rate


NODE_CLASS_MAPPINGS = {
    'SliceAudio': SliceAudio,
    'BatchToList': BatchToList,
    'ConcatAudioList': ConcatAudioList,
    'SequenceVariation': SequenceVariation
}
