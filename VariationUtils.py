import torch
import torchaudio
import os
import soundfile as sf
import numpy as np

from comfy.model_management import get_torch_device
from SampleDiffusion import AudioInference
from libs.diffusion_library.sampler import SamplerType
from libs.diffusion_library.scheduler import SchedulerType

# -------------
# LIST CREATION
# -------------


# Split up audio into a sequence of smaller clips (represented as a 4d tensor) to be processed individually
class SliceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "clip_size": ("INT", {"default": 1, "min": 1, "max": 1e9, "step": 1}),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("audio_list", "sample_rate")
    FUNCTION = "slice_audio"

    CATEGORY = "🎙️Jags_Audio/VariationUtils"

    def slice_audio(self, audio, clip_size, sample_rate):
        return list(torch.split(audio.clone(), clip_size, 2)), sample_rate


class BatchToList:
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

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("audio_list", "sample_rate")
    FUNCTION = "batch_to_list"

    CATEGORY = "🎙️Jags_Audio/VariationUtils"

    def batch_to_list(self, audio, sample_rate):
        return list(torch.split(audio.clone(), 1, 0)), sample_rate

class LoadAudioDir:
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "dir_path": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO_LIST", "INT")
    RETURN_NAMES = ("dir_path", "audio_list", "sample_rate")
    FUNCTION = "load_audio_dir"
    OUTPUT_NODE = True

    CATEGORY = "🎙️Jags_Audio/VariationUtils"

    def load_audio_dir(self, dir_path):
        tensor_list = []
        sample_rate = None

        if dir_path == '':
            return dir_path, tensor_list, sample_rate

        for file_path in os.listdir(dir_path):
            if os.path.isfile(f'{dir_path}/{file_path}'):
                if file_path.endswith('.mp3'):
                    if os.path.exists(file_path.replace('.mp3', '') + '.wav'):
                        file_path = file_path.replace('.mp3', '') + '.wav'
                    else:
                        data, sample_rate = sf.read(file_path)
                        sf.write(file_path.replace('.mp3', '') + '.wav', data, sample_rate)

                    os.remove(file_path.replace('.wav', '.mp3'))

                waveform, sample_rate = torchaudio.load(f'{dir_path}/{file_path}')
                waveform = waveform.to(get_torch_device())
                waveform = waveform.unsqueeze(0)
                tensor_list.append(waveform)

        return dir_path, tensor_list, sample_rate

# ----------------
# LIST AGGREGATION
# ----------------


class ListToBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO_LIST",),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "list_to_batch"

    CATEGORY = "🎙️Jags_Audio/VariationUtils"

    def list_to_batch(self, audio_list, sample_rate):
        max_len = 0
        batch_size = 0
        for tensor in audio_list:
            max_len = max(tensor.size(2), max_len)
            batch_size += tensor.size(0)
        tensor_out = torch.zeros((batch_size, 2, max_len), device=audio_list[0].device)

        batch_start = 0
        for tensor in audio_list:
            batch_end = batch_start + tensor.size(0)
            tensor_out[batch_start:batch_end, :, :tensor.size(2)] += tensor
            batch_start = batch_end

        return tensor_out, sample_rate


class ConcatAudioList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO_LIST",)
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "concat_audio"

    CATEGORY = "🎙️Jags_Audio/VariationUtils"

    def concat_audio(self, audio_list, sample_rate):
        return torch.cat(audio_list, 2), sample_rate


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
                "audio_list": ("AUDIO_LIST",)
            },
            "optional": {},
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("audio_list", "sample_rate")
    FUNCTION = "do_variation"

    CATEGORY = "🎙️Jags_Audio/VariationUtils"

    def do_variation(self, audio_model, batch_size, steps, sampler, sigma_min, sigma_max, rho, scheduler, audio_list, noise_level=0.7, seed=-1):
        audio_inference = AudioInference()
        tensor_list_out = []
        sample_rate = 44100
        for tensor in audio_list:
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

class GetSingle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO_LIST",),
                "index": ("INT", {"default": 1, "min": 1, "max": 1e9, "step": 1})
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "get_single"

    CATEGORY = "🎙️Jags_Audio/VariationUtils"

    def get_single(self, audio_list, index, sample_rate):
        return audio_list[index], sample_rate
# ----------
# PROCESSING
# ----------


# Perform variation on each audio tensor in a list
class BulkVariation:
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
                "audio_list": ("AUDIO_LIST",)
            },
            "optional": {},
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("audio_list", "sample_rate")
    FUNCTION = "do_variation"

    CATEGORY = "🎙️Jags_Audio/VariationUtils"

    def do_variation(self, audio_model, batch_size, steps, sampler, sigma_min, sigma_max, rho, scheduler, audio_list, noise_level=0.7, seed=-1):
        audio_inference = AudioInference()
        tensor_list_out = []
        sample_rate = 44100
        for tensor in audio_list:
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
    'LoadAudioDir': LoadAudioDir,
    'ListToBatch': ListToBatch,
    'ConcatAudioList': ConcatAudioList,
    'GetSingle': GetSingle,
    'SequenceVariation': SequenceVariation,
    'BulkVariation': BulkVariation
    
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'SliceAudio': 'Jags_SliceAudio',
    'BatchToList': 'Jags_BatchToList',
    'LoadAudioDir': 'Jags_LoadAudioDir',
    'ListToBatch': 'Jags_ListToBatch',
    'ConcatAudioList': 'Jags_ConcatAudioList',
    'SequenceVariation': 'Jags_SequenceVariation',
    'GetSingle': 'Jags_GetSingle',
    'BulkVariation': 'Jags_BulkVariation'
}
