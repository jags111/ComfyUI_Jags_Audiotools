# Imports

import subprocess, sys, os
import torch
import random
from pathlib import Path 


from server import PromptServer
from aiohttp import web
from folder_paths import models_dir, get_filename_list
from model_management import get_torch_device

import importlib
import yaml

def hijack_import(importname, installname):
    try:
        importlib.import_module(importname)
    except ModuleNotFoundError:
        print(f"Import failed for {importname}, Installing {installname}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", installname])

hijack_import("audio_diffusion_pytorch", "audio_diffusion_pytorch==0.0.96")
hijack_import("diffusion", "v-diffusion-pytorch")
hijack_import("k_diffusion", "k-diffusion")
hijack_import("soundfile", "soundfile")
hijack_import("torchaudio", "torchaudio")

import soundfile as sf
import torchaudio

def get_comfy_dir():
    dirs = __file__.split('\\')
    comfy_index = None
    for i, dir in enumerate(dirs):
        if dir == "ComfyUI":
            comfy_index = i
            break
    if comfy_index is not None:
        # Join the list up to the "ComfyUI" folder
        return '\\'.join(dirs[:comfy_index+1])
    else:
        return None

PromptServer.instance.app._client_max_size = 250 * 1024 * 1024 #  250 MB

# Add route for uploading audio, duplicates image upload but to audio_input
@PromptServer.instance.routes.post("/ComfyUI_Jags_Audiotools/upload/audio")
async def upload_audio(request):
    upload_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "audio_input")

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    post = await request.post()
    file = post.get("file")

    if file and file.file:
        filename = file.filename
        if not filename:
            return web.Response(status=400)

        if os.path.exists(os.path.join(upload_dir, filename)):
            os.remove(os.path.join(upload_dir, filename))

        filepath = os.path.join(upload_dir, filename)

        with open(filepath, "wb") as f:
            f.write(file.file.read())
        
        return web.json_response({"name" : filename})
    else:
        return web.Response(status=400)

# Add route for getting audio, duplicates view image but allows audio_input
@PromptServer.instance.routes.get("/ComfyUI_Jags_Audiotools/audio")
async def view_image(request):
    if "filename" in request.rel_url.query:
        type = request.rel_url.query.get("type", "audio_input")
        if type not in ["output", "input", "temp", "audio_input"]:
            return web.Response(status=400)

        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type)
        if "subfolder" in request.rel_url.query:
            full_output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])
            if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                return web.Response(status=403)
            output_dir = full_output_dir

        filename = request.rel_url.query["filename"]
        filename = os.path.basename(filename)
        file = os.path.join(output_dir, filename)

        if os.path.isfile(file):
            return web.FileResponse(file, headers={"Content-Disposition": f"filename=\"{filename}\""})
        
    return web.Response(status=404)


config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")


if not os.path.exists(config):
    with open(config, "w") as f:
        yaml.dump({"model_folder": f"{os.path.join(models_dir, 'audio_diffusion')}"}, f)

with open(config, "r") as f:
    config = yaml.safe_load(f)
models_folder = config["model_folder"]

# init and sample_diffusion lib load


comfy_dir = get_comfy_dir()   
#if not os.path.exists(os.path.join(comfy_dir, 'custom_nodes/ComfyUI_Jags_Audiotools/libs')):
    #os.makedirs(os.path.join(comfy_dir, 'custom_nodes/ComfyUI_Jags_Audiotools/libs'))
libs = os.path.join(comfy_dir, 'custom_nodes/ComfyUI_Jags_Audiotools/libs') 
if not os.path.exists(os.path.join(comfy_dir, libs)):
    os.system (os.path.join(comfy_dir, libs))
sys.path.append(os.path.join(comfy_dir, libs ))

from libs.util.util import load_audio, crop_audio
from libs.dance_diffusion.api import RequestHandler, Request, ModelType
from libs.diffusion_library.sampler import SamplerType
from libs.diffusion_library.scheduler import SchedulerType
from libs.dance_diffusion.dd.model import DDModelWrapper
from libs.dance_diffusion.dd.inference import DDInference

def save_audio(audio_out, output_path: str, sample_rate, id_str:str = None):
    out_files = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ix = 1
    for sample in audio_out:
        while True:
            output_file = os.path.join(output_path, f"sample_{id_str}_{ix}.wav" if id_str else f"sample_{ix}.wav")
            if not os.path.exists(output_file):
                break
            ix += 1
        
        open(output_file, "a").close()
        
        output = sample.cpu()
        torchaudio.save(output_file, output, sample_rate)
        out_files.append(output_file)
        ix += 1
    
    return out_files


# ****************************************************************************
# *                                   NODES                                  *
# ****************************************************************************





class AudioInference():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "audio_model": ("DD_MODEL", ),
                "mode": (['Generation', 'Variation'],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10000000000, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000000000, "step": 1}),
                "sampler": (SamplerType._member_names_, {"default": "V_IPLMS"}),
                "sigma_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1280, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 50, "min": 0.0, "max": 1280, "step": 0.01}),
                "rho": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 128.0, "step": 0.01}),
                "scheduler": (SchedulerType._member_names_, {"default": "V_CRASH"}),
                "noise_level": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1}),
                },
            "optional": {
                "input_tensor": ("AUDIO", {}),
                "input_audio_path": ("STRING", {"default": '', "forceInput": True}),
                },
            }

    RETURN_TYPES = ("LIST", "AUDIO", "INT")
    RETURN_NAMES = ("out_paths", "tensor", "sample_rate")
    FUNCTION = "do_sample"

    CATEGORY = "Audio/ComfyUI_Jags_Audiotools"

    def do_sample(self, audio_model, mode, batch_size, steps, sampler, sigma_min, sigma_max, rho, scheduler, input_audio_path='', input_tensor=None, noise_level=0.7, seed=-1):


        wrapper, inference = audio_model
        device_type_accelerator = get_torch_device()
        device_accelerator = torch.device(device_type_accelerator)
        device_offload = get_torch_device()

        crop = lambda audio: crop_audio(audio, wrapper.chunk_size, 0)

        if input_tensor is None:
            input_audio_path = None if input_audio_path == '' else input_audio_path
            load_input = lambda source: crop(load_audio(device_accelerator, source, wrapper.sample_rate)) if source is not None else None
            audio_source = load_input(input_audio_path)
        else:
            if len(input_tensor.shape) == 3:
            # remove first (batch) dimension
                input_tensor = input_tensor[0]
            if input_tensor.shape[0] != 2:
                channels, sample_length = input_tensor.shape
                input_tensor = input_tensor.view(1, sample_length).repeat(2, 1)
                input_tensor = input_tensor.to(get_torch_device())

            audio_source = crop(input_tensor)

        
        request_handler = RequestHandler(device_accelerator, device_offload, optimize_memory_use=False, use_autocast=True)
        
        seed = seed if(seed!=-1) else torch.randint(0, 4294967294, [1], device=device_type_accelerator).item()
        print(f"Using accelerator: {device_type_accelerator}, Seed: {seed}.")
        
        request = Request(
            request_type=mode,
            model_path=wrapper.path,
            model_type=ModelType.DD,
            model_chunk_size=wrapper.chunk_size,
            model_sample_rate=wrapper.sample_rate,
            model_wrapper=wrapper,
            model_inference=inference,
            
            seed=seed,
            batch_size=batch_size,
            
            audio_source=audio_source,
            audio_target=None,
            
            mask=None,
            
            noise_level=noise_level,
            interpolation_positions=None,
            resamples=None,
            keep_start=True,
                    
            steps=steps,
            
            sampler_type=SamplerType[sampler],
            sampler_args={'use_tqdm': True},
            
            scheduler_type=SchedulerType[scheduler],
            scheduler_args={
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'rho': rho,
            }
        )
        
        response = request_handler.process_request(request)#, lambda **kwargs: print(f"{kwargs['step'] / kwargs['x']}"))
        paths = save_audio(response.result, f"{comfy_dir}/temp", wrapper.sample_rate, f"{seed}_{random.randint(0, 100000)}")
        return (paths, response.result, wrapper.sample_rate)

class SaveAudio():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "tensor": ("AUDIO", ),
                "output_path": ("STRING", {"default": f'{comfy_dir}/output/audio_samples'}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1}),
                "id_string": ("STRING", {"default": 'ComfyUI'}),
                "tame": (['Enabled', 'Disabled'],)
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("out_paths", )
    FUNCTION = "save_audio_ui"
    OUTPUT_NODE = True

    CATEGORY = "Audio"

    def save_audio_ui(self, tensor, output_path, sample_rate, id_string, tame):
        return (save_audio(audio_out=(0.5 * tensor).clamp(-1,1) if(tame == 'Enabled') else tensor, output_path=output_path, sample_rate=sample_rate, id_str=id_string), )

class LoadAudio():
    def __init__(self):
        self.input_audio = os.listdir(f'{comfy_dir}/custom_nodes/ComfyUI_Jags_Audiotools/audio_input')
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                ""
                "file_path": ("STRING", {}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("STRING", "AUDIO", "INT")
    RETURN_NAMES = ("path", "tensor", "sample_rate")
    FUNCTION = "LoadAudio"
    OUTPUT_NODE = True

    CATEGORY = "Audio"

    def LoadAudio(self, file_path):
        if file_path == '':
            waveform, samplerate = None, None
            return (file_path, samplerate, waveform)

        file_path = f'{comfy_dir}/custom_nodes/ComfyUI_Jags_Audiotools/audio_input/{file_path}'

        if file_path.endswith('.mp3'):
            if os.path.exists(file_path.replace('.mp3', '')+'.wav'):
                file_path = file_path.replace('.mp3', '')+'.wav'
            else:
                data, samplerate = sf.read(file_path)
                sf.write(file_path.replace('.mp3', '')+'.wav', data, samplerate)

            os.remove(file_path.replace('.wav', '.mp3'))

        waveform, samplerate = torchaudio.load(file_path)
        waveform = waveform.to(get_torch_device())
        waveform = waveform.unsqueeze(0)

        return (file_path, waveform, samplerate)

class LoadAudioModelDD():    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        global models_folder
        models = os.listdir(models_folder)
        models = [x for x in models if x.endswith('.ckpt')]
        return {
            "required": {
                ""
                "model": (models, {}),
                "chunk_size": ("INT", {"default": 65536, "min": 32768, "max": 10000000000, "step": 32768}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1}),
                "optimize_memory_use": (['Enabled', 'Disabled'], {"default": 'Enabled'}),
                "autocast": (['Enabled', 'Disabled'], {"default": 'Enabled'}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("DD_MODEL", )
    RETURN_NAMES = ("audio_model", )
    FUNCTION = "DoLoadAudioModelDD"
    OUTPUT_NODE = True

    CATEGORY = "Audio/ComfyUI_Jags_Audiotools"

    def DoLoadAudioModelDD(self, model, chunk_size, sample_rate, optimize_memory_use, autocast):
        global models_folder
        model = os.path.join(models_folder, model)
        device = get_torch_device()
        wrapper = DDModelWrapper()
        wrapper.load(model, device, optimize_memory_use, chunk_size, sample_rate)
        inference = DDInference(device, device, optimize_memory_use, autocast, wrapper)


        loaded_model = (wrapper, inference)

        return (loaded_model, )

class PreviewAudioFile():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "paths": ("LIST",),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ()
    FUNCTION = "PreviewAudioFile"
    OUTPUT_NODE = True

    CATEGORY = "Audio"

    def PreviewAudioFile(self, paths):
        # fix slashes
        paths = [path.replace("\\", "/") for path in paths]
        # get filenames with extensions from paths

        filenames = [os.path.basename(path) for path in paths]
        return {"result": (filenames,), "ui": filenames}

class PreviewAudioTensor():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO",),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "tame": (['Enabled', 'Disabled'],)
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("paths", )
    FUNCTION = "PreviewAudioTensor"
    OUTPUT_NODE = True

    CATEGORY = "Audio"

    def PreviewAudioTensor(self, tensor, sample_rate, tame):
        # fix slashes
        paths = save_audio((0.5 * tensor).clamp(-1,1) if(tame == 'Enabled') else tensor, f"{comfy_dir}/temp", sample_rate, f"{random.randint(0, 10000000000)}")
        paths = [path.replace("\\", "/") for path in paths]
        # get filenames with extensions from paths
        paths = [os.path.basename(path) for path in paths]
        return {"result": (paths,), "ui": paths}

class MergeTensors():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor_1": ("AUDIO",),
                "tensor_2": ("AUDIO",),
                "tensor_1_volume": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "tensor_2_volume": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("tensor", "sample_rate")
    FUNCTION = "do_merge"

    CATEGORY = "Audio/Helpers"

    def do_merge(self, tensor_1, tensor_2, tensor_1_volume, tensor_2_volume, sample_rate):
        # Ensure both batches have the same size and number of channels
        assert tensor_1.size(0) == tensor_2.size(0) and tensor_1.size(1) == tensor_2.size(1), "Batches must have the same size and number of channels"

        # Pad or truncate the shorter waveforms in the batches to match the length of the longer ones
        max_length = max(tensor_1.size(2), tensor_2.size(2))
        tensor_1_padded = torch.zeros(tensor_1.size(0), tensor_1.size(1), max_length)
        tensor_2_padded = torch.zeros(tensor_2.size(0), tensor_2.size(1), max_length)

        tensor_1_padded[:, :, :tensor_1.size(2)] = tensor_1
        tensor_2_padded[:, :, :tensor_2.size(2)] = tensor_2

        # Mix the batches with specified volumes
        mixed_tensors = tensor_1_volume * tensor_1_padded + tensor_2_volume * tensor_2_padded

        return (mixed_tensors, sample_rate)


class StringListIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("LIST", ),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "doStuff"
    CATEGORY = "Audio/Helpers"

    def doStuff(self, list, index):
        return (list[index],)

class AudioIndex:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("tensor", "sample_rate", "filename")
    FUNCTION = "doStuff"
    CATEGORY = "Audio/Helpers"

    def doStuff(self, path, index):
        if not os.path.exists(path):
            raise Exception("Path does not exist")
        audios = []
        for audio in os.listdir(path):
            if any(audio.endswith(ext) for ext in [".wav", ".flac"]):
                audios.append(audio)
        filename = audios[index]
        audio, sample_rate = torchaudio.load(os.path.join(path, filename))
        # make stereo if mono
        audio = audio.unsqueeze(0)
        audio.to(get_torch_device())
        return (audio, sample_rate, filename)

NODE_CLASS_MAPPINGS = {
    "GenerateAudioSample": AudioInference,
    "SaveAudioTensor": SaveAudio,
    "LoadAudioFile": LoadAudio,
    "PreviewAudioFile": PreviewAudioFile,
    "PreviewAudioTensor": PreviewAudioTensor,
    "GetStringByIndex": StringListIndex,
    "LoadAudioModel (DD)": LoadAudioModelDD,
    "MixAudioTensors": MergeTensors,
    "GetAudioFromFolderIndex": AudioIndex,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateAudioSample": "Jags-AudioInference",
    "SaveAudioTensor": "Jags_SaveAudio",
    "LoadAudioFile": "Jags_LoadAudio",
    "PreviewAudioFile": "Jags_PreviewAudioFile",
    "PreviewAudioTensor": "Jags_PreviewAudioTensor",
    "GetStringByIndex": "Jags_StringListIndex",
    "LoadAudioModel (DD)": "Jags_LoadAudioModelDD",
    "MixAudioTensors": "Jags_MergeTensors",
    "GetAudioFromFolderIndex": "Jags_AudioIndex",
}

