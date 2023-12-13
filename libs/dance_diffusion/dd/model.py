import torch
from torch import nn
from typing import Callable

from .ddattnunet import DiffusionAttnUnet1D
from libs.dance_diffusion.base.model import ModelWrapperBase
from libs.dance_diffusion.base.type import ModelType


class DanceDiffusionInference(nn.Module):
    def __init__(self, n_attn_layers:int = 4, **kwargs):
        super().__init__()

        self.diffusion_ema = DiffusionAttnUnet1D(kwargs, n_attn_layers=n_attn_layers)

class DDModelWrapper(ModelWrapperBase):
    def __init__(self):
        
        super().__init__()
        
        self.module:DanceDiffusionInference = None
        self.model:Callable = None
        
    def load(
        self,
        path:str,
        device_accelerator:torch.device,
        optimize_memory_use:bool=False,
        chunk_size:int=None,
        sample_rate:int=None
    ):
        
        default_model_config = dict(
            version = [0, 0, 1],
            model_info = dict(
                name = 'Dance Diffusion Model',
                description = 'v1.0',
                type = ModelType.DD,
                native_chunk_size = 65536,
                sample_rate = 48000,
            ),
            diffusion_config = dict(
                n_attn_layers = 4
            )
        )
        
        file = torch.load(path, map_location='cpu')
        
        model_config = file.get('model_config')
        if not model_config:
            print(f"Model file {path} is invalid. Please run the conversion script.")
            print(f" - Default model config will be used, which may be inaccurate.")
            model_config = default_model_config
            
        model_info = model_config.get('model_info')
        diffusion_config = model_config.get('diffusion_config')

        self.path = path
        self.chunk_size =  model_info.get('native_chunk_size')if not chunk_size else chunk_size
        self.sample_rate = model_info.get('sample_rate')if not sample_rate else sample_rate
        
        self.module = DanceDiffusionInference(
            n_attn_layers=diffusion_config.get('n_attn_layers'),
            sample_size=chunk_size,
            sample_rate=sample_rate,
            latent_dim=0,
        )
        
        self.module.load_state_dict(
            file["state_dict"], 
            strict=False
        )
        self.module.eval().requires_grad_(False)
        
        self.model = self.module.diffusion_ema if (optimize_memory_use) else self.module.diffusion_ema.to(device_accelerator)
