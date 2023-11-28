import torch
import enum
import contextlib
from contextlib import nullcontext
from typing import Tuple

from .model import ModelWrapperBase

class InferenceBase():
    def __init__(
        self,
        device_accelerator: torch.device,
        device_offload: torch.device,
        optimize_memory_use: bool,
        use_autocast: bool,
        model: ModelWrapperBase
    ):
        self.device_accelerator = device_accelerator
        self.device_offload = device_offload if(optimize_memory_use==True) else None
        self.optimize_memory_use = optimize_memory_use
        self.use_autocast = use_autocast
        self.model = model
        self.generator = torch.Generator(device_accelerator)# if (device_accelerator.type != 'mps') else torch.device('cpu'))
        self.rng_state = None
        
    def set_device_accelerator(
        self,
        device: torch.device = None
    ):
        self.device_accelerator = device
    
    def get_device_accelerator(
        self
    ) -> torch.device:
        return self.device_accelerator
        
    def set_model(
        self,
        model: ModelWrapperBase = None
    ):
        self.model = model
    
    def get_model(
        self
    ) -> ModelWrapperBase:
        return self.model

    def expand(
        self,
        tensor: torch.Tensor,
        expansion_map: list[int]
    ) -> torch.Tensor:
        out = torch.empty([0], device=self.device_accelerator)
        
        for i in range(tensor.shape[0]):
            out = torch.cat([out, tensor[i,:,:].expand(expansion_map[i], -1, -1)], 0)
            
        return out
        
    
    # def cc_randn(self, shape:tuple, seed:int, device:torch.device, dtype = None, rng_state_in:torch.Tensor = None):
        
    #     initial_rng_state = self.generator.get_state()
    #     rng_state_out = torch.empty([shape[0], shape[1]], dtype=torch.ByteTensor,device=self.generator.device)
        
    #     rn = torch.empty(shape,device=device, dtype=dtype, device=device)
        
    #     for sample in range(shape[0]):
    #         for channel in range(shape[1]):
    #             self.generator.manual_seed(seed + sample * shape[1] + channel) if(rng_state_in == None) else self.generator.set_state(rng_state_in[sample, channel])
    #             rn[sample, channel] = torch.randn([shape[2]], generator=self.generator, dtype=dtype, device=device)
    #             rng_state_out[sample, channel] = self.generator.get_state()
        
    #     self.rng_state = rng_state_out
    #     self.generator.set_state(initial_rng_state)
    #     return rn
    
    # def cc_randn_like(self, input:torch.Tensor, seed:int, rng_state_in:torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
    #     initial_rng_state = self.generator.get_state()
    #     rng_state_out = torch.empty([input.shape[0], input.shape[1]], dtype=torch.ByteTensor,device=self.generator.device)
        
    #     rn = torch.empty_like(input)
        
    #     for sample in range(input.shape[0]):
    #         for channel in range(input.shape[1]):
    #             self.generator.manual_seed(seed + sample * input.shape[1] + channel) if(rng_state_in == None) else self.generator.set_state(rng_state_in[sample, channel])
    #             rn[sample, channel] = torch.randn([input.shape[2]], generator=self.generator, dtype=input.dtype, device=input.device)
    #             rng_state_out[sample, channel] = self.generator.get_state()
        
    #     self.rng_state = rng_state_out
    #     self.generator.set_state(initial_rng_state)
    #     return rn
        
    
    def autocast_context(self):
        if self.device_accelerator.type == 'cuda':
            return torch.cuda.amp.autocast()
        elif self.device_accelerator.type == 'cpu':
            return torch.cpu.amp.autocast()
        elif self.device_accelerator.type == 'mps':
            return nullcontext()
        else:
            return torch.autocast(self.device_accelerator.type, dtype=torch.float32)

    @contextlib.contextmanager
    def offload_context(self, model):
        """
            Used by inference implementations, this context manager moves the
            passed model to the inference's `device_accelerator` device on enter,
            and then returns it to the `device_offload` device on exit.

            It also wraps the `inference.autocast_context()` context.
        """

        autocast = self.autocast_context() if self.use_autocast else nullcontext()
        
        with autocast:
            if self.optimize_memory_use:
                model.to(self.device_accelerator)

            yield None

            if self.optimize_memory_use:
                model.to(self.device_offload)