import torch
import enum

from dataclasses import dataclass
from typing import Callable

from .base.type import ModelType
from .base.model import ModelWrapperBase
from .base.inference import InferenceBase

from .dd.model import DDModelWrapper
from .dd.inference import DDInference
    
class RequestType(str, enum.Enum):
    Generation = 'Generation'
    Variation = 'Variation'
    Interpolation = 'Interpolation'
    Inpainting = 'Inpainting'
    Extension = 'Extension'

class Request:
    def __init__(
        self,
        request_type: RequestType,
        model_path: str,
        model_type: ModelType,
        model_chunk_size: int,
        model_sample_rate: int,
        **kwargs
    ):
        self.request_type = request_type
        self.model_path = model_path
        self.model_type = model_type
        self.model_chunk_size = model_chunk_size
        self.model_sample_rate = model_sample_rate
        self.kwargs = kwargs


class Response:
    def __init__(
        self,
        result: torch.Tensor
    ):
        self.result = result


class RequestHandler:
    def __init__(
        self, 
        device_accelerator: torch.device, 
        device_offload: torch.device = None, 
        optimize_memory_use: bool = False,
        use_autocast: bool = True
    ):
        self.device_accelerator = device_accelerator
        self.device_offload = device_offload
        
        self.model_wrapper: ModelWrapperBase = None
        self.inference: InferenceBase = None 
        
        self.optimize_memory_use = optimize_memory_use
        self.use_autocast = use_autocast
        
    def process_request(
        self,
        request: Request,
        callback: Callable = None
    ) -> Response:
        # load the model from the request if it's not already loaded
        if (self.model_wrapper == None):
            self.load_model(
                request.model_type, 
                request.model_path, 
                request.model_chunk_size,
                request.model_sample_rate
            )
        elif (request.model_path != self.model_wrapper.path):
            del self.model_wrapper, self.inference
            self.load_model(
                request.model_type, 
                request.model_path, 
                request.model_chunk_size,
                request.model_sample_rate
            )
        
        handlers_by_request_type = {
            RequestType.Generation: self.handle_generation,
            RequestType.Variation: self.handle_variation,
            RequestType.Interpolation: self.handle_interpolation,
            RequestType.Inpainting: self.handle_inpainting,
            RequestType.Extension: self.handle_extension,
        }
        
        Handler = handlers_by_request_type.get(request.request_type)
        
        if Handler:
            tensor_result = Handler(request, callback)
        else:
            raise ValueError('Unexpected RequestType in process_request')
            
        return Response(tensor_result)

    def load_model(self, model_type, model_path, chunk_size, sample_rate):
        wrappers_by_model_type = {
            ModelType.DD: [DDModelWrapper, DDInference]
        }
        
        [Wrapper, Inference] = wrappers_by_model_type.get(model_type, [None, None])
        
        if Wrapper:
            self.model_wrapper = Wrapper()
            self.model_wrapper.load(
                model_path,
                self.device_accelerator,
                self.optimize_memory_use,
                chunk_size,
                sample_rate
            )
            self.inference = Inference(
                self.device_accelerator,
                self.device_offload,
                self.optimize_memory_use,
                self.use_autocast,
                self.model_wrapper
            )
        else:
            raise ValueError("Unexpected ModelType in load_model")

    def handle_generation(self, request: Request, callback: Callable) -> Response:
        kwargs = request.kwargs.copy()
        
        if request.model_type in [ModelType.DD]:
            return self.inference.generate(
                callback=callback,
                scheduler=kwargs['scheduler_type'],
                sampler=kwargs['sampler_type'],
                **kwargs
            )
        else:
            raise ValueError("Unexpected ModelType in handle_generation")

    def handle_variation(self, request: Request, callback: Callable) -> torch.Tensor:
        kwargs = request.kwargs.copy()
        kwargs.update(
            expansion_map = [kwargs['batch_size']],
            audio_source = kwargs['audio_source'][None,:,:]
        )
        
        if request.model_type in [ModelType.DD]:
            return self.inference.generate_variation(
                callback=callback,
                scheduler=kwargs['scheduler_type'],
                sampler=kwargs['sampler_type'],
                **kwargs
            )
        else:
            raise ValueError("Unexpected ModelType in handle_variation")

    def handle_interpolation(self, request: Request, callback: Callable) -> torch.Tensor:
        kwargs = request.kwargs.copy()
        kwargs.update(
            batch_size = len(kwargs['interpolation_positions']),
            audio_source = kwargs['audio_source'][None,:,:],
            audio_target = kwargs['audio_target'][None,:,:]
        )
        
        if request.model_type in [ModelType.DD]:
            return self.inference.generate_interpolation(
                callback=callback,
                scheduler=kwargs['scheduler_type'],
                sampler=kwargs['sampler_type'],
                **kwargs
            )
        else:
            raise ValueError("Unexpected ModelType in handle_interpolation")
        
    def handle_inpainting(self, request: Request, callback: Callable) -> torch.Tensor:
        kwargs = request.kwargs.copy()
        kwargs.update(
            expansion_map = [kwargs['batch_size']],
            audio_source = kwargs['audio_source'][None,:,:]
        )
        
        if request.model_type == [ModelType.DD]:
            return self.inference.generate_inpainting(
                callback=callback,
                scheduler=kwargs['scheduler_type'],
                sampler=kwargs['sampler_type'],
                **kwargs
            )
        else:
            raise ValueError("Unexpected ModelType in handle_inpainting")

    def handle_extension(self, request: Request, callback: Callable) -> torch.Tensor:
        kwargs = request.kwargs.copy()
        kwargs.update(
            expansion_map = [kwargs['batch_size']],
            audio_source = kwargs['audio_source'][None,:,:]
        )
        
        if request.model_type in [ModelType.DD]:
            return self.inference.generate_extension(
                callback=callback,
                scheduler=kwargs['scheduler_type'],
                sampler=kwargs['sampler_type'],
                **kwargs
            )
        else:
            raise ValueError("Unexpected ModelType in handle_extension")