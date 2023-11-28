import torch

class ModelWrapperBase():
    
    def __init__(self):
        #self.uuid: str = None
        #self.name: str = None
        self.path: str = None
        
        self.device_accelerator: torch.device = None
        
        self.chunk_size: int = None
        self.sample_rate: int = None
        
        
    def load(
        self,
        path: str,
        device_accelerator: torch.device,
        optimize_memory_use:bool=False,
        chunk_size: int=131072,
        sample_rate: int=48000
    ):
        raise NotImplementedError