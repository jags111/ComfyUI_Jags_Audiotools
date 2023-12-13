import torch


from tqdm.auto import trange

from diffusion.utils import t_to_alpha_sigma
from k_diffusion.external import VDenoiser

from typing import Tuple, Callable
from libs.diffusion_library.scheduler import SchedulerType
from libs.diffusion_library.sampler import SamplerType
from libs.dance_diffusion.base.model import ModelWrapperBase
from libs.dance_diffusion.base.inference import InferenceBase

from libs.util.util import tensor_slerp_2D, PosteriorSampling
    
class DDInference(InferenceBase):
    
    def __init__(
        self,
        device_accelerator: torch.device = None,
        device_offload: torch.device = None,
        optimize_memory_use: bool = False,
        use_autocast: bool = True,
        model: ModelWrapperBase = None
    ):
        super().__init__(device_accelerator, device_offload, optimize_memory_use, use_autocast, model)
        
    def generate(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args: dict = None,
        sampler: SamplerType = None,
        sampler_args: dict = None,
        **kwargs
    ):
        self.generator.manual_seed(seed)
        
        step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)#step_list = step_list[:-1] if sampler in [SamplerType.V_PRK, SamplerType.V_PLMS, SamplerType.V_PIE, SamplerType.V_PLMS2, SamplerType.V_IPLMS] else step_list
        
        if SamplerType.is_v_sampler(sampler):
            x_T = torch.randn([batch_size, 2, self.model.chunk_size], generator=self.generator, device=self.device_accelerator)
            model = self.model.model
        else:
            x_T = step_list[0] * torch.randn([batch_size, 2, self.model.chunk_size], generator=self.generator, device=self.device_accelerator)
            model = VDenoiser(self.model.model)
        
        with self.offload_context(self.model.model):
            return sampler.sample(
                model,
                x_T,
                step_list,
                callback,
                **sampler_args
            ).float()
            
            
    def generate_variation(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        expansion_map: list[int] = None,
        noise_level: float = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args = None,
        sampler: SamplerType = None,
        sampler_args = None,
        **kwargs
    ) -> torch.Tensor:
        self.generator.manual_seed(seed)
        
        audio_source = self.expand(audio_source, expansion_map)
        
        if SamplerType.is_v_sampler(sampler):
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            step_list = step_list[step_list < noise_level]
            alpha_T, sigma_T = t_to_alpha_sigma(step_list[0])
            x_T = alpha_T * audio_source + sigma_T * torch.randn(audio_source.shape, device=audio_source.device, generator=self.generator)
            model = self.model.model
        else:
            scheduler_args.update(sigma_max = scheduler_args.get('sigma_max', 1.0) * noise_level)
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            x_T = audio_source + step_list[0] * torch.randn(audio_source.shape, device=audio_source.device, generator=self.generator)
            model = VDenoiser(self.model.model)
        
        with self.offload_context(self.model.model):
            return sampler.sample(
                model,
                x_T,
                step_list,
                callback,
                **sampler_args
            ).float()
            
            
    def generate_interpolation(
        self,
        callback: Callable = None,
        batch_size: int = None,
        # seed: int = None,
        interpolation_positions: list[float] = None,
        audio_source: torch.Tensor = None,
        audio_target: torch.Tensor = None,
        expansion_map: list[int] = None,
        noise_level: float = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args = None,
        sampler: SamplerType = None,
        sampler_args = None,
        **kwargs
        ) -> torch.Tensor:
        
        if SamplerType.is_v_sampler(sampler):
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            step_list = step_list[step_list < noise_level]
            step_list[-1] += 1e-7 #HACK avoid division by 0 in reverse sampling
            model = self.model.model
        else:
            scheduler_args.update(sigma_max = scheduler_args.get('sigma_max', 1.0) * noise_level)
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            step_list = step_list[:-1] #HACK avoid division by 0 in reverse sampling
            model = VDenoiser(self.model.model)
        
        if self.optimize_memory_use and batch_size < 2:
            x_0_source = audio_source
            x_0_target = audio_target
            
            with self.offload_context(self.model.model):
                x_T_source = sampler.sample(
                    model,
                    x_0_source,
                    step_list.flip(0),
                    callback,
                    **sampler_args
                )
            
            with self.offload_context(self.model.model):
                x_T_target = sampler.sample(
                    model,
                    x_0_target,
                    step_list.flip(0),
                    callback,
                    **sampler_args
                )
            
            x_T = torch.cat([x_T_source, x_T_target], dim=0)
        else:
            x_0 = torch.cat([audio_source, audio_target], dim=0)
            
            with self.offload_context(self.model.model):
                x_T = sampler.sample(
                    model,
                    x_0,
                    step_list.flip(0),
                    callback,
                    **sampler_args
                )
        
        if SamplerType.is_v_sampler(sampler): #HACK reset schedule after hack
            step_list[-1] = 0.0
        else:
            step_list = torch.cat([step_list, step_list.new_zeros([1])])
        
        x_Int = torch.empty([batch_size, 2, self.model.chunk_size], device=self.device_accelerator)
        
        for pos in range(len(interpolation_positions)):
            x_Int[pos] = tensor_slerp_2D(x_T[0], x_T[1], interpolation_positions[pos])
        
        with self.offload_context(self.model.model):
            return sampler.sample(
                model,
                x_Int,
                step_list,
                callback,
                **sampler_args
            ).float()
            

    def generate_inpainting(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        expansion_map: list[int] = None,
        mask: torch.Tensor = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args = None,
        sampler: SamplerType = None,
        sampler_args = None,
        inpainting_args = None,
        **kwargs
    ) -> torch.Tensor:
        
        self.generator.manual_seed(seed)
        
        method = inpainting_args.get('method')
        
        if(method == 'repaint'):
            raise Exception("Repaint currently not supported due to changed requirements")
            
        elif(method == 'posterior_guidance'):
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            
            if SamplerType.is_v_sampler(sampler):
                raise Exception('V-Sampler currently not supported for posterior guidance. Please choose a K-Sampler.')
            else:
                x_T = audio_source + step_list[0] * torch.randn([batch_size, 2, self.model.chunk_size], generator=self.generator, device=self.device_accelerator)
                model = PosteriorSampling(
                    VDenoiser(self.model.model),
                    x_T,
                    audio_source,
                    mask,
                    inpainting_args.get('posterior_guidance_scale')
                )
                
                with self.offload_context(self.model.model):
                    return sampler.sample(
                        model,
                        x_T,
                        step_list,
                        callback,
                        **sampler_args
                    ).float()
                
                    
    def generate_extension(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        expansion_map: list[int] = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args = None,
        sampler: SamplerType = None,
        sampler_args = None,
        inpainting_args = None,
        keep_start: bool = None,
        **kwargs
    ) -> torch.Tensor:
        
        half_chunk_size = self.model.chunk_size // 2
        chunk = torch.cat([audio_source[:, :, -half_chunk_size:], torch.zeros([batch_size, 2, half_chunk_size], device=self.device_accelerator)], dim=2)
        #chunk = audio_source
        
        mask = torch.cat(
            [torch.ones([batch_size, 2, half_chunk_size], dtype=torch.bool, device=self.device_accelerator),
            torch.zeros([batch_size, 2, half_chunk_size], dtype=torch.bool, device=self.device_accelerator)],
            dim=2 
        )
        
        output = self.generate_inpainting(
            callback,
            batch_size,
            seed,
            chunk,
            expansion_map,
            mask,
            steps,
            scheduler,
            scheduler_args,
            sampler,
            sampler_args,
            inpainting_args
        )
        
        if (keep_start):
            return torch.cat(
                [audio_source,
                 output[:, :, -half_chunk_size:]],
                dim=2
            )
        else:
            return output[:, :, -half_chunk_size:]
from typing import Tuple, Callable
from diffusion_library.scheduler import SchedulerType
from diffusion_library.sampler import SamplerType
from dance_diffusion.base.model import ModelWrapperBase
from dance_diffusion.base.inference import InferenceBase

from libs.util.util import tensor_slerp_2D, PosteriorSampling
    
class DDInference(InferenceBase):
    
    def __init__(
        self,
        device_accelerator: torch.device = None,
        device_offload: torch.device = None,
        optimize_memory_use: bool = False,
        use_autocast: bool = True,
        model: ModelWrapperBase = None
    ):
        super().__init__(device_accelerator, device_offload, optimize_memory_use, use_autocast, model)
        
    def generate(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args: dict = None,
        sampler: SamplerType = None,
        sampler_args: dict = None,
        **kwargs
    ):
        self.generator.manual_seed(seed)
        
        step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)#step_list = step_list[:-1] if sampler in [SamplerType.V_PRK, SamplerType.V_PLMS, SamplerType.V_PIE, SamplerType.V_PLMS2, SamplerType.V_IPLMS] else step_list
        
        if SamplerType.is_v_sampler(sampler):
            x_T = torch.randn([batch_size, 2, self.model.chunk_size], generator=self.generator, device=self.device_accelerator)
            model = self.model.model
        else:
            x_T = step_list[0] * torch.randn([batch_size, 2, self.model.chunk_size], generator=self.generator, device=self.device_accelerator)
            model = VDenoiser(self.model.model)
        
        with self.offload_context(self.model.model):
            return sampler.sample(
                model,
                x_T,
                step_list,
                callback,
                **sampler_args
            ).float()
            
            
    def generate_variation(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        expansion_map: list[int] = None,
        noise_level: float = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args = None,
        sampler: SamplerType = None,
        sampler_args = None,
        **kwargs
    ) -> torch.Tensor:
        self.generator.manual_seed(seed)
        
        audio_source = self.expand(audio_source, expansion_map)
        
        if SamplerType.is_v_sampler(sampler):
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            step_list = step_list[step_list < noise_level]
            alpha_T, sigma_T = t_to_alpha_sigma(step_list[0])
            x_T = alpha_T * audio_source + sigma_T * torch.randn(audio_source.shape, device=audio_source.device, generator=self.generator)
            model = self.model.model
        else:
            scheduler_args.update(sigma_max = scheduler_args.get('sigma_max', 1.0) * noise_level)
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            x_T = audio_source + step_list[0] * torch.randn(audio_source.shape, device=audio_source.device, generator=self.generator)
            model = VDenoiser(self.model.model)
        
        with self.offload_context(self.model.model):
            return sampler.sample(
                model,
                x_T,
                step_list,
                callback,
                **sampler_args
            ).float()
            
            
    def generate_interpolation(
        self,
        callback: Callable = None,
        batch_size: int = None,
        # seed: int = None,
        interpolation_positions: list[float] = None,
        audio_source: torch.Tensor = None,
        audio_target: torch.Tensor = None,
        expansion_map: list[int] = None,
        noise_level: float = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args = None,
        sampler: SamplerType = None,
        sampler_args = None,
        **kwargs
        ) -> torch.Tensor:
        
        if SamplerType.is_v_sampler(sampler):
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            step_list = step_list[step_list < noise_level]
            step_list[-1] += 1e-7 #HACK avoid division by 0 in reverse sampling
            model = self.model.model
        else:
            scheduler_args.update(sigma_max = scheduler_args.get('sigma_max', 1.0) * noise_level)
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            step_list = step_list[:-1] #HACK avoid division by 0 in reverse sampling
            model = VDenoiser(self.model.model)
        
        if self.optimize_memory_use and batch_size < 2:
            x_0_source = audio_source
            x_0_target = audio_target
            
            with self.offload_context(self.model.model):
                x_T_source = sampler.sample(
                    model,
                    x_0_source,
                    step_list.flip(0),
                    callback,
                    **sampler_args
                )
            
            with self.offload_context(self.model.model):
                x_T_target = sampler.sample(
                    model,
                    x_0_target,
                    step_list.flip(0),
                    callback,
                    **sampler_args
                )
            
            x_T = torch.cat([x_T_source, x_T_target], dim=0)
        else:
            x_0 = torch.cat([audio_source, audio_target], dim=0)
            
            with self.offload_context(self.model.model):
                x_T = sampler.sample(
                    model,
                    x_0,
                    step_list.flip(0),
                    callback,
                    **sampler_args
                )
        
        if SamplerType.is_v_sampler(sampler): #HACK reset schedule after hack
            step_list[-1] = 0.0
        else:
            step_list = torch.cat([step_list, step_list.new_zeros([1])])
        
        x_Int = torch.empty([batch_size, 2, self.model.chunk_size], device=self.device_accelerator)
        
        for pos in range(len(interpolation_positions)):
            x_Int[pos] = tensor_slerp_2D(x_T[0], x_T[1], interpolation_positions[pos])
        
        with self.offload_context(self.model.model):
            return sampler.sample(
                model,
                x_Int,
                step_list,
                callback,
                **sampler_args
            ).float()
            

    def generate_inpainting(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        expansion_map: list[int] = None,
        mask: torch.Tensor = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args = None,
        sampler: SamplerType = None,
        sampler_args = None,
        inpainting_args = None,
        **kwargs
    ) -> torch.Tensor:
        
        self.generator.manual_seed(seed)
        
        method = inpainting_args.get('method')
        
        if(method == 'repaint'):
            raise Exception("Repaint currently not supported due to changed requirements")
            
        elif(method == 'posterior_guidance'):
            step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)
            
            if SamplerType.is_v_sampler(sampler):
                raise Exception('V-Sampler currently not supported for posterior guidance. Please choose a K-Sampler.')
            else:
                x_T = audio_source + step_list[0] * torch.randn([batch_size, 2, self.model.chunk_size], generator=self.generator, device=self.device_accelerator)
                model = PosteriorSampling(
                    VDenoiser(self.model.model),
                    x_T,
                    audio_source,
                    mask,
                    inpainting_args.get('posterior_guidance_scale')
                )
                
                with self.offload_context(self.model.model):
                    return sampler.sample(
                        model,
                        x_T,
                        step_list,
                        callback,
                        **sampler_args
                    ).float()
                
                    
    def generate_extension(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        expansion_map: list[int] = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args = None,
        sampler: SamplerType = None,
        sampler_args = None,
        inpainting_args = None,
        keep_start: bool = None,
        **kwargs
    ) -> torch.Tensor:
        
        half_chunk_size = self.model.chunk_size // 2
        chunk = torch.cat([audio_source[:, :, -half_chunk_size:], torch.zeros([batch_size, 2, half_chunk_size], device=self.device_accelerator)], dim=2)
        #chunk = audio_source
        
        mask = torch.cat(
            [torch.ones([batch_size, 2, half_chunk_size], dtype=torch.bool, device=self.device_accelerator),
            torch.zeros([batch_size, 2, half_chunk_size], dtype=torch.bool, device=self.device_accelerator)],
            dim=2 
        )
        
        output = self.generate_inpainting(
            callback,
            batch_size,
            seed,
            chunk,
            expansion_map,
            mask,
            steps,
            scheduler,
            scheduler_args,
            sampler,
            sampler_args,
            inpainting_args
        )
        
        if (keep_start):
            return torch.cat(
                [audio_source,
                 output[:, :, -half_chunk_size:]],
                dim=2
            )
        else:
            return output[:, :, -half_chunk_size:]
