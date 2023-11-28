import enum, torch
from diffusion import sampling as vsampling
from k_diffusion import sampling as ksampling


class SamplerType(str, enum.Enum):
    V_DDPM = 'V_DDPM'
    V_DDIM = 'V_DDIM'
    V_PRK = 'V_PRK'
    V_PIE = 'V_PIE'
    V_PLMS = 'V_PLMS'
    V_PLMS2 = 'V_PLMS2'
    V_IPLMS = 'V_IPLMS'
    
    K_EULER = 'K_EULER'
    K_EULERA = 'K_EULERA'
    K_HEUN = 'K_HEUN'
    K_DPM2 = 'K_DPM2'
    K_DPM2A = 'K_DPM2A'
    K_LMS = 'K_LMS'
    K_DPMF = 'K_DPMF'
    K_DPMA = 'K_DPMA'
    K_DPMPP2SA = 'K_DPMPP2SA'
    K_DPMPP2M = 'K_DPMPP2M'
    K_DPMPPSDE = 'K_DPMPPSDE'

    @classmethod
    def is_v_sampler(cls, value):
        return value[0] == 'V'

    def sample(self, model_fn, x_t, steps, callback, **sampler_args) -> torch.Tensor:
        if self == SamplerType.V_DDPM:
            if sampler_args.get('is_reverse'):
                return vsampling.reverse_sample(
                    model_fn,
                    x_t,
                    steps,
                    0.0,
                    sampler_args.get('extra_args', {}),
                    callback
                )
            else:
                return vsampling.sample(
                    model_fn,
                    x_t,
                    steps,
                    0.0,
                    sampler_args.get('extra_args', {}),
                    callback
                )
        elif self == SamplerType.V_DDIM:
            if sampler_args.get('is_reverse'): # HACK: Technically incorrect since DDIM implies eta > 0.0
                return vsampling.reverse_sample(
                    model_fn,
                    x_t,
                    steps,
                    0.0,
                    sampler_args.get('extra_args', {}),
                    callback
                )
            else:
                return vsampling.sample(
                    model_fn,
                    x_t,
                    steps,
                    sampler_args.get('eta', 0.1),
                    sampler_args.get('extra_args', {}),
                    callback
                )
        elif self == SamplerType.V_PRK:
            return vsampling.prk_sample(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                True,
                callback
            )
        elif self == SamplerType.V_PIE:
            return vsampling.pie_sample(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                True,
                callback
            )
        elif self == SamplerType.V_PLMS:
            return vsampling.plms_sample(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                True,
                callback
            )
        elif self == SamplerType.V_PLMS2:
            return vsampling.plms2_sample(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                True,
                callback
            )
        elif self == SamplerType.V_IPLMS:
            return vsampling.iplms_sample(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                True,
                callback
            )
        elif self == SamplerType.K_EULER:
            return ksampling.sample_euler(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('s_churn', 0.0),
                sampler_args.get('s_tmin', 0.0),
                sampler_args.get('s_tmax',float('inf')),
                sampler_args.get('s_noise', 1.0)
            )
        elif self == SamplerType.K_EULERA:
            return ksampling.sample_euler_ancestral(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('eta', 0.1),
                sampler_args.get('s_noise', 1.0),
                sampler_args.get('noise_sampler', None)
            )
        elif self == SamplerType.K_HEUN:
            return ksampling.sample_heun(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('s_churn', 0.0),
                sampler_args.get('s_tmin', 0.0),
                sampler_args.get('s_tmax',float('inf')),
                sampler_args.get('s_noise', 1.0)
            )
        elif self == SamplerType.K_DPM2:
            return ksampling.sample_dpm_2(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('s_churn', 0.0),
                sampler_args.get('s_tmin', 0.0),
                sampler_args.get('s_tmax',float('inf')),
                sampler_args.get('s_noise', 1.0)
            )
        elif self == SamplerType.K_DPM2A:
            return ksampling.sample_dpm_2_ancestral(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('eta', 0.1),
                sampler_args.get('s_noise', 1.0),
                sampler_args.get('noise_sampler', None)
            )
        elif self == SamplerType.K_LMS:
            return ksampling.sample_lms(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('order', 4)
            )
        elif self == SamplerType.K_DPMF:# sample_dpm_fast
            return ksampling.sample_dpm_fast(
                model_fn,
                x_t,
                sampler_args.get('sigma_min', 0.001),
                sampler_args.get('sigma_max', 1.0),
                sampler_args.get('n', 3),
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('eta', 0.1),
                sampler_args.get('s_noise', 1.0),
                sampler_args.get('noise_sampler', None)
            )
        elif self == SamplerType.K_DPMA:
            return ksampling.sample_dpm_adaptive(
                model_fn,
                x_t,
                sampler_args.get('sigma_min', 0.001),
                sampler_args.get('sigma_max', 1.0),
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('order', 3),
                sampler_args.get('rtol', 0.05),
                sampler_args.get('atol', 0.0078),
                sampler_args.get('h_init', 0.05),
                sampler_args.get('pcoeff', 0.0),
                sampler_args.get('icoeff', 1.0),
                sampler_args.get('dcoeff', 0.0),
                sampler_args.get('accept_safety', 0.81),
                sampler_args.get('eta', 0.1),
                sampler_args.get('s_noise', 1.0),
                sampler_args.get('noise_sampler', None),
                sampler_args.get('return_info', False)
            )
        elif self == SamplerType.K_DPMPP2SA:
            return ksampling.sample_dpmpp_2s_ancestral(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('eta', 0.1),
                sampler_args.get('s_noise', 1.0),
                sampler_args.get('noise_sampler', None)
            )
        elif self == SamplerType.K_DPMPP2M:
            return ksampling.sample_dpmpp_2m(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False)
            )
        elif self == SamplerType.K_DPMPPSDE:
            return ksampling.sample_dpmpp_sde(
                model_fn,
                x_t,
                steps,
                sampler_args.get('extra_args', {}),
                callback,
                sampler_args.get('disable', False),
                sampler_args.get('eta', 0.1),
                sampler_args.get('s_noise', 1.0),
                sampler_args.get('noise_sampler', None),
                sampler_args.get('r', 1/2)
            )
        else:
            raise Exception(f"No sample implementation for sampler_type '{self}'")
        
