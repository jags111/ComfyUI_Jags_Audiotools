import math
import enum
import torch

from diffusion import utils as vscheduling
from k_diffusion import sampling as kscheduling


class SchedulerType(str, enum.Enum):
    V_DDPM = 'V_DDPM'
    V_SPLICED_DDPM_COSINE = 'V_SPLICED_DDPM_COSINE'
    V_LOG = 'V_LOG'
    V_CRASH = 'V_CRASH'
    
    K_KARRAS = 'K_KARRAS'
    K_EXPONENTIAL = 'K_EXPONENTIAL'
    K_POLYEXPONENTIAL = 'K_POLYEXPONENTIAL'
    K_VP = 'K_VP'
    
    @classmethod
    def is_v_scheduler(cls, value):
        return value[0] == 'V'
        
    def get_step_list(self, n: int, device: str, **schedule_args):
        #if SchedulerType.is_v_scheduler(self):
        #    n -= 1

        if self == SchedulerType.V_DDPM:
            return torch.nn.functional.pad(vscheduling.get_ddpm_schedule(torch.linspace(1, 0, n)), [0,1], value=0.0).to(device)
        elif self == SchedulerType.V_SPLICED_DDPM_COSINE:
            return vscheduling.get_spliced_ddpm_cosine_schedule(torch.linspace(1, 0, n + 1)).to(device)
        elif self == SchedulerType.V_LOG:
            return torch.nn.functional.pad(
                vscheduling.get_log_schedule(
                    torch.linspace(1, 0, n),
                    schedule_args.get('min_log_snr', -10.0),
                    schedule_args.get('max_log_snr', 10.0)
                ),
                [0,1],
                value=0.0
            ).to(device)
        elif self == SchedulerType.V_CRASH:
            sigma = torch.sin(torch.linspace(1, 0, n + 1) * math.pi / 2) ** 2
            alpha = (1 - sigma ** 2) ** 0.5
            return vscheduling.alpha_sigma_to_t(alpha, sigma).to(device)
        elif self == SchedulerType.K_KARRAS:
            return kscheduling.get_sigmas_karras(
                n,
                schedule_args.get('sigma_min', 0.001),
                schedule_args.get('sigma_max', 1.0),
                schedule_args.get('rho', 7.0),
                device = device
            )
        elif self == SchedulerType.K_EXPONENTIAL:
            return kscheduling.get_sigmas_exponential(
                n,
                schedule_args.get('sigma_min', 0.001),
                schedule_args.get('sigma_max', 1.0),
                device = device
            )
        elif self == SchedulerType.K_POLYEXPONENTIAL:
            return kscheduling.get_sigmas_polyexponential(
                n,
                schedule_args.get('sigma_min', 0.001),
                schedule_args.get('sigma_max', 1.0),
                schedule_args.get('rho', 1.0),
                device = device
            )
        elif self == SchedulerType.K_VP:
            return kscheduling.get_sigmas_vp(
                n,
                schedule_args.get('beta_d', 1.205),
                schedule_args.get('beta_min', 0.09),
                schedule_args.get('eps_s', 0.001),
                device = device
            )
        else:
            raise Exception(f"No get_step_list implementation for scheduler_type '{self}'")
