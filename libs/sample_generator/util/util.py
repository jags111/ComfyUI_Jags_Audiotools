import os
import torch
import torchaudio
from k_diffusion.utils import append_dims


def tensor_slerp_2D(a: torch.Tensor, b: torch.Tensor, t: float):
    slerped = torch.empty_like(a)
    
    for channel in range(a.size(0)):
        slerped[channel] = tensor_slerp(a[channel], b[channel], t)
    
    return slerped


def tensor_slerp(a: torch.Tensor, b: torch.Tensor, t: float):
    omega = torch.arccos(torch.dot(a / torch.linalg.norm(a), b / torch.linalg.norm(b)))
    so = torch.sin(omega)
    return torch.sin((1.0 - t) * omega) / so * a + torch.sin(t * omega) / so * b


def load_audio(device, audio_path: str, sample_rate):
    
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")

    audio, file_sample_rate = torchaudio.load(audio_path)

    if file_sample_rate != sample_rate:
        resample = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        audio = resample(audio)

    return audio.to(device)


def save_audio(audio_out, output_path: str, sample_rate, id_str:str = None):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(output_path, f"sample_{id_str}_{ix + 1}.wav" if(id_str!=None) else f"sample_{ix + 1}.wav")
        open(output_file, "a").close()
        
        output = sample.cpu()

        torchaudio.save(output_file, output, sample_rate)


def crop_audio(source: torch.Tensor, chunk_size: int, crop_offset: int = 0) -> torch.Tensor:
    n_channels, n_samples = source.shape
    
    offset = 0
    if (crop_offset > 0):
        offset = min(crop_offset, n_samples - chunk_size)
    elif (crop_offset == -1):
        offset = torch.randint(0, max(0, n_samples - chunk_size) + 1, []).item()
    
    chunk = source.new_zeros([n_channels, chunk_size])
    chunk [:, :min(n_samples, chunk_size)] = source[:, offset:offset + chunk_size]
    
    return chunk


class PosteriorSampling(torch.nn.Module):
    def __init__(self, model, x_T, measurement, mask, scale):
        super().__init__()
        self.model = model
        self.x_prev = x_T
        self.measurement = measurement
        self.mask = mask
        self.scale = scale
    
    @torch.enable_grad()
    def forward(self, input, sigma, **kwargs):
        x_t = input.detach().requires_grad_()
        out = self.model(x_t, sigma, **kwargs)
        difference = (self.measurement - out) * self.mask
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t)[0].detach()
        
        return out.detach() - self.scale * norm_grad
        
        # x_t = input.detach().requires_grad_()
        # x_0_hat = self.model(input, sigma, **kwargs).detach().requires_grad_()
        
        # difference = (self.measurement - x_0_hat) * self.mask
        # norm = torch.linalg.norm(difference)
        # norm_grad = torch.autograd.grad(outputs=norm, inputs=self.x_prev)[0].detach()
        
        # self.x_prev = x_t.detach().requires_grad_()
        
        # return x_t.detach() - norm_grad * self.scale
    
# class PosteriorSampling(torch.nn.Module):
#     def __init__(self, model, measurement, mask, strength):
#         super().__init__()
#         self.model = model
#         self.mask = mask
#         self.measurement = measurement
#         self.strength = strength
    
#     @torch.enable_grad()
#     def forward(self, input, sigma, **kwargs):
#         input = input.detach().requires_grad_()
#         out = self.model(input, sigma, **kwargs)
#         difference = (self.measurement - out) * self.mask
#         norm = torch.linalg.norm(difference)
#         norm_grad = torch.autograd.grad(outputs=norm, inputs=input)[0].detach()
#         N = self.measurement.shape[-1]**0.5
#         step_size = -self.strength * N / (torch.linalg.norm(norm_grad) + 1e-4) * append_dims(sigma**2, input.ndim)
#         print('Norm:', norm.detach())
#         print('Step size:', step_size)
#         return out.detach() + step_size * norm_grad