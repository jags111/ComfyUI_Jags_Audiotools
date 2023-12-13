#!/usr/bin/env python

'''
Spectrology
This script is able to encode an image into audio file whose spectrogram represents input image.

License: MIT
Website: https://github.com/solusipse/spectrology
'''

from PIL import Image, ImageOps
import wave, math, array, sys
from tqdm.auto import tqdm
import torchaudio
import torch

import os
import scipy.signal as signal
import subprocess
try:
    import matplotlib
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib

# Install the missing package

# Import the required module
from comfy.model_management import get_torch_device

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import random
import numpy as np

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

class Plot_Spectrogram():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "audio_tensor": ("AUDIO", {}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1}),
                "window_size": ("INT", {"default": 512, "min": 1, "max": 10000000000, "step": 1}),
                "overlap_size": ("INT", {"default": 256, "min": 1, "max": 10000000000, "step": 1}),
                "color_map": (list(cm.datad.keys()), {"default": "Spectral"}),
                "labels": (['Enabled', 'Disabled'], {"default": 'Disabled'}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "PlotSpectrogram"
    OUTPUT_NODE = True

    CATEGORY = "Jags_Audio/Extra"

    def PlotSpectrogram(self, audio_tensor, sample_rate, window_size, overlap_size, color_map, labels):
        labels = labels == 'Enabled'
        images = []
        for image in audio_tensor:
            # get only last dim
            image = image[-1]
            image = Image.open(save_spectrogram_image(image.cpu(), sample_rate, window_size, overlap_size, color_map, labels))
            out_image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            out_image = torch.from_numpy(out_image).unsqueeze(0)
            images.append(out_image)
        #audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        images = torch.cat(images, dim=0)
        return (images, )

def save_spectrogram_image(audio_tensor, sample_rate=44100, nperseg=512, noverlap=256, cmap='Spectral', labels=False):
    # Compute the spectrogram
    freqs, times, spectrogram = signal.spectrogram(audio_tensor.numpy(), fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    # Convert the spectrogram to dB scale
    spectrogram = 10 * np.log10(spectrogram)

    # Plot the spectrogram
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(times, freqs, spectrogram, cmap=cmap)
    if labels:
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()
    filename = f'{get_comfy_dir()}\\temp\\spectrogram_{random.randint(0,1000000000000000000)}.png'
    # Save the spectrogram to a file
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.savefig(filename)

    # Close the plot to free up memory
    plt.close()
    return filename


class ImageToSpectral():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "input_images": ("IMAGE",{}),
                "bottom_frequency": ("INT",{"default": 200, "min": 0, "max": 24000, "step": 1}),
                "top_frequency": ("INT",{"default": 20000, "min": 0, "max": 24000, "step": 1}),
                "pixels_per_second": ("INT",{"default": 30, "min": 0, "max": 1000, "step": 1}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1}),
                "rotate": (["Enabled", "Disabled"], {"default": "Disabled"}),
                "invert": (["Enabled", "Disabled"], {"default": "Disabled"}),
                "width": ("INT", {"default": 256, "min": 0, "max": 10000000000, "step": 1}),
                "height": ("INT", {"default": 256, "min": 0, "max": 10000000000, "step": 1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("STRING", "AUDIO", "INT")
    RETURN_NAMES = ("path", "tensor", "sample_rate")
    FUNCTION = "DoImageToSpectral"
    OUTPUT_NODE = True

    CATEGORY = "Jags_Audio/Extra"

    def tensor_to_pil(self, img):
        if img is not None:
            i = 255. * img.cpu().numpy().squeeze()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def DoImageToSpectral(self, input_images, bottom_frequency, top_frequency, pixels_per_second, sample_rate, rotate, invert, width, height):
        audio_total = []
        size = width, height
        for image in input_images:
            # fix slashes
            rotate = rotate == "Enabled"
            invert = invert == "Enabled"
            name = str(random.randint(0,100000000))
            if not os.path.exists(os.path.join(get_comfy_dir(), 'temp\\')):
                os.makedirs(os.path.join(get_comfy_dir(), 'temp\\'))
            base_path = os.path.join(get_comfy_dir(), 'temp\\', name)
            png_input = os.path.join(base_path + '.png')
            audio_output = os.path.join(base_path + '_spectogram.wav')
            # save input tensor image
            image = self.tensor_to_pil(image)
            if image.size != size:
                image = image.resize(size)
            image.save(png_input)

            convert(png_input, audio_output, bottom_frequency, top_frequency, pixels_per_second, sample_rate, rotate, invert)
            tensor, sample_rate = torchaudio.load(audio_output)
            # add dimension for channel with value 2
            tensor = tensor.unsqueeze(0)
            audio_total.append(tensor)
        audio_total = torch.cat(audio_total, dim=0)
        audio_total = audio_total.to(get_torch_device())


        return (audio_output, audio_total, sample_rate)


def convert(inpt, output, minfreq, maxfreq, pxs, wavrate, rotate, invert):
    img = Image.open(inpt).convert('L')

    # rotate image if requested
    if rotate:
      img = img.rotate(90)

    # invert image if requested
    if invert:
      img = ImageOps.invert(img)

    output = wave.open(output, 'w')
    output.setparams((1, 2, wavrate, 0, 'NONE', 'not compressed'))

    freqrange = maxfreq - minfreq
    interval = freqrange / img.size[1]

    fpx = wavrate // pxs
    data = array.array('h')

    for x in tqdm(range(img.size[0])):
        row = []
        for y in range(img.size[1]):
            yinv = img.size[1] - y - 1
            amp = img.getpixel((x,y))
            if (amp > 0):
                row.append( genwave(yinv * interval + minfreq, amp, fpx, wavrate) )

        for i in range(fpx):
            for j in row:
                try:
                    data[i + x * fpx] += j[i]
                except(IndexError):
                    data.insert(i + x * fpx, j[i])
                except(OverflowError):
                    if j[i] > 0:
                      data[i + x * fpx] = 32767
                    else:
                      data[i + x * fpx] = -32768

    output.writeframes(data.tobytes())
    output.close()

def genwave(frequency, amplitude, samples, samplerate):
    cycles = samples * frequency / samplerate
    a = []
    for i in range(samples):
        x = math.sin(float(cycles) * 2 * math.pi * i / float(samples)) * float(amplitude)
        a.append(int(math.floor(x)))
    return a

NODE_CLASS_MAPPINGS = {
    'ImageToSpectral': ImageToSpectral,
    'PlotSpectrogram': Plot_Spectrogram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ImageToSpectral': 'Jags Image To Spectral',
    'PlotSpectrogram': 'Jags Plot Spectrogram',
}
