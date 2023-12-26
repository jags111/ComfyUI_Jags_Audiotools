import numpy as np
from scipy import signal as sig
import torch
import torchaudio
import os
import random


NOTES = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

def pitch2freq(pitch, A4=440):
    note = NOTES.index(pitch[:-1])
    octave = int(pitch[-1])
    distance_from_A4 = note + 12 * (octave - (4 if note < 3 else 5))

    return A4 * 2 ** (distance_from_A4/12)

def create_signal(keys, sample_rate, sample_length, amplitude, waveform='sine'):
    keys = keys.replace(',', '').split(' ')
    awaves = []
    for key in keys:
        key = key.replace(key[-1], str(int(key[-1]) - 2))
        freq = pitch2freq(key.upper())  # Frequency in Hz
        x = np.arange(sample_length)

        if waveform == 'sine':
            awave = 100*np.sin(2 * np.pi * freq * x / sample_rate)
        elif waveform == 'square':
            awave = 100*sig.square(2 * np.pi * freq * x / sample_rate)
        elif waveform == 'saw':
            awave = 100*sig.sawtooth(2 * np.pi * freq * x / sample_rate)
        awaves.append(awave)

    awaves = np.sum(awaves, axis=0)
    awaves = awaves / np.max(np.abs(awaves))
    awaves = awaves * amplitude
    awaves = torch.from_numpy(awaves)
    return awaves.unsqueeze(0).repeat(2, 1).unsqueeze(0).float()


class WaveGenerator():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keys": ("STRING", {'default': 'C5 C6 C7'}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1}),
                "chunk_size": ("INT", {"default": 65536, "min": 32768, "max": 10000000000, "step": 32768}),
                "amplitude": ("FLOAT", {'default': 1.0, 'min': 0.0, 'max': 2.0, 'step': 0.01}),
                "waveform": (["sine", "square", "saw"], {'default': 'sine'})
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("STRING", "AUDIO", "INT")
    RETURN_NAMES = ("path", "🎙️audio", "sample_rate")
    FUNCTION = "generate_wave"

    CATEGORY = "🎙️Jags_Audio/WaveGenerator"

    def generate_wave(self, keys, sample_rate, chunk_size, amplitude, waveform='sine'):
        tensor = create_signal(keys=keys, sample_rate=sample_rate, sample_length=chunk_size, amplitude=amplitude, waveform=waveform)
        rand = random.randint(0, 100000000000)
    
        dirs = __file__.split('\\')
        comfy_index = None
        for i, dir in enumerate(dirs):
            if dir == "ComfyUI":
                comfy_index = i
                break
        if comfy_index is not None:
            # Join the list up to the "ComfyUI" folder
            comfy_dir = '\\'.join(dirs[:comfy_index+1])


        for ix, sample in enumerate(tensor):
            if not os.path.exists(os.path.join(comfy_dir, 'temp')):
                os.makedirs(os.path.join(comfy_dir, 'temp'))
            path = os.path.join(comfy_dir, 'temp\\', f"sample_{rand}.wav")
            open(path, "a").close()
            
            output = sample.cpu()
            torchaudio.save(path, output, sample_rate)



        return (path, tensor, sample_rate)

NODE_CLASS_MAPPINGS = {
    "GenerateAudioWave": WaveGenerator
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateAudioWave": "Jags_Wave Generator"
}
