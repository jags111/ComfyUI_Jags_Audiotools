# Imports
from torch import from_numpy
from tqdm import tqdm
import torch
import subprocess, sys, os
try:
    from pedalboard import load_plugin, Pedalboard, Chorus, Reverb, Bitcrush, Delay, Clipping, Compressor, Distortion
    from pedalboard import Convolution, Gain, Invert, Limiter, NoiseGate, Phaser, PitchShift, MP3Compressor
    from pedalboard import HighShelfFilter, HighpassFilter, LowShelfFilter, LowpassFilter, LadderFilter, PeakFilter
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pedalboard"])
    from pedalboard import load_plugin, Pedalboard, Chorus, Reverb, Bitcrush, Delay, Clipping, Compressor, Distortion
    from pedalboard import Convolution, Gain, Invert, Limiter, NoiseGate, Phaser, PitchShift, MP3Compressor
    from pedalboard import HighShelfFilter, HighpassFilter, LowShelfFilter, LowpassFilter, LadderFilter, PeakFilter

# ****************************************************************************
# *                                  HELPERS                                 *
# ****************************************************************************


def apply_board_tensor(tensor, sample_rate, board):
    total_tensor = []
    for audio in tqdm(tensor):
        audio = audio.cpu()
        audio = audio.numpy()
        audio = board(audio, sample_rate)
        audio = pedal_to_tensor(audio)
        total_tensor.append(audio)
    return torch.cat(total_tensor, 0)  

def pedal_to_tensor(f):
    return from_numpy(f).unsqueeze(0)



def apply_vst3_tensor(tensor, sample_rate, vst3, params):
    total_tensor = []
    for audio in tqdm(tensor):
        audio = audio.cpu()
        audio = audio.numpy()
        for vst_param in vst3.parameters.keys():
            if vst_param in params:
                setattr(vst3, vst_param, params[vst_param])

        audio = vst3(audio, sample_rate)
        audio = pedal_to_tensor(audio)
        total_tensor.append(audio)

    total_tensor = torch.cat(total_tensor, 0)
    
    # add
    
    return total_tensor



class LoadVST():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vst3_path": ("STRING", {'default': ''})
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("VST3", )
    RETURN_NAMES = ("vst3", )
    FUNCTION = "load_vst3"

    CATEGORY = "🎙️Jags_Audio/Pedalboard"

    def load_vst3(self, vst3_path):
        return (load_plugin(vst3_path), )


# ****************************************************************************
# *                                 EXTERNAL                                 *
# ****************************************************************************


class OTT():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "vst3": ('VST3', ),
                "depth": ("FLOAT", {'default': 100.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "time": ("FLOAT", {'default': 5.0, "min": 0.0, "max": 1000.0, "step": 1.0}),
                "in_gain_db": ("FLOAT", {'default': 0.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                "out_gain_db": ("FLOAT", {'default': 0.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                "thresh_l": ("FLOAT", {'default': 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "thresh_m": ("FLOAT", {'default': 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "thresh_h": ("FLOAT", {'default': 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "gain_l_db": ("FLOAT", {'default': 0.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                "gain_m_db": ("FLOAT", {'default': 0.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                "gain_h_db": ("FLOAT", {'default': 0.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                "upwd_strght": ("FLOAT", {'default': 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "dnwd_strght": ("FLOAT", {'default': 100.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_ott"

    CATEGORY = "🎙️Jags_Audio/Pedalboard/VST3Wrappers"

    def apply_ott(
        self,
        audio,
        sample_rate,
        vst3,
        depth=100.0,
        time=150.0,
        in_gain_db=0.0, 
        out_gain_db=5.0, 
        thresh_l=100.0, 
        thresh_m=100.0, 
        thresh_h=100.0,
        gain_l_db=0.0,
        gain_m_db=0.0,
        gain_h_db=0.0,
        upwd_strght=100.0,
        dnwd_strght=100.0,
        ):
        params = locals()
        del params['vst3']
        total_tensor = apply_vst3_tensor(audio, sample_rate, vst3, params)

        return (total_tensor, sample_rate)


# ****************************************************************************
# *                                   CORE FX                                *
# ****************************************************************************



class BitCrushEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "bit_depth": ("FLOAT", {'default': 8.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_bitcrush"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_bitcrush(self, audio, sample_rate, bit_depth=8.0, ):
        board = Pedalboard([Bitcrush(bit_depth=bit_depth)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)


        return (total_tensor, sample_rate)

class ChorusEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "rate_hz": ("FLOAT", {'default': 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "depth": ("FLOAT", {'default': 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "centre_delay_ms": ("FLOAT", {'default': 0.0, "min": 0.0, "max": 80.0, "step": 0.1}),
                "feedback": ("FLOAT", {'default': 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mix": ("FLOAT", {'default': 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_chorus"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_chorus(self, audio, sample_rate, rate_hz=1.0, depth=0.25, centre_delay_ms=0.0, feedback=0.0, mix=0.5):
        board = Pedalboard([Chorus(rate_hz=rate_hz, depth=depth, centre_delay_ms=centre_delay_ms, feedback=feedback, mix=mix)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)


        return (total_tensor, sample_rate)

class ClippingEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "threshold_db": ("FLOAT", {'default': -6.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_clipping"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_clipping(self, audio, sample_rate, threshold_db):
        board = Pedalboard([Clipping(threshold_db=threshold_db)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)


        return (total_tensor, sample_rate)

class CompressorEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "threshold_db": ("FLOAT", {'default': -6.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "ratio": ("FLOAT", {'default': 1.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "attack_ms": ("FLOAT", {'default': 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "release_ms": ("FLOAT", {'default': 100.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_compressor"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_compressor(self, audio, sample_rate, threshold_db=-6.0, ratio=1.0, attack_ms=1.0, release_ms=100.0):
        board = Pedalboard([Compressor(threshold_db=threshold_db, ratio=ratio, attack_ms=attack_ms, release_ms=release_ms)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class ConvolutionEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "impulse_response_path": ("STRING", {"default": "", "forceInput": True}),
                "mix": ("FLOAT", {'default': 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_convolution"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_convolution(self, audio, sample_rate, impulse_response_path, mix=1.0):
        board = Pedalboard([Convolution(impulse_response_path=impulse_response_path, mix=mix)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class DelayEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "delay_seconds": ("FLOAT", {'default': 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feedback": ("FLOAT", {'default': 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mix": ("FLOAT", {'default': 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_delay"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_delay(self, audio, sample_rate, delay_seconds=0.5, feedback=0.0, mix=0.5):
        board = Pedalboard([Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class DistortionEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "drive_db": ("FLOAT", {'default': 25.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_distortion"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_distortion(self, audio, sample_rate, drive_db=25.0):
        board = Pedalboard([Distortion(drive_db=drive_db)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class GainEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "gain_db": ("FLOAT", {'default': 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_gain"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_gain(self, audio, sample_rate, gain_db=0.0):
        board = Pedalboard([Gain(gain_db=gain_db)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class InvertEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_invert"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_invert(self, audio, sample_rate):
        board = Pedalboard([Invert()])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)


class LimiterEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "threshold_db": ("FLOAT", {'default': -10.0, "min": -100.0, "max": 0.0, "step": 0.1}),
                "release_ms": ("FLOAT", {'default': 100.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_limiter"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_limiter(self, audio, sample_rate, threshold_db=-10.0, release_ms=100.0):
        board = Pedalboard([Limiter(threshold_db=threshold_db, release_ms=release_ms)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)


class MP3CompressorEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "vbr_quality": ("FLOAT", {'default': 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_mp3_compressor"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_mp3_compressor(self, audio, sample_rate, vbr_quality=2.0):
        board = Pedalboard([MP3Compressor(vbr_quality=vbr_quality)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)


class NoiseGateEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "threshold_db": ("FLOAT", {'default': -100.0, "min": -100.0, "max": 0.0, "step": 0.1}),
                "ratio": ("FLOAT", {'default': 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "attack_ms": ("FLOAT", {'default': 1.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "release_ms": ("FLOAT", {'default': 100.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_noise_gate"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_noise_gate(self, audio, sample_rate, threshold_db=-100.0, ratio=10.0, attack_ms=1.0, release_ms=100.0):
        board = Pedalboard([NoiseGate(threshold_db=threshold_db, ratio=ratio, attack_ms=attack_ms, release_ms=release_ms)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

# class pedalboard.PitchShift(semitones: float = 0.0)

class PitchShiftEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "semitones": ("FLOAT", {'default': 0.0, "min": -96.0, "max": 96.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_pitch_shift"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_pitch_shift(self,audio, sample_rate, semitones=0.0):
        board = Pedalboard([PitchShift(semitones=semitones)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class PhaserEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "rate_hz": ("FLOAT", {'default': 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "depth": ("FLOAT", {'default': 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "centre_frequency_hz": ("FLOAT", {'default': 1300.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "feedback": ("FLOAT", {'default': 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mix": ("FLOAT", {'default': 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_phaser"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_phaser(self, audio, sample_rate, rate_hz=1.0, depth=0.5, centre_frequency_hz=1300.0, feedback=0.0, mix=0.5):
        board = Pedalboard([Phaser(rate_hz=rate_hz, depth=depth, centre_frequency_hz=centre_frequency_hz, feedback=feedback, mix=mix)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class ReverbEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "room_size": ("FLOAT", {'default': 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "damping": ("FLOAT", {'default': 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "wet_level": ("FLOAT", {'default': 0.33, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dry_level": ("FLOAT", {'default': 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "width": ("FLOAT", {'default': 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "freeze_mode": ("FLOAT", {'default': 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_reverb"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/FX"

    def apply_reverb(self, audio, sample_rate, room_size=0.5, damping=0.5, wet_level=0.33, dry_level=0.4, width=1.0, freeze_mode=0.0):
        board = Pedalboard([Reverb(room_size=room_size, damping=damping, wet_level=wet_level, dry_level=dry_level, width=width, freeze_mode=freeze_mode)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

# ****************************************************************************
# *                                  FILTERS                                 *
# ****************************************************************************

class HighShelfFilterEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "cutoff_frequency_hz": ("FLOAT", {'default': 440.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "gain_db": ("FLOAT", {'default': 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "q": ("FLOAT", {'default': 0.7071067690849304, "min": 0.0, "max": 100.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_high_shelf_filter"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/Filters"

    def apply_high_shelf_filter(self, audio, sample_rate, cutoff_frequency_hz=440.0, gain_db=0.0, q=0.7071067690849304):
        board = Pedalboard([HighShelfFilter(cutoff_frequency_hz=cutoff_frequency_hz, gain_db=gain_db, q=q)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class HighpassFilterEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "cutoff_frequency_hz": ("FLOAT", {'default': 50.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_highpass_filter"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/Filters"

    def apply_highpass_filter(self, audio, sample_rate, cutoff_frequency_hz=50.0):
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff_frequency_hz)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class LadderFilterEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "mode": (['LPF12', 'LPF24', 'BPF12', 'BPF24', 'HPF12', 'HPF24'], {'default': 'LPF12'}),
                "cutoff_hz": ("FLOAT", {'default': 200.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "resonance": ("FLOAT", {'default': 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "drive": ("FLOAT", {'default': 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_ladder_filter"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/Filters"

    def apply_ladder_filter(self, audio, sample_rate, mode='LPF12', cutoff_hz=200.0, resonance=0.0, drive=1.0):
        mode = getattr(LadderFilter, mode)
        board = Pedalboard([LadderFilter(mode=mode, cutoff_hz=cutoff_hz, resonance=resonance, drive=drive)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class LowShelfFilterEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "cutoff_frequency_hz": ("FLOAT", {'default': 440.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "gain_db": ("FLOAT", {'default': 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "q": ("FLOAT", {'default': 0.7071067690849304, "min": 0.0, "max": 100.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_low_shelf_filter"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/Filters"

    def apply_low_shelf_filter(self, audio, sample_rate, cutoff_frequency_hz=440.0, gain_db=0.0, q=0.7071067690849304):
        board = Pedalboard([LowShelfFilter(cutoff_frequency_hz=cutoff_frequency_hz, gain_db=gain_db, q=q)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class LowpassFilterEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "cutoff_frequency_hz": ("FLOAT", {'default': 50.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_lowpass_filter"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/Filters"

    def apply_lowpass_filter(self, audio, sample_rate, cutoff_frequency_hz=50.0):
        board = Pedalboard([LowpassFilter(cutoff_frequency_hz=cutoff_frequency_hz)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

class PeakFilterEffect():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "cutoff_frequency_hz": ("FLOAT", {'default': 440.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "gain_db": ("FLOAT", {'default': 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "q": ("FLOAT", {'default': 0.7071067690849304, "min": 0.0, "max": 100.0, "step": 0.1}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("🎙️audio", "sample_rate")
    FUNCTION = "apply_peak_filter"
    CATEGORY = "🎙️Jags_Audio/Pedalboard/Filters"

    def apply_peak_filter(self, audio, sample_rate, cutoff_frequency_hz=440.0, gain_db=0.0, q=0.7071067690849304):
        board = Pedalboard([PeakFilter(cutoff_frequency_hz=cutoff_frequency_hz, gain_db=gain_db, q=q)])
        total_tensor = apply_board_tensor(audio, sample_rate, board)
        return (total_tensor, sample_rate)

NODE_CLASS_MAPPINGS = {
    "OTTAudioFX": OTT,
    "LoadVST3": LoadVST,
    "BitCrushAudioFX": BitCrushEffect,
    "ChorusAudioFX": ChorusEffect,
    "ClippingAudioFX": ClippingEffect,
    "CompressorAudioFX": CompressorEffect,
    "ConvolutionAudioFX": ConvolutionEffect,
    "DelayAudioFX": DelayEffect,
    "DistortionAudioFX": DistortionEffect,
    "GainAudioFX": GainEffect,
    "InvertAudioFX": InvertEffect,
    "LimiterAudioFX": LimiterEffect,
    "MP3CompressorAudioFX": MP3CompressorEffect,
    "NoiseGateAudioFX": NoiseGateEffect,
    "PitchShiftAudioFX": PitchShiftEffect,
    "PhaserEffectAudioFX": PhaserEffect,
    "ReverbAudioFX": ReverbEffect,
    "HighShelfFilter": HighShelfFilterEffect,
    "HighpassFilter": HighpassFilterEffect,
    "LadderFilter": LadderFilterEffect,
    "LowShelfFilter": LowShelfFilterEffect,
    "LowpassFilter": LowpassFilterEffect,
    "PeakFilter": PeakFilterEffect
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTTAudioFX": "OTT",
    "LoadVST3": "Load VST3",
    "BitCrushAudioFX": "Bit Crush",
    "ChorusAudioFX": "Chorus",
    "ClippingAudioFX": "Clipping",
    "CompressorAudioFX": "Compressor",
    "ConvolutionAudioFX": "Convolution",
    "DelayAudioFX": "Delay",
    "DistortionAudioFX": "Distortion",
    "GainAudioFX": "Gain",
    "InvertAudioFX": "Invert",
    "LimiterAudioFX": "Limiter",
    "MP3CompressorAudioFX": "MP3 Compressor",
    "NoiseGateAudioFX": "Noise Gate",
    "PitchShiftAudioFX": "Pitch Shift",
    "PhaserEffectAudioFX": "Phaser",
    "ReverbAudioFX": "Reverb",
    "HighShelfFilter": "High Shelf Filter",
    "HighpassFilter": "Highpass Filter",
    "LadderFilter": "Ladder Filter",
    "LowShelfFilter": "Low Shelf Filter",
    "LowpassFilter": "Lowpass Filter",
    "PeakFilter": "Peak Filter"
}
