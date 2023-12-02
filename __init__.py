# only import if running as a custom node
"""
@author: jags111
@title: Jags_Audiotools
@nickname: Audiotools
@description: This extension offers various audio generation tools
"""
import sys, os, shutil
import importlib
import traceback
import json

import folder_paths

custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
ComfyUI_Jags_Audiotools_path = os.path.join(custom_nodes_path, "ComfyUI_Jags_Audiotools")
sys.path.append(ComfyUI_Jags_Audiotools_path)

#from server import server

from .AudioManipulation import *
from .PedalBoard import *
from .SampleDiffusion import *
from .Spectrology import *
from .VariationUtils import *
from .WaveGen import *

"""module_js_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")
application_root_directory = os.path.dirname(folder_paths.__file__)
extension_web_extensions_directory = os.path.join(application_root_directory, "web", "extensions", "comfyui_jags_audiotools")
shutil.copytree(module_js_directory, extension_web_extensions_directory, dirs_exist_ok=True)"""



NODE_CLASS_MAPPINGS = {
    "JoinAudio": JoinAudio, 
    "BatchJoinAudio": BatchJoinAudio,
    "CutAudio": CutAudio,
    "DuplicateAudio": DuplicateAudio,
    "StretchAudio": StretchAudio,
   "ReverseAudio": ReverseAudio,
   "ResampleAudio": ResampleAudio, 
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
   "PeakFilter": PeakFilterEffect, 
   "GenerateAudioSample": AudioInference,
   "SaveAudioTensor": SaveAudio,
   "LoadAudioFile": LoadAudio,
   "PreviewAudioFile": PreviewAudioFile,
   "PreviewAudioTensor": PreviewAudioTensor,
   "GetStringByIndex": StringListIndex,
   "LoadAudioModel (DD)": LoadAudioModelDD,
   "MixAudioTensors": MergeTensors,
   "GetAudioFromFolderIndex": AudioIndex, 
   "ImageToSpectral": ImageToSpectral,
   "PlotSpectrogram": Plot_Spectrogram,
   "SliceAudio": SliceAudio,
   "BatchToList": BatchToList,  
   "ConcatAudioList": ConcatAudioList,  
   "GenerateAudioWave": WaveGenerator,
   "SequenceVariation": SequenceVariation
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'JoinAudio': 'Jags_JoinAudio',
    'BatchJoinAudio': 'Jags_BatchJoinAudio',
    'CutAudio': 'Jags_CutAudio',
    'DuplicateAudio': 'Jags_DuplicateAudio',
    'StretchAudio': 'Jags_StretchAudio',
    'ReverseAudio': 'Jags_ReverseAudio',
    'ResampleAudio': 'Jags_ResampleAudio',
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
    "PeakFilter": "Peak Filter",
    "GenerateAudioSample": "Jags-AudioInference",
    "SaveAudioTensor": "Jags_SaveAudio",
    "LoadAudioFile": "Jags_LoadAudio",
    "PreviewAudioFile": "Jags_PreviewAudioFile",
    "PreviewAudioTensor": "Jags_PreviewAudioTensor",
    "GetStringByIndex": "Jags_StringListIndex",
    "LoadAudioModel (DD)": "Jags_LoadAudioModelDD",
    "MixAudioTensors": "Jags_MergeTensors",
    "GetAudioFromFolderIndex": "Jags_AudioIndex",
    'ImageToSpectral': 'Jags Image To Spectral',
    'PlotSpectrogram': 'Jags Plot Spectrogram',
    'SliceAudio': 'Jags_SliceAudio',
    'BatchToList': 'Jags_BatchToList',
    'LoadAudioDir': 'Jags_LoadAudioDir',
    'ListToBatch': 'Jags_ListToBatch',
    'ConcatAudioList': 'Jags_ConcatAudioList',
    'SequenceVariation': 'Jags_SequenceVariation',
    'GetSingle': 'Jags_GetSingle',
    'BulkVariation': 'Jags_BulkVariation',
     "GenerateAudioWave": "Jags_Wave Generator"
}
CC_VERSION = 1.0

THIS_DIR=os.path.dirname(os.path.abspath(__file__))
DIR_DEV_JS=os.path.abspath(f'{THIS_DIR}/js')
DIR_PY=os.path.abspath(f'{THIS_DIR}/py')
DIR_WEB_JS=os.path.abspath(f'{THIS_DIR}/../../web/extensions/comfyui_jags_audiotools')

if not os.path.exists(DIR_WEB_JS):
    os.makedirs(DIR_WEB_JS)

shutil.copytree(DIR_DEV_JS, DIR_WEB_JS, dirs_exist_ok=True)

# web ui feature
WEB_DIRECTORY = "js"

#print confirmation

print('--------------')
print('*ComfyUI_Jags_Audiotools- nodes_loaded*')
print('--------------')

#__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

__ALL__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'CC_VERSION']

"""for node in os.listdir(os.path.dirname(__file__)):
    if node.startswith('EXT_'):
        node = node.split('.')[0]
        node_import = importlib.import_module('custom_nodes.ComfyUI_Jags_Audiotools.' + node)
        # get class node mappings from py file
        NODE_CLASS_MAPPINGS.update(node_import.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(node_import.NODE_DISPLAY_NAME_MAPPINGS)"""
#      NODE_CLASS_MAPPINGS.update(node_import.NODE_CLASS_MAPPINGS)

