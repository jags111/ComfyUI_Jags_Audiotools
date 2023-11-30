# only import if running as a custom node
"""
@author: jags111
@title: Jags_Audiotools
@nickname: Audiotools
@description: This extension offers various audio generation tools
"""
import sys, os, shutil
import importlib
import utils
import folder_paths
import traceback

custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
SampleDiffusion_path = os.path.join(custom_nodes_path, "ComfyUI_Jags_Audiotools")
sys.path.append(ComfyUI_Jags_Audiotools_path)

# from .server import server

from AudioManipulation import JoinAudio, BatchJoinAudio, LayerAudio, CutAudio, DuplicateAudio, StretchAudio, ReverseAudio, ResampleAudio, SeparatePercussion
from PedalBoard import ( OTT, LoadVST, BitCrushEffect, ChorusEffect, ClippingEffect, CompressorEffect, ConvolutionEffect, DelayEffect, DistortionEffect,GainEffect, InvertEffect, LimiterEffect, MP3CompressorEffect, NoiseGateEffect, PitchShiftEffect, PhaserEffect, ReverbEffect, HighShelfFilterEffect, HighpassFilterEffect, LadderFilterEffect, LowShelfFilterEffect, LowpassFilterEffect, PeakFilterEffect  )
from SampleDiffusion import AudioInference, SaveAudio, LoadAudio, PreviewAudioFile, PreviewAudioTensor, StringListIndex, LoadAudioModelDD, MergeTensors, AudioIndex
from Spectrology import ImageToSpectral, Plot_Spectrogram
from VariationUtils import SliceAudio, BatchToList, LoadAudioDir, ListToBatch, ConcatAudioList, GetSingle, BulkVariation
from WaveGen import WaveGenerator
from AudioManipulation import JoinAudio, BatchJoinAudio, CutAudio, DuplicateAudio, StretchAudio, ReverseAudio, ResampleAudio
from VariationUtils import SliceAudio, BatchToList, ConcatAudioList, SequenceVariation

NODE_CLASS_MAPPINGS = {
    
 "JoinAudio": JoinAudio, 
 "BatchJoinAudio": BatchJoinAudio,
 "LayerAudio": LayerAudio,
 "CutAudio": CutAudio,
 "DuplicateAudio": DuplicateAudio,
 "StretchAudio": StretchAudio,
 "ReverseAudio": ReverseAudio,
 "ResampleAudio": ResampleAudio,
 "SeparatePercussion": SeparatePercussion,

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
  "LoadAudioDir": LoadAudioDir,
  "ListToBatch": ListToBatch,
  "ConcatAudioList": ConcatAudioList,
  "GetSingle": GetSingle,
  "BulkVariation": BulkVariation,
  "GenerateAudioWave": WaveGenerator,
  "SequenceVariation": SequenceVariation

}

NODE_DISPLAY_NAME_MAPPINGS = {
 "JoinAudio": "JoinAudio", 
 "BatchJoinAudio": "BatchJoinAudio",
 "LayerAudio": "LayerAudio",
 "CutAudio": "CutAudio",
 "DuplicateAudio": "DuplicateAudio",
 "StretchAudio": "StretchAudio",
 "ReverseAudio": "ReverseAudio",
 "ResampleAudio": "ResampleAudio",
 "SeparatePercussion": "SeparatePercussion",
 "OTTAudioFX": "OTT",
 "LoadVST3": "LoadVST",
 "BitCrushAudioFX": "BitCrushEffect",
 "ChorusAudioFX": "ChorusEffect",
 "ClippingAudioFX": "ClippingEffect",
 "CompressorAudioFX": "CompressorEffect",
 "ConvolutionAudioFX": "ConvolutionEffect",
 "DelayAudioFX": "DelayEffect",
 "DistortionAudioFX": "DistortionEffect",
 "GainAudioFX": "GainEffect",
 "InvertAudioFX": "InvertEffect",
 "LimiterAudioFX": "LimiterEffect",
 "MP3CompressorAudioFX": "MP3CompressorEffect",
 "NoiseGateAudioFX": "NoiseGateEffect",
 "PitchShiftAudioFX": "PitchShiftEffect",
 "PhaserEffectAudioFX": "PhaserEffect",
 "ReverbAudioFX": "ReverbEffect",
 "HighShelfFilter": "HighShelfFilterEffect",
 "HighpassFilter": "HighpassFilterEffect",
 "LadderFilter": "LadderFilterEffect",
 "LowShelfFilter": "LowShelfFilterEffect",
 "LowpassFilter": "LowpassFilterEffect",
 "PeakFilter": "PeakFilterEffect", 
  "GenerateAudioSample": "AudioInference",
  "SaveAudioTensor": "SaveAudio",
  "LoadAudioFile": "LoadAudio",
  "PreviewAudioFile": "PreviewAudioFile",
  "PreviewAudioTensor": "PreviewAudioTensor",
  "GetStringByIndex": "StringListIndex",
  "LoadAudioModel (DD)": "LoadAudioModelDD",
  "MixAudioTensors": "MergeTensors",
  "GetAudioFromFolderIndex": "AudioIndex", 
  "ImageToSpectral": "ImageToSpectral",
  "PlotSpectrogram": "Plot_Spectrogram",
  "SliceAudio": "SliceAudio",
  "BatchToList": "BatchToList",
  "LoadAudioDir": "LoadAudioDir",
  "ListToBatch": "ListToBatch",
  "ConcatAudioList": "ConcatAudioList",
  "GetSingle": "GetSingle",
  "BulkVariation": "BulkVariation",
  "GenerateAudioWave": "WaveGenerator",
  "SequenceVariation": "SequenceVariation" 
}
CC_VERSION = 1.0


# web ui feature
WEB_DIRECTORY = "js"

#print confirmation

print('--------------')
print('*ComfyUI_Jags_Audiotools- nodes_loaded*')
print('--------------')

#__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__ALL__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'CC_VERSION']

#      NODE_CLASS_MAPPINGS.update(node_import.NODE_CLASS_MAPPINGS)

