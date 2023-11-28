# only import if running as a custom node
import sys, os, shutil
import importlib
import utils
import folder_paths

custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
SampleDiffusion_path = os.path.join(custom_nodes_path, "SampleDiffusion")
sys.path.append(SampleDiffusion_path)

# from .server import server

from .EXT_AudioManipulation import JoinAudio, BatchJoinAudio, LayerAudio, CutAudio, DuplicateAudio, StretchAudio, ReverseAudio, ResampleAudio, SeparatePercussion
from .EXT_PedalBoard import ( OTT, LoadVST, BitCrushEffect, ChorusEffect, ClippingEffect, CompressorEffect, ConvolutionEffect, DelayEffect, DistortionEffect,GainEffect, InvertEffect, LimiterEffect, MP3CompressorEffect, NoiseGateEffect, PitchShiftEffect, PhaserEffect, ReverbEffect, HighShelfFilterEffect, HighpassFilterEffect, LadderFilterEffect, LowShelfFilterEffect, LowpassFilterEffect, PeakFilterEffect  )
from .EXT_SampleDiffusion import AudioInference, SaveAudio, LoadAudio, PreviewAudioFile, PreviewAudioTensor, StringListIndex, LoadAudioModelDD, MergeTensors, AudioIndex
from .EXT_Spectrology import ImageToSpectral, Plot_Spectrogram
from .EXT_VariationUtils import SliceAudio, BatchToList, LoadAudioDir, ListToBatch, ConcatAudioList, GetSingle, BulkVariation
from .EXT_WaveGen import WaveGenerator


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
  
  "GenerateAudioWave": WaveGenerator

}

NODE_DISPLAY_NAME_MAPPINGS = {
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
  
  "GenerateAudioWave": WaveGenerator
    
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

#for node in os.listdir(os.path.dirname(__file__)):
 #   if node.startswith('EXT_'):
 #       node = node.split('.')[0]
  #      node_import = importlib.import_module('custom_nodes.SampleDiffusion.', node)
        # get class node mappings from py file
  #      NODE_CLASS_MAPPINGS.update(node_import.NODE_CLASS_MAPPINGS)

