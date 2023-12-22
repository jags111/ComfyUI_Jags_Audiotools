# Imports

import subprocess, sys, os
import torch
import random
from pathlib import Path 
import importlib
import yaml
import pygame
import time
import tkinter as tk
from server import PromptServer
from aiohttp import web
from pydub import AudioSegment
from pygame import mixer


pygame.init()
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

import soundfile as sf
import torchaudio
# message handler
PromptServer.instance.app._client_max_size = 250 * 1024 * 1024 #  250 MB
dictionary_of_stuff = {"something":"Playing Sound"}
PromptServer.instance.send_sync("my-message-handle", dictionary_of_stuff)

#Play the audio file sound using pydub or others

class SoundPlayer:
    """
    This node provides an interface to play, pause, and stop audio files (WAV and MP3),
    and to load and save audio files from/to a directory.
    """
    def __init__(self):
        self.audio = None
        mixer.init()

    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "tensor": ("AUDIO", ),
                "audio_file_path": ("STRING", {"default": ""}),
                "volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            "optional": {
                "save_file_path": ("STRING", {"default": ""})  # Path to save the audio file
            }
        }

    RETURN_TYPES = ("PATH","AUDIO")
    RETRUN_NAMES = ("STRING", "tensor")
    FUNCTION = "play_audio"  # Change as per the primary function
    CATEGORY = "Jags_Audio/AudioHelpers"

    def load_audio(self, path):
        if not os.path.exists(path):
            print("Audio file not found.")
            return
        if path.lower().endswith('.mp3'):
            self.audio = AudioSegment.from_mp3(path)
        elif path.lower().endswith('.wav'):
            self.audio = AudioSegment.from_wav(path)
        else:
            print("Unsupported file format.")
            return
        self.audio.export("temp.wav", format="wav")
        mixer.music.load("temp.wav")

    def play_audio(self, audio_file_path, volume):
        self.load_audio(audio_file_path)
        if self.audio:
            mixer.music.set_volume(volume)
            mixer.music.play()

    def pause_audio(self):
        if self.audio:
            mixer.music.pause()

    def stop_audio(self):
        if self.audio:
            mixer.music.stop()
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")

    def save_audio(self, path):
        if self.audio:
            file_format = "mp3" if path.lower().endswith('.mp3') else "wav"
            self.audio.export(path, format=file_format)

    # Handle different file formats
        

    # Export to a format pygame can handle (WAV) and load
       

    # Set volume and play
    # Remove temporary file
        os.remove("temp.wav")

    # No return since this node performs an action
        return ()
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "SoundPlayer": "Jags_SoundPlayer",
    }
