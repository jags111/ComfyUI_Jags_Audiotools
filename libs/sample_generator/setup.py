from setuptools import setup, find_packages

setup(
  name="sample-diffusion",
  version="0.0.1",
  description="",
  packages=find_packages(),
  install_requires=[
    "torch",
    "tqdm",
    "einops",
    "pandas",
    "prefigure", 
    "pytorch_lightning",
    "scipy",        
    "torchaudio",       
    "wandb",
  ]
)
